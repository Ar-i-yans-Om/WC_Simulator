"""
data_sources.py
===============

Real, deterministic public-data helpers used by the Researcher and Bookmaker.

Nothing here is mocked:

* `HOST_CITIES_2026` are the *actual* 16 host cities of the 2026 World Cup with
  real lat/long geocoordinates and stadium altitudes. `travel_distance_km`
  computes the real great-circle (Haversine) distance between two of them.

* `compute_fitness_profile` turns rest days, travel distance, altitude, climate
  and host status into the Physiologist `fitness_degradation_factor` via a
  transparent, documented formula (no black boxes).

* The the-odds-api.com helpers (`odds_api_fixture_odds` and friends) fetch live
  H2H odds for a single fixture when an ODDS_API_KEY is configured; otherwise
  the Bookmaker falls back to Gemini with Google Search grounding.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Real 2026 FIFA World Cup host cities: (latitude, longitude, stadium altitude m)
# Mexico City (Estadio Azteca, ~2240 m) is the only true altitude venue.
# ---------------------------------------------------------------------------
HOST_CITIES_2026: Dict[str, Tuple[float, float, float]] = {
    # United States (11)
    "Atlanta": (33.7554, -84.4008, 320),
    "Boston": (42.0909, -71.2643, 12),          # Gillette Stadium, Foxborough
    "Dallas": (32.7473, -97.0945, 180),         # AT&T Stadium, Arlington
    "Houston": (29.6847, -95.4107, 15),
    "Kansas City": (39.0489, -94.4839, 270),
    "Los Angeles": (33.9535, -118.3392, 30),    # SoFi Stadium, Inglewood
    "Miami": (25.9580, -80.2389, 3),            # Hard Rock Stadium
    "New York": (40.8135, -74.0745, 5),         # MetLife, East Rutherford NJ
    "Philadelphia": (39.9008, -75.1675, 12),
    "San Francisco": (37.4030, -121.9700, 5),   # Levi's Stadium, Santa Clara
    "Seattle": (47.5952, -122.3316, 5),         # Lumen Field
    # Canada (2)
    "Toronto": (43.6332, -79.4185, 90),         # BMO Field
    "Vancouver": (49.2768, -123.1119, 3),       # BC Place
    # Mexico (3)
    "Mexico City": (19.3029, -99.1505, 2240),   # Estadio Azteca - high altitude
    "Guadalajara": (20.6819, -103.4628, 1560),
    "Monterrey": (25.6690, -100.2440, 510),
}


# ---------------------------------------------------------------------------
# Venue climate during the tournament window (mid-June .. mid-July).
# (typical local match-time air temperature °C, relative humidity %).
# Mexico City reads mild despite its latitude because ~2240 m altitude keeps it
# cool; the Gulf/Texas venues are the hot-and-humid extreme.
# ---------------------------------------------------------------------------
VENUE_CLIMATE_2026: Dict[str, Tuple[float, float]] = {
    # United States
    "Atlanta": (29.0, 68.0),
    "Boston": (26.0, 62.0),
    "Dallas": (34.0, 52.0),
    "Houston": (33.0, 74.0),
    "Kansas City": (31.0, 60.0),
    "Los Angeles": (27.0, 55.0),
    "Miami": (31.0, 75.0),
    "New York": (28.0, 64.0),
    "Philadelphia": (29.0, 63.0),
    "San Francisco": (21.0, 60.0),
    "Seattle": (23.0, 55.0),
    # Canada
    "Toronto": (26.0, 60.0),
    "Vancouver": (22.0, 60.0),
    # Mexico
    "Mexico City": (22.0, 50.0),     # altitude keeps it mild
    "Guadalajara": (26.0, 45.0),
    "Monterrey": (33.0, 58.0),
}
# Sensible neutral fallback for an unknown venue: warm-temperate, no penalty bias.
_DEFAULT_VENUE_CLIMATE = (26.0, 60.0)


# ---------------------------------------------------------------------------
# Team "home" climate, by archetype. The penalty is driven by how much HARSHER
# (hotter / more humid) the venue is than the team's usual environment, so we
# only need a representative home baseline per nation, not a full climatology.
#   (home temp °C, home humidity %, acclimatised_to_altitude)
# ---------------------------------------------------------------------------
_CLIMATE_ARCHETYPES: Dict[str, Tuple[float, float, bool]] = {
    "cool_temperate":  (18.0, 65.0, False),  # N/W Europe, Canada, NZ
    "warm_temperate":  (24.0, 58.0, False),  # S Europe, USA, Southern Cone
    "hot_humid":       (30.0, 75.0, False),  # tropical coast / Gulf / SE Asia
    "hot_arid":        (33.0, 30.0, False),  # N Africa, Gulf, desert
    "high_altitude":   (20.0, 55.0, True),   # Andean — also altitude-acclimatised
}

# Each 2026 participant → its home-climate archetype. These are deliberately
# coarse and fully tunable; refine any nation individually if needed.
TEAM_CLIMATE_ARCHETYPE: Dict[str, str] = {
    # cool / temperate Europe + Canada + NZ
    "England": "cool_temperate", "France": "cool_temperate",
    "Netherlands": "cool_temperate", "Belgium": "cool_temperate",
    "Germany": "cool_temperate", "Austria": "cool_temperate",
    "Switzerland": "cool_temperate", "Norway": "cool_temperate",
    "Sweden": "cool_temperate", "Scotland": "cool_temperate",
    "Czechia": "cool_temperate", "Bosnia and Herzegovina": "cool_temperate",
    "Canada": "cool_temperate", "New Zealand": "cool_temperate",
    # warm-temperate (S Europe, USA, Southern Cone, etc.)
    "Spain": "warm_temperate", "Portugal": "warm_temperate",
    "Croatia": "warm_temperate", "Türkiye": "warm_temperate",
    "United States": "warm_temperate", "Argentina": "warm_temperate",
    "Uruguay": "warm_temperate", "Paraguay": "warm_temperate",
    "Japan": "warm_temperate", "Korea Republic": "warm_temperate",
    "Uzbekistan": "warm_temperate", "South Africa": "warm_temperate",
    # hot & humid (tropical / Gulf-coast / coastal W Africa / Caribbean)
    "Brazil": "hot_humid", "Senegal": "hot_humid", "Ivory Coast": "hot_humid",
    "Ghana": "hot_humid", "DR Congo": "hot_humid", "Cape Verde": "hot_humid",
    "Panama": "hot_humid", "Curaçao": "hot_humid", "Haiti": "hot_humid",
    "Australia": "hot_humid",
    # hot & arid (N Africa / Gulf / desert)
    "Morocco": "hot_arid", "Algeria": "hot_arid", "Egypt": "hot_arid",
    "Tunisia": "hot_arid", "Saudi Arabia": "hot_arid", "Iran": "hot_arid",
    "Iraq": "hot_arid", "Jordan": "hot_arid", "Qatar": "hot_arid",
    # Andean high-altitude (also altitude-acclimatised)
    "Ecuador": "high_altitude", "Colombia": "high_altitude",
    "Mexico": "high_altitude",
}
_DEFAULT_ARCHETYPE = "warm_temperate"


def venue_climate(city: str) -> Tuple[float, float]:
    """(temperature °C, humidity %) at `city` during the tournament window."""
    return VENUE_CLIMATE_2026.get(city, _DEFAULT_VENUE_CLIMATE)


def team_home_climate(team_name: str) -> Tuple[float, float, bool]:
    """(home temp °C, home humidity %, altitude_acclimatised) for a team."""
    arche = TEAM_CLIMATE_ARCHETYPE.get(team_name, _DEFAULT_ARCHETYPE)
    return _CLIMATE_ARCHETYPES.get(arche, _CLIMATE_ARCHETYPES[_DEFAULT_ARCHETYPE])



# ---------------------------------------------------------------------------
# City name aliases — maps schedule/venue city names to canonical host cities
# ---------------------------------------------------------------------------
# The fixtures schedule uses local borough/suburb names (e.g. "Inglewood" for
# SoFi Stadium) while our Haversine dict uses the major city name. This mapping
# normalises them transparently so match_runner.py can use either form.
CITY_ALIASES: dict = {
    # USA suburbs → major city
    "Inglewood":       "Los Angeles",   # SoFi Stadium
    "East Rutherford": "New York",      # MetLife Stadium
    "Foxboro":         "Boston",        # Gillette Stadium
    "Arlington":       "Dallas",        # AT&T Stadium
    "Santa Clara":     "San Francisco", # Levi's Stadium
    # Stadium names that some schedule sources put in the city field
    "AT&T Stadium":    "Dallas",
    "BMO Field":       "Toronto",
    "Arrowhead Stadium": "Kansas City",
    "SoFi Stadium":    "Los Angeles",
    "Lumen Field":     "Seattle",
    "MetLife Stadium": "New York",
    "NRG Stadium":     "Houston",
    "BC Place":        "Vancouver",
    # Identity mappings (already canonical)
    "Los Angeles":     "Los Angeles",
    "New York":        "New York",
    "Dallas":          "Dallas",
    "Houston":         "Houston",
    "Atlanta":         "Atlanta",
    "Miami":           "Miami",
    "Seattle":         "Seattle",
    "Philadelphia":    "Philadelphia",
    "Kansas City":     "Kansas City",
    "San Francisco":   "San Francisco",
    "Boston":          "Boston",
    "Toronto":         "Toronto",
    "Vancouver":       "Vancouver",
    "Mexico City":     "Mexico City",
    "Guadalajara":     "Guadalajara",
    "Monterrey":       "Monterrey",
}


def resolve_city(raw: str) -> str:
    """Return the canonical host-city name for a raw schedule city string."""
    return CITY_ALIASES.get(raw, raw)


def travel_distance_km(city_a: str, city_b: str) -> float:
    """Great-circle (Haversine) distance in km between two 2026 host cities."""
    if city_a == city_b:
        return 0.0
    if city_a not in HOST_CITIES_2026 or city_b not in HOST_CITIES_2026:
        return 0.0  # unknown city -> no penalty rather than a guess
    lat1, lon1, _ = HOST_CITIES_2026[city_a]
    lat2, lon2, _ = HOST_CITIES_2026[city_b]
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 1)


def venue_altitude_m(city: str) -> float:
    return HOST_CITIES_2026.get(city, (0.0, 0.0, 0.0))[2]


# ---------------------------------------------------------------------------
# The Physiologist Multiplier
# ---------------------------------------------------------------------------
# fitness_degradation_factor = clamp(1.0 - rest_pen - travel_pen - climate_pen
#                                        - unfamiliar_pen + host_bonus, 0.80, 1.06)
#
#   rest_pen       : each day of rest below a 4-day baseline costs 3% (capped 12%).
#   travel_pen     : up to 6% scaling linearly to ~8000 km of travel.
#   climate_pen    : altitude > 1500 m for a non-acclimatised team costs up to 5%.
#   unfamiliar_pen : a venue hotter / more humid than home costs up to 4%.
#   host_bonus     : +4% for the host nations (crowd / familiarity).
# ---------------------------------------------------------------------------
REST_BASELINE_DAYS = 4
REST_PENALTY_PER_DAY = 0.03
REST_PENALTY_CAP = 0.12
TRAVEL_REFERENCE_KM = 8000.0
TRAVEL_PENALTY_CAP = 0.06
ALTITUDE_THRESHOLD_M = 1500.0
ALTITUDE_PENALTY_CAP = 0.05
HOST_BONUS = 0.04

# Climate unfamiliarity: a team pays a (small) penalty only when the venue is
# HOTTER and/or MORE HUMID than its home baseline — moving to harsher conditions
# saps a side that is not used to it (e.g. England in a hot, humid US venue).
# Moving to a milder venue is never penalised. Deliberately gentle and capped.
CLIMATE_TEMP_PENALTY_PER_C    = 0.0035   # per °C the venue is hotter than home
CLIMATE_HUMIDITY_PENALTY_PER_PCT = 0.0009  # per %RH the venue is more humid
CLIMATE_UNFAMILIAR_CAP        = 0.04     # hard ceiling on the climate penalty


def compute_fitness_profile(
    rest_days: int,
    travel_km: float,
    altitude_m: float,
    is_host_nation: bool,
    venue_temp_c: float | None = None,
    venue_humidity: float | None = None,
    home_temp_c: float | None = None,
    home_humidity: float | None = None,
    acclimatised_to_altitude: bool = False,
) -> dict:
    """
    The Physiologist profile as a plain dict. `fitness_degradation_factor` is a
    multiplier on a team's baseline lambda: < 1.0 degraded (rest, travel,
    altitude, climate), 1.0 neutral, > 1.0 boosted (host nation).
    """
    rest_penalty = min(
        REST_PENALTY_CAP,
        max(0, REST_BASELINE_DAYS - rest_days) * REST_PENALTY_PER_DAY,
    )
    travel_penalty = min(TRAVEL_PENALTY_CAP, (travel_km / TRAVEL_REFERENCE_KM) * TRAVEL_PENALTY_CAP)

    climate_penalty = 0.0
    climate_note = "sea-level / temperate venue"
    if altitude_m >= ALTITUDE_THRESHOLD_M and not acclimatised_to_altitude:
        over = (altitude_m - ALTITUDE_THRESHOLD_M) / 1000.0
        climate_penalty = min(ALTITUDE_PENALTY_CAP, 0.025 + 0.02 * over)
        climate_note = f"high-altitude venue (~{int(altitude_m)} m)"
    elif altitude_m >= ALTITUDE_THRESHOLD_M and acclimatised_to_altitude:
        climate_note = f"high-altitude venue (~{int(altitude_m)} m, acclimatised)"

    # Heat / humidity unfamiliarity — only penalised when the venue is HARSHER
    # (hotter and/or more humid) than the team's home baseline.
    climate_unfamiliar_penalty = 0.0
    if None not in (venue_temp_c, venue_humidity, home_temp_c, home_humidity):
        heat_excess  = max(0.0, venue_temp_c - home_temp_c)
        humid_excess = max(0.0, venue_humidity - home_humidity)
        climate_unfamiliar_penalty = min(
            CLIMATE_UNFAMILIAR_CAP,
            heat_excess * CLIMATE_TEMP_PENALTY_PER_C
            + humid_excess * CLIMATE_HUMIDITY_PENALTY_PER_PCT,
        )
        if climate_unfamiliar_penalty > 0:
            climate_note += (
                f"; unfamiliar climate (venue {venue_temp_c:.0f}°C/{venue_humidity:.0f}% "
                f"vs home {home_temp_c:.0f}°C/{home_humidity:.0f}%)"
            )

    host_bonus = HOST_BONUS if is_host_nation else 0.0

    factor = (1.0 - rest_penalty - travel_penalty - climate_penalty
              - climate_unfamiliar_penalty + host_bonus)
    factor = max(0.80, min(1.06, factor))

    return {
        "rest_days": rest_days,
        "travel_km": travel_km,
        "is_host_nation": is_host_nation,
        "altitude_m": altitude_m,
        "climate_note": climate_note,
        "rest_penalty": round(rest_penalty, 4),
        "travel_penalty": round(travel_penalty, 4),
        "climate_penalty": round(climate_penalty, 4),
        "climate_unfamiliar_penalty": round(climate_unfamiliar_penalty, 4),
        "acclimatised_to_altitude": acclimatised_to_altitude,
        "host_bonus": round(host_bonus, 4),
        "fitness_degradation_factor": round(factor, 4),
    }


# ---------------------------------------------------------------------------
# Team-name aliasing: our canonical names vs the spellings the odds API uses.
# Only nations whose API spelling differs from ours need an entry.
# ---------------------------------------------------------------------------
TEAM_NAME_ALIASES: Dict[str, set] = {
    "United States": {"usa", "united states", "united states of america"},
    "Korea Republic": {"south korea", "korea republic", "republic of korea",
                       "korea, republic of"},
    "Türkiye": {"turkey", "turkiye", "türkiye"},
    "Czechia": {"czech republic", "czechia"},
    "Ivory Coast": {"ivory coast", "cote d'ivoire", "côte d'ivoire"},
    "DR Congo": {"dr congo", "democratic republic of the congo", "congo dr"},
    "Cape Verde": {"cape verde", "cabo verde"},
    "Bosnia and Herzegovina": {"bosnia and herzegovina", "bosnia & herzegovina",
                               "bosnia-herzegovina"},
    "Curaçao": {"curacao", "curaçao"},
    "Iran": {"iran", "ir iran", "iran islamic republic"},
}


def _norm_name(s: str) -> str:
    return (s or "").strip().lower()


def _alias_set(name: str) -> set:
    base = _norm_name(name)
    for canon, variants in TEAM_NAME_ALIASES.items():
        if base == _norm_name(canon) or base in {v.lower() for v in variants}:
            return {v.lower() for v in variants} | {_norm_name(canon), base}
    return {base}


def team_names_match(our_name: str, api_name: str) -> bool:
    """True if `api_name` (the odds API's spelling) refers to `our_name`."""
    api = _norm_name(api_name)
    if api in _alias_set(our_name):
        return True
    # Guarded substring fallback for unlisted spellings; min length 5 avoids
    # short-name collisions (e.g. Iran/Iraq).
    ours = _norm_name(our_name)
    return len(api) >= 5 and (api in ours or ours in api)


def _commence_window_utc(date_iso: str) -> Tuple[str, str]:
    """±1-day UTC window around an ISO date, to absorb timezone/kickoff roll."""
    from datetime import datetime, timedelta, timezone
    d = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lo = (d - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    hi = (d + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")
    return lo, hi


def odds_api_find_event(
    api_key: str,
    team_a: str,
    team_b: str,
    date: Optional[str] = None,
    sport: str = "soccer_fifa_world_cup",
) -> Optional[dict]:
    """
    Resolve a single fixture to its odds-API event via the FREE /events endpoint
    (costs 0 against the usage quota). Optionally narrows by an ISO match date.
    Returns the matching event dict ({id, home_team, away_team, ...}) or None.
    """
    try:
        import requests

        url = f"https://api.the-odds-api.com/v4/sports/{sport}/events"
        params = {"apiKey": api_key}
        if date:
            lo, hi = _commence_window_utc(date)
            params["commenceTimeFrom"] = lo
            params["commenceTimeTo"] = hi
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        for ev in r.json() or []:
            pair = (ev.get("home_team", ""), ev.get("away_team", ""))
            a_ok = any(team_names_match(team_a, n) for n in pair)
            b_ok = any(team_names_match(team_b, n) for n in pair)
            if a_ok and b_ok:
                return ev
        return None
    except Exception:
        return None


def odds_api_event_odds(
    api_key: str,
    event_id: str,
    regions: str = "uk,eu",
    sport: str = "soccer_fifa_world_cup",
) -> Optional[dict]:
    """
    Fetch H2H odds for ONE event (the match in question). Returns the single
    event dict with its bookmakers, or None. Costs 1 quota per region (h2h only).
    """
    try:
        import requests

        url = f"https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
        params = {"apiKey": api_key, "regions": regions,
                  "markets": "h2h", "oddsFormat": "decimal"}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def odds_api_fixture_odds(
    api_key: str,
    team_a: str,
    team_b: str,
    date: Optional[str] = None,
    regions: str = "uk,eu",
    sport: str = "soccer_fifa_world_cup",
) -> Optional[dict]:
    """
    One-shot helper: find the fixture's event id (free) then pull only that
    event's H2H odds. Returns a single event dict, or None if the fixture isn't
    listed. This is what the Bookmaker should call — no whole-slate download.
    """
    ev = odds_api_find_event(api_key, team_a, team_b, date=date, sport=sport)
    if not ev or not ev.get("id"):
        return None
    return odds_api_event_odds(api_key, ev["id"], regions=regions, sport=sport)


def implied_probs_from_decimal(home: float, draw: float, away: float) -> dict:
    """Convert decimal odds to overround-normalised implied probabilities."""
    inv = [1.0 / home, 1.0 / draw, 1.0 / away]
    overround = sum(inv) - 1.0
    total = sum(inv)
    return {
        "implied_home_win": round(inv[0] / total, 4),
        "implied_draw": round(inv[1] / total, 4),
        "implied_away_win": round(inv[2] / total, 4),
        "overround": round(overround, 4),
    }