"""
llm.py
======

Wrapper around the Google Gen AI SDK (google-genai >= 1.0).

Two public methods used by every agent:

1. `structured(schema, system, user, context)` -> Pydantic model instance
   Uses response_mime_type="application/json" so the model emits only JSON.
   The Pydantic model is passed as a native `response_schema` whenever Gemini
   can represent it; otherwise the JSON Schema is embedded in the prompt.

2. `research(system, user)` -> str
   Attaches Google Search grounding so the model pulls real current web data.
   Falls back to an ungrounded call if grounding fails OR returns empty.

DIAGNOSTICS
-----------
Every call logs, at INFO, whether the API was hit and whether usable data came
back; on failure the real error is logged, never swallowed.

RESILIENCE: each call runs through a retry/backoff loop, a multi-key failover
loop, AND a model-level capacity fallback. Transient server errors (5xx /
network) back off exponentially and retry the same key; auth/quota errors (401 /
403 / 429) fail over IMMEDIATELY to the next API key in the pool (no wasted
backoff); capacity-overload errors (503 / "overloaded" / UNAVAILABLE) switch to
the next MODEL in the fallback chain (a separate capacity pool) or, when no chain
is configured, rotate straight to the next key; fatal request errors (400 /
404 / 405 — bad prompt or wrong model) abort at once since no key can fix them.
Supply several keys via GEMINI_API_KEYS (comma/space/newline separated) to spread
free-tier quota across independent buckets, and a fallback chain via
FIFA_FALLBACK_MODELS. FIFA_LLM_STRICT defaults ON, so once every key/retry/model
is exhausted the call RAISES (aborting the fixture) instead of soft-failing to
schema-default placeholders; set FIFA_LLM_STRICT=0 to opt back out.

GROUNDING ON REASONING MODELS (Gemini 2.5 / 3.x)
------------------------------------------------
These are thinking models: internal reasoning tokens count against
max_output_tokens, so a small cap can leave the visible answer EMPTY
(finish_reason=MAX_TOKENS) — this shows up most on grounded research calls,
which carry extra search overhead. Two mitigations here:
  * research() uses a generous token floor (>=8192) so thinking + answer fit.
  * research() falls back to an UNGROUNDED call when grounding fails *or*
    returns empty text, and returns whatever the model produces.
Set FIFA_DISABLE_GROUNDING=1 to skip the grounded attempt entirely (use the
model's own knowledge), e.g. if your key/tier can't use the googleSearch tool.

The same thinking-token trap applies to prose-heavy STRUCTURED calls (e.g. the
Judge's JudgeProse, which must write four prose fields). structured() therefore
applies its own token floor (_STRUCTURED_TOKEN_FLOOR) and warns when a parse
succeeds but every content field comes back empty.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# HTTP status codes that will never succeed when retried on the same key.
_PERMANENT_CODES = {400, 401, 403, 404, 405}
_PREVIEW = 600  # chars of fetched data to echo into the log at INFO
_RESEARCH_TOKEN_FLOOR = 8192    # reasoning models need headroom for thinking+answer
_STRUCTURED_TOKEN_FLOOR = 4096  # prose-heavy structured calls starve without headroom
_STRUCTURED_EMPTY_RETRIES = 2   # re-rolls when a parse succeeds but is all-empty
# Re-rolls when a grounded call returns text but the model never actually searched
# (no grounding_metadata). Each re-roll rotates the key + adds a hard search nudge.
_GROUNDING_RETRIES = int(os.getenv("FIFA_GROUNDING_RETRIES", "2"))

# Native constrained decoding: hand Gemini the Pydantic model as `response_schema`
# so the output is STRUCTURALLY guaranteed (kills most parse failures + much of the
# all-empty/vacuous problem). Auto-skipped for any schema that uses an arbitrary-key
# Dict field (Gemini can't represent those). Disable entirely with FIFA_RESPONSE_SCHEMA=0.
_RESPONSE_SCHEMA = os.getenv("FIFA_RESPONSE_SCHEMA", "1").strip().lower() in {"1", "true", "yes", "on"}

# Explicit thinking budget (tokens the model may spend on internal reasoning).
#   -1  → leave it to the model default (no thinking_config sent; the safe default)
#    0  → disable thinking (fastest/cheapest; good for pure extraction)
#  N>0  → cap thinking at N tokens
# Opt-in via FIFA_THINKING_BUDGET. Guarded: if the SDK/model rejects it, it's ignored.
_THINKING_BUDGET = int(os.getenv("FIFA_THINKING_BUDGET", "-1"))


def _response_schema_ok(model) -> bool:
    """False if the schema uses an arbitrary-key map (Dict[str, X] → JSON
    'additionalProperties'), which Gemini's native response_schema can't represent.
    Such schemas fall back to the prompt-embedded-JSON path."""
    try:
        return '"additionalProperties"' not in json.dumps(model.model_json_schema())
    except Exception:
        return False


def _thinking_config(types):
    """Build a ThinkingConfig from FIFA_THINKING_BUDGET, or None to leave the model
    default. Returns None (rather than raising) if the SDK lacks ThinkingConfig."""
    if _THINKING_BUDGET < 0:
        return None
    try:
        return types.ThinkingConfig(thinking_budget=_THINKING_BUDGET)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Per-run QUALITY LEDGER                                                       #
# --------------------------------------------------------------------------- #
# Records every time a call settles for a SUB-OPTIMAL route (ungrounded research,
# a vacuous structured parse, or a soft-failed default) so the runner can stamp
# each fixture's prediction with whether it was clean — and re-run the dirty ones
# later. Fixtures run sequentially, so a module-level ledger reset per run is
# sufficient (not built for concurrent runs).
_QUALITY_EVENTS: list = []


def reset_run_quality() -> None:
    _QUALITY_EVENTS.clear()


def record_quality(call: str, issue: str, detail: str = "") -> None:
    _QUALITY_EVENTS.append({"call": call, "issue": issue, "detail": detail})


def run_quality_summary() -> dict:
    """Summarise the current run's degradations for persistence in predictions.json."""
    events = list(_QUALITY_EVENTS)
    grounded = not any(e["issue"].startswith("ungrounded") for e in events)
    return {
        "ok": len(events) == 0,        # True == every call took the optimal route
        "grounded": grounded,          # True == live web data backed the research
        "n_issues": len(events),
        "issues": events,
    }

def _is_vacuous(dumped: dict) -> bool:
    """True if a parsed model has content but every content field is empty.
    Ignores 'confidence' (a Literal that always has a non-empty default), so a
    genuine all-blank generation is detected while a normal report is not."""
    content_vals = [v for k, v in dumped.items() if k != "confidence"]
    return bool(content_vals) and all(v in ("", {}, [], None) for v in content_vals)


# --------------------------------------------------------------------------- #
# Response inspection helpers (the SDK's response.text can raise / be empty)   #
# --------------------------------------------------------------------------- #
def _describe_response(response) -> str:
    """A compact, log-friendly summary of what the API actually returned."""
    bits = []
    try:
        cands = getattr(response, "candidates", None) or []
        bits.append(f"candidates={len(cands)}")
        if cands:
            fr = getattr(cands[0], "finish_reason", None)
            bits.append(f"finish={getattr(fr, 'name', fr)}")
    except Exception:
        pass
    try:
        pf = getattr(response, "prompt_feedback", None)
        br = getattr(pf, "block_reason", None) if pf else None
        if br:
            bits.append(f"BLOCKED={getattr(br, 'name', br)}")
    except Exception:
        pass
    try:
        um = getattr(response, "usage_metadata", None)
        if um:
            bits.append(
                f"tokens p={getattr(um, 'prompt_token_count', '?')}/"
                f"c={getattr(um, 'candidates_token_count', '?')}/"
                f"think={getattr(um, 'thoughts_token_count', '?')}"
            )
    except Exception:
        pass
    return " | ".join(bits) or "no-metadata"


def _extract_text(response) -> str:
    """
    Pull text out of a response robustly. `response.text` is a property that
    raises when the candidate finished for a non-text reason (safety block,
    MAX_TOKENS with no text, tool-only turn), so we guard it and fall back to
    walking the candidate parts (skipping thought parts).
    """
    try:
        t = response.text
        if t:
            return t
    except Exception:
        pass
    chunks = []
    for cand in (getattr(response, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                continue  # skip internal thinking parts
            txt = getattr(part, "text", None)
            if txt:
                chunks.append(txt)
    return "".join(chunks)


def _has_grounding(response) -> bool:
    """True if the response carries real Google Search grounding metadata.

    A grounded model that *chooses not to search* still returns text but attaches
    no grounding_metadata — so non-empty text alone does NOT prove a live search
    happened. We inspect the metadata so the Researcher can tell live web data
    apart from the model answering purely from its (possibly stale) knowledge.
    """
    try:
        for cand in (getattr(response, "candidates", None) or []):
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            for attr in ("grounding_chunks", "grounding_supports",
                         "web_search_queries", "search_entry_point"):
                if getattr(gm, attr, None):
                    return True
    except Exception:
        pass
    return False


def _error_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of the HTTP status code from an SDK exception."""
    for attr in ("code", "status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    msg = str(exc)
    for code in _PERMANENT_CODES | {429, 500, 502, 503, 504}:
        if str(code) in msg:
            return code
    return None


# Request/model is wrong — no other key and no amount of retrying will fix it.
_FATAL_REQUEST_CODES = {400, 404, 405}
# The key itself is the problem (invalid / forbidden / out of quota) — the right
# move is to FAIL OVER to a different API key rather than retry the same one.
_KEY_RELATED_CODES = {401, 403, 429}
_QUOTA_HINTS = (
    "quota", "rate limit", "rate-limit", "resource_exhausted", "exhausted",
    "permission denied", "api key not valid", "invalid api key", "api_key_invalid",
)


def _is_fatal_request(exc: Exception) -> bool:
    return _error_code(exc) in _FATAL_REQUEST_CODES


def _is_key_related(exc: Exception) -> bool:
    """True if the failure is specific to this API key (auth/quota), so rotating
    to another key is the sensible recovery."""
    if _error_code(exc) in _KEY_RELATED_CODES:
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _QUOTA_HINTS)


# The MODEL is overloaded server-side (transient capacity, not this key's quota).
# Neither retrying the same model nor rotating keys helps — switch to a different
# model (a separate capacity pool) via the fallback chain.
_OVERLOAD_CODES = {503}
_OVERLOAD_HINTS = ("overloaded", "unavailable", "try again later", "model is overloaded")


def _is_overloaded(exc: Exception) -> bool:
    if _error_code(exc) in _OVERLOAD_CODES:
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _OVERLOAD_HINTS)


class LLM:
    def __init__(
        self,
        api_key=None,
        model: str = "",
        max_tokens: int = 3000,
        max_retries: int = 3,
        strict: bool = False,
        grounding_enabled: bool = False,
        api_keys: Optional[list] = None,
        fallback_models: Optional[list] = None,
    ):
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.strict = strict
        self.grounding_enabled = grounding_enabled
        self.tag = f"[LLM:{model}]"

        # Ordered models tried per call: the primary first, then every distinct
        # fallback (skipping the primary). On a 503/overload error _generate steps
        # to the next entry — a separate server capacity pool. De-duped, order kept.
        self.models: list[str] = [model]
        for m in (fallback_models or []):
            m = (m or "").strip()
            if m and m not in self.models:
                self.models.append(m)
        if len(self.models) > 1:
            logger.info("%s capacity-fallback chain: %s",
                        self.tag, " → ".join(self.models))

        # Accept either a single `api_key` (str) or an ordered `api_keys` pool.
        # The pool is the failover sequence: on an auth/quota error we rotate to
        # the next key. De-duplicate while preserving order.
        pool: list[str] = []
        for k in ([api_key] if isinstance(api_key, str) else (api_key or [])) + list(api_keys or []):
            k = (k or "").strip()
            if k and k not in pool:
                pool.append(k)

        # One SDK client per key. `_key_idx` round-robins across calls so load is
        # spread, and advances on failover within a call.
        self._clients: list = []
        self._key_labels: list = []
        self._key_idx = 0

        if pool:
            from google import genai
            for i, k in enumerate(pool):
                self._clients.append(genai.Client(api_key=k))
                self._key_labels.append(f"key#{i + 1}(…{k[-4:]})")
            logger.info("%s client initialised with %d API key(s) for failover.",
                        self.tag, len(self._clients))
        else:
            logger.warning("%s NO API KEY — client not built; calls will fail.", self.tag)

    def _rotate_key(self) -> None:
        if len(self._clients) > 1:
            self._key_idx = (self._key_idx + 1) % len(self._clients)

    # ------------------------------------------------------------------ #
    # Internal: a single guarded generate_content call with backoff       #
    # ------------------------------------------------------------------ #
    def _generate(self, *, contents, config, what: str):
        """
        Run generate_content with retries/backoff, multi-key failover, AND a
        model-level capacity fallback.

        Policy per call:
          • Try each model in `self.models` in order (primary, then fallbacks).
          • For each model, do one pass over the key pool (starting at the current
            round-robin key).
          • On a transient server error (5xx / network), back off exponentially
            and retry the SAME key up to `max_retries` times.
          • On an auth/quota error (401/403/429 or a quota message), do NOT waste
            time backing off — rotate immediately to the next key.
          • On a capacity-overload error (503 / "overloaded" / UNAVAILABLE),
            abandon this model entirely (no key or backoff fixes server capacity)
            and jump to the next model in the chain — a separate capacity pool.
          • On a fatal request error (400/404/405 — bad prompt or wrong model),
            stop: no key, retry, or model swap can fix it.
        Returns the SDK response on success, or raises the last exception. All
        outcomes are logged with which key + model were used.
        """
        if not self._clients:
            raise RuntimeError(f"{self.tag} no client (missing GEMINI_API_KEY)")

        last_err: Optional[Exception] = None
        num_keys = len(self._clients)
        num_models = len(self.models)

        for model_try, model in enumerate(self.models):
            if model_try > 0:
                logger.warning("%s ↳ %s: model '%s' overloaded — falling back to "
                               "model '%s' (%d/%d).", self.tag, what,
                               self.models[model_try - 1], model,
                               model_try + 1, num_models)
            overloaded = False

            for key_try in range(num_keys):             # one pass over the pool
                idx = self._key_idx
                client = self._clients[idx]
                label = self._key_labels[idx]
                rotate_after = False

                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = client.models.generate_content(
                            model=model, contents=contents, config=config,
                        )
                        logger.info("%s ← API responded for %s via %s [model=%s] | %s",
                                    self.tag, what, label, model,
                                    _describe_response(response))
                        return response
                    except Exception as exc:  # noqa: BLE001 - re-raised after logging
                        last_err = exc
                        code = _error_code(exc)
                        logger.error("%s ✗ %s API error on %s [model=%s] (key %d/%d, "
                                     "attempt %d/%d, code=%s): %s: %s",
                                     self.tag, what, label, model, key_try + 1, num_keys,
                                     attempt, self.max_retries, code,
                                     type(exc).__name__, str(exc)[:300])
                        logger.debug("%s full traceback for %s:", self.tag, what, exc_info=True)

                        if _is_fatal_request(exc):
                            logger.error("%s ✗ %s is a PERMANENT request error (code=%s) — "
                                         "no key, retry, or model swap will help. Check the "
                                         "prompt and that model '%s' exists/is enabled.",
                                         self.tag, what, code, model)
                            raise

                        # Server capacity overload (503). No key/backoff fixes the
                        # model itself, so prefer to move OFF this overloaded path:
                        #   1) jump to the next fallback MODEL (separate capacity
                        #      pool), if one is configured; else
                        #   2) rotate to the next KEY immediately — empirically a
                        #      different key's backend often serves the same model
                        #      fine during a demand spike (no wasted backoff).
                        if _is_overloaded(exc):
                            if model_try < num_models - 1:
                                logger.warning("%s … model '%s' is OVERLOADED (code=%s); "
                                               "switching to the next fallback model.",
                                               self.tag, model, code)
                                overloaded = True
                                break
                            if num_keys > 1:
                                logger.warning("%s … model '%s' is OVERLOADED (code=%s) and no "
                                               "fallback model left; rotating to the next key.",
                                               self.tag, model, code)
                                rotate_after = True
                                break

                        if num_keys > 1 and _is_key_related(exc):
                            logger.warning("%s … key %s is auth/quota-limited (code=%s); "
                                           "failing over to the next key.",
                                           self.tag, label, code)
                            rotate_after = True
                            break  # stop retrying this key; outer loop rotates

                        if attempt < self.max_retries:
                            backoff = min(30.0, 4.0 ** attempt)
                            logger.warning("%s … retrying %s on %s in %.1fs",
                                           self.tag, what, label, backoff)
                            time.sleep(backoff)
                        else:
                            rotate_after = True  # exhausted this key → try the next

                if overloaded:
                    break  # leave the key loop → outer model loop tries fallback

                if rotate_after and num_keys > 1 and key_try < num_keys - 1:
                    self._rotate_key()
                    continue
                break  # single key, or nothing left to fail over to

            if overloaded:
                continue  # try the next model in the chain
            break  # returned already, or no fallback model left to try

        raise last_err if last_err else RuntimeError(f"{self.tag} {what} failed")

    # ------------------------------------------------------------------ #
    # Structured output via JSON mode                                     #
    # ------------------------------------------------------------------ #
    def structured(self, schema: Type[T], system: str, user: str, context: str = "") -> T:
        from google.genai import types

        what = f"structured({schema.__name__})"
        # Native constrained decoding when the schema is representable (no Dict
        # fields) and not disabled. Otherwise embed the JSON schema in the prompt.
        use_native = _RESPONSE_SCHEMA and _response_schema_ok(schema)

        parts = [f"SYSTEM INSTRUCTION:\n{system}\n"]
        if context:
            parts.append(f"REFERENCE DATA:\n{context}\n")
        parts.append(f"TASK:\n{user}")
        if use_native:
            # The schema is enforced by the API; just remind it to emit raw JSON.
            parts.append(
                "\nRESPONSE FORMAT:\nReturn ONLY a single JSON object conforming to "
                "the response schema. Raw JSON — no markdown, no commentary, no extra keys."
            )
        else:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            parts.append(
                "\nRESPONSE FORMAT:\n"
                "You MUST respond with ONLY a single valid JSON object matching "
                "the schema below exactly. Raw JSON only — no markdown fences, "
                "no explanation, no extra keys, no trailing text.\n\n" + schema_json
            )
        prompt = "\n".join(parts)
        logger.info("%s → %s …%s", self.tag, what, " [native schema]" if use_native else "")

        # Reasoning models spend output tokens on internal thinking; a small cap
        # can leave the visible JSON empty (the trap research() guards against).
        # Apply a structured-output floor so prose-heavy schemas (e.g.
        # JudgeProse) have room for thinking + answer.
        tokens = max(self.max_tokens, _STRUCTURED_TOKEN_FLOOR)

        def _attempt():
            """One generate+parse. Returns (obj, dumped, char_len) or raises."""
            cfg_kwargs = dict(response_mime_type="application/json", max_output_tokens=tokens)
            if use_native:
                cfg_kwargs["response_schema"] = schema
            tc = _thinking_config(types)
            if tc is not None:
                cfg_kwargs["thinking_config"] = tc
            response = self._generate(
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
                what=what,
            )
            text = (_extract_text(response) or "").strip()
            if not text:
                raise ValueError(f"empty/blocked response ({_describe_response(response)})")
            # Strip markdown fences defensively.
            if text.startswith("```"):
                lines = text.splitlines()
                inner = lines[1:] if lines[0].startswith("```") else lines
                if inner and inner[-1].strip() == "```":
                    inner = inner[:-1]
                text = "\n".join(inner).strip()
            obj = schema.model_validate(json.loads(text))
            return obj, obj.model_dump(), len(text)

        try:
            # Some small models intermittently emit a schema-valid but ALL-EMPTY
            # object (every content field ""). That parses fine yet carries no
            # data. It's flaky run-to-run, so re-roll a few times before giving
            # up — far more reliable than prompt-tuning alone.
            obj = dumped = None
            for roll in range(_STRUCTURED_EMPTY_RETRIES + 1):
                obj, dumped, char_len = _attempt()
                if not _is_vacuous(dumped):
                    break
                if roll < _STRUCTURED_EMPTY_RETRIES:
                    logger.warning("%s ⚠ %s parsed OK but ALL content fields are EMPTY "
                                   "— re-rolling (%d/%d).",
                                   self.tag, what, roll + 1, _STRUCTURED_EMPTY_RETRIES)
                else:
                    logger.warning("%s ⚠ %s still EMPTY after %d re-rolls — accepting "
                                   "vacuous result; caller's deterministic fallback "
                                   "(if any) will apply.",
                                   self.tag, what, _STRUCTURED_EMPTY_RETRIES)
                    record_quality(what, "vacuous",
                                   "all content fields empty after re-rolls")

            logger.info("%s ✓ %s OK | %d chars", self.tag, what, char_len)
            logger.info("%s   data: %s", self.tag,
                        json.dumps(dumped, default=str)[:_PREVIEW])
            return obj

        except Exception as exc:  # noqa: BLE001
            logger.error("%s ✗ %s FAILED: %s: %s", self.tag, what,
                         type(exc).__name__, str(exc)[:300])
            if self.strict:
                raise
            logger.error("%s ⚠ RETURNING SCHEMA DEFAULTS for %s — downstream "
                         "values will be PLACEHOLDERS, not real data.", self.tag, what)
            record_quality(what, "soft-fail-defaults", f"{type(exc).__name__}: {str(exc)[:120]}")
            return schema()

    # ------------------------------------------------------------------ #
    # Web research via Google Search grounding (robust fallback chain)    #
    # ------------------------------------------------------------------ #
    def research(self, system: str, user: str) -> str:
        from google.genai import types

        # Reasoning models spend output tokens on thinking; give headroom so the
        # visible answer isn't truncated to empty (esp. with search overhead).
        tokens = max(self.max_tokens, _RESEARCH_TOKEN_FLOOR)

        def _log_ok(what: str, text: str) -> str:
            logger.info("%s ✓ %s OK | %d chars retrieved", self.tag, what, len(text))
            logger.info("%s   fetched: %s", self.tag, text[:_PREVIEW].replace("\n", " "))
            return text

        # 1) Grounded Google Search (best quality) — unless globally disabled.
        # Whether the model SEARCHES is a stochastic (model+prompt) decision, not a
        # function of the API key — so a grounding-miss isn't a key failure. We
        # re-roll a few times to coax a real search (each re-roll rotates the key
        # too: free, and covers the rare case where one key's project has the
        # Search tool enabled and another doesn't, plus adds a hard search nudge).
        # The best-effort ungrounded text is KEPT and only used if grounding never
        # engages, so no information is thrown away.
        if self.grounding_enabled:
            attempts = 1 + max(0, _GROUNDING_RETRIES)
            last_text = ""
            for gi in range(attempts):
                what = "research(grounded)" if gi == 0 else f"research(grounded#retry{gi})"
                if gi > 0:
                    self._rotate_key()
                grounded_user = user if gi == 0 else (
                    "IMPORTANT: You MUST use Google Search to verify current facts "
                    "before answering. Do not answer from memory.\n\n" + user)
                logger.info("%s → %s …", self.tag, what)
                try:
                    g_cfg = dict(
                        system_instruction=system,
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        max_output_tokens=tokens,
                    )
                    tc = _thinking_config(types)
                    if tc is not None:
                        g_cfg["thinking_config"] = tc
                    response = self._generate(
                        contents=grounded_user,
                        config=types.GenerateContentConfig(**g_cfg),
                        what=what,
                    )
                    text = (_extract_text(response) or "").strip()
                    if text and _has_grounding(response):
                        return _log_ok(what, text)
                    if text:
                        last_text = text  # keep best-effort answer
                        more = (f"re-rolling on next key ({gi + 1}/{attempts - 1})"
                                if gi < attempts - 1 else "out of grounding retries")
                        logger.warning(
                            "%s %s produced text but NO grounding metadata — the "
                            "model did NOT search; %s.", self.tag, what, more)
                    else:
                        logger.warning("%s %s returned EMPTY (%s).",
                                       self.tag, what, _describe_response(response))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s %s failed (%s: %s).", self.tag, what,
                                   type(exc).__name__, str(exc)[:200])

            if last_text:
                logger.warning("%s grounding never engaged after %d attempt(s) — "
                               "returning UNGROUNDED model-knowledge text (NOT live "
                               "web data).", self.tag, attempts)
                record_quality("research", "ungrounded-no-search",
                               f"model never searched after {attempts} attempts")
                return _log_ok("research(ungrounded-no-search)", last_text)
            logger.warning("%s grounded attempts yielded no usable text — falling back UNGROUNDED.",
                           self.tag)
            grounding_issue = "ungrounded-fallback"
        else:
            logger.info("%s grounding disabled (FIFA_DISABLE_GROUNDING) — using model knowledge.",
                        self.tag)
            grounding_issue = "ungrounded-disabled"

        # 2) Ungrounded fallback — model knowledge only (acceptable per design).
        what = "research(ungrounded)"
        logger.info("%s → %s …", self.tag, what)
        try:
            u_cfg = dict(system_instruction=system, max_output_tokens=tokens)
            tc = _thinking_config(types)
            if tc is not None:
                u_cfg["thinking_config"] = tc
            response = self._generate(
                contents=user,
                config=types.GenerateContentConfig(**u_cfg),
                what=what,
            )
            text = (_extract_text(response) or "").strip()
            if text:
                record_quality("research", grounding_issue, "no live web grounding")
                return _log_ok(what, text)
            logger.error("%s ✗ %s also returned EMPTY (%s).",
                         self.tag, what, _describe_response(response))
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.error("%s ✗ %s FAILED: %s: %s", self.tag, what,
                         type(exc).__name__, str(exc)[:300])
            if self.strict:
                raise
            return f"[Research unavailable: {type(exc).__name__}: {exc}]"


def build_llms(settings) -> dict:
    """Construct the three model tiers once and share them across all agents."""
    def _flag(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None or not v.strip():
            return default
        return v.strip().lower() in {"1", "true", "yes", "on"}

    # Strict is ON by default: a soft-failed LLM call should ABORT the fixture
    # rather than silently substitute schema-default placeholders that then look
    # like real predictions downstream. Set FIFA_LLM_STRICT=0 to opt back into
    # soft-fail behaviour.
    strict = _flag("FIFA_LLM_STRICT", default=True)
    grounding_enabled = not _flag("FIFA_DISABLE_GROUNDING")

    # Ordered failover pool: prefer the explicit list, fall back to the single key.
    key_pool = list(getattr(settings, "gemini_api_keys", None) or [])
    if not key_pool and settings.gemini_api_key:
        key_pool = [settings.gemini_api_key]

    common = dict(
        api_keys=key_pool,
        max_tokens=settings.max_output_tokens,
        strict=strict,
        grounding_enabled=grounding_enabled,
        fallback_models=list(getattr(settings, "fallback_models", None) or []),
    )
    llms = {
        "research":  LLM(model=settings.research_model,  **common),
        "reasoning": LLM(model=settings.reasoning_model, **common),
        "judge":     LLM(model=settings.judge_model,     **common),
    }
    if len(key_pool) > 1:
        logger.info("[LLM] %d Gemini API keys loaded — auto-failover on auth/quota errors.",
                    len(key_pool))
    if strict:
        logger.info("[LLM] FIFA_LLM_STRICT is ON (default) — a failed call RAISES "
                    "and aborts the fixture instead of returning placeholders.")
    else:
        logger.warning("[LLM] FIFA_LLM_STRICT is OFF — failed calls soft-fail to "
                       "schema DEFAULTS; placeholder data may flow downstream.")
    if not grounding_enabled:
        logger.warning("[LLM] FIFA_DISABLE_GROUNDING is ON — research uses model knowledge only.")
    return llms