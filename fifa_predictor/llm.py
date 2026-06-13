"""
llm.py  (google-genai edition)
================================

Wrapper around the Google Gen AI SDK (google-genai >= 1.0).

Two public methods (interface unchanged from the rest of the codebase):

1. `structured(schema, system, user, context)` -> Pydantic model instance
   Uses response_mime_type="application/json" so the model is forced to emit
   only JSON. The full JSON Schema is embedded in the prompt.

2. `research(system, user, max_searches)` -> str
   Attaches Google Search grounding so the model pulls real current web data.
   Falls back to an ungrounded call if grounding fails OR returns empty.

DIAGNOSTICS
-----------
Every call logs, at INFO, whether the API was hit and whether usable data came
back; on failure the REAL error is logged (no longer swallowed). Transient
errors (429 / 5xx / network) retry with backoff; permanent errors (400 / 401 /
403 / 404) are not retried and logged loudly. FIFA_LLM_STRICT=1 raises on
failure instead of soft-failing to schema defaults.

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
Judge's FinalReport, which must write four prose fields). structured() therefore
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

# HTTP status codes that will never succeed on retry.
_PERMANENT_CODES = {400, 401, 403, 404, 405}
_PREVIEW = 600  # chars of fetched data to echo into the log at INFO
_RESEARCH_TOKEN_FLOOR = 8192    # reasoning models need headroom for thinking+answer
_STRUCTURED_TOKEN_FLOOR = 4096  # prose-heavy structured calls starve without headroom
_STRUCTURED_EMPTY_RETRIES = 2   # re-rolls when a parse succeeds but is all-empty

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


def _error_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of the HTTP status code from an SDK exception."""
    for attr in ("code", "status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    msg = str(exc)
    for code in _PERMANENT_CODES | {429, 500, 503, 504}:
        if str(code) in msg:
            return code
    return None


def _is_permanent(exc: Exception) -> bool:
    return _error_code(exc) in _PERMANENT_CODES


class LLM:
    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        max_tokens: int = 3000,
        max_retries: int = 3,
        strict: bool = False,
        grounding_enabled: bool = False,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.strict = strict
        self.grounding_enabled = grounding_enabled
        self._client = None
        self.tag = f"[LLM:{model}]"

        # Lightweight call counters so you can see the hit/success ratio.
        self.attempts = 0
        self.successes = 0
        self.failures = 0

        if api_key:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            logger.info("%s client initialised (key set, %d chars).", self.tag, len(api_key))
        else:
            logger.warning("%s NO API KEY — client not built; calls will fail.", self.tag)

    # ------------------------------------------------------------------ #
    # Internal: a single guarded generate_content call with backoff       #
    # ------------------------------------------------------------------ #
    def _generate(self, *, contents, config, what: str):
        """
        Run generate_content with retries/backoff. Returns the SDK response on
        success or raises the last exception. All outcomes are logged.
        """
        if self._client is None:
            raise RuntimeError(f"{self.tag} no client (missing GEMINI_API_KEY)")

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self.attempts += 1
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
                logger.info("%s ← API responded for %s | %s",
                            self.tag, what, _describe_response(response))
                return response
            except Exception as exc:  # noqa: BLE001 - we re-raise after logging
                last_err = exc
                code = _error_code(exc)
                permanent = _is_permanent(exc)
                logger.error("%s ✗ %s API error (attempt %d/%d, code=%s): %s: %s",
                             self.tag, what, attempt, self.max_retries, code,
                             type(exc).__name__, str(exc)[:300])
                logger.debug("%s full traceback for %s:", self.tag, what, exc_info=True)
                if permanent:
                    logger.error("%s ✗ %s error is PERMANENT (code=%s) — not retrying. "
                                 "Check the API key and that model '%s' exists/is enabled.",
                                 self.tag, what, code, self.model)
                    break
                if attempt < self.max_retries:
                    backoff = min(60.0, 4.0 ** attempt)
                    logger.warning("%s … retrying %s in %.1fs", self.tag, what, backoff)
                    time.sleep(backoff)
        raise last_err if last_err else RuntimeError(f"{self.tag} {what} failed")

    # ------------------------------------------------------------------ #
    # Structured output via JSON mode                                     #
    # ------------------------------------------------------------------ #
    def structured(self, schema: Type[T], system: str, user: str, context: str = "") -> T:
        from google.genai import types

        what = f"structured({schema.__name__})"
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        parts = [f"SYSTEM INSTRUCTION:\n{system}\n"]
        if context:
            parts.append(f"REFERENCE DATA:\n{context}\n")
        parts.append(f"TASK:\n{user}")
        parts.append(
            "\nRESPONSE FORMAT:\n"
            "You MUST respond with ONLY a single valid JSON object matching "
            "the schema below exactly. Raw JSON only — no markdown fences, "
            "no explanation, no extra keys, no trailing text.\n\n" + schema_json
        )
        prompt = "\n".join(parts)
        logger.info("%s → %s …", self.tag, what)

        # Reasoning models spend output tokens on internal thinking; a small cap
        # can leave the visible JSON empty (the trap research() guards against).
        # Apply a structured-output floor so prose-heavy schemas (e.g.
        # FinalReport) have room for thinking + answer.
        tokens = max(self.max_tokens, _STRUCTURED_TOKEN_FLOOR)

        def _attempt():
            """One generate+parse. Returns (obj, dumped, char_len) or raises."""
            response = self._generate(
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=tokens,
                ),
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

            self.successes += 1
            logger.info("%s ✓ %s OK | %d chars", self.tag, what, char_len)
            logger.info("%s   data: %s", self.tag,
                        json.dumps(dumped, default=str)[:_PREVIEW])
            return obj

        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            logger.error("%s ✗ %s FAILED: %s: %s", self.tag, what,
                         type(exc).__name__, str(exc)[:300])
            if self.strict:
                raise
            logger.error("%s ⚠ RETURNING SCHEMA DEFAULTS for %s — downstream "
                         "values will be PLACEHOLDERS, not real data.", self.tag, what)
            return schema()

    # ------------------------------------------------------------------ #
    # Web research via Google Search grounding (robust fallback chain)    #
    # ------------------------------------------------------------------ #
    def research(self, system: str, user: str, max_searches: int = 6) -> str:
        from google.genai import types

        # Reasoning models spend output tokens on thinking; give headroom so the
        # visible answer isn't truncated to empty (esp. with search overhead).
        tokens = max(self.max_tokens, _RESEARCH_TOKEN_FLOOR)

        def _log_ok(what: str, text: str) -> str:
            self.successes += 1
            logger.info("%s ✓ %s OK | %d chars retrieved", self.tag, what, len(text))
            logger.info("%s   fetched: %s", self.tag, text[:_PREVIEW].replace("\n", " "))
            return text

        # 1) Grounded Google Search (best quality) — unless globally disabled.
        if self.grounding_enabled:
            what = "research(grounded)"
            logger.info("%s → %s …", self.tag, what)
            try:
                response = self._generate(
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        max_output_tokens=tokens,
                    ),
                    what=what,
                )
                text = (_extract_text(response) or "").strip()
                if text:
                    return _log_ok(what, text)
                logger.warning("%s %s returned EMPTY (%s) — falling back UNGROUNDED.",
                               self.tag, what, _describe_response(response))
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s %s failed (%s: %s) — falling back UNGROUNDED.",
                               self.tag, what, type(exc).__name__, str(exc)[:200])
        else:
            logger.info("%s grounding disabled (FIFA_DISABLE_GROUNDING) — using model knowledge.",
                        self.tag)

        # 2) Ungrounded fallback — model knowledge only (acceptable per design).
        what = "research(ungrounded)"
        logger.info("%s → %s …", self.tag, what)
        try:
            response = self._generate(
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=tokens,
                ),
                what=what,
            )
            text = (_extract_text(response) or "").strip()
            if text:
                return _log_ok(what, text)
            self.failures += 1
            logger.error("%s ✗ %s also returned EMPTY (%s).",
                         self.tag, what, _describe_response(response))
            return ""
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            logger.error("%s ✗ %s FAILED: %s: %s", self.tag, what,
                         type(exc).__name__, str(exc)[:300])
            if self.strict:
                raise
            return f"[Research unavailable: {type(exc).__name__}: {exc}]"

    # ------------------------------------------------------------------ #
    # Health check — confirm the key + model actually work before a run   #
    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        """Fire one trivial call; log and return whether the API is reachable."""
        from google.genai import types
        logger.info("%s → ping …", self.tag)
        try:
            resp = self._generate(
                contents="Reply with the single word: OK",
                config=types.GenerateContentConfig(max_output_tokens=2048),
                what="ping",
            )
            txt = (_extract_text(resp) or "").strip()
            ok = bool(txt)
            logger.info("%s ✓ ping reachable, replied %r", self.tag, txt[:40]) if ok \
                else logger.error("%s ✗ ping reachable but empty reply.", self.tag)
            return ok
        except Exception as exc:  # noqa: BLE001
            logger.error("%s ✗ ping FAILED: %s: %s", self.tag,
                         type(exc).__name__, str(exc)[:300])
            return False


def build_llms(settings) -> dict:
    """Construct the three model tiers once and share them across all agents."""
    def _flag(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    strict = _flag("FIFA_LLM_STRICT")
    grounding_enabled = not _flag("FIFA_DISABLE_GROUNDING")
    common = dict(
        api_key=settings.gemini_api_key,
        max_tokens=settings.max_output_tokens,
        strict=strict,
        grounding_enabled=grounding_enabled,
    )
    llms = {
        "research":  LLM(model=settings.research_model,  **common),
        "reasoning": LLM(model=settings.reasoning_model, **common),
        "judge":     LLM(model=settings.judge_model,     **common),
    }
    if strict:
        logger.warning("[LLM] FIFA_LLM_STRICT is ON — calls will RAISE on failure.")
    if not grounding_enabled:
        logger.warning("[LLM] FIFA_DISABLE_GROUNDING is ON — research uses model knowledge only.")
    return llms