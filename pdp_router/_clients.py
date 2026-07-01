# Description: Unified LLM client wrappers with a common completion interface.
# Description: AnthropicClient wraps the SDK; OllamaClient is a stub for Mac Studio.

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import anthropic
from opentelemetry import trace

from pdp_router._cost import estimate_cost
from pdp_router._effort import (
    anthropic_output_config,
    is_anthropic_effort_model,
    openrouter_reasoning_body,
)
from pdp_router._models import CreditExhaustionError

log = logging.getLogger(__name__)


# Anthropic server-side web search tool. Version pinned and verified against
# live Anthropic docs 2026-05-31. web_search_20250305 is the self-contained
# tool (no code-execution dependency); bumping to web_search_20260209 (dynamic
# filtering) additionally requires enabling the code-execution tool, so it is a
# deliberate future step, not a drop-in. Priced at $10 per 1,000 searches plus
# the search-result tokens that land in the input context. max_uses caps the
# searches per request as a direct cost lever.
WEB_SEARCH_TOOL_VERSION = "web_search_20250305"
WEB_SEARCH_MAX_USES = 5


def _anthropic_web_search_tool() -> dict:
    """Return the Anthropic web_search server-tool definition (max_uses-capped)."""
    return {
        "type": WEB_SEARCH_TOOL_VERSION,
        "name": "web_search",
        "max_uses": WEB_SEARCH_MAX_USES,
    }


@dataclass(frozen=True)
class CompletionResult:
    """Structured result from any LLM completion call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    estimated_cost_usd: float
    # Count of provider-side web searches performed for this completion. Anthropic
    # reports it in usage.server_tool_use.web_search_requests; Gemini grounding
    # exposes it best-effort via candidate grounding_metadata. 0 when web search
    # was off or the model chose not to search. Used for cost visibility.
    web_search_requests: int = 0


class LLMClient(Protocol):
    """Protocol for LLM client wrappers.

    enable_web_search opts a single call into provider-native web search. It is
    a per-call flag (not client state) so the proxy can enable it on the cascade
    path while leaving the panel path and every other pdp-router consumer
    untouched. Providers without native web search (DeepSeek, Ollama) accept and
    ignore it to satisfy this protocol.

    effort opts a single call into a provider reasoning-effort knob (low/medium/
    high). Like enable_web_search it is a per-call value, None meaning "do not set
    the knob." The proxy only passes a non-None effort to models that support it
    (see _effort.supports_effort); providers/models with no dial (Gemini, DeepSeek,
    Llama, Haiku, Ollama) accept and ignore it to satisfy this protocol.
    """

    def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        langsmith_extra: dict | None = None,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult: ...

    def complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult: ...

    def stream_complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens as they arrive from the backend."""
        ...

    def stream_complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens for multi-turn conversations."""
        ...


class AnthropicClient:
    """Wraps the Anthropic SDK messages API behind the LLMClient interface.

    Decoupled from any config object -- takes explicit credentials.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str = "",
        auth_token: str = "",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._auth_token = auth_token

        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
        elif auth_token:
            self._client = anthropic.Anthropic(auth_token=auth_token)
        else:
            raise ValueError("No Anthropic credentials provided. Pass api_key or auth_token.")

        # AsyncAnthropic is lazily constructed only when streaming is invoked;
        # most callers stay on the sync code path and pay no cost for it.
        self._async_client: anthropic.AsyncAnthropic | None = None

    def _get_async_client(self) -> anthropic.AsyncAnthropic:
        """Return (creating if needed) the async Anthropic client."""
        if self._async_client is None:
            if self._api_key:
                self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
            else:
                self._async_client = anthropic.AsyncAnthropic(auth_token=self._auth_token)
        return self._async_client

    def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        langsmith_extra: dict | None = None,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        """Send a single-turn completion request.

        Args:
            system: System prompt.
            user_message: User message content.
            max_tokens: Maximum tokens to generate.
            langsmith_extra: Per-call LangSmith metadata (passed through to SDK).
            enable_web_search: Attach the web_search server tool. The model still
                decides whether to actually search.
            effort: Reasoning-effort level (low/medium/high). When set on a non-Haiku
                model, attached as output_config.effort; Haiku rejects the parameter
                so it is dropped here (and the proxy also gates it out upstream).

        Raises:
            CreditExhaustionError: On HTTP 402 or billing-related errors.
            anthropic.APIStatusError: On other API errors.
        """
        extra_kwargs: dict = {}
        if langsmith_extra is not None:
            extra_kwargs["langsmith_extra"] = langsmith_extra
        if enable_web_search:
            extra_kwargs["tools"] = [_anthropic_web_search_tool()]
        if effort and is_anthropic_effort_model(self._model):
            extra_kwargs["output_config"] = anthropic_output_config(effort)

        message = self._make_api_call(
            system=system,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        return self._process_response(message)

    def complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        """Send a multi-turn completion request.

        Args:
            system: System prompt.
            messages: Conversation history as a list of role/content dicts.
            max_tokens: Maximum tokens to generate.
            enable_web_search: Attach the web_search server tool. The model still
                decides whether to actually search.
            effort: Reasoning-effort level (low/medium/high). When set, attached as
                output_config.effort.

        Raises:
            CreditExhaustionError: On HTTP 402 or billing-related errors.
            anthropic.APIStatusError: On other API errors.
        """
        extra_kwargs: dict = {}
        if enable_web_search:
            extra_kwargs["tools"] = [_anthropic_web_search_tool()]
        if effort and is_anthropic_effort_model(self._model):
            extra_kwargs["output_config"] = anthropic_output_config(effort)

        message = self._make_api_call(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        return self._process_response(message)

    def _make_api_call(self, **kwargs: object) -> anthropic.types.Message:
        """Call the Anthropic messages API, converting billing errors."""
        try:
            return self._client.messages.create(model=self._model, **kwargs)
        except anthropic.APIStatusError as e:
            error_msg = str(e).lower()
            if e.status_code == 402 or any(
                kw in error_msg for kw in ("credit", "billing", "insufficient")
            ):
                span = trace.get_current_span()
                span.set_attribute("pdp_router.credit_exhaustion", True)
                raise CreditExhaustionError(
                    f"Anthropic API credit/billing error (HTTP {e.status_code}): {e}"
                ) from e
            raise

    def _process_response(self, message: anthropic.types.Message) -> CompletionResult:
        """Extract tokens, estimate cost, record span attributes.

        Concatenates every text block rather than reading content[0]. When the
        web_search tool fires, the response interleaves text / server_tool_use /
        web_search_tool_result / text blocks -- content[0] is only Claude's
        "I'll search for..." preamble, and the real answer lives in later text
        blocks. Joining all type=="text" blocks yields the full answer in both
        the search and no-search cases; non-text blocks (tool use, results) are
        skipped.
        """
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )

        # Web search billing is separate ($10/1k) from token cost. The count
        # rides on the result for the proxy's cost-visibility log line.
        server_tool_use = getattr(message.usage, "server_tool_use", None)
        raw_searches = getattr(server_tool_use, "web_search_requests", 0)
        web_search_requests = raw_searches if isinstance(raw_searches, int) else 0

        token_usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        cost = estimate_cost(self._model, token_usage)

        span = trace.get_current_span()
        span.set_attribute("gen_ai.usage.input_tokens", token_usage["input_tokens"])
        span.set_attribute("gen_ai.usage.output_tokens", token_usage["output_tokens"])
        span.set_attribute("pdp_router.estimated_cost_usd", cost)
        if web_search_requests:
            span.set_attribute("pdp_router.web_search_requests", web_search_requests)

        return CompletionResult(
            text=text,
            input_tokens=token_usage["input_tokens"],
            output_tokens=token_usage["output_tokens"],
            model=self._model,
            estimated_cost_usd=cost,
            web_search_requests=web_search_requests,
        )

    async def stream_complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens as they arrive from the Anthropic backend."""
        async for token in self._stream(
            system=system,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
            effort=effort,
        ):
            yield token

    async def stream_complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens for a multi-turn conversation."""
        async for token in self._stream(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
            effort=effort,
        ):
            yield token

    async def _stream(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Iterate the async messages stream, yielding content_block_delta text.

        With web search on, the stream interleaves server_tool_use and
        web_search_tool_result blocks (and a pause while the search runs); the
        text_delta filter below skips them and yields only answer text. With
        effort set, output_config.effort rides on the stream request.
        """
        client = self._get_async_client()
        stream_kwargs: dict = {}
        if enable_web_search:
            stream_kwargs["tools"] = [_anthropic_web_search_tool()]
        if effort and is_anthropic_effort_model(self._model):
            stream_kwargs["output_config"] = anthropic_output_config(effort)
        try:
            async with client.messages.stream(
                model=self._model,
                system=system,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                **stream_kwargs,
            ) as stream:
                async for event in stream:
                    if (
                        getattr(event, "type", None) == "content_block_delta"
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        text = getattr(event.delta, "text", "")
                        if text:
                            yield text
        except anthropic.APIStatusError as e:
            error_msg = str(e).lower()
            if e.status_code == 402 or any(
                kw in error_msg for kw in ("credit", "billing", "insufficient")
            ):
                span = trace.get_current_span()
                span.set_attribute("pdp_router.credit_exhaustion", True)
                raise CreditExhaustionError(
                    f"Anthropic API credit/billing error (HTTP {e.status_code}): {e}"
                ) from e
            raise


class GeminiClient:
    """Wraps the Google Gen AI SDK behind the LLMClient interface.

    Supports two modes:
    - API key mode (for gemini-* models): pass api_key
    - Vertex AI mode (for meta/* partner models): pass project + location
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str = "",
        project: str = "",
        location: str = "",
    ) -> None:
        self._model = model

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai SDK not installed. Run: pip install 'pdp-router[gemini]'"
            ) from exc

        if api_key:
            self._client = genai.Client(api_key=api_key)
        elif project:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location or "us-central1",
            )
        else:
            raise ValueError(
                "No Gemini credentials provided. Pass api_key (Gemini API) or "
                "project + location (Vertex AI)."
            )

    def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        langsmith_extra: dict | None = None,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        """Send a single-turn completion request.

        effort is accepted for protocol parity and ignored: Gemini's thinking
        knob (thinking_budget/level) is Gemini-version dependent on the 2.5 roster
        and is not wired in v1 (see _effort.supports_effort).
        """
        response = self._make_api_call(
            contents=user_message,
            system=system,
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
        )
        return self._process_response(response)

    def complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        """Send a multi-turn completion request (effort accepted, ignored in v1)."""
        from google.genai import types

        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        response = self._make_api_call(
            contents=contents,
            system=system,
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
        )
        return self._process_response(response)

    def _grounding_tools(self, enable_web_search: bool) -> list | None:
        """Return the Google Search grounding tool list, or None.

        Grounding is a Gemini-model feature. This client also fronts meta/* Llama
        partner models on Vertex (get_client routes them here), which reject the
        google_search tool, so the tool is gated on a gemini- model id.
        """
        if not enable_web_search or not self._model.startswith("gemini"):
            return None
        from google.genai import types

        return [types.Tool(google_search=types.GoogleSearch())]

    def _make_api_call(
        self,
        contents: object,
        system: str,
        max_tokens: int,
        enable_web_search: bool = False,
    ) -> object:
        """Call the Google Gen AI generate_content API, converting billing errors."""
        from google.genai import types

        try:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    tools=self._grounding_tools(enable_web_search),
                ),
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ("quota", "billing", "resource_exhausted", "429")):
                span = trace.get_current_span()
                span.set_attribute("pdp_router.credit_exhaustion", True)
                raise CreditExhaustionError(f"Gemini API quota/billing error: {e}") from e
            raise

    def _process_response(self, response: object) -> CompletionResult:
        """Extract tokens, estimate cost, record span attributes."""
        text = response.text  # type: ignore[attr-defined]
        usage = response.usage_metadata  # type: ignore[attr-defined]
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        cost = estimate_cost(self._model, token_usage)

        # Best-effort grounding-search count for cost visibility. Gemini bills per
        # search query the model executes; webSearchQueries on the candidate's
        # grounding_metadata enumerates them. Guarded: absent when grounding was
        # off or the model did not search, and the attribute chain is optional.
        web_search_requests = 0
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                meta = getattr(candidates[0], "grounding_metadata", None)
                queries = getattr(meta, "web_search_queries", None) if meta else None
                if isinstance(queries, (list, tuple)):
                    web_search_requests = len(queries)
        except Exception:
            # Optional cost metric only -- never disrupt the (already extracted)
            # completion. Log at debug with the trace rather than swallow silently.
            log.debug("grounding web_search_queries extraction failed", exc_info=True)
            web_search_requests = 0

        span = trace.get_current_span()
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("pdp_router.estimated_cost_usd", cost)
        if web_search_requests:
            span.set_attribute("pdp_router.web_search_requests", web_search_requests)

        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            estimated_cost_usd=cost,
            web_search_requests=web_search_requests,
        )

    async def stream_complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens as they arrive from the Gemini backend.

        effort is accepted for protocol parity and ignored (no Gemini dial in v1).
        """
        async for token in self._stream(
            contents=user_message,
            system=system,
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
        ):
            yield token

    async def stream_complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens for a multi-turn Gemini conversation.

        effort is accepted for protocol parity and ignored (no Gemini dial in v1).
        """
        from google.genai import types

        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        async for token in self._stream(
            contents=contents,
            system=system,
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
        ):
            yield token

    async def _stream(
        self,
        *,
        contents: object,
        system: str,
        max_tokens: int,
        enable_web_search: bool = False,
    ) -> AsyncIterator[str]:
        """Iterate the Gemini async stream, yielding chunk.text when present."""
        from google.genai import types

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    tools=self._grounding_tools(enable_web_search),
                ),
            )
            async for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ("quota", "billing", "resource_exhausted", "429")):
                span = trace.get_current_span()
                span.set_attribute("pdp_router.credit_exhaustion", True)
                raise CreditExhaustionError(f"Gemini API quota/billing error: {e}") from e
            raise


class OpenAICompatibleClient:
    """Wraps any OpenAI-compatible /chat/completions API behind the LLMClient interface.

    Uses httpx directly. base_url must include the version segment (for example
    "https://openrouter.ai/api/v1"); every request POSTs to the absolute
    "{base_url}/chat/completions". label names the provider in error messages.
    Fronts OpenRouter (openai/* and qwen/* slugs); DeepSeekClient subclasses it.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str = "",
        base_url: str = "",
        label: str = "OpenAI-compatible",
        supports_reasoning_effort: bool = True,
    ) -> None:
        self._model = model
        self._label = label
        # OpenRouter normalizes a unified reasoning.effort param across its arms;
        # DeepSeek's direct API has no such param (its reasoning is a separate
        # model, not a request field), so the DeepSeek subclass disables this.
        # Guards against ever putting an unsupported field on a direct provider.
        self._supports_reasoning_effort = supports_reasoning_effort

        if not api_key:
            raise ValueError(f"No {label} credentials provided. Pass api_key.")
        if not base_url:
            raise ValueError(f"No base_url provided for the {label} client.")

        import httpx

        self._api_key = api_key
        self._chat_url = f"{base_url.rstrip('/')}/chat/completions"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
        )
        self._async_client: httpx.AsyncClient | None = None

    def _get_async_client(self) -> object:
        """Return (creating if needed) the async httpx client."""
        if self._async_client is None:
            import httpx

            self._async_client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            )
        return self._async_client

    def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        langsmith_extra: dict | None = None,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        """Send a single-turn completion request.

        enable_web_search is accepted for protocol parity and ignored: these
        OpenAI-compatible endpoints have no native web-search server tool. effort,
        when set on a reasoning-effort-capable endpoint (OpenRouter), rides as the
        unified reasoning.effort body param.
        """
        return self._call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            effort=effort,
        )

    def complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        """Send a multi-turn completion request (enable_web_search ignored)."""
        full_messages = [{"role": "system", "content": system}, *messages]
        return self._call(messages=full_messages, max_tokens=max_tokens, effort=effort)

    def _call(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        effort: str | None = None,
    ) -> CompletionResult:
        """Make the API call and process the response."""
        body: dict = {"model": self._model, "messages": messages, "max_tokens": max_tokens}
        if effort and self._supports_reasoning_effort:
            body.update(openrouter_reasoning_body(effort))
        response = self._client.post(self._chat_url, json=body)

        if response.status_code == 402 or (
            response.status_code >= 400
            and any(
                kw in response.text.lower() for kw in ("insufficient", "billing", "quota", "credit")
            )
        ):
            span = trace.get_current_span()
            span.set_attribute("pdp_router.credit_exhaustion", True)
            raise CreditExhaustionError(
                f"{self._label} API billing error (HTTP {response.status_code}): {response.text}"
            )

        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        cost = estimate_cost(self._model, token_usage)

        span = trace.get_current_span()
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("pdp_router.estimated_cost_usd", cost)

        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            estimated_cost_usd=cost,
        )

    async def stream_complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens as they arrive from the endpoint (enable_web_search ignored)."""
        async for token in self._stream(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            effort=effort,
        ):
            yield token

    async def stream_complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text tokens for a multi-turn conversation (enable_web_search ignored)."""
        full_messages = [{"role": "system", "content": system}, *messages]
        async for token in self._stream(
            messages=full_messages, max_tokens=max_tokens, effort=effort
        ):
            yield token

    async def _stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Open an SSE stream against the endpoint, yield delta content."""
        client = self._get_async_client()
        body: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if effort and self._supports_reasoning_effort:
            body.update(openrouter_reasoning_body(effort))
        async with client.stream(  # type: ignore[attr-defined]
            "POST",
            self._chat_url,
            json=body,
        ) as response:
            if response.status_code == 402 or (
                response.status_code >= 400
                and any(
                    kw in (await response.aread()).decode("utf-8", "replace").lower()
                    for kw in ("insufficient", "billing", "quota", "credit")
                )
            ):
                span = trace.get_current_span()
                span.set_attribute("pdp_router.credit_exhaustion", True)
                raise CreditExhaustionError(
                    f"{self._label} API billing error (HTTP {response.status_code})"
                )
            response.raise_for_status()

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content


class DeepSeekClient(OpenAICompatibleClient):
    """OpenAI-compatible client pinned to DeepSeek's endpoint.

    DeepSeek is a genuinely decorrelated training lineage; this thin subclass
    fixes the base_url and the credential-error label.
    """

    def __init__(self, model: str, *, api_key: str = "") -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            label="DeepSeek",
            # DeepSeek's reasoning is a separate model (deepseek-reasoner), not a
            # request param, so no unified reasoning.effort knob: do not emit it.
            supports_reasoning_effort=False,
        )


class OllamaClient:
    """Stub client for local Ollama models (DGX Spark integration pending)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url

    def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        langsmith_extra: dict | None = None,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        raise RuntimeError(f"Ollama not yet available -- Mac Studio pending. Model: {self._model}")

    def complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> CompletionResult:
        raise RuntimeError(f"Ollama not yet available -- Mac Studio pending. Model: {self._model}")

    async def stream_complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("Streaming not implemented for Ollama")
        # pragma: no cover -- unreachable, satisfies AsyncIterator typing.
        if False:
            yield ""

    async def stream_complete_multi(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        effort: str | None = None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("Streaming not implemented for Ollama")
        # pragma: no cover -- unreachable, satisfies AsyncIterator typing.
        if False:
            yield ""


def get_client(
    model_name: str,
    *,
    api_key: str = "",
    auth_token: str = "",
    base_url: str = "",
    project: str = "",
    location: str = "",
    deepseek_api_key: str = "",
) -> AnthropicClient | GeminiClient | OpenAICompatibleClient | OllamaClient:
    """Return the appropriate client wrapper for a model name.

    Args:
        model_name: Model identifier (e.g. "claude-sonnet-4-...", "gemini-2.5-flash",
            "meta/llama-4-scout-...", "deepseek-chat", or "ollama/llama3").
        api_key: API key for Anthropic or Gemini models.
        auth_token: OAuth token for Anthropic models (fallback if no api_key).
        base_url: Base URL for Ollama and OpenAI-compatible (OpenRouter) models.
        project: GCP project ID for Vertex AI partner models (meta/*).
        location: GCP region for Vertex AI partner models (default: us-central1).
        deepseek_api_key: API key for DeepSeek models.

    Returns:
        Client wrapper implementing the LLMClient protocol.

    Raises:
        ValueError: If model_name does not match a known provider prefix.
    """
    if model_name.startswith("claude-"):
        return AnthropicClient(model_name, api_key=api_key, auth_token=auth_token)
    if model_name.startswith("deepseek-"):
        return DeepSeekClient(model_name, api_key=deepseek_api_key or api_key)
    if model_name.startswith("gemini-"):
        return GeminiClient(model_name, api_key=api_key)
    if model_name.startswith("meta/"):
        return GeminiClient(model_name, project=project, location=location)
    if model_name.startswith("ollama/") or model_name.startswith("local/"):
        return OllamaClient(model=model_name, base_url=base_url or "http://localhost:11434")
    if model_name.startswith("openai/") or model_name.startswith("qwen"):
        return OpenAICompatibleClient(
            model_name,
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            label="OpenRouter",
        )
    raise ValueError(f"Unknown model provider for: {model_name}")
