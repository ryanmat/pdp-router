# Description: Tests for GeminiClient wrapper (API key and Vertex AI modes).
# Description: Uses pytest.importorskip and mock patching for the optional google-genai dep.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from pdp_router._clients import CompletionResult, GeminiClient, get_client
from pdp_router._models import CreditExhaustionError


def _mock_response(
    text: str = "response text", input_tokens: int = 100, output_tokens: int = 50
) -> MagicMock:
    """Build a mock GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata.prompt_token_count = input_tokens
    resp.usage_metadata.candidates_token_count = output_tokens
    return resp


class TestGeminiClientInit:
    def test_init_api_key_mode(self) -> None:
        with patch("pdp_router._clients.genai", create=True) as mock_genai:
            mock_genai.Client = MagicMock
            GeminiClient("gemini-2.5-flash", api_key="test-key")

    def test_init_vertex_mode(self) -> None:
        with patch("pdp_router._clients.genai", create=True) as mock_genai:
            mock_genai.Client = MagicMock
            GeminiClient(
                "meta/llama-4-scout-17b-16e-instruct-maas",
                project="my-project",
                location="us-east5",
            )

    def test_init_vertex_default_location(self) -> None:
        with patch("pdp_router._clients.genai", create=True) as mock_genai:
            mock_genai.Client = MagicMock
            GeminiClient(
                "meta/llama-4-scout-17b-16e-instruct-maas",
                project="my-project",
            )

    def test_init_no_credentials_raises(self) -> None:
        with pytest.raises(ValueError, match="No Gemini credentials"):
            GeminiClient("gemini-2.5-flash")

    def test_init_missing_sdk_raises(self) -> None:
        with (
            patch.dict("sys.modules", {"google": None, "google.genai": None}),
            pytest.raises(ImportError, match="google-genai SDK not installed"),
        ):
            GeminiClient("gemini-2.5-flash", api_key="test-key")


class TestGeminiClientComplete:
    @patch("google.genai.Client")
    def test_complete_returns_completion_result(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        result = client.complete("system prompt", "user message")

        assert isinstance(result, CompletionResult)
        assert result.text == "response text"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.model == "gemini-2.5-flash"
        assert result.estimated_cost_usd > 0

    @patch("google.genai.Client")
    def test_complete_passes_system_and_max_tokens(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        client.complete("test system", "test message", max_tokens=2048)

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.system_instruction == "test system"
        assert config.max_output_tokens == 2048

    @patch("google.genai.Client")
    def test_complete_multi_returns_completion_result(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response("multi reply", 200, 100)

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        result = client.complete_multi("system", messages)

        assert result.text == "multi reply"
        assert result.input_tokens == 200
        assert result.output_tokens == 100

    @patch("google.genai.Client")
    def test_complete_multi_maps_assistant_to_model(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "follow up"},
        ]
        client.complete_multi("system", messages)

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"

    @patch("google.genai.Client")
    def test_langsmith_extra_ignored(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        result = client.complete("system", "message", langsmith_extra={"key": "val"})
        assert isinstance(result, CompletionResult)


class TestGeminiClientErrors:
    @patch("google.genai.Client")
    def test_quota_error_raises_credit_exhaustion(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: quota exceeded"
        )

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        with pytest.raises(CreditExhaustionError, match="quota/billing"):
            client.complete("system", "message")

    @patch("google.genai.Client")
    def test_billing_error_raises_credit_exhaustion(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("billing account disabled")

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        with pytest.raises(CreditExhaustionError):
            client.complete("system", "message")

    @patch("google.genai.Client")
    def test_non_billing_error_reraises(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("model not found")

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        with pytest.raises(Exception, match="model not found"):
            client.complete("system", "message")

    @patch("google.genai.Client")
    def test_none_usage_metadata_defaults_to_zero(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        resp = MagicMock()
        resp.text = "ok"
        resp.usage_metadata.prompt_token_count = None
        resp.usage_metadata.candidates_token_count = None
        mock_client.models.generate_content.return_value = resp

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        result = client.complete("system", "message")
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestGeminiWebSearch:
    """Google Search grounding wiring (proxy_web_search_enabled cascade path)."""

    @staticmethod
    def _config_of(mock_client: MagicMock) -> object:
        call = mock_client.models.generate_content.call_args
        return call.kwargs.get("config") or call[1].get("config")

    @patch("google.genai.Client")
    def test_default_off_no_grounding_tool(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        client.complete("sys", "msg")
        assert not self._config_of(mock_client).tools

    @patch("google.genai.Client")
    def test_enable_web_search_attaches_grounding(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient("gemini-2.5-flash", api_key="test-key")
        client.complete("sys", "msg", enable_web_search=True)
        tools = self._config_of(mock_client).tools
        assert tools and len(tools) == 1
        assert tools[0].google_search is not None

    @patch("google.genai.Client")
    def test_meta_model_never_grounds(self, mock_client_cls: MagicMock) -> None:
        """meta/* Llama partner models route through GeminiClient but reject the
        google_search tool; the gemini-only gate must withhold it even with the
        flag on."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _mock_response()

        client = GeminiClient(
            "meta/llama-4-scout-17b-16e-instruct-maas",
            project="my-project",
            location="us-east5",
        )
        client.complete("sys", "msg", enable_web_search=True)
        assert not self._config_of(mock_client).tools


class TestGetClientGemini:
    def test_gemini_prefix_returns_gemini_client(self) -> None:
        with patch("google.genai.Client"):
            client = get_client("gemini-2.5-flash", api_key="test-key")
            assert isinstance(client, GeminiClient)

    def test_meta_prefix_returns_gemini_client(self) -> None:
        with patch("google.genai.Client"):
            client = get_client(
                "meta/llama-4-scout-17b-16e-instruct-maas",
                project="my-project",
                location="us-east5",
            )
            assert isinstance(client, GeminiClient)
