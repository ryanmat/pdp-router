# Description: Tests for LLM response parsing utilities.
# Description: Covers fence stripping, JSON parsing, and fallback behavior.

from __future__ import annotations

from pdp_router._utils import parse_llm_json, strip_markdown_fences


class TestStripMarkdownFences:
    def test_json_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_plain_fence(self) -> None:
        text = "```\nsome text\n```"
        assert strip_markdown_fences(text) == "some text"

    def test_no_fences_passthrough(self) -> None:
        text = '{"key": "value"}'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_whitespace_handling(self) -> None:
        text = '  \n```json\n{"a": 1}\n```\n  '
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_empty_string(self) -> None:
        assert strip_markdown_fences("") == ""

    def test_only_opening_fence(self) -> None:
        text = '```json\n{"key": "value"}'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_only_closing_fence(self) -> None:
        text = '{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_nested_backticks_in_content(self) -> None:
        text = '```json\n{"code": "use `backticks` here"}\n```'
        assert strip_markdown_fences(text) == '{"code": "use `backticks` here"}'


class TestParseLlmJson:
    def test_valid_json(self) -> None:
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self) -> None:
        result = parse_llm_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_trailing_text(self) -> None:
        """raw_decode fallback handles trailing stderr from mcporter."""
        result = parse_llm_json('{"key": "value"}\nsome trailing stderr text')
        assert result == {"key": "value"}

    def test_unparseable_returns_string(self) -> None:
        result = parse_llm_json("this is not json at all")
        assert isinstance(result, str)
        assert result == "this is not json at all"

    def test_empty_string_returns_string(self) -> None:
        result = parse_llm_json("")
        assert isinstance(result, str)

    def test_fenced_json_with_trailing_text(self) -> None:
        text = '```json\n{"a": 1}\n```\nSome extra output'
        result = parse_llm_json(text)
        assert result == {"a": 1}

    def test_nested_json_object(self) -> None:
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_llm_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_json_array_returns_list(self) -> None:
        result = parse_llm_json("[1, 2, 3]")
        assert result == [1, 2, 3]
