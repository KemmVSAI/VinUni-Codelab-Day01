"""
Day 1 — LLM API Foundation
Test suite for student solutions and master keys.

Run from the 02-lab/ folder:
    pytest tests/ -v

All external API calls are mocked — no real API keys or internet connection required.
"""

import importlib.util
import inspect
import io
import contextlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Locate directories relative to this test file
TESTS_DIR = Path(__file__).parent
LAB_DIR = TESTS_DIR.parent

# Resolve module paths
SOLUTION_PATH = LAB_DIR / "solution-code" / "solution.py"
TEMPLATE_PATH = LAB_DIR / "starter-code" / "template.py"

def _load_module(path: Path, unique_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Target file not found at: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Dynamically load the solution module if it exists (for grading), otherwise test the starter template
if SOLUTION_PATH.exists():
    _m = _load_module(SOLUTION_PATH, "solution")
else:
    _m = _load_module(TEMPLATE_PATH, "template")

# Import target functions
call_openai = getattr(_m, "call_openai")
call_gemini = getattr(_m, "call_gemini")
call_anthropic = getattr(_m, "call_anthropic")
compare_models = getattr(_m, "compare_models")
streaming_chatbot = getattr(_m, "streaming_chatbot")
retry_with_backoff = getattr(_m, "retry_with_backoff")
batch_compare = getattr(_m, "batch_compare")
format_comparison_table = getattr(_m, "format_comparison_table")


# ---------------------------------------------------------------------------
# Mock Generators
# ---------------------------------------------------------------------------
def _make_openai_response(text: str = "Hello from OpenAI"):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 25
    resp.usage_metadata = usage
    return resp


def _make_anthropic_response(text: str = "Hello from Anthropic"):
    resp = MagicMock()
    content_part = MagicMock()
    content_part.text = text
    resp.content = [content_part]
    usage = MagicMock()
    usage.input_tokens = 15
    usage.output_tokens = 30
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestCallOpenAI(unittest.TestCase):

    @patch("openai.OpenAI")
    def test_openai_returns_tuple_structure(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("OpenAI Mock Response")

        text, latency, usage = call_openai("Hello")

        self.assertEqual(text, "OpenAI Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 20)


class TestCallGemini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_gemini_returns_tuple_structure_new_sdk(self, MockGenAIClient):
        mock_client = MagicMock()
        MockGenAIClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response("Gemini Mock Response")

        text, latency, usage = call_gemini("Hello")

        self.assertEqual(text, "Gemini Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 25)


class TestCallAnthropic(unittest.TestCase):

    @patch("openai.OpenAI")
    def test_anthropic_name_maps_to_openrouter_free_router(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response(
            "Free Router Mock Response"
        )

        text, latency, usage = call_anthropic("Hello")

        self.assertEqual(text, "Free Router Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 20)
        self.assertEqual(
            MockOpenAI.call_args.kwargs["base_url"],
            "https://openrouter.ai/api/v1",
        )

    def test_openrouter_key_is_used_for_free_router_default(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openrouter-key"},
        ), patch("openai.OpenAI") as mock_openai:
            call_anthropic("Hello")

        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "openrouter-key")

    def test_missing_openrouter_key_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True), patch("openai.OpenAI") as mock_openai:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                call_anthropic("Hello")

        mock_openai.assert_not_called()

    def test_explicit_groq_override_uses_groq_key(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_BASE_URL": "https://api.groq.com/openai/v1",
                "GROQ_API_KEY": "groq-key",
                "OPENAI_API_KEY": "openrouter-key",
            },
        ), patch("openai.OpenAI") as mock_openai:
            call_anthropic("Hello")

        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "groq-key")
        self.assertEqual(
            mock_openai.call_args.kwargs["base_url"],
            "https://api.groq.com/openai/v1",
        )


class TestProviderConfiguration(unittest.TestCase):
    def test_free_tier_defaults_and_base_urls(self):
        self.assertEqual(_m.OPENROUTER_BASE_URL, "https://openrouter.ai/api/v1")
        self.assertEqual(_m.COMPAT_BASE_URL, "https://openrouter.ai/api/v1")
        self.assertTrue(_m.OPENAI_MODEL.endswith(":free"))
        self.assertTrue(_m.OPENAI_MINI_MODEL.endswith(":free"))
        self.assertEqual(_m.GEMINI_MODEL, "gemini-3.8-flash")
        self.assertEqual(_m.COMPAT_MODEL, "openrouter/free")

    def test_public_signatures_are_preserved(self):
        expected_names = {
            "call_openai": {"prompt", "model", "temperature", "top_p", "max_tokens"},
            "call_gemini": {"prompt", "model", "temperature", "top_p", "max_tokens"},
            "call_anthropic": {"prompt", "model", "temperature", "top_p", "max_tokens"},
            "compare_models": {"prompt"},
            "streaming_chatbot": set(),
            "retry_with_backoff": {"fn", "max_retries", "base_delay"},
            "batch_compare": {"prompts"},
            "format_comparison_table": {"results"},
        }
        for function_name, parameter_names in expected_names.items():
            function = getattr(_m, function_name)
            self.assertEqual(
                set(inspect.signature(function).parameters),
                parameter_names,
            )


class TestCompareModels(unittest.TestCase):

    def test_compare_models_evaluates_correct_costs(self):
        with patch.object(_m, "call_openai") as mock_openai, \
             patch.object(_m, "call_gemini") as mock_gemini, \
             patch.object(_m, "call_anthropic") as mock_compat:
             
            # Setup mock returns
            mock_openai.side_effect = [
                ("GPT-4o Response", 0.5, {"input_tokens": 10, "output_tokens": 20}),       # OpenRouter gpt-4o
                ("GPT-4o-Mini Response", 0.3, {"input_tokens": 10, "output_tokens": 20}),  # OpenRouter gpt-4o-mini
            ]
            mock_gemini.return_value = (
                "Gemini Response",
                0.4,
                {"input_tokens": 10, "output_tokens": 20}
            )
            mock_compat.return_value = (
                "Free Router Response",
                0.2,
                {"input_tokens": 10, "output_tokens": 20}
            )

            result = compare_models("Prompt")

            # Verify keys exist
            self.assertIn("gpt4o", result)
            self.assertIn("gpt4o_mini", result)
            self.assertIn("gemini_flash", result)
            self.assertIn("free_router", result)

            # Verify exact cost calculations
            # Both OpenRouter selections are explicit :free models.
            self.assertAlmostEqual(result["gpt4o"]["cost"], 0.0, places=8)
            
            self.assertAlmostEqual(result["gpt4o_mini"]["cost"], 0.0, places=8)
            
            # Gemini 3.8 Flash is free of charge on Google's documented Free Tier.
            self.assertAlmostEqual(result["gemini_flash"]["cost"], 0.0, places=8)

            # The Free Models Router is documented at USD 0 for both directions.
            self.assertAlmostEqual(result["free_router"]["cost"], 0.0, places=8)

    def test_compare_models_continues_when_one_provider_fails(self):
        with patch.object(_m, "call_openai") as mock_openai, \
             patch.object(_m, "call_gemini") as mock_gemini, \
             patch.object(_m, "call_anthropic") as mock_compat:
            mock_openai.side_effect = [
                ("GPT-4o Response", 0.5, {"input_tokens": 10, "output_tokens": 20}),
                ("GPT-4o-Mini Response", 0.3, {"input_tokens": 10, "output_tokens": 20}),
            ]
            mock_gemini.return_value = (
                "Gemini Response",
                0.4,
                {"input_tokens": 10, "output_tokens": 20},
            )
            mock_compat.side_effect = RuntimeError("Provider unavailable")

            result = compare_models("Prompt")

            self.assertEqual(result["gpt4o"]["response"], "GPT-4o Response")
            self.assertEqual(result["gemini_flash"]["response"], "Gemini Response")
            self.assertIn("error", result["free_router"])
            self.assertEqual(result["free_router"]["provider"], "OpenRouter Free Router")

    def test_compare_models_isolates_invalid_model_error(self):
        class FakeNotFoundError(Exception):
            status_code = 404

        with patch.object(_m, "call_openai") as mock_openai, \
             patch.object(_m, "call_gemini") as mock_gemini, \
             patch.object(_m, "call_anthropic") as mock_compat:
            mock_openai.side_effect = FakeNotFoundError("Model unavailable")
            mock_gemini.return_value = (
                "Gemini Response",
                0.4,
                {"input_tokens": 10, "output_tokens": 20},
            )
            mock_compat.return_value = (
                "Free Router Response",
                0.2,
                {"input_tokens": 10, "output_tokens": 20},
            )

            result = compare_models("Prompt")

        self.assertIn("error", result["gpt4o"])
        self.assertEqual(result["gemini_flash"]["response"], "Gemini Response")
        self.assertEqual(result["free_router"]["response"], "Free Router Response")


class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_immediately(self):
        val = retry_with_backoff(lambda: "success")
        self.assertEqual(val, "success")

    def test_succeeds_after_retries(self):
        state = {"attempts": 0}

        def flaky_fn():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("Fail")
            return "ok"

        val = retry_with_backoff(flaky_fn, max_retries=3, base_delay=0.001)
        self.assertEqual(val, "ok")
        self.assertEqual(state["attempts"], 3)

    def test_raises_permanent_exception(self):
        def permanent_fail():
            raise RuntimeError("Permanent")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(permanent_fail, max_retries=2, base_delay=0.001)

    def test_does_not_retry_authentication_error(self):
        class FakeAuthenticationError(Exception):
            status_code = 401

        attempts = 0

        def authenticated_failure():
            nonlocal attempts
            attempts += 1
            raise FakeAuthenticationError("Invalid credentials")

        with self.assertRaises(FakeAuthenticationError):
            retry_with_backoff(authenticated_failure, max_retries=3, base_delay=0.001)

        self.assertEqual(attempts, 1)

    def test_uses_retry_after_header_for_rate_limit(self):
        class FakeRateLimitError(Exception):
            status_code = 429

        response = MagicMock()
        response.headers = {"retry-after": "2"}
        error = FakeRateLimitError("Too many requests")
        error.response = response
        attempts = 0

        def rate_limited():
            nonlocal attempts
            attempts += 1
            raise error

        with patch.object(_m.time, "sleep") as mock_sleep:
            with self.assertRaises(FakeRateLimitError):
                retry_with_backoff(rate_limited, max_retries=1, base_delay=0.001)

        self.assertEqual(attempts, 2)
        mock_sleep.assert_called_once_with(2.0)


class TestBatchCompareAndFormat(unittest.TestCase):

    def test_batch_runs_and_formats_table(self):
        with patch.object(_m, "compare_models") as mock_comp:
            def _get_mock():
                return {
                    "gpt4o": {"response": "GPT-4o response content here", "latency": 0.5, "cost": 0.00045, "input_tokens": 10, "output_tokens": 20},
                    "gpt4o_mini": {"response": "GPT-4o-Mini response content here", "latency": 0.3, "cost": 0.0000135, "input_tokens": 10, "output_tokens": 20},
                    "gemini_flash": {"response": "Gemini response content here", "latency": 0.4, "cost": 0.00000675, "input_tokens": 10, "output_tokens": 20}
                    ,
                    "free_router": {"response": "", "latency": 0.2, "cost": None, "input_tokens": 0, "output_tokens": 0, "provider": "OpenRouter Free Router", "model": "openrouter/free", "error": "RuntimeError: unavailable"}
                }
            mock_comp.side_effect = _get_mock

            results = batch_compare(["Q1", "Q2"])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["prompt"], "Q1")

            table = format_comparison_table(results)
            self.assertIsInstance(table, str)
            self.assertIn("Prompt", table)
            self.assertIn("GPT-4o", table)
            self.assertIn("gemini-3.8-flash", table)
            self.assertIn("OpenRouter Free Router", table)
            self.assertIn("unavailable", table)
            self.assertIn("Q1", table)


class TestStreamingChatbot(unittest.TestCase):

    @patch("builtins.input", side_effect=["quit"])
    @patch("google.genai.Client")
    def test_exits_cleanly(self, MockClient, mock_input):
        # Ensure it quits cleanly without entering infinite loop
        try:
            streaming_chatbot()
        except SystemExit:
            pass

    @patch("builtins.input", side_effect=["hello", "quit"])
    @patch("google.genai.Client")
    def test_handles_streaming_error(self, MockClient, mock_input):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = RuntimeError("offline")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            streaming_chatbot()

        self.assertIn("Gemini streaming error", output.getvalue())
        self.assertIn("offline", output.getvalue())

    @patch("builtins.input", side_effect=["hello", "quit"])
    @patch("google.genai.Client")
    def test_handles_empty_stream(self, MockClient, mock_input):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.return_value = iter([])
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            streaming_chatbot()

        self.assertIn("empty stream", output.getvalue())


if __name__ == "__main__":
    unittest.main()
