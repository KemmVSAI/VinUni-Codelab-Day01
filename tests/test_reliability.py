import contextlib
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


LAB_DIR = Path(__file__).parent.parent
SOLUTION_PATH = LAB_DIR / "solution-code" / "solution.py"
spec = importlib.util.spec_from_file_location("reliability_solution", SOLUTION_PATH)
solution = importlib.util.module_from_spec(spec)
sys.modules["reliability_solution"] = solution
spec.loader.exec_module(solution)


def _openai_response(text, model=None):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = 5
    response.usage.completion_tokens = 7
    if model is not None:
        response.model = model
    return response


def _gemini_response(text, model):
    response = MagicMock()
    response.text = text
    response.usage_metadata.prompt_token_count = 8
    response.usage_metadata.candidates_token_count = 9
    response.model = model
    return response


def _gemini_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk


class TestOpenRouterFallback(unittest.TestCase):
    def test_rate_limited_free_model_falls_back_to_free_router(self):
        class RateLimitError(Exception):
            status_code = 429

        responses = [
            RateLimitError("Free model is rate limited"),
            _openai_response("router response"),
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "openai.OpenAI"
        ) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = responses
            text, latency, usage = solution.call_openai(
                "Hello",
                solution.OPENAI_MODEL,
            )

        calls = mock_openai.return_value.chat.completions.create.call_args_list
        self.assertEqual(text, "router response")
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(
            usage["model"],
            "openrouter/free",
            msg=f"calls={calls} usage={usage}",
        )
        self.assertEqual(usage["input_tokens"], 5)
        self.assertEqual(usage["output_tokens"], 7)
        self.assertEqual(calls[0].kwargs["model"], solution.OPENAI_MODEL)
        self.assertEqual(calls[1].kwargs["model"], "openrouter/free")

    def test_paid_model_is_never_used_as_fallback(self):
        class RateLimitError(Exception):
            status_code = 429

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "openai.OpenAI"
        ) as mock_openai, patch.object(
            solution,
            "OPENAI_FALLBACK_MODELS",
            ["openai/gpt-4o"],
        ):
            mock_openai.return_value.chat.completions.create.side_effect = RateLimitError(
                "rate limited"
            )
            with self.assertRaises(RateLimitError):
                solution.call_openai("Hello", solution.OPENAI_MODEL)

        create = mock_openai.return_value.chat.completions.create
        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.kwargs["model"], solution.OPENAI_MODEL)

    def test_fallback_chain_is_bounded(self):
        class ServerError(Exception):
            status_code = 503

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "openai.OpenAI"
        ) as mock_openai, patch.object(
            solution,
            "OPENAI_FALLBACK_MODELS",
            ["fallback-a:free", "fallback-b:free"],
        ):
            mock_openai.return_value.chat.completions.create.side_effect = ServerError(
                "unavailable"
            )
            with self.assertRaises(ServerError):
                solution.call_openai("Hello", solution.OPENAI_MODEL)

        self.assertEqual(
            mock_openai.return_value.chat.completions.create.call_count,
            3,
        )

    def test_http_200_empty_reasoning_response_falls_back(self):
        class RateLimitError(Exception):
            status_code = 429

        empty_response = _openai_response("")
        empty_response.model = "nvidia/nemotron-3.5-lightning:free"
        empty_response.provider = "Nvidia"
        empty_response.choices[0].finish_reason = "length"
        content_response = _openai_response("Fallback response")
        content_response.model = "google/gemma-4-26b-a4b-it:free"
        content_response.provider = "Google AI Studio"
        content_response.choices[0].finish_reason = "stop"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "openai.OpenAI"
        ) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = [
                RateLimitError("primary rate limited"),
                empty_response,
                content_response,
            ]
            text, latency, usage = solution.call_openai(
                "Hello",
                solution.OPENAI_MODEL,
            )

        self.assertEqual(text, "Fallback response")
        self.assertEqual(usage["model"], "google/gemma-4-26b-a4b-it:free")
        self.assertEqual(usage["finish_reason"], "stop")
        calls = mock_openai.return_value.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], solution.OPENAI_MODEL)
        self.assertEqual(calls[1].kwargs["model"], "openrouter/free")
        self.assertEqual(
            calls[2].kwargs["model"],
            "google/gemma-4-26b-a4b-it:free",
        )


class TestGeminiFallback(unittest.TestCase):
    def test_verified_free_models_use_bounded_thinking(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _gemini_response("OK", "gemini-3.8-flash")
            )
            solution.call_gemini("Hello")

        config = mock_client.return_value.models.generate_content.call_args.kwargs[
            "config"
        ]
        self.assertEqual(config.thinking_config.thinking_level.value, "LOW")
        self.assertEqual(config.max_output_tokens, 256)

    def test_overloaded_flash_falls_back_to_verified_free_model(self):
        class ServerError(Exception):
            code = 503

        responses = [
            ServerError("overloaded"),
            _gemini_response("fallback response", "gemini-3.6-flash"),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as mock_client:
            mock_client.return_value.models.generate_content.side_effect = responses
            text, latency, usage = solution.call_gemini("Hello")

        self.assertEqual(text, "fallback response")
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["model"], "gemini-3.6-flash")
        calls = mock_client.return_value.models.generate_content.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(calls[1].kwargs["model"], "gemini-3.6-flash")

    def test_unknown_model_does_not_receive_hidden_fallback(self):
        class ServerError(Exception):
            code = 503

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as mock_client:
            mock_client.return_value.models.generate_content.side_effect = ServerError(
                "unavailable"
            )
            with self.assertRaises(ServerError):
                solution.call_gemini("Hello", model="custom-model")

        self.assertEqual(
            mock_client.return_value.models.generate_content.call_count,
            1,
        )


class TestRetryPolicy(unittest.TestCase):
    def test_exponential_backoff_adds_bounded_jitter(self):
        class ServerError(Exception):
            status_code = 500

        with patch.object(solution.random, "uniform", return_value=0.05), patch.object(
            solution.time,
            "sleep",
        ) as mock_sleep:
            with self.assertRaises(ServerError):
                solution.retry_with_backoff(
                    lambda: (_ for _ in ()).throw(ServerError("failed")),
                    max_retries=1,
                    base_delay=0.1,
                )

        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args.args[0], 0.15)

    def test_retry_after_and_maximum_delay_are_respected(self):
        class RateLimitError(Exception):
            status_code = 429

        error = RateLimitError("too many requests")
        error.response = MagicMock()
        error.response.headers = {"retry-after": "10"}

        with patch.object(solution, "MAX_RETRY_DELAY_SECONDS", 2.0), patch.object(
            solution.time,
            "sleep",
        ) as mock_sleep:
            with self.assertRaises(RateLimitError):
                solution.retry_with_backoff(
                    lambda: (_ for _ in ()).throw(error),
                    max_retries=1,
                    base_delay=0.1,
                )

        mock_sleep.assert_called_once_with(2.0)

    def test_quota_error_is_not_retried(self):
        class QuotaError(Exception):
            status_code = 402

        attempts = 0

        def fail():
            nonlocal attempts
            attempts += 1
            raise QuotaError("credits required")

        with self.assertRaises(QuotaError):
            solution.retry_with_backoff(fail, max_retries=3, base_delay=0.001)

        self.assertEqual(attempts, 1)


class TestStreamingReliability(unittest.TestCase):
    def test_history_is_limited_to_six_messages(self):
        streams = [
            iter([_gemini_chunk("1")]),
            iter([_gemini_chunk("2")]),
            iter([_gemini_chunk("3")]),
            iter([_gemini_chunk("4")]),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as mock_client, patch(
            "builtins.input",
            side_effect=["one", "two", "three", "four", "quit"],
        ):
            mock_client.return_value.models.generate_content_stream.side_effect = streams
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                solution.streaming_chatbot()

        calls = mock_client.return_value.models.generate_content_stream.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertLessEqual(len(calls[-1].kwargs["contents"]), 6)

    def test_stream_creation_failure_uses_one_free_fallback(self):
        class ServerError(Exception):
            code = 503

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as mock_client, patch(
            "builtins.input",
            side_effect=["hello", "quit"],
        ):
            mock_client.return_value.models.generate_content_stream.side_effect = [
                ServerError("overloaded"),
                iter([_gemini_chunk("fallback text")]),
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                solution.streaming_chatbot()

        self.assertIn("fallback text", output.getvalue())
        calls = mock_client.return_value.models.generate_content_stream.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(calls[1].kwargs["model"], "gemini-3.6-flash")

    def test_interrupted_stream_does_not_append_partial_history(self):
        class InterruptedStream:
            def __iter__(self):
                yield _gemini_chunk("partial ")
                raise RuntimeError("connection lost")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as mock_client, patch(
            "builtins.input",
            side_effect=["hello", "quit"],
        ):
            mock_client.return_value.models.generate_content_stream.side_effect = [
                InterruptedStream()
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                solution.streaming_chatbot()

        self.assertIn("partial ", output.getvalue())
        self.assertIn("connection lost", output.getvalue())
        self.assertEqual(
            mock_client.return_value.models.generate_content_stream.call_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
