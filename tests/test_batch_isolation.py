import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


LAB_DIR = Path(__file__).parent.parent
SOLUTION_PATH = LAB_DIR / "solution-code" / "solution.py"

spec = importlib.util.spec_from_file_location("batch_solution", SOLUTION_PATH)
solution = importlib.util.module_from_spec(spec)
sys.modules["batch_solution"] = solution
spec.loader.exec_module(solution)


class TestBatchIsolation(unittest.TestCase):
    def test_batch_compare_continues_after_a_prompt_fails(self):
        successful_result = {
            "gpt4o": {
                "response": "OK",
                "latency": 0.1,
                "cost": 0.0001,
                "input_tokens": 1,
                "output_tokens": 1,
                "provider": "OpenRouter",
                "model": "google/gemma-4-31b-it:free",
            }
        }

        with patch.object(solution, "compare_models") as mock_compare:
            mock_compare.side_effect = [successful_result, RuntimeError("Provider down")]
            results = solution.batch_compare(["first", "second"])

        self.assertEqual(results[0]["prompt"], "first")
        self.assertEqual(results[0]["gpt4o"]["response"], "OK")
        self.assertEqual(results[1]["prompt"], "second")
        self.assertIn("error", results[1])


if __name__ == "__main__":
    unittest.main()
