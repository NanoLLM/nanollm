import unittest

from scripts.benchmark_public_llm import format_sample, prepare_continuation, wilson_interval


class BenchmarkHelpersTest(unittest.TestCase):
    def test_prepare_continuation_keeps_rightmost_context(self):
        full_ids, continuation_len, truncated = prepare_continuation(
            prompt_ids=[1, 2, 3, 4],
            full_ids=[1, 2, 3, 4, 5, 6],
            max_context=3,
        )
        self.assertEqual(full_ids, [3, 4, 5, 6])
        self.assertEqual(continuation_len, 2)
        self.assertTrue(truncated)

    def test_wilson_interval_contains_accuracy(self):
        low, high = wilson_interval(30, 100)
        self.assertLess(low, 0.30)
        self.assertGreater(high, 0.30)

    def test_winogrande_formats_shared_prefix(self):
        prompt, choices, answer = format_sample(
            "winogrande",
            {
                "sentence": "The trophy does not fit because _ is too large.",
                "option1": "the trophy",
                "option2": "the suitcase",
                "answer": "1",
            },
        )
        self.assertEqual(prompt, "The trophy does not fit because")
        self.assertEqual(
            choices,
            [" the trophy is too large.", " the suitcase is too large."],
        )
        self.assertEqual(answer, 0)

    def test_arc_maps_non_numeric_labels(self):
        prompt, choices, answer = format_sample(
            "arc_easy",
            {
                "question": "What is 2+2?",
                "choices": {"label": ["A", "B"], "text": ["3", "4"]},
                "answerKey": "B",
            },
        )
        self.assertEqual(prompt, "What is 2+2?\nAnswer:")
        self.assertEqual(choices, [" 3", " 4"])
        self.assertEqual(answer, 1)


if __name__ == "__main__":
    unittest.main()
