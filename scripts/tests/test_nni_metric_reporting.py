from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from utils import nni_encoding_param_optimization_backbone as backbone
from utils.nni_braille_reading_rsnn import make_accuracy_metrics


class TestNniMetricReporting(TestCase):
    def test_accuracy_payload_uses_validation_as_default(self):
        metrics = make_accuracy_metrics(
            validation_accuracy=0.75,
            training_accuracy=0.90,
            test_accuracy=0.70,
        )

        self.assertEqual(
            list(metrics),
            ["default", "training_accuracy", "test_accuracy"],
        )
        self.assertEqual(metrics, {
            "default": 0.75,
            "training_accuracy": 0.90,
            "test_accuracy": 0.70,
        })

    def test_accuracy_payload_rejects_invalid_values(self):
        for invalid_accuracy in (-0.01, 1.01, float("nan")):
            with self.subTest(invalid_accuracy=invalid_accuracy):
                with self.assertRaises(ValueError):
                    make_accuracy_metrics(
                        validation_accuracy=invalid_accuracy,
                        training_accuracy=0.90,
                        test_accuracy=0.70,
                    )

    def test_train_and_evaluate_returns_best_validation_repetition_metrics(self):
        first = make_accuracy_metrics(
            validation_accuracy=0.60,
            training_accuracy=0.95,
            test_accuracy=0.58,
        )
        best = make_accuracy_metrics(
            validation_accuracy=0.80,
            training_accuracy=0.85,
            test_accuracy=0.76,
        )

        def fake_run_training(**kwargs):
            callback = kwargs["intermediate_result_callback"]
            callback(first)
            callback(best)
            return best["default"]

        with (
            patch.object(backbone, "run_training", side_effect=fake_run_training),
            patch.object(backbone.nni, "report_intermediate_result") as report,
        ):
            result = backbone.train_and_evaluate(Path("encoded.pkl"), "trial-1")

        self.assertEqual(result, best)
        self.assertEqual(report.call_args_list[0].args, (first,))
        self.assertEqual(report.call_args_list[1].args, (best,))

    def test_train_and_evaluate_rejects_mismatched_optimization_metric(self):
        metrics = make_accuracy_metrics(
            validation_accuracy=0.80,
            training_accuracy=0.85,
            test_accuracy=0.76,
        )

        def fake_run_training(**kwargs):
            kwargs["intermediate_result_callback"](metrics)
            return 0.70

        with (
            patch.object(backbone, "run_training", side_effect=fake_run_training),
            patch.object(backbone.nni, "report_intermediate_result"),
            self.assertRaisesRegex(
                RuntimeError,
                "best reported validation accuracy",
            ),
        ):
            backbone.train_and_evaluate(Path("encoded.pkl"), "trial-1")
