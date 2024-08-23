import pytest

from classy_bench._results import _clean_metric_name, get_metrics_results


@pytest.mark.parametrize(
    ("avg_type", "metric_name", "expected"),
    [("micro avg", "precision", "micro_precision"), ("macro avg", "f1-score", "macro_f1_score")],
)
def test_clean_metric_name(avg_type: str, metric_name: str, expected: str):
    assert _clean_metric_name(avg_type=avg_type, metric_name=metric_name) == expected


CSV_DATA = """
,precision,recall,f1-score,support
micro avg,0.1,0.2,0.3,13683.0
macro avg,0.3,0.4,0.5,13683.0
weighted avg,0.5,0.6,0.7,13683.0
samples avg,0.12,0.12,0.12,13683.0
"""


def test_get_metrics_results():
    expected = {
        "micro_precision": 0.1,
        "macro_precision": 0.3,
        "weighted_precision": 0.5,
        "micro_recall": 0.2,
        "macro_recall": 0.4,
        "weighted_recall": 0.6,
        "micro_f1_score": 0.3,
        "macro_f1_score": 0.5,
        "weighted_f1_score": 0.7,
    }
    output = get_metrics_results(csv_string=CSV_DATA)
    assert output == expected


def test_get_metrics_results_error():
    wrong_csv_data = """
    ,precision,recall,f1-score,support
    micro avg,0.1,0.2,0.3,13683.0"""
    with pytest.raises(ValueError):
        get_metrics_results(csv_string=wrong_csv_data)
