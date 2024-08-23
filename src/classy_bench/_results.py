import io
from typing import Dict

import pandas as pd

INDEX_AVG_TYPES = ["micro avg", "macro avg", "weighted avg"]
METRICS_COL_NAMES = ["precision", "recall", "f1-score"]


def _clean_metric_name(avg_type: str, metric_name: str) -> str:
    return "_".join([x for x in [*avg_type.split(), metric_name.replace("-", "_")] if x != "avg"])


def _check_index_and_column_names(df: pd.DataFrame) -> None:
    index_names_correct = all(expected_index in df.index for expected_index in INDEX_AVG_TYPES)
    column_names_correct = all(expected_col_name in df.columns for expected_col_name in METRICS_COL_NAMES)
    if index_names_correct and column_names_correct:
        return

    err_str = (
        "The expected columns and index names are not found. "
        f"Got index names: {df.index.names} and column names: {df.columns}"
    )
    raise ValueError(err_str)


def get_metrics_results(csv_string: str) -> Dict[str, float]:
    """Get the metrics results as a dictionary."""
    df = pd.read_csv(io.StringIO(csv_string), index_col=0)
    _check_index_and_column_names(df)
    df = df.loc[INDEX_AVG_TYPES, METRICS_COL_NAMES]
    metrics_dict = {}
    for metric_name in METRICS_COL_NAMES:
        new_metrics = (
            df[metric_name]
            .rename(lambda x, metric_name=metric_name: _clean_metric_name(avg_type=x, metric_name=metric_name))
            .to_dict()
        )
        metrics_dict = {**metrics_dict, **new_metrics}

    return metrics_dict
