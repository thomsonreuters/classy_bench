import ast
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from classy_bench.models.bm25 import evaluate, preprocess, train
from tests.testing_utils import TEST_DATA_PATH, TEST_MODELS_PATH

BM25_TESTS_PATH = TEST_MODELS_PATH / "bm25"


def test_preprocess_main(output_data_location: Path):
    train_df, _, _ = preprocess.main(
        input_data_location=str(TEST_DATA_PATH),
        output_data_location=str(output_data_location),
        output_model_location=str(output_data_location),
        output_report_location=str(output_data_location),
        subset=True,
        samples=10,
        preprocess=True,
    )
    expected_train_df = pd.read_csv(str(BM25_TESTS_PATH / "expected_preprocess_output_train.csv"))
    expected_train_df["labels"] = expected_train_df["labels"].apply(ast.literal_eval)
    assert_frame_equal(train_df, expected_train_df)


def test_train_main(output_data_location: Path):
    train.main(
        input_data_location=str(TEST_DATA_PATH),
        output_checkpoints_location=str(output_data_location),
        output_model_artifacts_location=str(output_data_location),
        output_dir_location=str(output_data_location),
        subset=True,
        samples=1000,
    )
    expected_files = ["bm25_model.pkl"]
    output_files = [f_.name for f_ in output_data_location.glob("*")]
    assert all(file in output_files for file in expected_files)


def test_evaluate_main(output_data_location: Path):
    evaluate.main(
        input_data_location=str(TEST_DATA_PATH),
        input_model_location=str(output_data_location),
        output_data_location=str(output_data_location),
        output_report_location=str(output_data_location),
        subset=True,
        samples=1000,
    )
    expected_files = sorted(["bm25_model.pkl", "predictions_bm25.csv", "metrics.csv"])
    assert sorted([f_.name for f_ in output_data_location.glob("*")]) == expected_files
