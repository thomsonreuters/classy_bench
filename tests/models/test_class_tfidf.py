import ast
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from classy_bench.models.class_tfidf import evaluate, preprocess, train
from tests.testing_utils import TEST_DATA_PATH, TEST_MODELS_PATH

CLASS_TFIDF_TESTS_PATH = TEST_MODELS_PATH / "class_tfidf"


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
    expected_train_df = pd.read_csv(str(CLASS_TFIDF_TESTS_PATH / "expected_preprocess_output_train.csv"))
    expected_train_df["labels"] = expected_train_df["labels"].apply(ast.literal_eval)
    assert_frame_equal(train_df, expected_train_df)


# test individual functions for main also
def test_train_main(output_data_location: Path):
    train.main(
        input_data_location=str(TEST_DATA_PATH),
        output_checkpoints_location=str(output_data_location),
        output_model_artifacts_location=str(output_data_location),
        output_dir_location=str(output_data_location),
        subset=True,
        samples=1000,
        tfidf_max_features=50000,
    )
    expected_files = [
        "num_labels_train.npz",
        "label_encoder.pkl",
        "tfidf_vectorizer.pkl",
        "tfidf_m.npz",
        "bm25.pkl",
    ]
    output_files = [f_.name for f_ in output_data_location.glob("*")]
    assert all(file in output_files for file in expected_files)


def test_evaluate_main(output_data_location: Path):
    evaluate.main(
        input_data_location=str(TEST_DATA_PATH),
        input_model_location=str(output_data_location),
        output_data_location=str(output_data_location),
        output_model_location=str(output_data_location),
        output_report_location=str(output_data_location),
        subset=True,
        samples=1000,
        thresholds_str="0.1,0.2",
    )

    expected_files = [
        "num_labels_train.npz",
        "label_encoder.pkl",
        "tfidf_vectorizer.pkl",
        "tfidf_m.npz",
        "bm25.pkl",
        "predictions_list_topn.csv",
        "metrics_top_n.csv",
        "predictions_list_th0.1.csv",
        "metrics_threshold_0.1.csv",
        "predictions_list_th0.2.csv",
        "metrics_threshold_0.2.csv",
        "predictions_list_bm25.csv",
        "metrics_bm25.csv",
    ]
    output_files = [f_.name for f_ in output_data_location.glob("*")]
    assert all(file in output_files for file in expected_files)
