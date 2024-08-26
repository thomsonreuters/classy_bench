import ast
from pathlib import Path
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

from classy_bench.models.t5 import preprocess, train
from tests.testing_utils import TEST_DATA_PATH, TEST_MODELS_PATH

T5_TESTS_PATH = TEST_MODELS_PATH / "t5"


def test_preprocess(output_data_location: Path):
    train_df, _, _ = preprocess.main(
        input_data_location=str(TEST_DATA_PATH),
        output_data_location=str(output_data_location),
        output_model_location=str(output_data_location),
        output_report_location=str(output_data_location),
        subset=True,
        samples=10,
        preprocess=True,
    )
    expected_train_df = pd.read_csv(str(T5_TESTS_PATH / "expected_preprocess_output_train.csv"), index_col=False)
    expected_train_df["labels"] = expected_train_df["labels"].apply(ast.literal_eval)
    assert_frame_equal(train_df, expected_train_df)


def test_train(output_data_location: Path):
    with mock.patch("transformers.Seq2SeqTrainer.train") as mocker_trainer_train, mock.patch(
        "transformers.Seq2SeqTrainer.save_model"
    ) as mock_trainer_save:
        train.main(
            input_data_location=str(TEST_DATA_PATH),
            output_checkpoints_location=str(output_data_location),
            output_model_artifacts_location=str(output_data_location),
            output_dir_location=str(output_data_location),
            subset=True,
            samples=10,
            epochs=1,
            batch_size=2,
            learning_rate=0.0001,
            model_identifier="t5-small",
            early_stopping_patience=3,
            task_prefix="summarize",
        )
        mocker_trainer_train.assert_called_once()
        mock_trainer_save.assert_called_once()
