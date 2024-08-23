import ast
from pathlib import Path
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

from classy_bench.models.roberta import preprocess, train
from tests.testing_utils import TEST_DATA_PATH, TEST_MODELS_PATH

ROBERTA_TESTS_PATH = TEST_MODELS_PATH / "roberta"


def test_preprocess(output_data_location: Path):
    preprocess.main(
        input_data_location=str(TEST_DATA_PATH),
        output_data_location=str(output_data_location),
        output_model_location=str(output_data_location),
        output_report_location=str(output_data_location),
        subset=True,
        samples=10,
    )
    expected_train_df = pd.read_csv(str(ROBERTA_TESTS_PATH / "expected_preprocess_output_train.csv"), index_col=False)
    expected_train_df["labels"] = expected_train_df["labels"].apply(ast.literal_eval)
    output_train_df = pd.read_csv(output_data_location / "train.csv", index_col=False)
    output_train_df["labels"] = output_train_df["labels"].apply(ast.literal_eval)
    assert_frame_equal(output_train_df, expected_train_df)


def test_train(output_data_location: Path):
    with mock.patch("transformers.Trainer.train") as mocker_trainer_train, mock.patch(
        "transformers.Trainer.save_model"
    ) as mock_trainer_save:
        train.main(
            input_data_location=str(TEST_DATA_PATH),
            output_checkpoints_location=str(output_data_location),
            output_model_artifacts_location=str(output_data_location),
            output_dir_location=str(output_data_location),
            tokenizer_model_ckpt="distilroberta-base",
            tokenizer_do_lower_case=True,
            training_arguments_batch_size=2,
            training_arguments_log_per_epoch=2,
            training_arguments_eval_per_epoch=2,
            training_arguments_lr=0.0001,
            training_arguments_n_epochs=1,
            training_arguments_loss_func_name="sigmoid_bce_loss",
            training_arguments_early_stopping_patience=8,
        )
        mocker_trainer_train.assert_called_once()
        mock_trainer_save.assert_called_once()
