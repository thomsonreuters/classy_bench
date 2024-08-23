import ast
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sentence_transformers import InputExample

from classy_bench.models.biencoder import evaluate, preprocess, train
from tests.testing_utils import TEST_DATA_PATH, TEST_MODELS_PATH

BIENCODER_TESTS_PATH = TEST_MODELS_PATH / "biencoder"


def test_preprocess(output_data_location: Path):
    train_data, train_df, _, _ = preprocess.main(
        input_data_location=str(TEST_DATA_PATH),
        output_data_location=str(output_data_location),
        output_model_location=str(output_data_location),
        output_report_location=str(output_data_location),
        subset=True,
        samples=10,
        preprocess=True,
        num_negative=2,
    )
    expected_train_df = pd.read_csv(str(BIENCODER_TESTS_PATH / "expected_preprocess_output_train.csv"), index_col=False)
    expected_train_df["labels"] = expected_train_df["labels"].apply(ast.literal_eval)
    assert_frame_equal(train_df, expected_train_df)
    assert all(isinstance(example, InputExample) for example in train_data)
    assert {e.texts[0] for e in train_data if e.label == 1} == set(train_df["text"].to_list())


def test_train(output_data_location: Path):
    with mock.patch("sentence_transformers.SentenceTransformer.fit") as mock_sentence_transformer_fit, mock.patch(
        "sentence_transformers.SentenceTransformer.save"
    ) as mock_sentence_transformer_save, mock.patch(
        "classy_bench.models.biencoder.train._load_training_examples"
    ) as mock_load_training_examples:
        mock_load_training_examples.return_value = [InputExample(texts=["A", "B"], label=1) for _ in range(2)]
        train.main(
            input_data_location=str(TEST_DATA_PATH),
            output_checkpoints_location=str(output_data_location),
            output_model_artifacts_location=str(output_data_location),
            output_dir_location=str(output_data_location),
            subset=True,
            samples=10,
            epochs=1,
            batch_size=2,
            base_model="casehold/custom-legalbert",
            loss="CosineSimilarityLoss",
        )
        mock_sentence_transformer_fit.assert_called_once()
        mock_sentence_transformer_save.assert_called_once()


def test_evaluate(output_data_location: Path):

    with mock.patch(
        "classy_bench.models.biencoder.evaluate.predict_test_sample"
    ) as mock_predict_test_sample, mock.patch(
        "classy_bench.models.biencoder.evaluate.model_predict_biencoder"
    ) as mock_model_predict_biencoder:
        mock_model_predict_biencoder.return_value = np.random.rand(2, 2)
        mock_predict_test_sample.return_value = ["A" for _ in range(3)]
        evaluate.main(
            input_data_location=str(TEST_DATA_PATH),
            input_model_location=str(output_data_location),
            output_data_location=str(output_data_location),
            output_report_location=str(output_data_location),
            subset=True,
            samples=10,
            thresholds="0.1,0.2",
        )
        assert mock_predict_test_sample.call_count == 6
        assert mock_model_predict_biencoder.call_count == 1
