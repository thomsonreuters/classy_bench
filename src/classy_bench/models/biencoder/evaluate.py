import logging
import os
import subprocess
import sys
from typing import List

if os.path.exists("/opt/ml/processing/input/code/my_package/requirements.txt"):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "/opt/ml/processing/input/code/my_package/requirements.txt",
        ]
    )

import argparse
import ast
import random
import tarfile
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


def model_predict_biencoder(texts: np.ndarray, queries: np.ndarray, model_folder: str) -> np.ndarray:
    model = SentenceTransformer(model_folder, device="cuda")
    encoded_texts = model.encode(texts, device="cuda", convert_to_numpy=True)
    encoded_queries = model.encode(queries, device="cuda", convert_to_numpy=True)
    return cosine_similarity(encoded_texts, encoded_queries)


def predict_test_sample(
    sim_scores: np.ndarray, training_labels: List[str], threshold=None, median_num_train_labels=None
) -> List[str]:
    """
    Get predictions given the similarity scores and the training labels.
    If median_num_train_labels is set, that many labels will be assigned.
    If threshold is set, all labels with similarity above the threshold will be assigned.
    Either median_num_train_labels or threshold needs to be set.
    """
    if threshold is None and median_num_train_labels is None:
        err_str = "Please set threshold or median_num_train_labels!"
        raise ValueError(err_str)
    if threshold is not None and median_num_train_labels is not None:
        err_str = "Please set either threshold or median_num_train_labels, but not both!"
        raise ValueError(err_str)

    # indices of the training labels descending in similarity score, with the according similarity score
    closest_training_label_indices = sim_scores.argsort()[::-1]
    closest_training_label_scores = sorted(sim_scores, reverse=True)

    if median_num_train_labels is not None:
        # we pick the top_{median_num_train_labels} labels
        predicted_labels = [training_labels[i] for i in closest_training_label_indices[:median_num_train_labels]]
    else:
        # we assign all labels with the score above the threshold
        predicted_labels = [
            training_labels[i]
            for i, score in zip(closest_training_label_indices, closest_training_label_scores)
            if score >= threshold
        ]
    return predicted_labels


def get_index_of_labels_with_support(real_list: np.ndarray, target_names: np.ndarray) -> List[int]:
    # only compute classification report on labels that occur in set to evaluate
    real_count = np.sum(real_list, axis=0)
    labels_with_support = [i for i in range(len(target_names)) if real_count[i] > 0]
    return labels_with_support


def compute_evaluation_metrics(y_pred: pd.Series, y_true: pd.Series) -> pd.DataFrame:
    y_pred_list = y_pred.tolist()
    y_true_list = y_true.tolist()
    all_labels = [labels for label_sets in [y_true_list, y_pred_list] for labels in label_sets]
    mlb = MultiLabelBinarizer().fit(all_labels)
    y_test = mlb.transform(pd.Series(y_true_list))
    y_pred = mlb.transform(pd.Series(y_pred_list))
    label_set = mlb.classes_
    logger.info(f"number of classes in MultiLabelBinarizer: {len(label_set)}")
    labels_with_support = get_index_of_labels_with_support(y_test, label_set)

    target_names = [label_set[i] for i in labels_with_support]
    all_metric = classification_report(
        y_test,
        y_pred,
        digits=3,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
        labels=labels_with_support,
    )
    return pd.DataFrame(all_metric)


def main(
    input_data_location: str,
    input_model_location: str,
    output_data_location: str,
    output_report_location: str,
    subset: bool,
    samples: int,
    thresholds: str,
):

    # Required as you used sagemaker training before evaluation, so the model artifacts are in model.tar.gz.
    # OS check is needed for testing and also for local usage of script
    if os.path.exists(os.path.join(input_model_location, "model.tar.gz")):
        model_path = os.path.join(input_model_location, "model.tar.gz")
        logger.info(f"Extracting model from path: {model_path}")

        with tarfile.open(model_path) as tar:
            tar.extractall(path=input_model_location)
        logger.debug("Input model location files: %s", os.listdir(input_model_location))

    subdirectories = [output_data_location, output_report_location]

    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    start_time = time.time()

    train_df = pd.read_csv(os.path.join(input_data_location, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_data_location, "test.csv"))

    if subset:
        train_df = train_df[:samples]
        test_df = test_df[:samples]

    train_df["labels"] = train_df["labels"].apply(ast.literal_eval)
    test_df["labels"] = test_df["labels"].apply(ast.literal_eval)

    logger.info(f"test_df.shape: {test_df.shape}")
    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    # get predictions for test and save the predictions to output_data_location
    # for each document in test, we obtain the similarity scores (document, label_i) for each label_i in train
    # then, we pick the top_{median_number_of_labels_in_train} labels
    training_labels = train_df.explode("labels")["labels"].unique()
    logger.info(f"Num training labels: {len(training_labels)}")
    train_df["n_labels"] = train_df["labels"].apply(len)
    median_num_train_labels = int(np.median(train_df["n_labels"]))
    logger.info(f"Median number of training labels: {median_num_train_labels}")

    predict_start_time = time.time()
    sim_scores = model_predict_biencoder(test_df["text"].to_numpy(), training_labels, input_model_location)
    logger.debug(f"sim_scores.shape: {sim_scores.shape}")
    logger.debug(f"max sim_scores: {np.max(sim_scores)}")
    logger.debug(f"min sim_scores: {np.min(sim_scores)}")
    logger.debug(f"mean sim_scores: {np.mean(sim_scores)}")
    logger.debug(f"median sim_scores: {np.median(sim_scores)}")
    logger.info(f"prediction took {time.time() - predict_start_time} seconds")

    # get predictions for median num train labels
    test_df["predicted_labels_median"] = [
        predict_test_sample(sims, training_labels, median_num_train_labels=median_num_train_labels)
        for sims in sim_scores
    ]
    test_df["predicted_labels_median"].to_csv(os.path.join(output_report_location, "predictions_top_train_median.csv"))
    # get classification report with the metrics for each class (and micro, macro, weighted F1) and save it
    # to output_report_location
    all_metrics_df = compute_evaluation_metrics(test_df["predicted_labels_median"], test_df["labels"])
    all_metrics_df.transpose().to_csv(os.path.join(output_report_location, "metrics_top_train_median.csv"))

    predict_start_time = time.time()
    # get predictions for thresholds
    thresholds_list = [float(i) for i in thresholds.split(",")]
    for thresh in thresholds_list:
        test_df["predicted_labels_threshold"] = [
            predict_test_sample(sims, training_labels, threshold=thresh) for sims in sim_scores
        ]
        test_df["predicted_labels_threshold"].to_csv(
            os.path.join(output_report_location, f"predictions_threshold_{thresh}.csv")
        )
        # get classification report with the metrics for each class (and micro, macro, weighted F1) and save it
        # to output_report_location
        all_metrics_df_threshold = compute_evaluation_metrics(test_df["predicted_labels_threshold"], test_df["labels"])
        all_metrics_df_threshold.transpose().to_csv(
            os.path.join(output_report_location, f"metrics_threshold_{thresh}.csv")
        )

    logger.info(f"threshold predictions took {time.time() - predict_start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the model with the preprocessed data and store the output in output_data_location."
    )

    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: dev.csv, test.csv.",
        type=str,
        default="/opt/ml/processing/input/data/",
    )
    parser.add_argument(
        "--input_model_location",
        help="Directory containing any models from the previous processing step.",
        type=str,
        default="/opt/ml/processing/input/model/",
    )
    parser.add_argument(
        "--output_data_location",
        help="Directory where the data after this preprocessing step will get stored.",
        type=str,
        default="/opt/ml/processing/output/data/",
    )
    parser.add_argument(
        "--output_report_location",
        help="Directory location for storing reports/logs for this processing step.",
        type=str,
        default="/opt/ml/processing/output/report/",
    )

    parser.add_argument("--subset", action="store_true")
    parser.add_argument(
        "--samples", help="number of samples that should be used (for debugging)", type=int, default=1000
    )
    parser.add_argument(
        "--thresholds", help="delimited ',' list of thresholds as string", type=str, default="0.4,0.5,0.6,0.7"
    )

    parsed_args, _ = parser.parse_known_args()

    main(
        input_data_location=parsed_args.input_data_location,
        input_model_location=parsed_args.input_model_location,
        output_data_location=parsed_args.output_data_location,
        output_report_location=parsed_args.output_report_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        thresholds=parsed_args.thresholds,
    )
