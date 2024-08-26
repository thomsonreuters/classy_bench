"""class_tfidf Evaluate."""

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
import logging
import os
import pickle
import tarfile

import nltk
import pandas as pd
from scipy import sparse

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

import multiprocessing
import random
import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

seed = 42
random.seed(seed)
np.random.seed(seed)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


def get_index_of_labels_with_support(real_list: np.ndarray, target_names: np.ndarray) -> List[int]:
    # only compute classification report on labels that occur in set to evaluate
    real_count = np.sum(real_list, axis=0)
    labels_with_support = [i for i in range(len(target_names)) if real_count[i] > 0]
    return labels_with_support


def get_all_metrics_df(test_df: pd.DataFrame, label_encoder: LabelEncoder) -> pd.DataFrame:
    # inverse transform the label encoding
    test_df["predicted_labels"] = test_df["predictions"].apply(lambda x: label_encoder.inverse_transform(x[0]))

    all_labels = pd.concat([pd.Series(test_df["labels"]), pd.Series(test_df["predicted_labels"])])
    mlb = MultiLabelBinarizer().fit(all_labels)

    y_test = mlb.transform(pd.Series(test_df["labels"]))
    y_pred = mlb.transform(pd.Series(test_df["predicted_labels"]))

    label_set = mlb.classes_
    logger.info(f"number of classes in MultiLabelBinarizer for top top_n: {len(label_set)}")
    labels_with_support = get_index_of_labels_with_support(y_test, label_set)

    all_metric = classification_report(
        y_test,
        y_pred,
        digits=3,
        target_names=list(mlb.classes_),
        output_dict=True,
        zero_division=0,
        labels=labels_with_support,
    )
    return pd.DataFrame(all_metric).T


def _apply_df(args):
    """Apply df."""
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    """Apply multiprocessing."""
    workers = kwargs.get("workers", multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


def predict_labels(txt, tfidf_m, tfidf_vectorizer, top_n=5, threshold=None):
    """Identify the top_n most similar documents for the given text."""
    new_doc_tfidf = tfidf_vectorizer.transform([txt])

    sim_score = cosine_similarity(new_doc_tfidf, tfidf_m)

    if threshold:
        top_n = len(np.argwhere(sim_score >= threshold)[:, 1])

    top_n_index = sim_score[0].argsort()[::-1][:top_n]
    top_n_score = sim_score[0][sim_score[0].argsort()[::-1][:top_n]]
    return top_n_index, top_n_score


def get_prediction_results_for_top_n(tfidf_m, test_df, tfidf_vectorizer, label_encoder, top_n):
    """Get evaluation results for test_df, assuming you want to assign exactly top_n labels."""
    logger.info(f"Predicting labels for top_n={top_n}")
    start = time.time()
    test_df["predictions"] = apply_by_multiprocessing(
        test_df["text"], predict_labels, tfidf_m=tfidf_m, tfidf_vectorizer=tfidf_vectorizer, top_n=top_n
    )
    logger.info(f"Prediction Time {round(time.time()-start,3)} seconds")

    return get_all_metrics_df(test_df, label_encoder)


def get_threshold_based_predictions(tfidf_m, test_df, tfidf_vectorizer, label_encoder, threshold):
    """Get evaluation results for test_df, assuming you don't know how many labels you want to assign,
    and you just pick a similarity threshold for label assignment (the same threshold for each label class).
    """
    logger.info(f"Predicting labels for threshold={threshold}")
    start = time.time()
    test_df["predictions"] = apply_by_multiprocessing(
        test_df["text"], predict_labels, tfidf_m=tfidf_m, tfidf_vectorizer=tfidf_vectorizer, threshold=threshold
    )
    logger.info(f"Prediction Time {round(time.time()-start,3)} seconds")
    return get_all_metrics_df(test_df, label_encoder)


def get_top_n_bm25(txt: str, bm25, top_n):
    """Get_top_n_bm25."""
    tokenized_query = txt.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    top_n_index = doc_scores.argsort()[::-1][:top_n]
    top_n_score = doc_scores[doc_scores.argsort()[::-1][:top_n]]

    return top_n_index, top_n_score


def get_top_n_with_bm25(bm25, test_df, label_encoder, top_n):
    """Get evaluation results for test_df, assuming you want to assign top_n labels using BM25."""
    # Evaluation on test using BM25
    start = time.time()
    test_df["predictions"] = apply_by_multiprocessing(test_df["text"], get_top_n_bm25, bm25=bm25, top_n=top_n)
    logger.info(f"Prediction Time: {time.time()-start} seconds")
    return get_all_metrics_df(test_df, label_encoder)


def main(
    input_data_location: str,
    input_model_location: str,
    output_data_location: str,
    output_model_location: str,
    output_report_location: str,
    subset: bool,
    samples: int,
    thresholds_str: str,
):
    # Required as you used sagemaker-training before evaluation. Since sagemaker-training will tar all model artifacts
    # this Extraction is needed . But can be commented on Local Usage.
    if os.path.exists(os.path.join(input_model_location, "model.tar.gz")):
        model_path = os.path.join(input_model_location, "model.tar.gz")
        logger.info(f"Extracting model from path: {model_path}")

        with tarfile.open(model_path) as tar:
            tar.extractall(path=input_model_location)

    subdirectories = [output_data_location, output_report_location, output_model_location]

    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    start_time = time.time()

    dev_df = pd.read_csv(os.path.join(input_data_location, "dev.csv"))
    test_df = pd.read_csv(os.path.join(input_data_location, "test.csv"))

    if subset:
        dev_df = dev_df[:samples]
        test_df = test_df[:samples]

    dev_df["labels"] = dev_df["labels"].apply(ast.literal_eval)
    test_df["labels"] = test_df["labels"].apply(ast.literal_eval)

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    logger.info(f"test_df.shape: {test_df.shape}")

    logger.info("Loading num_labels_train")
    num_labels_train_npz = np.load(os.path.join(input_model_location, "num_labels_train.npz"))
    num_labels_train = num_labels_train_npz["num_labels_train_array"]

    top_n_based_on_train = int(np.median(num_labels_train))
    logger.info(f"Using top_n={top_n_based_on_train} because it's the median number of labels in train!")

    tfidf_m_file = os.path.join(input_model_location, "tfidf_m.npz")
    tfidf_m = sparse.load_npz(tfidf_m_file)

    tfidf_vectorizer_file = os.path.join(input_model_location, "tfidf_vectorizer.pkl")
    with open(tfidf_vectorizer_file, "rb") as file:
        tfidf_vectorizer = pickle.load(file)

    label_encoder_file = os.path.join(input_model_location, "label_encoder.pkl")
    with open(label_encoder_file, "rb") as file:
        label_encoder = pickle.load(file)

    # Evaluate on top_n
    all_metric_for_top_n_df = get_prediction_results_for_top_n(
        tfidf_m, test_df, tfidf_vectorizer, label_encoder, top_n_based_on_train
    )
    test_df["predicted_labels"].to_csv(os.path.join(output_report_location, "predictions_list_topn.csv"), header=False)
    all_metric_for_top_n_df.to_csv(os.path.join(output_report_location, "metrics_top_n.csv"))

    # Evaluate on thresholds
    thresholds = [float(i) for i in thresholds_str.split(",")]

    for threshold in thresholds:
        all_metric_for_threshold_df = get_threshold_based_predictions(
            tfidf_m, test_df, tfidf_vectorizer, label_encoder, threshold
        )
        test_df["predicted_labels"].to_csv(
            os.path.join(output_report_location, f"predictions_list_th{threshold}.csv"), header=False
        )
        all_metric_for_threshold_df.to_csv(os.path.join(output_report_location, f"metrics_threshold_{threshold}.csv"))

    bm25_file = os.path.join(input_model_location, "bm25.pkl")

    with open(bm25_file, "rb") as file:
        bm25 = pickle.load(file)

    all_metric_for_bm25_df = get_top_n_with_bm25(bm25, test_df, label_encoder, top_n_based_on_train)

    test_df["predicted_labels"].to_csv(os.path.join(output_report_location, "predictions_list_bm25.csv"), header=False)

    all_metric_for_bm25_df.to_csv(os.path.join(output_report_location, "metrics_bm25.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the model with preprocessed data, and store the results in output_model_location."
    )

    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: train.csv, dev.csv, test.csv",
        type=str,
        default="/opt/ml/processing/input/data/",
    )
    parser.add_argument(
        "--input_model_location",
        help="Directory containing the trained models from the previous processing step",
        type=str,
        default="/opt/ml/processing/input/model/",
    )
    parser.add_argument(
        "--output_data_location",
        help="Directory where the data after this processing step will get stored",
        type=str,
        default="/opt/ml/processing/output/data/",
    )
    parser.add_argument(
        "--output_model_location",
        help="Directory location for storing models for this processing step",
        type=str,
        default="/opt/ml/processing/output/model/",
    )
    parser.add_argument(
        "--output_report_location",
        help="Directory location for storing reports/logs for this processing step",
        type=str,
        default="/opt/ml/processing/output/report/",
    )
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.add_argument("--no_subset", dest="subset", action="store_false")
    parser.set_defaults(subset=False)
    parser.add_argument(
        "--samples", help="number of samples that should be used (for debugging)", type=int, default=1000
    )
    parser.add_argument("--thresholds", help="delimited ',' list of thresholds as string", type=str, default="0.1,0.2")

    parsed_args, _ = parser.parse_known_args()

    main(
        input_data_location=parsed_args.input_data_location,
        input_model_location=parsed_args.input_model_location,
        output_data_location=parsed_args.output_data_location,
        output_model_location=parsed_args.output_model_location,
        output_report_location=parsed_args.output_report_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        thresholds_str=parsed_args.thresholds,
    )
