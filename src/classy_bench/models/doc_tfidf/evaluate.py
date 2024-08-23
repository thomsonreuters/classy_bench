import logging
import os
import subprocess
import sys

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
import multiprocessing
import pickle
import random
import tarfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

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


def get_index_of_labels_with_support(real_list: np.ndarray, target_names: np.ndarray):
    # only compute classification report on labels that occur in set to evaluate
    real_count = np.sum(real_list, axis=0)
    labels_with_support = [i for i in range(len(target_names)) if real_count[i] > 0]
    return labels_with_support


def get_closest_with_bm25(txt, training_data, bm25):
    tokenized_query = txt.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    closest_doc_index = doc_scores.argsort()[::-1][0]

    return training_data["labels"][closest_doc_index]


def get_closest_training_sample(v, m):
    """
    Calculate cosine similarity and return the labels of the closest training sample.
    """
    sim_score = cosine_similarity(v, m)
    closest_training_sample_index = sim_score[0].argsort()[::-1][0]
    return closest_training_sample_index


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.get("workers", multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


def predict_labels(txt, training_data, training_data_tfidf, tfidf_vectorizer):
    """
        Identify the sample in train which is most similar to the given txt
    args:
        txt: input text for which to get predictions
        training_data: training dataframe
    """
    new_doc_tfidf = tfidf_vectorizer.transform([txt])
    closest_training_sample_index = get_closest_training_sample(new_doc_tfidf, training_data_tfidf)

    return training_data["labels"][closest_training_sample_index]


def main(
    input_data_location: str,
    input_model_location: str,
    output_data_location: str,
    output_report_location: str,
    subset: bool,
    samples: int,
):
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(output_report_location, "results_evaluate.log"))

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

    # DocTFIDF: Load the tfidf vectorizer and predict labels for test data
    logger.info("Loading tfidf vectorizer...")
    tfidf_vectorizer_file = os.path.join(input_model_location, "tfidf_vectorizer.pkl")
    with open(tfidf_vectorizer_file, "rb") as file:
        tfidf_vectorizer = pickle.load(file)

    training_data_tfidf = tfidf_vectorizer.transform(train_df["text"])

    logger.info("Predicting on test data...")
    test_df["predicted_labels"] = apply_by_multiprocessing(
        test_df["text"],
        predict_labels,
        training_data=train_df,
        training_data_tfidf=training_data_tfidf,
        tfidf_vectorizer=tfidf_vectorizer,
    )

    logger.info("Save doc-tfidf predictions...")
    test_df["predicted_labels"].to_csv(os.path.join(output_report_location, "predictions.csv"))

    # 2. get classification report with the metrics for each class (and micro, macro, weighted F1) and save it
    #    to output_report_location

    all_labels = pd.concat([pd.Series(test_df["labels"]), pd.Series(test_df["predicted_labels"])])
    mlb = MultiLabelBinarizer().fit(all_labels)

    y_test = mlb.transform(pd.Series(test_df["labels"]))
    y_pred = mlb.transform(pd.Series(test_df["predicted_labels"]))

    label_set = mlb.classes_
    logger.info(f"number of classes in MultiLabelBinarizer: {len(label_set)}")
    labels_with_support = get_index_of_labels_with_support(y_test, label_set)

    all_metric = classification_report(
        y_test, y_pred, digits=3, target_names=list(mlb.classes_), output_dict=True, labels=labels_with_support
    )

    pd.DataFrame(all_metric).transpose().to_csv(os.path.join(output_report_location, "metrics.csv"))


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
        help="Directory containing the trained models from the previous processing step",
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

    parsed_args, _ = parser.parse_known_args()
    main(
        input_data_location=parsed_args.input_data_location,
        input_model_location=parsed_args.input_model_location,
        output_data_location=parsed_args.output_data_location,
        output_report_location=parsed_args.output_report_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
    )
