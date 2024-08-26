import argparse
import ast
import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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


def main(
    input_data_location: str,
    output_checkpoints_location: str,
    output_model_artifacts_location: str,
    output_dir_location: str,
    subset: bool,
    samples: int,
    tfidf_max_features: int,
):

    subdirectories = [output_checkpoints_location, output_model_artifacts_location, output_dir_location]

    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    # train classifier
    start_time = time.time()

    # Load data
    train_df = pd.read_csv(os.path.join(input_data_location, "train.csv"))

    if subset:
        train_df = train_df[:samples]

    # load list of labels correctly (from string to list)
    train_df["labels"] = train_df["labels"].apply(ast.literal_eval)
    logger.info(f"Loading data took: {time.time()-start_time} seconds.")
    logger.info(f"Shape of training data: {train_df.shape}")

    # TFIDF Transformation
    start = time.time()
    tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
    tfidf_vectorizer.fit(train_df["text"])
    logger.info(f"Number of Features : {len(tfidf_vectorizer.get_feature_names_out())}")
    logger.info(f"Training took {time.time() - start} seconds")
    tfidf_vectorizer_file = os.path.join(output_model_artifacts_location, "tfidf_vectorizer.pkl")
    logger.info(f"Saving fitted tfidf_vectorizer to: {tfidf_vectorizer_file}")
    with open(tfidf_vectorizer_file, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)


if __name__ == "__main__":
    """Loads the preprocessed data, then trains a model with that data,
    and then stores model artifacts in output_model_location .
    Args:
        input_data_location (str)  : Directory containing the preprocessed input data in the files train.csv,
                                     dev.csv, test.csv. The expected columns of csv file are "labels","text"
        output_model_location(str) : Directory location for storing models for this training step.
                                     This directory contains model.tar.gz file containing all
                                     model artifacts.
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Train a model with preprocessed data, and stores model artifacts in output_model_location."
    )
    parser.add_argument(
        "--input_data_location",
        help="Directory containing the preprocessed input data in files train.csv, dev.csv, test.csv.",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
    )
    parser.add_argument(
        "--output_checkpoints_location",
        help="Directory location for storing models for this processing step.",
        type=str,
        default="/opt/ml/checkpoints/",
    )
    parser.add_argument(
        "--output_model_artifacts_location",
        help="Directory location for storing model artifacts for this training step.",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--output_dir_location",
        help="Directory location for storing reports/logs for this processing step.",
        type=str,
        default=os.environ["SM_OUTPUT_DIR"],
    )
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.add_argument("--no_subset", dest="subset", action="store_false")
    parser.set_defaults(subset=False)
    parser.add_argument(
        "--samples", help="number of samples that should be used (for debugging)", type=int, default=1000
    )
    parser.add_argument(
        "--tfidf_max_features", help="max_features parameter of TfidfVectorizer", type=int, default=50000
    )
    parsed_args, _ = parser.parse_known_args()
    main(
        input_data_location=parsed_args.input_data_location,
        output_checkpoints_location=parsed_args.output_checkpoints_location,
        output_model_artifacts_location=parsed_args.output_model_artifacts_location,
        output_dir_location=parsed_args.output_dir_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        tfidf_max_features=parsed_args.tfidf_max_features,
    )
