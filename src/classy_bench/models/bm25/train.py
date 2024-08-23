import argparse
import ast
import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

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
):

    subdirectories = [output_checkpoints_location, output_model_artifacts_location, output_dir_location]

    for subdir in subdirectories:
        try:
            # subdir_path = os.path.join(experiment_directory, subdir)
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
    logger.info("Fitting BM25...")
    tokenized_corpus = train_df["text"].str.split(" ").tolist()
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_file_name = os.path.join(output_model_artifacts_location, "bm25_model.pkl")
    logger.info(f"Saving fitted BM25 to: {bm25_file_name}")
    with open(bm25_file_name, "wb") as f:
        pickle.dump(bm25, f)

    logger.info(f"Done. Took {time.time()-start_time} seconds.")


if __name__ == "__main__":
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
    parsed_args, _ = parser.parse_known_args()
    main(
        input_data_location=parsed_args.input_data_location,
        output_checkpoints_location=parsed_args.output_checkpoints_location,
        output_model_artifacts_location=parsed_args.output_model_artifacts_location,
        output_dir_location=parsed_args.output_dir_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
    )
