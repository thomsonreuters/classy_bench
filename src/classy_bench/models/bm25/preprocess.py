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
import logging
import re
import time
from typing import List

import nltk
import pandas as pd

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


def _text_normalization(
    txt: str, stemmer: PorterStemmer, stop_words: List[str], to_lower: bool = True, no_punct: bool = True
) -> str:
    # apply lowercase
    if to_lower:
        txt = txt.lower()
    # remove all non-alphabet items including punctuation except the dash i.e. for well-thought
    if no_punct:
        txt = re.sub(r"(^\(?[^()]*\))|([^a-zA-Z0-9\s]+)", "", txt)
    # remove stopwords and stem the words
    words = word_tokenize(str(txt))
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 1]
    txt = " ".join(words)
    # remove multiple empty spaces and replace with one
    txt = re.sub(r"[^\S]+", " ", txt)
    # remove any remaining white spaces at the beginning and end of the sentence
    txt = txt.strip()
    return txt


def preprocess_text(full_df: pd.DataFrame) -> pd.DataFrame:
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    full_df["text"] = full_df.apply(lambda x: _text_normalization(x["text"], stemmer, stop_words), axis=1)
    return full_df


def main(
    input_data_location: str,
    output_data_location: str,
    output_model_location: str,
    output_report_location: str,
    subset: bool,
    samples: int,
    preprocess: bool = True,
):
    """
    Preprocess the input data and store the preprocessed data in output_data_location.
    """
    start_time = time.time()

    subdirectories = [output_data_location, output_model_location, output_report_location]
    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    # Load data
    logger.info(f"Loading dataset from {input_data_location}")
    train_df = pd.read_csv(os.path.join(input_data_location, "train.csv"))
    dev_df = pd.read_csv(os.path.join(input_data_location, "dev.csv"))
    test_df = pd.read_csv(os.path.join(input_data_location, "test.csv"))

    # load list of labels correctly (from string to list)
    train_df["labels"] = train_df["labels"].apply(ast.literal_eval)
    dev_df["labels"] = dev_df["labels"].apply(ast.literal_eval)
    test_df["labels"] = test_df["labels"].apply(ast.literal_eval)

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    if subset:
        train_df = train_df[:samples]
        dev_df = dev_df[:samples]
        test_df = test_df[:samples]

    if preprocess:
        train_df = preprocess_text(train_df)
        dev_df = preprocess_text(dev_df)
        test_df = preprocess_text(test_df)

    logger.info(f"Preprocessing of data took: {time.time()-start_time} seconds.")

    logger.debug(f"train data shape: {train_df.shape}")
    logger.debug(f"dev data shape: {dev_df.shape}")
    logger.debug(f"test data shape: {test_df.shape}")

    return train_df, dev_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the input data and store the preprocessed data in output_data_location."
    )
    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: train.csv, dev.csv, test.csv.",
        type=str,
        default="/opt/ml/processing/input/data/",
    )
    parser.add_argument(
        "--output_data_location",
        help="Directory where the data will get stored after this preprocessing step.",
        type=str,
        default="/opt/ml/processing/output/data/",
    )
    parser.add_argument(
        "--output_model_location",
        help="Directory location for storing models for this processing step.",
        type=str,
        default="/opt/ml/processing/output/model/",
    )
    parser.add_argument(
        "--output_report_location",
        help="Directory location for storing reports/logs for this processing step.",
        type=str,
        default="/opt/ml/processing/output/report/",
    )
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.add_argument("--no_subset", dest="subset", action="store_false")
    parser.set_defaults(subset=False)
    parser.add_argument(
        "--samples", help="number of samples that should be used (for debugging)", type=int, default=1000
    )
    parser.add_argument("--preprocess", dest="preprocess", action="store_true")
    parser.add_argument("--no_preprocess", dest="preprocess", action="store_false")
    parser.set_defaults(preprocess=True)

    parsed_args, _ = parser.parse_known_args()

    train_df, dev_df, test_df = main(
        input_data_location=parsed_args.input_data_location,
        output_data_location=parsed_args.output_data_location,
        output_model_location=parsed_args.output_model_location,
        output_report_location=parsed_args.output_report_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        preprocess=parsed_args.preprocess,
    )

    logger.info("Saving data...")
    train_df.to_csv(os.path.join(parsed_args.output_data_location, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(parsed_args.output_data_location, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(parsed_args.output_data_location, "test.csv"), index=False)
