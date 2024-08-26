import logging
import os
import subprocess
import sys
from typing import List, Set

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
import nltk

nltk.download("wordnet")

import argparse
import ast
import pickle
import random
import time

import pandas as pd
from sentence_transformers import InputExample

SEED = 42
random.seed(SEED)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


def create_negative_samples(
    df_row: pd.Series, labels_in_train: Set[str], num_negative_samples: int
) -> List[InputExample]:
    labels_in_row = set(df_row["labels"])
    negative_labels = list(labels_in_train - labels_in_row)
    # make sure the script doesn't fail if there are not enough to sample from
    num_neg_to_sample = min(len(negative_labels), num_negative_samples)
    return [
        InputExample(texts=[df_row["text"], neg_label], label=0)
        for neg_label in random.sample(negative_labels, num_neg_to_sample)
    ]


def get_distinct_labels(df: pd.DataFrame) -> Set[str]:
    set_of_labels = set(df.explode("labels")["labels"].unique())
    logger.info(f"Number of labels: {len(set_of_labels)}")
    return set_of_labels


def create_input_examples(full_df: pd.DataFrame, num_negative_factor: int) -> List[InputExample]:
    """
    Converts the data in full_df to a list of InputExamples, and it augments the data
    by adding `num_negative_factor` negative samples per positive sample.
    """

    logging.info(f"Num of samples before augmentation: {len(full_df)}")

    distinct_labels = get_distinct_labels(full_df)
    augmented_samples = []

    for _, row in full_df.iterrows():
        for label in row["labels"]:
            augmented_samples.append(InputExample(texts=[row["text"], label], label=1))
            # for each positive label, we add num_negative negative labels
            num_pos_labels = len(row["labels"])
            negative_samples = create_negative_samples(
                row, labels_in_train=distinct_labels, num_negative_samples=num_negative_factor * num_pos_labels
            )
            augmented_samples.extend(negative_samples)

    logging.info(f"Num of samples after augmentation: {len(augmented_samples)}")

    return augmented_samples


def _log_positive_negative_ratio(data: List[InputExample]) -> None:
    num_pos = len([t for t in data if t.label == 1])
    num_neg = len([t for t in data if t.label == 0])
    logger.info(f"Number of positive samples: {num_pos}")
    logger.info(f"Number of negative samples: {num_neg}")
    logger.info(f"Ratio of negative samples per positive sample: {num_neg/num_pos}")


def main(
    input_data_location: str,
    output_data_location: str,
    output_model_location: str,
    output_report_location: str,
    subset: bool,
    samples: int,
    preprocess: bool,
    num_negative: int,
):

    start_time = time.time()

    subdirectories = [output_data_location, output_model_location, output_report_location]
    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    logger.info(f"Loading dataset from {input_data_location}")
    train_df = pd.read_csv(os.path.join(input_data_location, "train.csv"))
    dev_df = pd.read_csv(os.path.join(input_data_location, "dev.csv"))
    test_df = pd.read_csv(os.path.join(input_data_location, "test.csv"))

    if subset:
        train_df = train_df[:samples]
        dev_df = dev_df[:samples]
        test_df = test_df[:samples]

    train_df["labels"] = train_df["labels"].apply(ast.literal_eval)
    dev_df["labels"] = dev_df["labels"].apply(ast.literal_eval)
    test_df["labels"] = test_df["labels"].apply(ast.literal_eval)

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    if preprocess:
        # augment training data with num_negative negative samples
        train_data = create_input_examples(train_df, num_negative_factor=num_negative)
    else:
        # do not augment training data (augment with 0 negative samples)
        train_data = create_input_examples(train_df, num_negative_factor=0)

    dev_data = create_input_examples(dev_df, num_negative_factor=0)

    logger.debug(f"train data length: {len(train_data)}")
    _log_positive_negative_ratio(train_data)

    logger.debug(f"dev data length: {len(dev_data)}")
    _log_positive_negative_ratio(dev_data)

    return train_data, dev_data, train_df, dev_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the input data and store the preprocessed data in output_data_location."
    )

    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: train.csv,dev.csv,test.csv.",
        type=str,
        default="/opt/ml/processing/input/data/",
    )
    parser.add_argument(
        "--output_data_location",
        help="Directory where the data after this preprocessing step will get stored.",
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

    parser.add_argument("--num_negative", dest="num_negative", type=int)

    parsed_args, _ = parser.parse_known_args()

    train_data, dev_data, train_df, dev_df, test_df = main(
        input_data_location=parsed_args.input_data_location,
        output_data_location=parsed_args.output_data_location,
        output_model_location=parsed_args.output_model_location,
        output_report_location=parsed_args.output_report_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        preprocess=parsed_args.preprocess,
        num_negative=parsed_args.num_negative,
    )

    # saving is different due to the augmented data being in list form
    with open(os.path.join(parsed_args.output_data_location, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(parsed_args.output_data_location, "dev.pkl"), "wb") as f:
        pickle.dump(dev_data, f)
    # also copy the original train, dev and test set to the output_data_location (for evaluation)
    train_df.to_csv(os.path.join(parsed_args.output_data_location, "train.csv"))
    dev_df.to_csv(os.path.join(parsed_args.output_data_location, "dev.csv"))
    test_df.to_csv(os.path.join(parsed_args.output_data_location, "test.csv"))
