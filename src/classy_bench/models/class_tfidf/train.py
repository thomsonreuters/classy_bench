"""class_tfidf Train."""

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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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


def _num_distinct_labels(label_series: pd.Series) -> int:
    label_set = {label for labels in label_series.to_list() for label in labels}
    return len(label_set)


def save_object(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


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

    start_time = time.time()

    train_df = pd.read_csv(os.path.join(input_data_location, "train.csv"))

    if subset:
        train_df = train_df[:samples]

    train_df["labels"] = train_df["labels"].apply(ast.literal_eval)
    logger.info(f"Loading data took: {time.time()-start_time} seconds.")
    logger.info(f"Shape of training data: {train_df.shape}")
    num_labels_train = _num_distinct_labels(train_df["labels"])
    logger.info(f"Number of different labels in train: {num_labels_train}")

    # Adding column containing number of labels for a text observation(row)
    train_df["n_labels"] = train_df["labels"].apply(lambda x: len(x))

    num_labels_train_array = train_df["n_labels"].to_numpy()
    np.savez_compressed(
        os.path.join(output_model_artifacts_location, "num_labels_train.npz"),
        num_labels_train_array=num_labels_train_array,
    )

    # Explode will Transform each element of a list-like label in a row to multiple rows.
    train_df = train_df.explode("labels")
    logger.info(f"Shape of training data after exploding: {train_df.shape}")

    label_encoder_file = os.path.join(output_model_artifacts_location, "label_encoder.pkl")
    label_encoder = LabelEncoder()
    train_df["labels.label_encoder"] = label_encoder.fit_transform(train_df["labels"])
    logger.info(f"Shape of training data after label encoding : {train_df.shape}")
    save_object(label_encoder_file, label_encoder)

    # Create documents per label
    docs = pd.DataFrame({"Document": train_df["text"], "Class": train_df["labels.label_encoder"]})
    docs_per_class = docs.groupby(["Class"], as_index=False).agg({"Document": " ".join})
    logger.info(f"Docs per class dataframe shape : {docs_per_class.shape}")

    # variation in the combined document length
    docs_per_class["doc_length"] = docs_per_class["Document"].apply(lambda x: len(x.split()))
    docs_per_class["doc_length_percent"] = docs_per_class["doc_length"].apply(
        lambda x: x / np.sum(docs_per_class["doc_length"]) * 100
    )
    logger.info("Doc length per class:")
    logger.info(
        "%s",
        docs_per_class[["Class", "doc_length", "doc_length_percent"]].sort_values(by="doc_length", ascending=False),
    )

    # TF-IDF Transformation
    start = time.time()

    tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
    tfidf_vectorizer.fit(docs_per_class.Document)
    tfidf_m = tfidf_vectorizer.transform(docs_per_class.Document)
    tfidf_vectorizer_file_name = os.path.join(output_model_artifacts_location, "tfidf_vectorizer.pkl")
    save_object(tfidf_vectorizer_file_name, tfidf_vectorizer)

    sparse.save_npz(os.path.join(output_model_artifacts_location, "tfidf_m.npz"), tfidf_m)

    logger.info(f"Number of Features : {len(tfidf_vectorizer.get_feature_names_out())}")
    logger.info(f"TFIDF conversion took {time.time()-start} seconds")

    tokenized_corpus = [doc.split(" ") for doc in docs_per_class.Document]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_file_name = os.path.join(output_model_artifacts_location, "bm25.pkl")
    save_object(bm25_file_name, bm25)
    logger.info(f"Total TrainingTook {time.time()-start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with preprocessed data, and stores model artifacts in output_model_location."
    )

    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: train.csv, dev.csv, test.csv",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
    )
    parser.add_argument(
        "--input_model_location",
        help="Directory containing any models from the previous processing step",
        type=str,
        default="/opt/ml/input/model/",
    )
    parser.add_argument(
        "--output_data_location",
        help="Directory where the data after this preprocessing step will get stored.",
        type=str,
        default="/opt/ml/processing/output/data/",
    )
    parser.add_argument(
        "--output_checkpoints_location",
        help="Directory location for storing models for this processing step.",
        type=str,
        default="/opt/ml/checkpoints/",
    )
    parser.add_argument(
        "--output_model_artifacts_location",
        help="Directory location for storing model artifacts for this trainng step.",
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
        "--tfidf_max_features", help="max_features parameter of TfidfVectorizer", type=int, default=5000
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
