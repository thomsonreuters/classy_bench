import logging
import os
import subprocess
import sys
from typing import Dict

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

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


def _get_input_data(df: pd.DataFrame):
    return [
        {
            "id": row["id"],
            "text": row["text"],
            "labels_text": row["labels_text"],
            "real_labels": ast.literal_eval(idx),
        }
        for idx, row in df.iterrows()
    ]


def _get_all_labels_in_df(df: pd.DataFrame):
    all_labels = set()
    for _, row in df.iterrows():
        all_labels.update(row["labels_text"])
    return all_labels


def _load_model(model_path: str, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    # for gpu, enable fp16 precision to make calculations quicker
    if device.type == "cuda":
        model.half()
    model.eval()
    return model


def _get_batched_predictions(tokenizer, model, device, input_data_list: Dict, batch_size: int, threshold: float):
    preds_data = []
    for batch in tqdm([input_data_list[x : x + batch_size] for x in range(0, len(input_data_list), batch_size)]):
        # tokenize
        encoded = tokenizer(
            [d["text"] for d in batch],  # list of text inputs to encode
            max_length=512,  # Pad & truncate all sentences.
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        # predict
        with torch.no_grad():
            outputs = model(
                encoded["input_ids"].to(device),
                token_type_ids=None,
                attention_mask=encoded["attention_mask"].to(device),
            )
        probs = np.array(expit(outputs[0].to(device="cpu")))

        # append predictions
        for i, b in enumerate(batch):
            preds_data.append(
                {
                    "id": b["id"],
                    "probabilities": probs[i],
                    "predictions": list(map(lambda x: int(x > threshold), probs[i])),  # noqa: C417
                    "real_labels_text": b["labels_text"],
                    "real_labels": b["real_labels"],
                }
            )

    return preds_data


def main(
    input_data_location: str,
    input_model_location: str,
    output_data_location: str,
    output_model_location: str,
    output_report_location: str,
    tokenizer_model_ckpt: str,
    tokenizer_do_lower_case: bool,
    eval_parameters_batch_size: int,
    eval_parameters_threshold: float,
):
    # Required as you used sagemaker training before evaluation, so the model artifacts are in model.tar.gz.
    # OS check is needed for testing and also for local usage of script
    if os.path.exists(os.path.join(input_model_location, "model.tar.gz")):
        model_path = os.path.join(input_model_location, "model.tar.gz")
        logger.info(f"Extracting model from path: {model_path}")

        with tarfile.open(model_path) as tar:
            tar.extractall(path=input_model_location)
        logger.debug("Input model location files: %s", os.listdir(input_model_location))

    subdirectories = [output_data_location, output_report_location, output_model_location]

    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    label_set_df = pd.read_csv(os.path.join(input_model_location, "label_classes.csv"))
    label_set = label_set_df["label"].tolist()

    df_test = pd.read_csv(os.path.join(input_data_location, "test.csv"), index_col=0)
    df_test["labels_text"] = df_test["labels_text"].apply(ast.literal_eval)
    logger.info(f"test_df.shape: {df_test.shape}")

    test_data_list = _get_input_data(df_test)

    df_train = pd.read_csv(os.path.join(input_data_location, "train.csv"), index_col=0)
    df_train["labels_text"] = df_train["labels_text"].apply(ast.literal_eval)
    labels_in_train = _get_all_labels_in_df(df_train)
    logger.info(f"Number of labels in train: {len(labels_in_train)}")
    labels_in_df_to_eval = _get_all_labels_in_df(df_test)
    logger.info(f"Number of labels in test: {len(labels_in_df_to_eval)}")
    additional_labels_in_eval = sorted([l for l in labels_in_df_to_eval if l not in labels_in_train])

    # determine cuda availability and set device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("using cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_ckpt, do_lower_case=tokenizer_do_lower_case)
    model = _load_model(input_model_location, device)

    test_preds_data_list = _get_batched_predictions(
        tokenizer,
        model,
        device,
        test_data_list,
        batch_size=eval_parameters_batch_size,
        threshold=eval_parameters_threshold,
    )
    logger.info("Number of test predictions: %s", len(test_preds_data_list))

    # add the labels that are in eval but not in train to y_pred and y_true
    preds_list = [o["predictions"] for o in test_preds_data_list]
    real_list = [o["real_labels"] for o in test_preds_data_list]
    for o in test_preds_data_list:
        for additional_label in additional_labels_in_eval:
            if additional_label in o["real_labels_text"]:
                o["real_labels"].append(1)
            else:
                o["real_labels"].append(0)
            o["predictions"].append(0)

    logger.info("Calculating evaluation metrics...")
    target_names = label_set
    target_names.extend(additional_labels_in_eval)
    logger.info("Number of target names: %s", len(target_names))

    # only compute classification report on labels that occur in set to evaluate
    real_count = np.sum(real_list, axis=0)
    labels_with_support = [i for i in range(len(target_names)) if real_count[i] > 0]

    all_metrics = classification_report(
        real_list, preds_list, digits=3, output_dict=True, target_names=target_names, labels=labels_with_support
    )
    pd.DataFrame(all_metrics).transpose().to_csv(os.path.join(output_report_location, "metrics.csv"))

    predictions_thresholded = [
        {
            "id": test_pred["id"],
            "predictions": [
                {
                    "name": label_set[i],
                    "prob": v,
                }
                for i, v in enumerate(test_pred["probabilities"])
                if v > eval_parameters_threshold
            ],
        }
        for test_pred in test_preds_data_list
    ]
    pd.DataFrame(predictions_thresholded).to_csv(
        os.path.join(output_report_location, f"predictions_thresh_{eval_parameters_threshold}_test.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the model with the preprocessed data and store the output in output_data_location."
    )
    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: train.csv, dev.csv, test.csv",
        type=str,
        default="/opt/ml/processing/input/data/",
    )
    parser.add_argument(
        "--input_model_location",
        help="Directory containing model artifacts obtained from training step",
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
    parser.add_argument(
        "--tokenizer_model_ckpt",
        help="Huggingface model that should be used",
        type=str,
        default="distilroberta-base",
    )
    parser.add_argument("--tokenizer_do_lower_case", dest="tokenizer_do_lower_case", action="store_true")
    parser.add_argument("--no_tokenizer_do_lower_case", dest="tokenizer_do_lower_case", action="store_false")
    parser.set_defaults(tokenizer_do_lower_case=False)
    parser.add_argument(
        "--eval_parameters_batch_size",
        help="Batch size for evaluation",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval_parameters_threshold",
        help="Threshold for evaluation",
        type=float,
        default=0.5,
    )
    parsed_args, _ = parser.parse_known_args()
    main(
        input_data_location=parsed_args.input_data_location,
        input_model_location=parsed_args.input_model_location,
        output_data_location=parsed_args.output_data_location,
        output_model_location=parsed_args.output_model_location,
        output_report_location=parsed_args.output_report_location,
        tokenizer_model_ckpt=parsed_args.tokenizer_model_ckpt,
        tokenizer_do_lower_case=parsed_args.tokenizer_do_lower_case,
        eval_parameters_batch_size=parsed_args.eval_parameters_batch_size,
        eval_parameters_threshold=parsed_args.eval_parameters_threshold,
    )
