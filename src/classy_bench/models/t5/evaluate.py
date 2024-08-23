"""t5 evaluate."""

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
import os
import random
import tarfile
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

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


def _postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def get_accuracy_metric(predictions, references):
    correct = 0
    total = 0
    for pred, refs in zip(predictions, references):
        total += 1
        sorted_pred = sorted([p.strip() for p in pred.split("[SEP]")])
        sorted_ref = sorted(refs)
        if sorted_pred == sorted_ref:
            correct += 1
    return correct / total * 1.0


def clean_prediction(pred):
    no_s = pred.replace("<s>", "").replace("</s>", "")
    list_of_labels = no_s.split("[SEP]")
    list_of_labels_strip = [l.strip() for l in list_of_labels]
    return list_of_labels_strip


def get_all_labels_in(df):
    all_labels = set()
    for labels in df["labels"]:
        for label in labels:
            all_labels.add(label)
    return list(all_labels)


def compute_evaluation_metric(predictions, references):
    mlb = MultiLabelBinarizer()
    predictions_as_list = [p.split(" [SEP] ") for p in predictions]
    references_as_list = [r.split(" [SEP] ") for ref in references for r in ref]
    refs_binarized = mlb.fit_transform(references_as_list)
    preds_binarized = mlb.transform(predictions_as_list)
    all_reference_labels = mlb.classes_
    class_report = classification_report(
        refs_binarized, preds_binarized, digits=3, output_dict=True, target_names=all_reference_labels
    )
    metrics_df = pd.DataFrame(class_report).transpose()
    print(metrics_df.to_string())
    return {
        "macro f1": metrics.f1_score(refs_binarized, preds_binarized, average="macro", zero_division=0),
        "micro f1": metrics.f1_score(refs_binarized, preds_binarized, average="micro", zero_division=0),
        "weighted f1": metrics.f1_score(refs_binarized, preds_binarized, average="weighted", zero_division=0),
        "accuracy": get_accuracy_metric(predictions, references),
    }


def main(
    input_data_location: str,
    input_model_location: str,
    output_data_location: str,
    output_model_location: str,
    output_report_location: str,
    subset: bool,
    samples: int,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    model_identifier: str,
    task_prefix: str,
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

    # determine cuda availability and set device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using cuda")
    else:
        device = torch.device("cpu")

    start_time = time.time()

    test_df = pd.read_csv(os.path.join(input_data_location, "test.csv"))
    dev_df = pd.read_csv(os.path.join(input_data_location, "dev.csv"))

    if subset:
        dev_df = dev_df[:samples]
        test_df = test_df[:samples]

    test_df["labels"] = test_df["labels"].apply(ast.literal_eval)
    dev_df["labels"] = dev_df["labels"].apply(ast.literal_eval)

    # create Dataset from Dataframe
    ds = DatasetDict()

    ds["test"] = Dataset.from_pandas(test_df)
    ds["dev"] = Dataset.from_pandas(dev_df)

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    # special model prefixes
    prefix = f"{task_prefix}: "

    tokenizer = AutoTokenizer.from_pretrained(model_identifier, do_lower_case=False)

    max_input_length = 512
    max_target_length = 512

    def _preprocess_function(examples):
        inputs = [f"{prefix}{ex}" for ex in examples["text"]]
        targets = [" [SEP] ".join(ex) for ex in examples["labels"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # tokenize data
    tokenized_datasets = ds.map(_preprocess_function, batched=True)

    # load classifier model
    model = AutoModelForSeq2SeqLM.from_pretrained(input_model_location)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    model_args = Seq2SeqTrainingArguments(
        input_model_location,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=device.type != "cpu",
        metric_for_best_model="macro f1",
        load_best_model_at_end=True,
        save_strategy="epoch",
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = _postprocess_text(decoded_preds, decoded_labels)

        result = compute_evaluation_metric(predictions=decoded_preds, references=decoded_labels)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model,
        model_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    logger.info("Starting evaluation...")
    for split in ["dev", "test"]:

        preds_test = trainer.predict(tokenized_datasets[split])
        decoded_preds = tokenizer.batch_decode(preds_test.predictions, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        pd.DataFrame(decoded_preds).to_csv(
            os.path.join(output_report_location, f"predictions_{split}.csv"), header="Predictions", index=None
        )
        decoded_preds_list = [clean_prediction(d) for d in decoded_preds]
        pd.DataFrame([[d] for d in decoded_preds_list]).to_csv(
            os.path.join(output_report_location, f"predictions_list_{split}.csv"), header="Predictions", index=None
        )

        with open(os.path.join(output_report_location, f"metrics_{split}.txt"), "w") as f:
            f.write(f"---------------Metrics on {split} :---------------\n")
            f.write(str(preds_test.metrics))

        # also store the class-wise metrics as a dataframe
        mlb = MultiLabelBinarizer()

        # Replace -100 in the labels as we can't decode them.
        references_ids = [
            np.where(label != -100, label, tokenizer.pad_token_id) for label in tokenized_datasets[split]["labels"]
        ]
        references_as_list = tokenizer.batch_decode(references_ids, skip_special_tokens=True)
        references_as_list = [ref.split(" [SEP] ") for ref in references_as_list]
        binary_true_list = mlb.fit_transform(references_as_list)
        binary_preds_list = mlb.transform(decoded_preds_list)
        all_reference_labels = mlb.classes_
        real_count = np.sum(binary_true_list, axis=0)
        labels_with_support = [i for i in range(len(all_reference_labels)) if real_count[i] > 0]
        all_metrics = classification_report(
            binary_true_list,
            binary_preds_list,
            digits=3,
            output_dict=True,
            target_names=all_reference_labels,
            labels=labels_with_support,
        )
        metrics_df = pd.DataFrame(all_metrics).transpose()
        metrics_df.to_csv(os.path.join(output_report_location, f"metrics_{split}.csv"))


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
        help="Directory containing model artifacts obtained from training step (models.tar.gz)",
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

    parser.add_argument("--subset", action="store_true")
    parser.add_argument(
        "--samples", help="number of samples that should be used (for debugging)", type=int, default=1000
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs the model should be fine-tuned")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of epochs the model should be fine-tuned")

    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")

    parser.add_argument(
        "--model",
        dest="model_identifier",
        default="t5-small",
        help="Model identifier of the huggingface model to use. Default: t5-small",
    )
    parser.add_argument(
        "--task_prefix", default="summarize", help="Task prefix that should be used. Default: summarize"
    )

    parsed_args, _ = parser.parse_known_args()

    main(
        input_data_location=parsed_args.input_data_location,
        input_model_location=parsed_args.input_model_location,
        output_data_location=parsed_args.output_data_location,
        output_model_location=parsed_args.output_model_location,
        output_report_location=parsed_args.output_report_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        epochs=parsed_args.epochs,
        batch_size=parsed_args.batch_size,
        early_stopping_patience=parsed_args.early_stopping_patience,
        model_identifier=parsed_args.model_identifier,
        task_prefix=parsed_args.task_prefix,
    )
