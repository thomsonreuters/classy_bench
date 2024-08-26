import nltk

nltk.download("stopwords")
nltk.download("punkt")

import argparse
import ast
import logging
import os
import random
import time

import numpy as np

# Training Imports
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


def _postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def get_accuracy_metric(predictions, references):
    """Get_accuracy_metric."""
    correct = 0
    total = 0
    for pred, refs in zip(predictions, references):
        total += 1
        sorted_pred = sorted([p.strip() for p in pred.split("[SEP]")])
        sorted_ref = sorted(refs)
        if sorted_pred == sorted_ref:
            correct += 1
    return correct / total * 1.0


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


def get_tokenized_datasets(train_df: pd.DataFrame, dev_df: pd.DataFrame, task_prefix: str, tokenizer):

    train_df["labels"] = train_df["labels"].apply(ast.literal_eval)
    dev_df["labels"] = dev_df["labels"].apply(ast.literal_eval)

    # create Dataset from Dataframe
    ds = DatasetDict()
    ds["train"] = Dataset.from_pandas(train_df)
    ds["validation"] = Dataset.from_pandas(dev_df)

    # special model prefixes
    prefix = f"{task_prefix}: "
    max_input_length = 512
    max_target_length = 512

    def preprocess_function(examples):
        """Preprocess_function"""
        inputs = [f"{prefix}{ex}" for ex in examples["text"]]
        targets = [" [SEP] ".join(ex) for ex in examples["labels"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # tokenize data
    tokenized_datasets = ds.map(preprocess_function, batched=True)
    return tokenized_datasets


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
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_identifier: str,
    early_stopping_patience: int,
    task_prefix: str,
):

    subdirectories = [output_checkpoints_location, output_model_artifacts_location, output_dir_location]

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

    train_df = pd.read_csv(os.path.join(input_data_location, "train.csv"))
    dev_df = pd.read_csv(os.path.join(input_data_location, "dev.csv"))

    if subset:
        train_df = train_df[:samples]
        dev_df = dev_df[:samples]

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    logger.info("Tokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, do_lower_case=False)
    tokenized_datasets = get_tokenized_datasets(train_df, dev_df, task_prefix=task_prefix, tokenizer=tokenizer)

    logger.info(f"Loading pretrained classifier from huggingface {model_identifier}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)

    fp16 = device.type != "cpu"
    model_args = Seq2SeqTrainingArguments(
        output_model_artifacts_location,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=fp16,
        metric_for_best_model="macro f1",
        load_best_model_at_end=True,
        save_strategy="epoch",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def _compute_metrics(eval_preds):
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    logger.info("Start training...")
    start_time_training = time.time()
    trainer.train()
    logger.info("Training took: %s seconds", time.time() - start_time_training)
    # Saving model
    model.save_pretrained(output_model_artifacts_location)
    trainer.save_model(
        output_model_artifacts_location,
    )
    # Saving tokenizer
    tokenizer.save_pretrained(output_model_artifacts_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with the preprocessed data and store the model artifacts in output_model_location"
    )

    parser.add_argument(
        "--input_data_location",
        help="Directory containing the input data as files: train.csv, dev.csv, test.csv",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
    )
    parser.add_argument(
        "--output_data_location",
        help="Directory where the data after this preprocessing step will get stored",
        type=str,
        default="/opt/ml/processing/output/data/",
    )
    parser.add_argument(
        "--output_checkpoints_location",
        help="Directory location for storing model checkpoints for this processing step",
        type=str,
        default="/opt/ml/checkpoints/",
    )
    parser.add_argument(
        "--output_model_artifacts_location",
        help="Directory location for storing model artifacts for this trainng step",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--output_dir_location",
        help="Directory location for storing reports/logs for this processing step",
        type=str,
        default=os.environ["SM_OUTPUT_DIR"],
    )

    parser.add_argument("--subset", action="store_true")
    parser.add_argument(
        "--samples", help="number of samples that should be used (for debugging)", type=int, default=1000
    )

    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs the model should be fine-tuned")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of epochs the model should be fine-tuned")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--model",
        dest="model_identifier",
        default="t5-small",
        help="Model identifier of the huggingface model to use.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--task_prefix", default="summarize", help="Task prefix that should be used, e.g., summarize")

    parsed_args, _ = parser.parse_known_args()

    main(
        input_data_location=parsed_args.input_data_location,
        output_checkpoints_location=parsed_args.output_checkpoints_location,
        output_model_artifacts_location=parsed_args.output_model_artifacts_location,
        output_dir_location=parsed_args.output_dir_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        epochs=parsed_args.epochs,
        batch_size=parsed_args.batch_size,
        learning_rate=parsed_args.learning_rate,
        model_identifier=parsed_args.model_identifier,
        early_stopping_patience=parsed_args.early_stopping_patience,
        task_prefix=parsed_args.task_prefix,
    )
