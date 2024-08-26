import argparse
import ast
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn import metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


class FocalLoss(torch.nn.Module):
    """
    alpha needs to be above 0.5 to lean things in favor of learning the positive labels.
    If it is set to 0.5 then alpha applies equally to positive and negative class, and allows for beta and gamma
    (class balanced parameters) to show their effect.
    """

    def __init__(self, gamma, beta, alpha=0.5, class_freq=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.class_freq = class_freq

    def weight_balance(self, batch_size, beta, device="cuda"):  # balancing can be None or CB (class balance)
        """
        balancing will use class-balanced focal loss to
        apply weights to loss from different classes based on this paper: https://arxiv.org/abs/1901.05555
        The formula for the weight coefficient that will be multiplied by class i's loss is:
        (1-beta)/(1-(beta**classfreq[i]))
        beta is a balancing term with values in range [0, 1). When it is zero, no balancing will be done.
        """
        beta = self.beta
        class_balanced_coeffs = (1 - beta) / (1 - torch.pow(beta, torch.tensor(self.class_freq)))
        class_balanced_coeffs = class_balanced_coeffs.repeat(batch_size, 1).to(device)
        return class_balanced_coeffs

    def forward(self, inputs, targets, device="cuda"):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)

        # balancing factor v1: using alpha favouring pos label learning over neg label learning:
        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha).to(device)
        else:
            at = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # balancing factor v2: using beta applying class-balanced coefficients to favour low-freq classes learning
        # if beta is set to >0, coeffs will apply class-balanced multiplier to loss
        # if beta=0 loss will remain as is

        batch_size = targets.size(dim=0)
        coeffs = self.weight_balance(batch_size=batch_size, beta=self.beta)

        pt = torch.exp(-BCE_loss)
        floss = at * coeffs * (1 - pt) ** self.gamma * BCE_loss
        return floss.mean()


def get_class_weights(df: pd.DataFrame, mode: str, device: str) -> torch.Tensor:
    """Get class weights that depending on the class frequency.

    Args:
        df: Pandas Dataframe.
        mode: choose from ['equal', 'unnormalized', 'normalized', 'weighted'], where:
          - 'equal': each class has weight 1
          - 'unnormalized': class counts, i.e., number of samples for each class
          - 'normalized': weight for a class is the class count divided by total count
          - 'weighted': 1-(class count divided by total count), so the weight for low-resource classes is higher
        device: whether to run on cpu or gpu device.
    """
    binary_labels = df["labels"].apply(ast.literal_eval).apply(np.array)
    if mode == "equal":
        # each class has weight 1
        class_weights = np.ones(len(binary_labels[0]))
    elif mode == "unnormalized":
        # class counts, i.e., number of samples for each class
        class_weights = np.asarray(sum(binary_labels))
    elif mode == "normalized":
        # weight for a class is the class count divided by total count
        frequencies = np.asarray(sum(binary_labels))
        class_weights = frequencies / sum(frequencies)
    elif mode == "weighted":
        # 1-(class count divided by total count), so the weight for low-resource classes is higher
        # than for high-resource
        frequencies = np.asarray(sum(binary_labels))
        class_weights = 1 - (frequencies / sum(frequencies))
    else:
        err_str = f"Invalid weighting mode: {mode}"
        raise ValueError(err_str)
    return torch.from_numpy(class_weights).float().to(device)


def get_loss_function(loss_function, class_weights):
    if loss_function in {"sigmoid_bce_loss", "weighted_sigmoid_bce_loss"}:
        return torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    elif loss_function == "focal_loss":
        return FocalLoss(alpha=0.5, beta=0.7, gamma=3, class_freq=class_weights)
    else:
        err_str = f"{loss_function} has not been included as part of the experiment yet!"
        raise ValueError(err_str)


# get set of labels that occur in train
def get_all_labels_in_df(df, labels_text_col_name):
    """Get_all_labels_in_df."""
    all_labels = set()
    for _, row in df.iterrows():
        all_labels.update(row[labels_text_col_name])
    return all_labels


def _load_labels(input_data_location: str):
    all_labels = pd.read_csv(os.path.join(input_data_location, "label_classes.csv")).values.tolist()
    for l in all_labels:
        assert len(l) == 1
    return [label for labels in all_labels for label in labels]


def main(
    input_data_location: str,
    output_checkpoints_location: str,
    output_model_artifacts_location: str,
    output_dir_location: str,
    tokenizer_model_ckpt: str,
    tokenizer_do_lower_case: bool,
    training_arguments_batch_size: int,
    training_arguments_log_per_epoch: int,
    training_arguments_eval_per_epoch: int,
    training_arguments_lr: float,
    training_arguments_n_epochs: int,
    training_arguments_loss_func_name: str,
    training_arguments_early_stopping_patience: int,
):

    start_time = time.time()
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
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")

    # read train and val datasets:
    df_train = pd.read_csv(os.path.join(input_data_location, "train.csv"))
    df_dev = pd.read_csv(os.path.join(input_data_location, "dev.csv"))

    labels = _load_labels(input_data_location)
    logger.info("Number of classes: %d", len(labels))
    logger.info("First 5 classes in the label set as an example: %s", str(labels[:5]))

    label2id = {}
    id2label = {}
    for idx, label in enumerate(labels):
        id2label[idx] = label
        label2id[label] = idx

    df_train = df_train.fillna("")
    df_dev = df_dev.fillna("")
    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    logger.info("Load tokenizer from huggingface %s", tokenizer_model_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_ckpt, do_lower_case=tokenizer_do_lower_case)

    # tokenize train and test text values
    text_col_name = "text"
    dev_encodings = tokenizer(
        df_dev[text_col_name].values.tolist(),
        truncation=True,
        return_token_type_ids=False,
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(device)
    train_encodings = tokenizer(
        df_train[text_col_name].values.tolist(),
        truncation=True,
        return_token_type_ids=False,
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(device)

    def labels_to_float(labels):
        return np.fromstring(labels[1:-1], dtype=float, sep=", ")

    train_labels = df_train["labels"].apply(labels_to_float)
    test_labels = df_dev["labels"].apply(labels_to_float)

    # Define CustomDataset to use with the Trainer object
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, index):
            item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
            # labels need to be converted to float for multilabel loss calculation
            item["labels"] = torch.tensor(self.labels[index]).float()  # long()  #.float()
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(dev_encodings, test_labels)

    logger.info("Load model from huggingface %s", tokenizer_model_ckpt)

    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_model_ckpt, num_labels=len(labels), label2id=label2id, id2label=id2label
    ).to(device)
    model.gradient_checkpointing_enable()

    def compute_metrics(eval_pred):
        """
        Calculates metrics we want to track during model training.
        It will be passed on to the compute_metrics arg in Trainer object instantiation.
        """
        preds, labels = eval_pred
        preds = np.array(expit(preds) > 0.5, dtype=int)
        macro_f1 = metrics.f1_score(labels, preds, average="macro")
        micro_f1 = metrics.f1_score(labels, preds, average="micro")
        weighted_f1 = metrics.f1_score(labels, preds, average="weighted")
        samples_f1 = metrics.f1_score(labels, preds, average="samples")
        return {"macro_f1": macro_f1, "micro_f1": micro_f1, "weighted_f1": weighted_f1, "samples_f1": samples_f1}

    # set training args
    logger.info("Setting training args from config file")
    batch_size = training_arguments_batch_size
    log_per_epoch = training_arguments_log_per_epoch
    logging_steps = max(int((len(df_train) / batch_size) / log_per_epoch), 1)
    eval_per_epoch = training_arguments_eval_per_epoch
    eval_steps = max(int((len(df_train) / batch_size) / eval_per_epoch), 1)
    lr = float(training_arguments_lr)
    n_epochs = training_arguments_n_epochs
    loss_func_name = training_arguments_loss_func_name
    early_stopping_patience = training_arguments_early_stopping_patience
    fp16 = device.type != "cpu"

    args = TrainingArguments(
        output_dir=output_model_artifacts_location,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        fp16=fp16,
        warmup_ratio=0.1,
        logging_steps=logging_steps,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=5,
        metric_for_best_model="macro_f1",
        save_steps=eval_steps,
    )

    if loss_func_name == "focal_loss":
        class_weights = get_class_weights(df=df_train, mode="unnormalized", device=device)
    elif loss_func_name == "sigmoid_bce_loss":
        class_weights = get_class_weights(df=df_train, mode="equal", device=device)
    elif loss_func_name == "weighted_sigmoid_bce_loss":
        class_weights = get_class_weights(df=df_train, mode="weighted", device=device)
    else:
        err_str = "Unsupported loss function: {loss_func_name}"
        raise ValueError(err_str)

    # define custom Trainer class set to train multilabel classifiers
    # with dbloss or BCE with logits loss
    class MLBTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            # Feed inputs to model and extract logits
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # Extract labels
            labels = inputs.get("labels")
            labels = labels.view(-1, len(class_weights))
            # Compute loss based on selected loss function
            if loss_func_name == "focal_loss":
                loss_fc = get_loss_function(loss_function=loss_func_name, class_weights=class_weights)
                loss = loss_fc(logits, labels)
            else:
                loss_fc = get_loss_function(loss_function=loss_func_name, class_weights=class_weights)
                loss = loss_fc(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = MLBTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    logger.info("Start training...")
    start_time_training = time.time()
    trainer.train()
    trainer.save_model(output_model_artifacts_location)
    logger.info("Training took: %s seconds", time.time() - start_time_training)

    logger.info("Save trainer log history")
    with open(os.path.join(output_model_artifacts_location, "log_history.txt"), "w") as log_file:
        for line in trainer.state.log_history:
            log_file.write(str(line))
            log_file.write("\n")

    # Save again labels classes for evaluation
    label_classes = pd.read_csv(os.path.join(input_data_location, "label_classes.csv"))
    label_classes.to_csv(os.path.join(output_model_artifacts_location, "label_classes.csv"), index=False)


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
        "--input_model_location",
        help="Directory containing any models from the previous processing step",
        type=str,
        default="/opt/ml/input/model/",
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
        help="Directory location for storing model artifacts for this training step",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--output_dir_location",
        help="Directory location for storing reports/logs for this processing step",
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
        "--tokenizer_model_ckpt",
        help="Huggingface model that should be used",
        type=str,
        default="distilroberta-base",
    )

    parser.add_argument("--tokenizer_do_lower_case", dest="tokenizer_do_lower_case", action="store_true")
    parser.add_argument("--no_tokenizer_do_lower_case", dest="tokenizer_do_lower_case", action="store_false")
    parser.set_defaults(tokenizer_do_lower_case=False)

    parser.add_argument("--tokenizer_max_length", help="Max length of tokenizer", type=int, default=512)

    parser.add_argument("--tokenizer_padding", dest="tokenizer_padding", action="store_true")
    parser.add_argument("--no_tokenizer_padding", dest="tokenizer_padding", action="store_false")
    parser.set_defaults(tokenizer_padding=True)

    parser.add_argument("--tokenizer_truncation", dest="tokenizer_truncation", action="store_true")
    parser.add_argument("--no_tokenizer_truncation", dest="tokenizer_truncation", action="store_false")
    parser.set_defaults(tokenizer_truncation=True)

    parser.add_argument(
        "--tokenizer_return_token_type_ids", dest="tokenizer_return_token_type_ids", action="store_true"
    )
    parser.add_argument(
        "--no_tokenizer_return_token_type_ids", dest="tokenizer_return_token_type_ids", action="store_false"
    )
    parser.set_defaults(tokenizer_return_token_type_ids=False)

    parser.add_argument(
        "--training_arguments_lr",
        help="Learning rate for training",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--training_arguments_n_epochs",
        help="Number of epochs for training",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--training_arguments_batch_size",
        help="Batch size for training",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--training_arguments_loss_func_name",
        help="Loss function for training",
        type=str,
        default="TorchBCE",
    )

    parser.add_argument("--training_arguments_fp16", dest="training_arguments_fp16", action="store_true")
    parser.add_argument("--no_training_arguments_fp16", dest="training_arguments_fp16", action="store_false")
    parser.set_defaults(training_arguments_fp16=True)

    parser.add_argument(
        "--training_arguments_gradient_checkpointing",
        dest="training_arguments_gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--no_training_arguments_gradient_checkpointing",
        dest="training_arguments_gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(training_arguments_gradient_checkpointing=True)

    parser.add_argument(
        "--training_arguments_lr_scheduler_type",
        help="Learning rate scheduler type",
        type=str,
        default="cosine",
    )
    parser.add_argument(
        "--training_arguments_evaluation_strategy",
        help="Evaluation strategy (steps, epoch)",
        type=str,
        default="steps",
    )
    parser.add_argument(
        "--training_arguments_log_per_epoch",
        help="Log per epoch",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--training_arguments_eval_per_epoch",
        help="Eval per epoch",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--training_arguments_eval_thresh",
        help="Evaluation threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--training_arguments_early_stopping_patience",
        help="Early stopping patience",
        type=int,
        default=8,
    )

    parsed_args, _ = parser.parse_known_args()
    main(
        input_data_location=parsed_args.input_data_location,
        output_checkpoints_location=parsed_args.output_checkpoints_location,
        output_model_artifacts_location=parsed_args.output_model_artifacts_location,
        output_dir_location=parsed_args.output_dir_location,
        tokenizer_model_ckpt=parsed_args.tokenizer_model_ckpt,
        tokenizer_do_lower_case=parsed_args.tokenizer_do_lower_case,
        training_arguments_batch_size=parsed_args.training_arguments_batch_size,
        training_arguments_log_per_epoch=parsed_args.training_arguments_log_per_epoch,
        training_arguments_eval_per_epoch=parsed_args.training_arguments_eval_per_epoch,
        training_arguments_lr=parsed_args.training_arguments_lr,
        training_arguments_n_epochs=parsed_args.training_arguments_n_epochs,
        training_arguments_loss_func_name=parsed_args.training_arguments_loss_func_name,
        training_arguments_early_stopping_patience=parsed_args.training_arguments_early_stopping_patience,
    )
