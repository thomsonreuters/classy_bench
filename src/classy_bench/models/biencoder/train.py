import argparse
import logging
import math
import os
import pickle
import random
import time
from typing import List

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


logger = _get_logger()


def _load_training_examples(input_data_location: str) -> List[InputExample]:
    with open(os.path.join(input_data_location, "train.pkl"), "rb") as f:
        loaded_train_data: List[InputExample] = pickle.load(f)
    return loaded_train_data


def main(
    input_data_location: str,
    output_checkpoints_location: str,
    output_model_artifacts_location: str,
    output_dir_location: str,
    subset: bool,
    samples: int,
    base_model: str,
    batch_size: int,
    epochs: int,
    loss: str,
):
    subdirectories = [output_checkpoints_location, output_model_artifacts_location, output_dir_location]

    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    start_time = time.time()

    loaded_train_data = _load_training_examples(input_data_location)

    if subset:
        loaded_train_data = loaded_train_data[:samples]

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    model = SentenceTransformer(base_model)

    # determine the loss to use
    train_samples = []

    if loss == "CosineSimilarityLoss":
        train_loss = losses.CosineSimilarityLoss(model)
        # expects float label
        train_samples = [InputExample(texts=sample.texts, label=float(sample.label)) for sample in loaded_train_data]

    elif loss == "MultipleNegativesRankingLoss":
        train_loss = losses.MultipleNegativesRankingLoss(model)
        # only use the positive samples
        train_samples = [InputExample(texts=sample.texts) for sample in loaded_train_data if sample.label == 1]

    elif loss == "MegaBatchMarginLoss":
        train_loss = losses.MegaBatchMarginLoss(model, use_mini_batched_version=True, mini_batch_size=batch_size)
        # only use the positive samples
        train_samples = [InputExample(texts=sample.texts) for sample in loaded_train_data if sample.label == 1]

    elif loss == "ContrastiveLoss":
        train_loss = losses.ContrastiveLoss(model)
        # expects 0 or 1 as labels
        train_samples = [*loaded_train_data]

    elif loss in ["TripletLossOne", "TripletLossAll"]:
        train_loss = losses.TripletLoss(model)
        all_distinct_texts = {s.texts[0] for s in loaded_train_data}
        logger.info(f"Number of different texts in train: {len(all_distinct_texts)}")
        train_samples = []
        for text in all_distinct_texts:
            text_group = [t for t in loaded_train_data if t.texts[0] == text]
            pos = [t for t in text_group if t.label == 1]
            neg = [t for t in text_group if t.label == 0]
            if loss == "TripletLossOne":
                for p in pos:
                    sampled_neg = neg[np.random.randint(0, len(neg))]
                    # for each text and each positive label associated with it, create a triplet with a
                    # sampled negative label
                    train_samples.append(InputExample(texts=[text, p.texts[1], sampled_neg.texts[1]]))
            else:
                for p in pos:
                    for n in neg:
                        # for each text and each positive label associated with it, create a triplet
                        # with each negative label
                        train_samples.append(InputExample(texts=[text, p.texts[1], n.texts[1]]))

    else:
        err_str = f"Invalid loss: {loss}."
        raise ValueError(err_str)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    # 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)

    time_before_train = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_model_artifacts_location,
        show_progress_bar=True,
    )
    logger.info(f"Training took {(time.time()-time_before_train)/60} minutes.")
    model.save(output_model_artifacts_location)


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

    parser.add_argument("--epochs", help="Number of epochs that the Biencoder should be trained", type=int)
    parser.add_argument("--batch_size", help="Batch size during training", type=int)
    parser.add_argument("--base_model", type=str, help="Pretrained model path")
    parser.add_argument("--loss", type=str, help="Which loss to use for training (e.g. CosineSimilarityLoss)")

    parsed_args, _ = parser.parse_known_args()
    main(
        input_data_location=parsed_args.input_data_location,
        output_checkpoints_location=parsed_args.output_checkpoints_location,
        output_model_artifacts_location=parsed_args.output_model_artifacts_location,
        output_dir_location=parsed_args.output_dir_location,
        subset=parsed_args.subset,
        samples=parsed_args.samples,
        base_model=parsed_args.base_model,
        batch_size=parsed_args.batch_size,
        epochs=parsed_args.epochs,
        loss=parsed_args.loss,
    )
