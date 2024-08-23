import argparse
import logging
import math
import os
import pickle
import random
import time
from typing import List

import numpy as np
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
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


def _load_input_examples(input_examples_path: str) -> List[InputExample]:
    with open(input_examples_path, "rb") as f:
        loaded_input_examples: List[InputExample] = pickle.load(f)
    return loaded_input_examples


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
) -> None:

    subdirectories = [output_checkpoints_location, output_model_artifacts_location, output_dir_location]

    for subdir in subdirectories:
        try:
            os.mkdir(subdir)
        except Exception as e:
            logger.warning(e)
            continue

    start_time = time.time()

    loaded_train_samples = _load_input_examples(os.path.join(input_data_location, "train.pkl"))
    loaded_dev_samples = _load_input_examples(os.path.join(input_data_location, "dev.pkl"))

    if subset:
        loaded_train_samples = loaded_train_samples[:samples]
        loaded_dev_samples = loaded_dev_samples[:samples]

    logger.info(f"Loading data took: {time.time()-start_time} seconds.")

    train_samples = [InputExample(texts=sample.texts, label=float(sample.label)) for sample in loaded_train_samples]
    dev_samples = [InputExample(texts=sample.texts, label=float(sample.label)) for sample in loaded_dev_samples]

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    # We add an evaluator to evaluates the performance during training
    evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name="evaluator-dev")

    # 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)

    model = CrossEncoder(base_model, num_labels=1, max_length=512)

    # Train the model
    time_before_train = time.time()
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_model_artifacts_location,
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

    parser.add_argument("--epochs", help="Number of epochs that the Biencoder should be trained", type=int, default=1)
    parser.add_argument("--batch_size", help="Batch size during training", type=int, default=8)
    parser.add_argument("--base_model", type=str, help="Pretrained model path", default="casehold/custom-legalbert")

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
    )
