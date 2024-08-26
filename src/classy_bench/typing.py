import re
from enum import Enum
from typing import Literal

from pydantic import StringConstraints
from typing_extensions import Annotated

# copied from https://gist.github.com/rajivnarayan/c38f01b89de852b3e7d459cfde067f3f#file-s3path-py
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
_s3_pattern = re.compile(
    r"^s3://"
    # Bucket name must start with a letter or digit
    r"(?=[a-z0-9])"
    # Bucket name must not start with xn--, sthree-, sthree-configurator or end with -s3alias
    r"(?!(^xn--|sthree-|sthree-configurator|.+-s3alias$))"
    # Bucket name must not contain two adjacent periods
    r"(?!.*\.\.)"
    # Bucket naming constraints
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]"
    # Bucket name must not end with a period followed by a hyphen
    r"(?<!\.-$)"
    # Bucket name must not end with a period
    r"(?<!\.$)"
    # Bucket name must not end with a hyphen
    r"(?<!-$)"
    # key naming constraints
    r"(/([a-zA-Z0-9._-]+/?)*)?$"
)

S3PathStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=1023, pattern=_s3_pattern)]


class TaskType(str, Enum):
    """The type of a task."""

    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATION = "evaluation"


# The frameworks supported by the library
# https://sagemaker.readthedocs.io/en/stable/frameworks/index.html
FrameworkStr = Literal["SKLearn", "HuggingFace"]

PipelineTypeStr = Literal["biencoder", "bm25", "class_tfidf", "crossencoder", "doc_tfidf", "roberta", "t5"]

PipelineExecutionStatusStr = Literal["Executing", "Stopping", "Stopped", "Failed", "Succeeded"]
