"""Utils functions."""

import random
import string
from typing import Any, Dict, List


def get_random_string(length: int = 8) -> str:
    """Get a random string of a given length."""
    return "".join(
        random.choices(  # noqa: S311
            [*string.ascii_lowercase, *string.digits],
            k=length,
        )
    )


def get_arguments(aws_task_inputs: Dict[str, Any]) -> List[str]:
    """Get CLI arguments from the pipeline task inputs."""
    arguments = []
    if "parameters" in aws_task_inputs.keys():
        for key, value in aws_task_inputs["parameters"].items():
            arguments.append(f"--{key}")
            arguments.append(value)
    arguments.append("--sample_empty_param")
    arguments.append("sample_empty_value")
    return arguments
