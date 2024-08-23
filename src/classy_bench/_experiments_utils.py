from __future__ import annotations

from typing import TYPE_CHECKING

from botocore.exceptions import ClientError
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

if TYPE_CHECKING:
    from mypy_boto3_sagemaker import SageMakerClient


def create_or_load_experiment(*, experiment_name: str, sagemaker_boto_client: SageMakerClient) -> Experiment:
    """Try to load an existing experiment or create a new one."""
    try:
        return Experiment.load(experiment_name=experiment_name, sagemaker_boto_client=sagemaker_boto_client)
    except ClientError as error:
        if error.response["Error"]["Code"] == "ResourceNotFound":
            return Experiment.create(experiment_name=experiment_name, sagemaker_boto_client=sagemaker_boto_client)


def create_or_load_trial(*, experiment_name: str, trial_name: str, sagemaker_boto_client: SageMakerClient) -> Trial:
    """Try to load an existing trial or create a new one."""
    try:
        return Trial.load(trial_name=trial_name, sagemaker_boto_client=sagemaker_boto_client)
    except ClientError as error:
        if error.response["Error"]["Code"] == "ResourceNotFound":
            return Trial.create(
                experiment_name=experiment_name, trial_name=trial_name, sagemaker_boto_client=sagemaker_boto_client
            )


def check_trial_exists(trial_name: str, experiment: Experiment) -> None:
    """Check if a trial exists in an experiment."""
    existing_trial_names = [trial.trial_name for trial in experiment.list_trials()]
    if trial_name in existing_trial_names:
        error_message = (
            f"Trial '{trial_name}' already exists. Please provide a new trial name."
            f"\nExisting trials are: {existing_trial_names}"
            "\nIf you want to proceed with the same trial name, please set `use_existing_trial` to True."
        )
        raise ValueError(error_message)
