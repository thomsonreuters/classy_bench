from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from mypy_boto3_sagemaker import SageMakerClient


class ProcessingTaskInstanceStats(BaseModel):
    task_type: Literal["preprocessing", "evaluation"]
    pipeline_name: str
    failure_reason: Union[str, None]
    processing_start_time: Union[datetime, None] = Field(..., exclude=True)
    processing_end_time: Union[datetime, None] = Field(..., exclude=True)
    instance_type: str
    instance_count: int

    @property
    def _total_processing_time_sec(self) -> Union[int, None]:
        if self.processing_start_time and self.processing_end_time:
            return (self.processing_end_time - self.processing_start_time).seconds
        return None

    @computed_field
    @property
    def total_time_sec(self) -> Union[int, None]:
        return self._total_processing_time_sec


class TrainingTaskInstanceStats(BaseModel):
    task_type: Literal["training"] = Field("training", init=False)
    pieline_name: str
    failure_reason: Union[str, None]
    total_training_time_sec: Union[int, None] = Field(..., exclude=True)
    instance_type: str
    instance_count: int

    @computed_field
    @property
    def total_time_sec(self) -> Union[int, None]:
        return self.total_training_time_sec


def is_step_in_trial(step_name: str, trial_component_name: str) -> bool:
    """Check if a step is in a trial component name."""
    return step_name[0:20].replace("_", "-") in trial_component_name


def _get_job_name(source_arn: str) -> str:
    return source_arn.split("/")[1]


def get_processing_statistics(
    task_type: Literal["processing", "evaluation"],
    pipeline_name: str,
    processing_source_arn: str,
    sagemaker_client: SageMakerClient,
) -> ProcessingTaskInstanceStats:
    processing_job_name = _get_job_name(source_arn=processing_source_arn)
    response = sagemaker_client.describe_processing_job(ProcessingJobName=processing_job_name)
    return ProcessingTaskInstanceStats(
        task_type=task_type,
        pipeline_name=pipeline_name,
        failure_reason=response.get("FailureReason"),
        processing_start_time=response.get("ProcessingStartTime"),
        processing_end_time=response.get("ProcessingEndTime"),
        instance_type=response["ProcessingResources"]["ClusterConfig"]["InstanceType"],
        instance_count=response["ProcessingResources"]["ClusterConfig"]["InstanceCount"],
    )


def get_training_statistics(
    pipeline_name: str, training_souce_arn: str, sagemaker_client: SageMakerClient
) -> TrainingTaskInstanceStats:
    training_job_name = _get_job_name(source_arn=training_souce_arn)
    response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
    return TrainingTaskInstanceStats(
        pieline_name=pipeline_name,
        failure_reason=response.get("FailureReason"),
        total_training_time_sec=response.get("TrainingTimeInSeconds"),
        instance_type=response["ResourceConfig"]["InstanceType"],
        instance_count=response["ResourceConfig"]["InstanceCount"],
    )


def aggregate_instance_info(
    instance_stats_list: List[Union[ProcessingTaskInstanceStats, TrainingTaskInstanceStats]]
) -> Dict[str, Any]:
    return {
        "total_time_sec": sum([instance.total_time_sec for instance in instance_stats_list if instance.total_time_sec]),
        "instance_types": {instance.task_type: instance.instance_type for instance in instance_stats_list},
        "instance_counts": {instance.task_type: instance.instance_count for instance in instance_stats_list},
    }
