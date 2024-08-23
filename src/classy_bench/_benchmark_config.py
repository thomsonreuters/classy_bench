"""Parsing benchmark configurations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, computed_field, model_validator

from classy_bench._config import MODEL_SCRIPTS_PATH
from classy_bench.typing import FrameworkStr, PipelineTypeStr, S3PathStr, TaskType


class AWSMetricDefinition(BaseModel):
    """AWS Sagemaker metric definition.

    More info: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_MetricDefinition.html
    """

    name: str = Field(serialization_alias="Name")
    regex: str = Field(serialization_alias="Regex")


DEFAULT_METRIC_DEFINITION = [
    AWSMetricDefinition(name="val:loss", regex="eval_loss': (.*?),"),
    AWSMetricDefinition(name="val:microf1", regex="eval_micro_f1': (.*?),"),
    AWSMetricDefinition(name="val:macrof1", regex="eval_macro_f1': (.*?),"),
    AWSMetricDefinition(name="val:weighted1", regex="eval_weighted_f1': (.*?),"),
    AWSMetricDefinition(name="epoch", regex="epoch': (.*?)}"),
]


class AWSTask(BaseModel):
    """AWS task configuration.

    Args:
        framework: The framework to use for the task. Optional, if not provided,
        it is inferred from the instance type.
        instance_type: The type of EC2 instance to use for the task.
        instance_count: The number of instances to use for the task.
        max_runtime_in_hours: The maximum runtime in hours for the task.
        parameters: The parameters to pass to the task script.
    """

    framework: Union[FrameworkStr, None] = None
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    max_runtime_in_hours: int = 24
    parameters: dict

    @computed_field
    @property
    def valid_framework(self) -> FrameworkStr:
        """Return a framework string even if it is not provided."""
        if self.framework:
            return self.framework
        if self.instance_type.split(".")[1][:2] in ["g4", "g5", "p3", "p4"]:
            # GPU instances default to HuggingFace
            return "HuggingFace"
        return "SKLearn"

    @computed_field
    @property
    def max_runtime_in_seconds(self) -> int:
        """Get the maximum runtime in seconds."""
        return self.max_runtime_in_hours * 3600


class AWSProcessingTask(AWSTask):
    """AWS processing task configuration.

    Available frameworks: `SKLearn`, `HuggingFace`.
    See docs for [Scikit Learn Processor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor) and [HuggingFace Processor](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#hugging-face-processor)
    for more information.

    Args:
        framework: The framework to use for the task. Optional, if not
        provided, it is inferred from the instance type.
        instance_type: The type of EC2 instance to use for the task.
        instance_count: The number of instances to use for the task.
        max_runtime_in_hours: The maximum runtime in hours for the task.
        parameters: The parameters to pass to the task script.
        processor_kwargs: Any other kwargs argument that the processor accepts.
    """  # noqa: E501

    processor_kwargs: Union[Dict[str, Any], None] = None


class AWSTrainingTask(AWSTask):
    """AWS training task configuration.

    Args:
        framework: The framework to use for the task. Optional, if not provided,
        it is inferred from the instance type.
        instance_type: The type of EC2 instance to use for the task.
        instance_count: The number of instances to use for the task.
        max_runtime_in_hours: The maximum runtime in hours for the task.
        parameters: The parameters to pass to the task script.
        enable_smdistributed: Whether to enable distributed training using
        SMDistributed Data Parallel.
        metric_definitions: The metric definitions to use for the task.
        estimator_kwargs: Any other kwargs argument that the estimator accepts.
    """

    instance_type: str = "ml.m5.xlarge"
    enable_smdistributed: bool = False
    metric_definitions: List[AWSMetricDefinition] = DEFAULT_METRIC_DEFINITION
    estimator_kwargs: Union[Dict[str, Any], None] = None

    @computed_field
    @property
    def distribution(self) -> Union[Dict, None]:
        """Get the config for distributed training."""
        if self.enable_smdistributed:
            return {"smdistributed": {"dataparallel": {"enabled": True}}}
        return None

    @computed_field
    @property
    def metric_definitions_parsed(self) -> List[Dict[str, Any]]:
        """Get the metric definitions in a list of dictionaries."""
        return [metric_def.model_dump(by_alias=True) for metric_def in self.metric_definitions]


class TaskInstanceDetails(BaseModel):
    """Details of the EC2 instance used in a pipeline task."""

    pipeline_name: str
    task_type: TaskType
    instance_type: str
    instance_count: int


class AWSPipeline(BaseModel):
    """AWS pipeline configuration.

    Args:
        name: The name of the pipeline.
        s3_input_data_uri: The S3 URI of the location of the input data (i.e. train.csv, dev.csv, test.csv).
        s3_model_uri: The S3 URI of the location to save the model artifacts.
        pipeline_type: Which supported pipeline to use. Optional if custom_code_location is provided.
        custom_code_location: The path to the custom code to use for the pipeline.
        preprocessing: The configuration for the preprocessing step.
        training: The configuration for the training step.
        evaluation: The configuration for the evaluation step.
        metrics_file_names: The names of the files containing the metrics that should be shown in the report.
    """

    name: str = Field(..., max_length=20)
    s3_input_data_uri: Union[S3PathStr, None] = None
    s3_model_uri: Union[S3PathStr, None] = None
    pipeline_type: Union[PipelineTypeStr, None] = None
    custom_code_location: Union[Path, None] = None
    preprocessing: Union[AWSProcessingTask, None] = None
    training: Union[AWSTrainingTask, None] = None
    evaluation: Union[AWSProcessingTask, None] = None
    metrics_file_names: List[str] = ["metrics.csv"]

    @model_validator(mode="after")
    def custom_code_location_or_pipeline_type(self):
        """Ensure that only custom code location or model type is provided and at least one is provided."""
        if self.custom_code_location and self.pipeline_type:
            err_str = "Only one of custom_code_location or pipeline_type can be provided."
            raise ValueError(err_str)
        if not self.custom_code_location and not self.pipeline_type:
            err_str = "Please provide either custom_code_location or pipeline_type."
            raise ValueError(err_str)
        return self

    @computed_field
    @property
    def is_custom(self) -> bool:
        """Check if the pipeline uses custom code."""
        return self.custom_code_location is not None

    @computed_field
    @property
    def code_location(self) -> Path:
        """Get the location of the code to use in the pipeline steps."""
        if self.is_custom:
            return self.custom_code_location
        return MODEL_SCRIPTS_PATH / self.pipeline_type

    def get_tasks_instances_details(self) -> List[TaskInstanceDetails]:
        """Get the details of the instances used in the tasks."""
        task_types_map = {
            TaskType.PREPROCESSING: self.preprocessing,
            TaskType.TRAINING: self.training,
            TaskType.EVALUATION: self.evaluation,
        }
        tasks_details = [
            TaskInstanceDetails(
                pipeline_name=self.name,
                task_type=task_type,
                instance_type=task.instance_type,
                instance_count=task.instance_count,
            )
            for task_type, task in task_types_map.items()
            if isinstance(task, AWSTask)
        ]
        return tasks_details

    @computed_field
    @property
    def default_preprocessing_step_name(self) -> str:
        return f"{self.name}-preprocessing"

    @computed_field
    @property
    def default_training_step_name(self) -> str:
        return f"{self.name}-training"

    @computed_field
    @property
    def default_evaluation_step_name(self) -> str:
        return f"{self.name}-evaluation"


class AWSBenchmarkConfig(BaseModel):
    """The Benchmark configuration.

    Args:
        pipeline_list: The list of pipelines to run in the benchmark.
    """

    pipeline_list: List[AWSPipeline]

    def get_all_pipeline_names(self) -> List[str]:
        """Get all pipeline names."""
        return [pipeline.name for pipeline in self.pipeline_list]

    def get_all_tasks_instances_details(self) -> List[TaskInstanceDetails]:
        """Get the details of the instances used in the tasks."""
        tasks_details = [
            task_instance_details
            for pipeline in self.pipeline_list
            for task_instance_details in pipeline.get_tasks_instances_details()
        ]
        return tasks_details


def read_benchmark_config(config_file: Path) -> AWSBenchmarkConfig:
    """Read a benchmark configuration file.

    Args:
        config_file: The path to the benchmark configuration JSON file.
    """
    with open(config_file) as file:
        data = json.load(file)
    return AWSBenchmarkConfig.model_validate(data)
