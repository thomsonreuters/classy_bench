import pytest
from pydantic import ValidationError

from classy_bench._benchmark_config import (
    AWSMetricDefinition,
    AWSPipeline,
    AWSProcessingTask,
    AWSTrainingTask,
    read_benchmark_config,
)
from tests.testing_utils import TEST_CONFIG_PATH, TESTS_PATH


def test_aws_metric_definition():
    metric = AWSMetricDefinition(name="val:loss", regex="eval_loss': (.*?),")
    assert metric.name == "val:loss"
    assert metric.regex == "eval_loss': (.*?),"
    assert metric.model_dump(by_alias=True) == {"Name": "val:loss", "Regex": "eval_loss': (.*?),"}


def test_aws_processing_task():
    input_config = {
        "framework": "HuggingFace",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "max_runtime_in_hours": 24,
        "parameters": {"input_data_location": "s3://my-bucket/input", "output_data_location": "s3://my-bucket/output"},
        "processor_kwargs": {"image_uri": "my-image-uri"},
    }
    task = AWSProcessingTask.model_validate(input_config)
    assert task.valid_framework == "HuggingFace"
    assert task.max_runtime_in_seconds == 24 * 3600

    # Test the valid_framework property when framework is not provided
    input_config.pop("framework")
    task = AWSProcessingTask.model_validate(input_config)
    assert task.valid_framework == "SKLearn"


def test_aws_training_task():
    input_config = {
        "framework": "HuggingFace",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "max_runtime_in_hours": 24,
        "parameters": {"input_data_location": "s3://my-bucket/input", "output_data_location": "s3://my-bucket/output"},
        "enable_smdistributed": True,
        "metric_definitions": [
            {"name": "val:weighted1", "regex": "eval_weighted_f1': (.*?),"},
            {"name": "epoch", "regex": "epoch': (.*?)}"},
        ],
    }
    task = AWSTrainingTask.model_validate(input_config)
    assert task.valid_framework == "HuggingFace"
    assert task.max_runtime_in_seconds == 24 * 3600
    assert task.distribution == {"smdistributed": {"dataparallel": {"enabled": True}}}
    assert task.metric_definitions_parsed == [
        {"Name": "val:weighted1", "Regex": "eval_weighted_f1': (.*?),"},
        {"Name": "epoch", "Regex": "epoch': (.*?)}"},
    ]

    # Test the valid_framework property when framework is not provided
    input_config.pop("framework")
    task = AWSTrainingTask.model_validate(input_config)
    assert task.valid_framework == "SKLearn"


TEST_PIPELINE_CONFIG = {
    "name": "my-pipeline",
    "pipeline_type": "crossencoder",
    "preprocessing": {
        "framework": "HuggingFace",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "max_runtime_in_hours": 24,
        "parameters": {
            "input_data_location": "s3://my-bucket/input",
            "output_data_location": "s3://my-bucket/output",
        },
        "processor_kwargs": {"process": True},
    },
    "training": {
        "framework": "HuggingFace",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "max_runtime_in_hours": 24,
        "parameters": {
            "input_data_location": "s3://my-bucket/input",
            "output_data_location": "s3://my-bucket/output",
        },
        "enable_smdistributed": True,
        "metric_definitions": [
            {"name": "val:weighted1", "regex": "eval_weighted_f1': (.*?),"},
            {"name": "epoch", "regex": "epoch': (.*?)}"},
        ],
    },
    "evaluation": {
        "framework": "HuggingFace",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "max_runtime_in_hours": 24,
        "parameters": {
            "input_data_location": "s3://my-bucket/input",
            "output_data_location": "s3://my-bucket/output",
        },
    },
    "metrics_file_names": ["metrics.csv"],
}
EXPECTED_INSTANCE_DETAILS = [
    {
        "pipeline_name": "my-pipeline",
        "task_type": "preprocessing",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
    },
    {
        "pipeline_name": "my-pipeline",
        "task_type": "training",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
    },
    {
        "pipeline_name": "my-pipeline",
        "task_type": "evaluation",
        "instance_type": "ml.m5.large",
        "instance_count": 1,
    },
]


def test_aws_pipeline():
    pipeline = AWSPipeline.model_validate(TEST_PIPELINE_CONFIG)
    assert not pipeline.is_custom
    assert pipeline.code_location == TESTS_PATH.parent / "src" / "classy_bench" / "models" / "crossencoder"
    assert [td.model_dump() for td in pipeline.get_tasks_instances_details()] == EXPECTED_INSTANCE_DETAILS
    assert pipeline.default_preprocessing_step_name == "my-pipeline-preprocessing"
    assert pipeline.default_training_step_name == "my-pipeline-training"
    assert pipeline.default_evaluation_step_name == "my-pipeline-evaluation"


def test_pipeline_custom():
    pipeline_config = TEST_PIPELINE_CONFIG.copy()
    pipeline_config.pop("pipeline_type")
    pipeline_config["custom_code_location"] = "/path/to/code"
    pipeline = AWSPipeline.model_validate(pipeline_config)
    assert pipeline.is_custom
    assert pipeline.code_location.as_posix() == "/path/to/code"


def test_pipeline_value_error():
    pipeline_config = TEST_PIPELINE_CONFIG.copy()
    pipeline_config["custom_code_location"] = "/path/to/code"
    with pytest.raises(ValueError) as e:
        AWSPipeline.model_validate(pipeline_config)
        assert str(e.value) == "Only one of custom_code_location or pipeline_type can be provided."

    pipeline_config = TEST_PIPELINE_CONFIG.copy()
    pipeline_config.pop("pipeline_type")
    with pytest.raises(ValueError) as e:
        AWSPipeline.model_validate(pipeline_config)
        assert str(e.value) == "Please provide either custom_code_location or pipeline_type."

    pipeline_config = TEST_PIPELINE_CONFIG.copy()
    pipeline_config["pipeline_type"] = "invalid"
    with pytest.raises(ValidationError) as e:
        AWSPipeline.model_validate(pipeline_config)


def test_read_benchmark_config():
    config = read_benchmark_config(TEST_CONFIG_PATH)
    assert config.get_all_pipeline_names() == ["my-pipeline"]
    assert [td.model_dump() for td in config.get_all_tasks_instances_details()] == EXPECTED_INSTANCE_DETAILS
