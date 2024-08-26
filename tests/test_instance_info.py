from datetime import datetime, timezone

import pytest

from classy_bench._instance_info import (
    ProcessingTaskInstanceStats,
    _get_job_name,
    aggregate_instance_info,
    get_processing_statistics,
    get_training_statistics,
    is_step_in_trial,
)

TEST_SOURCE_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789:processing-job/pipelines-123456abcd-class-tfidf-evaluati-abcdef12345"
)


@pytest.mark.parametrize("processing_end_time", [datetime(2000, 1, 1, 2, 0, 0, tzinfo=timezone.utc), None])
def test_processing_task_instance_stats(processing_end_time):
    stats = ProcessingTaskInstanceStats(
        task_type="preprocessing",
        pipeline_name="test",
        failure_reason=None,
        processing_start_time=datetime(2000, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        processing_end_time=processing_end_time,
        instance_type="ml.test",
        instance_count=1,
    )
    if processing_end_time is None:
        assert stats.total_time_sec is None
    else:
        assert stats.total_time_sec == 1 * 60 * 60


def test_is_step_in_trial():
    step_name = "class-tfidf-evaluation"
    trial_component_name = "pipelines-123456abcd-class-tfidf-evaluati-abcdef12345-aws-processing-job"
    assert is_step_in_trial(step_name, trial_component_name)
    step_name = "custom-algo-evaluation"
    assert not is_step_in_trial(step_name, trial_component_name)


def test_get_job_name():
    assert _get_job_name(TEST_SOURCE_ARN) == "pipelines-123456abcd-class-tfidf-evaluati-abcdef12345"


class MockSagemakerClient:

    def __init__(self, mock_response):
        self._mock_response = mock_response

    def describe_processing_job(self, ProcessingJobName):  # noqa: N803
        _ = ProcessingJobName
        return self._mock_response

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        _ = TrainingJobName
        return self._mock_response


def test_get_processing_statistics():
    start_time = datetime(2000, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2000, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    mock_response = {
        "ProcessingStartTime": start_time,
        "ProcessingEndTime": end_time,
        "ProcessingResources": {"ClusterConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 2}},
    }
    output = get_processing_statistics(
        "preprocessing", "some-algo", TEST_SOURCE_ARN, MockSagemakerClient(mock_response)
    )
    assert output.task_type == "preprocessing"
    assert output.pipeline_name == "some-algo"
    assert output.failure_reason is None
    assert output.processing_start_time == start_time
    assert output.processing_end_time == end_time
    assert output.instance_type == "ml.m5.large"
    assert output.instance_count == 2


def test_get_training_statistics():
    mock_response = {
        "TrainingTimeInSeconds": 60,
        "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 2},
    }
    output = get_training_statistics("some-algo", TEST_SOURCE_ARN, MockSagemakerClient(mock_response))
    assert output.pieline_name == "some-algo"
    assert output.failure_reason is None
    assert output.total_time_sec == 60
    assert output.instance_type == "ml.m5.large"
    assert output.instance_count == 2


def test_aggregate_instance_info():
    start_time = datetime(2000, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2000, 1, 1, 1, 1, 0, tzinfo=timezone.utc)
    mock_processing_response = {
        "ProcessingStartTime": start_time,
        "ProcessingEndTime": end_time,
        "ProcessingResources": {"ClusterConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 2}},
    }
    mock_training_response = {
        "TrainingTimeInSeconds": 60,
        "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 2},
    }
    instance_info_list = [
        get_processing_statistics(
            "preprocessing", "some-algo", TEST_SOURCE_ARN, MockSagemakerClient(mock_processing_response)
        ),
        get_training_statistics("some-algo", TEST_SOURCE_ARN, MockSagemakerClient(mock_training_response)),
    ]
    output = aggregate_instance_info(instance_info_list)
    assert output["total_time_sec"] == 120
    assert output["instance_types"] == {"preprocessing": "ml.m5.large", "training": "ml.m5.large"}
    assert output["instance_counts"] == {"preprocessing": 2, "training": 2}
