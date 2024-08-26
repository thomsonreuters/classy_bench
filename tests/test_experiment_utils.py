from unittest import mock

import pytest
from botocore.exceptions import ClientError

from classy_bench._experiments_utils import (
    check_trial_exists,
    create_or_load_experiment,
    create_or_load_trial,
)


def test_create_or_load_experiment_loads_existing():
    with mock.patch("smexperiments.experiment.Experiment.load") as mock_experiment_load:
        mock_experiment_load.return_value = mock.MagicMock()
        sagemaker_boto_client = mock.MagicMock()

        _ = create_or_load_experiment(experiment_name="test_experiment", sagemaker_boto_client=sagemaker_boto_client)

        mock_experiment_load.assert_called_once_with(
            experiment_name="test_experiment", sagemaker_boto_client=sagemaker_boto_client
        )


def test_create_or_load_experiment_creates_new():
    with mock.patch("smexperiments.experiment.Experiment.load") as mock_experiment_load, mock.patch(
        "smexperiments.experiment.Experiment.create"
    ) as mock_experiment_create:
        mock_experiment_load.side_effect = ClientError({"Error": {"Code": "ResourceNotFound"}}, "load")
        mock_experiment_create.return_value = mock.MagicMock()
        sagemaker_boto_client = mock.MagicMock()

        _ = create_or_load_experiment(experiment_name="test_experiment", sagemaker_boto_client=sagemaker_boto_client)

        mock_experiment_load.assert_called_once_with(
            experiment_name="test_experiment", sagemaker_boto_client=sagemaker_boto_client
        )
        mock_experiment_create.assert_called_once_with(
            experiment_name="test_experiment", sagemaker_boto_client=sagemaker_boto_client
        )


def test_create_or_load_trial_loads_existing():
    with mock.patch("smexperiments.trial.Trial.load") as mock_trial_load:
        mock_trial_load.return_value = mock.MagicMock()
        sagemaker_boto_client = mock.MagicMock()

        _ = create_or_load_trial(
            experiment_name="test_experiment", trial_name="test_trial", sagemaker_boto_client=sagemaker_boto_client
        )

        mock_trial_load.assert_called_once_with(trial_name="test_trial", sagemaker_boto_client=sagemaker_boto_client)


def test_create_or_load_trial_creates_new():
    with mock.patch("smexperiments.trial.Trial.load") as mock_trial_load, mock.patch(
        "smexperiments.trial.Trial.create"
    ) as mock_trial_create:
        mock_trial_load.side_effect = ClientError({"Error": {"Code": "ResourceNotFound"}}, "load")
        mock_trial_create.return_value = mock.MagicMock()
        sagemaker_boto_client = mock.MagicMock()

        _ = create_or_load_trial(
            experiment_name="test_experiment", trial_name="test_trial", sagemaker_boto_client=sagemaker_boto_client
        )

        mock_trial_load.assert_called_once_with(trial_name="test_trial", sagemaker_boto_client=sagemaker_boto_client)
        mock_trial_create.assert_called_once_with(
            experiment_name="test_experiment", trial_name="test_trial", sagemaker_boto_client=sagemaker_boto_client
        )


def test_check_trial_exists():
    with mock.patch("smexperiments.experiment.Experiment") as mock_experiment:
        mock_experiment.list_trials.return_value = [
            mock.MagicMock(trial_name="test_trial_1"),
            mock.MagicMock(trial_name="test_trial_2"),
        ]
        check_trial_exists(trial_name="test_trial_3", experiment=mock_experiment)
        mock_experiment.list_trials.assert_called_once()

    with mock.patch("smexperiments.experiment.Experiment") as mock_experiment:
        mock_experiment.list_trials.return_value = [
            mock.MagicMock(trial_name="test_trial_1"),
            mock.MagicMock(trial_name="test_trial_2"),
        ]
        with pytest.raises(ValueError) as err:
            check_trial_exists(trial_name="test_trial_1", experiment=mock_experiment)
        assert "Trial 'test_trial_1' already exists" in str(err.value)
