from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List
from warnings import warn

if TYPE_CHECKING:

    import pandas as pd
    from sagemaker.analytics import ExperimentAnalytics


_COLUMNS_TO_KEEP = ["Trials", "TrialComponentName", "SourceArn", "SageMaker.InstanceType"]


@dataclass
class TrialMetadata:
    """The metadata of a trial."""

    name: str
    component_name: str
    source_arn: str
    sagemaker_instance_type: str

    @classmethod
    def from_experiment_analytics_row(cls, row: pd.Series):
        """Create a TrialMetadata object from a row of an ExperimentAnalytics dataframe."""
        trial_name = row["Trials"]
        if len(trial_name) != 1:
            warn("Some trials have more than one trial name. The first trial name will be used.")  # noqa: B028
        return cls(
            name=row["Trials"][0],
            component_name=row["TrialComponentName"],
            source_arn=row["SourceArn"],
            sagemaker_instance_type=row["SageMaker.InstanceType"],
        )


def _get_trial_names(trial_metadata_list: List[TrialMetadata]) -> List[str]:
    return list({trial_metadata.name for trial_metadata in trial_metadata_list})


@dataclass
class ExperimentReport:
    """A report of a SageMaker experiment."""

    trial_names: List[str]
    trial_metadata_list: List[TrialMetadata]

    @classmethod
    def from_experiment_analytics(cls, experiment_analytics: ExperimentAnalytics):
        """Create an ExperimentReport from an ExperimentAnalytics object."""
        experiment_analytics_df: pd.DataFrame = experiment_analytics.dataframe(force_refresh=True)
        if "Trials" not in experiment_analytics_df.columns:
            err_str = "Experiment statistics and results are not ready yet."
            raise ValueError(err_str)
        experiment_analytics_df = experiment_analytics_df[_COLUMNS_TO_KEEP]
        trial_metadata_list = [
            TrialMetadata.from_experiment_analytics_row(row) for _, row in experiment_analytics_df.iterrows()
        ]
        trial_names = _get_trial_names(trial_metadata_list)
        return cls(trial_names=trial_names, trial_metadata_list=trial_metadata_list)

    def get_trial_metadata(self, trial_name: str) -> List[TrialMetadata]:
        """Get the metadata of a trial."""
        return [trial_metadata for trial_metadata in self.trial_metadata_list if trial_metadata.name == trial_name]


@dataclass
class PipelineDetails:
    """The details of a pipeline execution."""

    pipeline_name: str
    pipeline_execution_arn: str


@dataclass
class PipelineExecutionHistory:
    """The history of a pipeline execution."""

    experiment_name: str
    trial_name: str
    history: List[PipelineDetails] = field(default_factory=list)

    def add_pipeline_details(self, pipeline_name: str, pipeline_execution_arn: str):
        """Attach pipeline details to the history."""
        self.history.append(PipelineDetails(pipeline_name=pipeline_name, pipeline_execution_arn=pipeline_execution_arn))
