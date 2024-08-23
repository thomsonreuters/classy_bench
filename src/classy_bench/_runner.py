from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Union

import boto3.session
import pandas as pd
import sagemaker
from botocore.config import Config
from cloudpathlib import S3Client, S3Path
from s3fs import S3FileSystem
from sagemaker.analytics import ExperimentAnalytics
from sagemaker.huggingface import HuggingFace, HuggingFaceProcessor
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from classy_bench._benchmark_config import (
    AWSPipeline,
    AWSProcessingTask,
    AWSTrainingTask,
    read_benchmark_config,
)
from classy_bench._config import DEFAULT_CONFIG_PATH
from classy_bench._defaults import (
    EvaluationDefaults,
    PreProcessingDefaults,
    TrainingDefaults,
)
from classy_bench._experiment_analytics import (
    ExperimentReport,
    PipelineExecutionHistory,
)
from classy_bench._experiments_utils import (
    check_trial_exists,
    create_or_load_experiment,
    create_or_load_trial,
)
from classy_bench._instance_info import (
    ProcessingTaskInstanceStats,
    TrainingTaskInstanceStats,
    aggregate_instance_info,
    get_processing_statistics,
    get_training_statistics,
    is_step_in_trial,
)
from classy_bench._results import get_metrics_results
from classy_bench.logging_utils import logger
from classy_bench.utils import get_arguments

if TYPE_CHECKING:
    from pathlib import Path

    from boto3.session import Session
    from mypy_boto3_sagemaker import SageMakerClient
    from sagemaker.workflow.entities import PipelineVariable
    from smexperiments.experiment import Experiment
    from smexperiments.trial import Trial

    from classy_bench.typing import FrameworkStr, PipelineExecutionStatusStr, S3PathStr


# Latest version: https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html
SKLEARN_FRAMEWORK_VERSION = "1.2-1"


class AWSBenchmarkRunner:
    """
    Main runner for orchestrating the benchmark execution on AWS.
    """

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        sagemaker_session_s3_bucket: str,
        role_arn: str,
        benchmark_config_path: Path = DEFAULT_CONFIG_PATH,
        use_existing_trial: bool = False,
        boto3_session: Union[Session, None] = None,
    ):
        """
        Create a new benchmark runner.

        Args:
            experiment_name: The name of the experiment.
            trial_name: The name of the trial.
            boto_session: The underlying Boto3 session which AWS service calls are delegated to. For more details: https://sagemaker.readthedocs.io/en/stable/session.html
            sagemaker_session_s3_bucket: The default S3 bucket of the SageMaker session. For more details: https://sagemaker.readthedocs.io/en/stable/session.html
            role_arn: The role arn that is assumed by workflow to create step artifacts. For more details: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
            benchmark_config_path: The path to the benchmark configuration file. Default to the path of the
            configuration file in the library.
            use_existing_trial: Whether to use an existing trial or create a new one. Default is False.
        """
        self.experiment_name = experiment_name

        # AWS and S3 setup
        self._boto3_session = boto3_session or boto3.session.Session()
        self.s3_fs = S3FileSystem(anon=False)
        self.role_arn = role_arn

        # SageMaker setup
        self.sagemaker_session_s3_bucket = sagemaker_session_s3_bucket
        self._sagemaker_session = sagemaker.Session(
            boto_session=self._boto3_session, default_bucket=self.sagemaker_session_s3_bucket
        )
        # SageMaker experiments details
        self.experiment = self._create_or_load_experiment()
        self.pipeline_session = PipelineSession(
            boto_session=self._boto3_session, default_bucket=self.sagemaker_session_s3_bucket
        )
        if not trial_name:
            err_str = "`trial-name` is empty, please provide a trial name."
            raise ValueError(err_str)
        self.trial_name = trial_name
        self.use_existing_trial = use_existing_trial
        if not self.use_existing_trial:
            check_trial_exists(trial_name=self.trial_name_sagemaker, experiment=self.experiment)
        self.trial = self._create_or_load_trial()

        logger.info("Creating an experiment with name: %s", self.experiment_name_sagemaker)

        # Benchmark config setup
        self._benchmark_config_path = benchmark_config_path
        self.benchmark_config = read_benchmark_config(config_file=self._benchmark_config_path)

        # Pipeline execution history
        self.pipeline_execution_history = PipelineExecutionHistory(
            experiment_name=self.experiment_name_sagemaker, trial_name=self.trial_name_sagemaker
        )

    def _get_sagemaker_client(self) -> SageMakerClient:
        return self._boto3_session.client(
            "sagemaker", config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 30})
        )

    @property
    def experiment_name_sagemaker(self) -> str:
        """The experiment name shown in SageMaker."""
        # Subclass to customize how the experiment name is passed to Sagemaker.
        return self.experiment_name

    @property
    def trial_name_sagemaker(self) -> str:
        """The trial name shown in SageMaker."""
        # Subclass to customize how the trial name is passed to Sagemaker.
        return self.trial_name

    @property
    def s3_trial_path(self) -> S3Path:
        """The S3 location where the trial artifacts are saved."""
        return (
            S3Path(f"s3://{self.sagemaker_session_s3_bucket}", client=S3Client(boto3_session=self._boto3_session))
            / self.experiment_name
            / self.trial_name
        )

    @property
    def s3_all_metrics_file_path(self) -> S3Path:
        """The S3 location of the aggregated metrics dataframe file."""
        return self.s3_trial_path / "all_algo_metrics.csv"

    def _create_or_load_experiment(self) -> Experiment:
        return create_or_load_experiment(
            experiment_name=self.experiment_name_sagemaker, sagemaker_boto_client=self._get_sagemaker_client()
        )

    def _create_or_load_trial(self) -> Trial:
        return create_or_load_trial(
            experiment_name=self.experiment_name_sagemaker,
            trial_name=self.trial_name_sagemaker,
            sagemaker_boto_client=self._get_sagemaker_client(),
        )

    def _get_pipeline_output_s3_path(self, pipeline_name: str) -> S3Path:
        return self.s3_trial_path / pipeline_name / "output"

    def _get_pipeline_reporting_s3_path(self, pipeine_name: str) -> S3Path:
        return self._get_pipeline_output_s3_path(pipeine_name) / "08_reporting"

    def _get_full_pipeline_name(self, pipeline_name: str) -> str:
        pipeline_name = pipeline_name.replace("_", "-")
        return f"{self.experiment_name_sagemaker}-{self.trial_name}-{pipeline_name}-Pipeline"

    def run_trial(self, s3_input_data_uri: Union[S3PathStr, None] = None) -> None:
        """Run a trial based on the given configuration file.

        Args:
            s3_input_data_uri: The S3 URI of the input data. If not provided, the input data URI
            from the config file will be used.
        """
        logger.info("Creating a trial with name: %s", self.trial_name_sagemaker)
        logger.info("These are the jobs that will be run in this trial:")
        for task_details in self.benchmark_config.get_all_tasks_instances_details():
            logger.info("  - %s", task_details.model_dump_json(by_alias=True))

        job_name = f"{self.trial_name_sagemaker}-{datetime.datetime.now(tz=datetime.timezone.utc)}"
        self._start_all_pipelines(job_name=job_name, s3_input_data_uri=s3_input_data_uri)

    def _get_sagemaker_processor(
        self,
        *,
        framework: FrameworkStr,
        base_job_name: str,
        instance_count: int,
        instance_type: str,
        max_runtime_in_seconds: int,
        **kwargs,
    ) -> Union[HuggingFaceProcessor, SKLearnProcessor]:
        if framework == "HuggingFace":
            logger.debug("Creating HuggingFaceProcessor...")
            processor_class_ = HuggingFaceProcessor
            processor_specific_kwargs = {
                # Only supported versions
                # https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#hugging-face-processor
                "transformers_version": "4.4.2",
                "pytorch_version": "1.6.0",
            }
        else:
            logger.debug("Creating SKLearnProcessor...")
            processor_class_ = SKLearnProcessor
            processor_specific_kwargs = {"framework_version": SKLEARN_FRAMEWORK_VERSION}

        other_kwargs = {**processor_specific_kwargs, **kwargs}

        return processor_class_(
            base_job_name=base_job_name,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=max_runtime_in_seconds,
            **other_kwargs,
        )

    def _get_sagemaker_estimator(
        self,
        *,
        framework: FrameworkStr,
        entry_point: str,
        source_dir: str,
        base_job_name: str,
        output_path: str,
        hyperparameters: dict,
        checkpoint_s3_uri: str,
        instance_type: str,
        instance_count: int,
        max_run: int,
        distribution: Union[dict, None],
        metric_definitions: Union[list, None],
        **kwargs,
    ) -> Union[HuggingFace, SKLearn]:
        if framework == "HuggingFace":
            logger.debug("Creating HuggingFace estimator...")
            estimator_class_ = HuggingFace
            estimator_specific_kwargs = {
                "transformers_version": "4.6.1",
                "pytorch_version": "1.7.1",
                # Only supported version
                # https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#hugging-face-model
                "py_version": "py36",
                "distribution": distribution,
            }
        else:
            logger.debug("Creating SKLearn estimator...")
            estimator_class_ = SKLearn
            estimator_specific_kwargs = {"framework_version": SKLEARN_FRAMEWORK_VERSION}

        other_kwargs = {**estimator_specific_kwargs, **kwargs}
        return estimator_class_(
            entry_point=entry_point,
            source_dir=source_dir,
            base_job_name=base_job_name,
            output_path=output_path,
            hyperparameters=hyperparameters,
            sagemaker_session=self.pipeline_session,
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type=instance_type,
            instance_count=instance_count,
            max_run=max_run,
            metric_definitions=metric_definitions,
            **other_kwargs,
        )

    def _get_preprocessing_step(
        self,
        pipeline: AWSPipeline,
        job_name: str,
        preprocessing_task_config: AWSProcessingTask,
        preprocessing_defaults: PreProcessingDefaults,
    ) -> ProcessingStep:
        logger.info("Defining preprocessing step for %s", pipeline.name)

        processing_arguments = {
            **preprocessing_task_config.model_dump(),
            "input_data_location": preprocessing_defaults.input_data_location.destination,
            "output_data_location": preprocessing_defaults.output_data_location.source,
            "output_model_location": preprocessing_defaults.output_model_location.source,
            "output_report_location": preprocessing_defaults.output_report_location.source,
        }
        processing_cli_arguments = get_arguments(processing_arguments)
        processor_kwargs = preprocessing_task_config.processor_kwargs or {}

        processor = self._get_sagemaker_processor(
            framework=preprocessing_task_config.valid_framework,
            base_job_name=job_name,
            instance_count=preprocessing_task_config.instance_count,
            instance_type=preprocessing_task_config.instance_type,
            max_runtime_in_seconds=preprocessing_task_config.max_runtime_in_seconds,
            **processor_kwargs,
        )

        logger.info("Input data path: %s", preprocessing_defaults.input_data_location.source)
        logger.info("Code path: %s", preprocessing_defaults.input_code_location.source)

        step_args = processor.run(
            job_name=job_name,
            code=str(preprocessing_defaults.processing_script_path),
            inputs=[preprocessing_defaults.input_code_location, preprocessing_defaults.input_data_location],
            outputs=[
                preprocessing_defaults.output_data_location,
                preprocessing_defaults.output_model_location,
                preprocessing_defaults.output_report_location,
            ],
            arguments=processing_cli_arguments,
            wait=False,
            logs=False,
        )
        return ProcessingStep(name=pipeline.default_preprocessing_step_name, step_args=step_args)

    def _get_training_step(
        self,
        pipeline: AWSPipeline,
        training_task_config: AWSTrainingTask,
        training_defaults: TrainingDefaults,
        depends_on: Union[List[str], None] = None,
    ) -> TrainingStep:

        logger.info("Defining training step for %s", pipeline.name)

        estimator_kwargs = training_task_config.estimator_kwargs or {}

        estimator = self._get_sagemaker_estimator(
            framework=training_task_config.valid_framework,
            entry_point=training_defaults.training_script_name,
            source_dir=str(pipeline.code_location),
            base_job_name=training_defaults.base_job_name,
            output_path=training_defaults.output_root_path,
            hyperparameters=training_task_config.parameters,
            checkpoint_s3_uri=training_defaults.model_checkpoint_output_path,
            instance_type=training_task_config.instance_type,
            instance_count=training_task_config.instance_count,
            max_run=training_task_config.max_runtime_in_seconds,
            distribution=training_task_config.distribution,
            metric_definitions=training_task_config.metric_definitions_parsed,
            **estimator_kwargs,
        )

        step_args = estimator.fit(
            job_name=training_defaults.base_job_name, inputs={"train": training_defaults.input_data_location}
        )
        return TrainingStep(name=pipeline.default_training_step_name, step_args=step_args, depends_on=depends_on)

    def _get_evaluation_step(
        self,
        pipeline: AWSPipeline,
        job_name: str,
        evaluation_task_config: AWSProcessingTask,
        evaluation_defaults: EvaluationDefaults,
        evaluation_input_model_s3_path: Union[S3PathStr, PipelineVariable],
        depends_on: Union[List[str], None] = None,
    ) -> ProcessingStep:

        logger.info("Defining evaluation step for %s", pipeline.name)

        input_model_location = evaluation_defaults.get_input_model_location(source=evaluation_input_model_s3_path)

        evaluation_arguments = {
            **evaluation_task_config.model_dump(),
            "input_data_location": evaluation_defaults.input_data_location.destination,
            "input_model_location": input_model_location.destination,
            "output_data_location": evaluation_defaults.output_data_location.source,
            "output_model_location": evaluation_defaults.output_model_location.source,
            "output_report_location": evaluation_defaults.output_report_location.source,
        }
        evaluation_cli_arguments = get_arguments(evaluation_arguments)
        processor_kwargs = evaluation_task_config.processor_kwargs or {}

        processor = self._get_sagemaker_processor(
            framework=evaluation_task_config.valid_framework,
            base_job_name=job_name,
            instance_count=evaluation_task_config.instance_count,
            instance_type=evaluation_task_config.instance_type,
            max_runtime_in_seconds=evaluation_task_config.max_runtime_in_seconds,
            **processor_kwargs,
        )

        step_args = processor.run(
            job_name=job_name,
            code=str(evaluation_defaults.evaluation_script_path),
            inputs=[
                evaluation_defaults.input_code_location,
                evaluation_defaults.input_data_location,
                input_model_location,
            ],
            outputs=[
                evaluation_defaults.output_data_location,
                evaluation_defaults.output_model_location,
                evaluation_defaults.output_report_location,
            ],
            arguments=evaluation_cli_arguments,  # optional
            wait=False,
            logs=False,
        )
        return ProcessingStep(name=pipeline.default_evaluation_step_name, step_args=step_args, depends_on=depends_on)

    def _validate_pipeline(
        self, *, pipeline: AWSPipeline, job_name: str, s3_input_data_uri: Union[S3PathStr, None] = None
    ) -> Pipeline:
        logger.info("Validating pipeline '%s'", pipeline.name)
        input_data_location = s3_input_data_uri if s3_input_data_uri else pipeline.s3_input_data_uri
        if not input_data_location:
            err_str = "`s3_input_data_uri` must be provided in the config file or in `run_trial` method!"
            raise ValueError(err_str)

        if pipeline.evaluation and not pipeline.training:
            if not pipeline.s3_model_uri:
                err_str = "When training step is not present, the `s3_model_uri` must be provided in the config file!"
                raise ValueError(err_str)

        output_location = str(self._get_pipeline_output_s3_path(pipeline.name))
        logger.info("Using output location: %s", output_location)

        pipeline_steps: List[ProcessingStep, TrainingStep] = []
        if preprocessing_task := pipeline.preprocessing:
            logger.info("Validating preprocessing step...")
            preprocessing_defaults = PreProcessingDefaults(
                code_path=pipeline.code_location,
                input_data_path=input_data_location,
                output_root_path=output_location,
            )
            preprocessing_step = self._get_preprocessing_step(
                pipeline=pipeline,
                job_name=job_name,
                preprocessing_task_config=preprocessing_task,
                preprocessing_defaults=preprocessing_defaults,
            )
            pipeline_steps.append(preprocessing_step)

        if training_task := pipeline.training:
            logger.info("Validating training step...")
            if preprocessing_task:
                training_input_data_location = preprocessing_defaults.output_data_location.destination
            else:
                training_input_data_location = input_data_location
            training_defaults = TrainingDefaults(
                trial_name=self.trial.trial_name,
                code_path=pipeline.code_location,
                input_data_path=training_input_data_location,
                output_root_path=output_location,
            )
            training_step = self._get_training_step(
                pipeline=pipeline,
                training_task_config=training_task,
                training_defaults=training_defaults,
                depends_on=[preprocessing_step.name] if preprocessing_task else None,
            )
            pipeline_steps.append(training_step)

        if evaluation_task := pipeline.evaluation:
            logger.info("Validating evaluation step...")
            evaluation_input_model_s3_path = (
                training_step.properties.ModelArtifacts.S3ModelArtifacts if training_task else pipeline.s3_model_uri
            )
            evaluation_input_data_location = (
                preprocessing_defaults.output_data_location.destination if preprocessing_task else input_data_location
            )
            evaluation_defaults = EvaluationDefaults(
                code_path=pipeline.code_location,
                input_data_path=evaluation_input_data_location,
                output_root_path=output_location,
            )
            evaluation_step = self._get_evaluation_step(
                pipeline=pipeline,
                job_name=job_name,
                evaluation_task_config=evaluation_task,
                evaluation_defaults=evaluation_defaults,
                evaluation_input_model_s3_path=evaluation_input_model_s3_path,
                depends_on=[training_step.name] if training_task else None,
            )
            pipeline_steps.append(evaluation_step)

        logger.info("Pipeline steps: %s", [f"{step.__class__.__name__}: {step.name}" for step in pipeline_steps])
        pipeline_name = self._get_full_pipeline_name(pipeline_name=pipeline.name)
        logger.info("Creating pipeline with name: %s", pipeline_name)
        pipeline = Pipeline(
            name=pipeline_name,
            pipeline_experiment_config=PipelineExperimentConfig(
                self.experiment_name_sagemaker, self.trial_name_sagemaker
            ),
            steps=pipeline_steps,
            sagemaker_session=self.pipeline_session,
        )
        pipeline.upsert(role_arn=self.role_arn)
        return pipeline

    def _start_all_pipelines(self, job_name: str, s3_input_data_uri: Union[S3PathStr, None] = None):
        """Start all pipelines in the benchmark in AWS."""
        pipeline_list = [
            self._validate_pipeline(pipeline=pipeline, job_name=job_name, s3_input_data_uri=s3_input_data_uri)
            for pipeline in self.benchmark_config.pipeline_list
        ]
        for pipeline in pipeline_list:
            logger.info("Starting pipeline '%s' execution in AWS...", pipeline.name)
            pipeline_execution = pipeline.start()
            self.pipeline_execution_history.add_pipeline_details(
                pipeline_name=pipeline.name, pipeline_execution_arn=pipeline_execution.arn
            )

    @staticmethod
    def _get_status_with_emoji(execution_status: PipelineExecutionStatusStr) -> str:
        status_emoji_map = {
            "Executing": "🟡",
            "Stopping": "🟠",
            "Stopped": "🔴",
            "Failed": "🔴",
            "Succeeded": "🟢",
        }
        return f"{status_emoji_map.get(execution_status)} {execution_status}"

    def view_current_benchmark_execution_status(self) -> pd.DataFrame:
        """View the execution status of the current trial for each pipeline in the benchmark."""
        if not self.pipeline_execution_history.history:
            err_str = "Pipeline execution history is empty. Please run a trial first."
            raise ValueError(err_str)

        sagemaker_client = self._get_sagemaker_client()

        pipeline_execution_list = []
        for pipeline_details in self.pipeline_execution_history.history:
            pipeline_execution = sagemaker_client.describe_pipeline_execution(
                PipelineExecutionArn=pipeline_details.pipeline_execution_arn
            )
            execution_status = pipeline_execution["PipelineExecutionStatus"]
            pipeline_execution_list.append(
                {
                    "experiment_name": self.pipeline_execution_history.experiment_name,
                    "trial_name": self.pipeline_execution_history.trial_name,
                    "pipeline_name": pipeline_details.pipeline_name,
                    "pipeline_execution_arn": pipeline_details.pipeline_execution_arn,
                    "pipeline_execution_status": self._get_status_with_emoji(execution_status),
                    "created_time": pipeline_execution["CreationTime"],
                    "last_modified_time": pipeline_execution["LastModifiedTime"],
                }
            )
        return pd.DataFrame(pipeline_execution_list)

    def view_all_benchmark_execution_status(self) -> pd.DataFrame:
        """View the execution status of all (past and current) trials for each pipeline in the benchmark."""
        sagemaker_client = self._get_sagemaker_client()

        all_pipeline_executions = []
        for pipeline in self.benchmark_config.pipeline_list:
            pipeline_name = self._get_full_pipeline_name(pipeline_name=pipeline.name)
            pipeline_execution_list = sagemaker_client.list_pipeline_executions(PipelineName=pipeline_name)[
                "PipelineExecutionSummaries"
            ]
            all_pipeline_executions.extend(
                [
                    {
                        "experiment_name": self.pipeline_execution_history.experiment_name,
                        "trial_name": self.pipeline_execution_history.trial_name,
                        "pipeline_name": pipeline_name,
                        "pipeline_execution_arn": pipeline_execution["PipelineExecutionArn"],
                        "pipeline_execution_status": self._get_status_with_emoji(
                            pipeline_execution["PipelineExecutionStatus"]
                        ),
                        "start_time": pipeline_execution["StartTime"],
                    }
                    for pipeline_execution in pipeline_execution_list
                ]
            )
        return pd.DataFrame(all_pipeline_executions)

    def get_experiment_analytics(self) -> ExperimentAnalytics:
        """Return the Sagemaker ExperimentAnalytics object for the experiment.

        More info: https://sagemaker.readthedocs.io/en/stable/api/training/analytics.html#sagemaker.analytics.ExperimentAnalytics
        """
        return ExperimentAnalytics(
            sagemaker_session=self._sagemaker_session,
            experiment_name=self.experiment_name_sagemaker,
        )

    def _get_trial_instances_info(
        self, pipeline: AWSPipeline
    ) -> List[Union[ProcessingTaskInstanceStats, TrainingTaskInstanceStats]]:
        experiment_analytics = self.get_experiment_analytics()

        experiment_report = ExperimentReport.from_experiment_analytics(experiment_analytics)

        if self.trial_name_sagemaker not in experiment_report.trial_names:
            err_str = f"Trial '{self.trial_name_sagemaker}' not found."
            raise RuntimeError(err_str)

        sagemaker_client = self._get_sagemaker_client()
        all_instance_stats = []
        for trial_meta in experiment_report.trial_metadata_list:
            if trial_meta.name == self.trial_name_sagemaker:
                if is_step_in_trial(
                    step_name=pipeline.default_preprocessing_step_name, trial_component_name=trial_meta.component_name
                ):
                    preprocessing_stats = get_processing_statistics(
                        task_type="preprocessing",
                        pipeline_name=pipeline.name,
                        processing_source_arn=trial_meta.source_arn,
                        sagemaker_client=sagemaker_client,
                    )
                    all_instance_stats.append(preprocessing_stats)
                if is_step_in_trial(
                    step_name=pipeline.default_training_step_name, trial_component_name=trial_meta.component_name
                ):
                    training_stats = get_training_statistics(
                        pipeline_name=pipeline.name,
                        training_souce_arn=trial_meta.source_arn,
                        sagemaker_client=sagemaker_client,
                    )
                    all_instance_stats.append(training_stats)
                if is_step_in_trial(
                    step_name=pipeline.default_evaluation_step_name, trial_component_name=trial_meta.component_name
                ):
                    evaluation_stats = get_processing_statistics(
                        task_type="evaluation",
                        pipeline_name=pipeline.name,
                        processing_source_arn=trial_meta.source_arn,
                        sagemaker_client=sagemaker_client,
                    )
                    all_instance_stats.append(evaluation_stats)
        return all_instance_stats

    def _get_aggregated_instance_info(self, pipeline: AWSPipeline) -> Dict[str, Any]:
        instance_stats_list = self._get_trial_instances_info(pipeline=pipeline)
        return aggregate_instance_info(instance_stats_list=instance_stats_list)

    def create_aggregated_results(self):
        """Create aggregated metrics results for the benchmark trial and save them to S3."""
        logger.info("Creating aggregated metrics results for %s", self.trial_name)
        all_metrics_data = []
        for pipeline in self.benchmark_config.pipeline_list:
            logger.info("Getting results for %s", pipeline.name)
            instance_info_dict = self._get_aggregated_instance_info(pipeline=pipeline)
            for metric_file_name in pipeline.metrics_file_names:
                logger.info("Getting results for %s", metric_file_name)
                metric_file_s3_path = str(
                    self._get_pipeline_reporting_s3_path(pipeine_name=pipeline.name) / metric_file_name
                )
                if self.s3_fs.exists(metric_file_s3_path):
                    with self.s3_fs.open(metric_file_s3_path, "r") as f:
                        metrics_csv_string = f.read()
                else:
                    logger.warning("Metrics file not found in '%s'. It will be skipped.", metric_file_s3_path)
                    continue
                metrics_dict = get_metrics_results(metrics_csv_string)
                all_metrics_data.append(
                    {
                        "pipeline_name": pipeline.name,
                        "metric_file_name": metric_file_name,
                        **metrics_dict,
                        **instance_info_dict,
                    }
                )

        all_metrics_df = pd.DataFrame(all_metrics_data)
        all_metrics_file_path = str(self.s3_all_metrics_file_path)

        with self.s3_fs.open(all_metrics_file_path, "w") as f:
            all_metrics_df.to_csv(f, index=False)

        logger.info("All metrics saved to S3 in: %s", all_metrics_file_path)

    def get_metric_results_dataframe(self) -> pd.DataFrame:
        """Return the aggregated metrics of the benchmark trial as a pandas DataFrame."""
        all_metrics_file_path = str(self.s3_all_metrics_file_path)
        if not self.s3_fs.exists(all_metrics_file_path):
            logger.warning("Metrics file not found in '%s'. It will be created.", all_metrics_file_path)
            self.create_aggregated_results()
        with self.s3_fs.open(all_metrics_file_path, "r") as f:
            return pd.read_csv(f)
