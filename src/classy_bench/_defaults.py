"""Default values for input, outputs, paths, etc. in AWS."""

from pathlib import Path
from typing import Union

from cloudpathlib import S3Path
from pydantic import BaseModel, Field, computed_field
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.entities import PipelineVariable

from classy_bench.typing import S3PathStr
from classy_bench.utils import get_random_string


class _BaseProcessTaskDefaults(BaseModel, arbitrary_types_allowed=True):
    """Base class for processing task defaults.

    Preprocessing and evaluation tasks are processing tasks.
    """

    code_path: Path
    input_data_path: S3PathStr
    output_root_path: S3PathStr
    random_string: str = Field(default_factory=get_random_string, init=False)

    @computed_field
    @property
    def input_code_location(self) -> ProcessingInput:
        """Get the input code location."""
        return ProcessingInput(
            source=str(self.code_path),
            destination="/opt/ml/processing/input/code/my_package/",
            input_name=f"input_code_location_{self.random_string}",
        )

    @computed_field
    @property
    def input_data_location(self) -> ProcessingInput:
        """Get the input data location."""
        return ProcessingInput(
            source=str(self.input_data_path),
            destination="/opt/ml/processing/input/data/",
            input_name=f"input_data_location_{self.random_string}",
        )

    @computed_field
    @property
    def output_data_location(self) -> ProcessingOutput:
        """Get the output data location."""
        return ProcessingOutput(
            source="/opt/ml/processing/output/data/",
            destination=str(S3Path(self.output_root_path) / "05_model_input"),
            output_name=f"output_data_location_{self.random_string}",
        )

    @computed_field
    @property
    def output_model_location(self) -> ProcessingOutput:
        """Get the output model location."""
        return ProcessingOutput(
            source="/opt/ml/processing/output/model/",
            destination=str(S3Path(self.output_root_path) / "06_models"),
            output_name=f"output_model_location_{self.random_string}",
        )

    @computed_field
    @property
    def output_report_location(self) -> ProcessingOutput:
        """Get the output report location."""
        return ProcessingOutput(
            source="/opt/ml/processing/output/report/",
            destination=str(S3Path(self.output_root_path) / "08_reporting"),
            output_name=f"output_report_location_{self.random_string}",
        )


class PreProcessingDefaults(_BaseProcessTaskDefaults):
    """Default values used in the processing task."""

    @computed_field
    @property
    def processing_script_path(self) -> Path:
        """Get the processing script path."""
        return self.code_path / "preprocess.py"


class TrainingDefaults(BaseModel, arbitrary_types_allowed=True):
    """Default values used in the training task."""

    trial_name: str
    code_path: Path
    input_data_path: S3PathStr
    output_root_path: S3PathStr
    random_string: str = Field(default_factory=get_random_string, init=False)
    training_script_name: str = Field(default="train.py", init=False)

    @computed_field
    @property
    def base_job_name(self) -> str:
        """Get the base job name."""
        # TODO this could be the same as in the other steps
        return f"{self.trial_name}{self.random_string}"

    @computed_field
    @property
    def input_data_location(self) -> TrainingInput:
        """Get the input data location."""
        return TrainingInput(s3_data=self.input_data_path)

    @computed_field
    @property
    def training_output_path(self) -> S3PathStr:
        """Get the training output path."""
        return str(S3Path(self.output_root_path) / "06_models")

    @computed_field
    @property
    def model_checkpoint_output_path(self) -> S3PathStr:
        """Get the model checkpoint output path."""
        return str(S3Path(self.training_output_path) / "checkpoints")


class EvaluationDefaults(_BaseProcessTaskDefaults):
    """Default values used in the evaluation task."""

    code_path: Path
    input_data_path: S3PathStr
    output_root_path: S3PathStr
    random_string: str = Field(default_factory=get_random_string, init=False)

    @computed_field
    @property
    def evaluation_script_path(self) -> Path:
        """Get the evaluation script path."""
        return self.code_path / "evaluate.py"

    def get_input_model_location(self, source: Union[str, PipelineVariable]) -> ProcessingInput:
        """Get the input model location."""
        return ProcessingInput(
            source=source,
            destination="/opt/ml/processing/input/model/",
            input_name=f"input_model_location_{self.random_string}",
        )
