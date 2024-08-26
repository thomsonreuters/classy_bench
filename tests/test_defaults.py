from pathlib import Path

from classy_bench._defaults import (
    EvaluationDefaults,
    PreProcessingDefaults,
    TrainingDefaults,
)


def test_processing_defaults():
    defaults = PreProcessingDefaults(
        code_path="/code/path",
        input_data_path="s3://inputdata/path",
        output_root_path="s3://outputroot/path",
    )
    assert defaults.processing_script_path.as_posix() == "/code/path/preprocess.py"
    input_code_location = defaults.input_code_location
    assert Path(input_code_location.source).as_posix() == "/code/path"
    assert input_code_location.destination == "/opt/ml/processing/input/code/my_package/"
    assert input_code_location.input_name.startswith("input_code_location_")
    input_data_location = defaults.input_data_location
    assert input_data_location.source == "s3://inputdata/path"
    assert input_data_location.destination == "/opt/ml/processing/input/data/"
    assert input_data_location.input_name.startswith("input_data_location_")
    output_data_location = defaults.output_data_location
    assert output_data_location.source == "/opt/ml/processing/output/data/"
    assert output_data_location.destination == "s3://outputroot/path/05_model_input"
    assert output_data_location.output_name.startswith("output_data_location_")
    output_model_location = defaults.output_model_location
    assert output_model_location.source == "/opt/ml/processing/output/model/"
    assert output_model_location.destination == "s3://outputroot/path/06_models"
    assert output_model_location.output_name.startswith("output_model_location_")
    output_report_location = defaults.output_report_location
    assert output_report_location.source == "/opt/ml/processing/output/report/"
    assert output_report_location.destination == "s3://outputroot/path/08_reporting"
    assert output_report_location.output_name.startswith("output_report_location_")


def test_training_defaults():
    defaults = TrainingDefaults(
        trial_name="trial_test",
        code_path="/code/path",
        input_data_path="s3://inputdata/path",
        output_root_path="s3://outputroot/path",
    )
    assert defaults.base_job_name.startswith("trial_test")
    assert defaults.input_data_location.config["DataSource"]["S3DataSource"]["S3Uri"] == "s3://inputdata/path"
    assert defaults.training_output_path == "s3://outputroot/path/06_models"
    assert defaults.model_checkpoint_output_path == "s3://outputroot/path/06_models/checkpoints"


def test_evaluation_defaults():
    defaults = EvaluationDefaults(
        code_path="/code/path",
        input_data_path="s3://inputdata/path",
        output_root_path="s3://outputroot/path",
    )
    assert defaults.evaluation_script_path.as_posix() == "/code/path/evaluate.py"
    input_code_location = defaults.input_code_location
    assert Path(input_code_location.source).as_posix() == "/code/path"
    assert input_code_location.destination == "/opt/ml/processing/input/code/my_package/"
    assert input_code_location.input_name.startswith("input_code_location_")
    input_data_location = defaults.input_data_location
    assert input_data_location.source == "s3://inputdata/path"
    assert input_data_location.destination == "/opt/ml/processing/input/data/"
    assert input_data_location.input_name.startswith("input_data_location_")
    input_model_location = defaults.get_input_model_location("s3://model/path")
    assert input_model_location.source == "s3://model/path"
    assert input_model_location.destination == "/opt/ml/processing/input/model/"
    assert input_model_location.input_name.startswith("input_model_location_")
    output_data_location = defaults.output_data_location
    assert output_data_location.source == "/opt/ml/processing/output/data/"
    assert output_data_location.destination == "s3://outputroot/path/05_model_input"
    assert output_data_location.output_name.startswith("output_data_location_")
    output_model_location = defaults.output_model_location
    assert output_model_location.source == "/opt/ml/processing/output/model/"
    assert output_model_location.destination == "s3://outputroot/path/06_models"
    assert output_model_location.output_name.startswith("output_model_location_")
    output_report_location = defaults.output_report_location
    assert output_report_location.source == "/opt/ml/processing/output/report/"
    assert output_report_location.destination == "s3://outputroot/path/08_reporting"
    assert output_report_location.output_name.startswith("output_report_location_")
