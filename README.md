# 🧪 `classy_bench`

-----

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Wiki](#wiki)
- [Authors](#authors)
- [Contributing and Support](#contributing-and-support)
- [License](#license)

## Description

`classy-bench` is a low-code Python library that simplifies the process of training and evaluating baseline models for real-world multi-label classification applications. Simply provide your datasets, and quickly get a benchmark of multiple models tailored to your specific use case.

Features and benefits:

- **Ready-to-use pipelines**: 7 built-in configurable pipelines (BM25, Class TF-IDF, Doc TF-IDF, Bi-Encoder, Cross-Encoder, RoBERTa and T5)
- **Customizable**: support for bring-your-own AWS Sagemaker pipeline code
- **Scalable**: run training pipeline on any type and any number of instances
- **Faster experimentation**: quickly understand which model performs best on your data
- **Low-code**: any member of the team can run the benchmark with confidence

This library was created as part of a research project _"The Right Model for the Job: An Evaluation of Legal Multi-Label Classification Baselines". (Forster, M., Schulz, C., Nokku, P., Mirsafian, M., Kasundra, J. and Skylaki, S.)_. See the [paper on arXiv](https://arxiv.org/abs/2401.11852) for more details.

## Installation

```console
pip install git+https://github.com/thomsonreuters/classy_bench.git
```

PyPI link coming soon!

## Usage

:warning: Access to AWS Sagemaker is required to use the library.

### Run the benchmark with the existing pipelines

#### Requirements

- A dataset that is split into 3 files (`train.csv`, `dev.csv` and `test.csv`) that contain train, validation and test sets respectively.
  Each file must have the following columns:
    - `id`: an identifier for each sample, e.g. a document id
    - `text`: the input text
    - `labels`: the labels list as a string (e.g. `"[LabelA, OtherLabel, LabelB]"`)
- Provide a `config.json` file that specifies which classifiers you want to run. In this config file, you can also set hyperparameters for training and evaluation.
  We recommend that you start by using the [default_config.json](https://github.com/thomsonreuters/classy_bench/blob/main/src/classy_bench/default_config.json) and adjust it as needed.
- Run the benchmark as shown in the [`notebooks/example.ipynb`](https://github.com/thomsonreuters/classy_bench/blob/main/notebooks/example.ipynb) notebook.

### Add your own pipelines to the benchmark

#### Requirements

- If you are planning to use any of the included pipelines, you must have a dataset split into 3 files (`train.csv`, `dev.csv` and `test.csv`) that contain train, validation and test sets respectively.
  Each file must have the following columns:
    - `id`: an identifier for each sample, e.g. a document id
    - `text`: the input text
    - `labels`: the labels list as a string (e.g. `"[LabelA, OtherLabel, LabelB]"`)

  If you are planning to only use custom pipelines, you can set your own rules. :)
- Provide a `config.json` file that specifies which classifiers you want to run. Please refer the [**Custom Pipeline**](https://github.com/thomsonreuters/classy_bench/wiki/Custom-Pipeline) page for an example on how to set up the config file. We recommend that you start by using the [default_config.json](https://github.com/thomsonreuters/classy_bench/blob/main/src/classy_bench/default_config.json) and adjust it as needed.
- Run the benchmark as shown in the [`notebooks/example.ipynb`](https://github.com/thomsonreuters/classy_bench/blob/main/notebooks/example.ipynb) notebook.

## Wiki

See the [Wiki](https://github.com/thomsonreuters/classy_bench/wiki) for more information on how to use the library.

## Authors

- Claudia Schulz
- Edoardo Abati
- Laura Skylaki
- Martina Forster
- Prudhvi Nokku
- Sammy Hannat

## Contributing and Support

Please note we are unable to promise immediate support and/or regular updates to this project at this stage.
However, if you need help or have any improvement ideas, please feel free to open an issue or a PR. We'd love to hear your feedback!

### Setup development environment

Requirements: [`hatch`](https://hatch.pypa.io/latest/install/), [`pre-commit`](https://pre-commit.com/#install)

1. Clone the repository
1. Run `hatch shell` to create and activate a virtual environment
1. Run `pre-commit install` to install the pre-commit hooks. This will force the linting and formatting checks.

### Run tests

- Linting and formatting checks: `hatch run lint:fmt`
- Unit tests: `hatch run test`

## License

`classy-bench` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
