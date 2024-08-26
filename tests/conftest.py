from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def output_data_location(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fn = tmp_path_factory.mktemp("output", numbered=True)
    return fn
