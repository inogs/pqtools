import pytest

from pqtools.components import dataset_versions
from pqtools.components.basin_data import source_dataset_ids
from pqtools.components.data_retrievers import DATASETS
from pqtools.components.data_retrievers import DataSource
from pqtools.components.masks import ANALYSIS_DATASET
from pqtools.components.masks import REANALYSIS_DATASET
from pqtools.components.masks import SATELLITE_L4_DATASET


@pytest.fixture
def fake_versions(monkeypatch):
    versions = {"a": "202611", "b": "202211"}
    monkeypatch.setattr(
        dataset_versions, "get_dataset_version", versions.__getitem__
    )


@pytest.mark.usefixtures("fake_versions")
def test_describe_dataset_versions():
    assert (
        dataset_versions.describe_dataset_versions(["b", "a", "b"])
        == "b (version 202211); a (version 202611)"
    )


def test_source_dataset_ids():
    model_masks = [ANALYSIS_DATASET, REANALYSIS_DATASET]
    assert source_dataset_ids([DataSource.ANALYSIS]) == [
        DATASETS[DataSource.ANALYSIS],
        *model_masks,
    ]
    assert source_dataset_ids(
        [DataSource.REANALYSIS, DataSource.SATELLITE, DataSource.ANALYSIS]
    ) == [
        DATASETS[DataSource.REANALYSIS],
        *model_masks,
        DATASETS[DataSource.SATELLITE],
        SATELLITE_L4_DATASET,
        DATASETS[DataSource.ANALYSIS],
    ]
