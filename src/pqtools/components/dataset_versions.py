from collections.abc import Iterable
from functools import cache

import copernicusmarine


@cache
def get_dataset_version(dataset_id: str) -> str:
    """Returns the version of a dataset used by pqtools.

    This is the version that `copernicusmarine` selects when no version is
    requested, i.e. the most recent one that has been released and has not
    been retired. pqtools passes it explicitly when it opens the dataset, so
    that the version recorded in the output files is exactly the one of the
    data.

    The result is cached: within the same Python process, a dataset is
    always read with the same version, even if a new one is released in the
    meantime.

    Raises:
        copernicusmarine.DatasetNotFound: If the dataset is not in the
            catalogue of the Copernicus Marine Service.
    """
    catalogue = copernicusmarine.describe(
        dataset_id=dataset_id,
        # Without this option, the catalogue may hide the version that
        # `copernicusmarine.open_dataset` would select
        show_all_versions=True,
        disable_progress_bar=True,
    )
    for product in catalogue.products:
        for dataset in product.datasets:
            if dataset.dataset_id == dataset_id:
                return dataset.get_version(None).label
    raise copernicusmarine.DatasetNotFound(dataset_id)


def describe_dataset_versions(dataset_ids: Iterable[str]) -> str:
    """Describes the datasets and their versions, for the output files.

    Returns:
        A string like ``"dataset_a (version 202611); dataset_b (version
        202511)"``, with the datasets in the given order (without
        duplicates).
    """
    return "; ".join(
        f"{dataset_id} (version {get_dataset_version(dataset_id)})"
        for dataset_id in dict.fromkeys(dataset_ids)
    )
