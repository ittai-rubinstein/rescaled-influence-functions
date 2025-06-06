from datasets.katl19.cdr_data import extract_cdr_dataset
from datasets.common import SplitDataset, DatasetName

from datasets.frozen_embeddings.loader import EmbeddedDataset, load_embedding_dataset

from typing import Union

from datasets.katl19.diabetes import load_diabetes
from datasets.katl19.dogfish import load_dogfish
from datasets.katl19.enron import load_enron


def load_dataset(
        dataset: Union[str, DatasetName, EmbeddedDataset],
        verbose: bool = False,
        **kwargs
) -> SplitDataset:
    """
    Load a dataset, supporting traditional and embedded formats.

    Parameters:
        dataset (Union[str, DatasetName, EmbeddedDataset]): Dataset identifier.
        verbose (bool): If True, print dataset details.
        **kwargs: Passed to loader (e.g., seed, train_size, etc.).

    Returns:
        SplitDataset: Train/test data.
    """

    # If dataset is EmbeddedDataset, load directly
    if isinstance(dataset, EmbeddedDataset):
        if verbose:
            print(f"Loading embedded dataset: {dataset.name.value} (model: {dataset.model})")
        return load_embedding_dataset(dataset, **kwargs)

    # If dataset is a DatasetName enum or str, normalize to string
    if isinstance(dataset, DatasetName):
        dataset_name = dataset.value
    elif isinstance(dataset, str):
        dataset_name = dataset
    else:
        raise TypeError("dataset must be a str, DatasetName, or EmbeddedDataset")

    if verbose:
        print(f"Loading dataset: {dataset_name}")

    # Handle traditional datasets
    if dataset_name == "DogFish":
        experiment = load_dogfish()
    elif dataset_name == "Enron":
        experiment = load_enron(**kwargs)
    elif dataset_name == "Diabetes":
        experiment = load_diabetes(**kwargs)
    elif dataset_name == "CDR":
        experiment = extract_cdr_dataset(**kwargs)
    else:
        # Assume it's an embedded dataset without a model specified
        try:
            name_enum = DatasetName(dataset_name.lower())
            experiment = load_embedding_dataset(EmbeddedDataset(name_enum), **kwargs)
        except ValueError:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    if verbose:
        print("Train set dimensions:")
        print(experiment.train.features.shape, experiment.train.labels.shape)

    return experiment
