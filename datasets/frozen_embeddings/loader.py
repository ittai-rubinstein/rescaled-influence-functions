from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from datasets.common import Dataset, SplitDataset, DatasetName
from datasets.directories import DATASETS_PATH


@dataclass(frozen=True)
class EmbeddedDataset:
    name: DatasetName
    model: Optional[str] = None

    def filename(self) -> str:
        if self.model is not None:
            return f"{self.name.value}_{self.model}.npz"
        return f"{self.name.value}.npz"

    def file_path(self, base_dir: Path = DATASETS_PATH) -> Path:
        return base_dir / "embeddings" / self.filename()

    def __str__(self):
        if self.model is None:
            return str(self.name.value)
        else:
            return str(self.name.value) + "_" + self.model

    def __repr__(self):
        return str(self)


VISION_MODELS = ["resnet18", "resnet50", "efficientnet_b0", "vit_b_16", "swin_t"]

FLOWER_CSV = Path(__file__).parent / "uk_flowers" / "uk_flower_data.csv"


def load_embedding_dataset(
        dataset: EmbeddedDataset,
        train_size: float = 0.8,
        seed: int = 42,
        max_samples: Optional[int] = None,
) -> SplitDataset:
    file_path = dataset.file_path(DATASETS_PATH)
    data = np.load(file_path)

    X = data["features"]
    X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

    name = dataset.name

    if name == DatasetName.TAPE_FLUORESCENCE:
        y = data["labels"]
        median = np.median(y)
        y = (y >= median).astype(np.int8)


    elif name == DatasetName.OXFORD_FLOWERS:

        y_raw = data["labels"].astype(np.int64)  # True labels from Oxford dataset

        df = pd.read_csv(FLOWER_CSV)

        # Build mapping from Flowers102 Label to metadata (index-aligned to y_raw)

        label_map = df.set_index("Flowers102 Label").sort_index()

        # Validate counts

        label_counts = np.bincount(y_raw)

        expected_counts = label_map["Num Samples"].values

        print("Expected label distribution from CSV:", expected_counts)

        print("Actual label distribution in NPZ:", label_counts)

        if len(expected_counts) != len(label_counts) or not np.allclose(expected_counts, label_counts, atol=1):
            raise ValueError("Mismatch between label histogram and CSV metadata")

        # Extract flower type metadata for each class (0 = single, 1 = multiple)

        flower_type_per_label = label_map["Flower Type"].str.lower().map({"single": 0, "multiple": 1}).values

        # Convert per-sample flower type using label indices

        y = flower_type_per_label[y_raw]
    elif name in {DatasetName.CIFAR10_CAT_DOG, DatasetName.CIFAR10_DEER_HORSE, DatasetName.CIFAR10_AUTO_TRUCK}:
        y_raw = data["labels"].astype(np.int8)
        y = (y_raw >= np.mean(y_raw.astype(np.float64))).astype(np.int8)
    elif name == DatasetName.ESC50:
        y_raw = data["labels"].astype(np.int64)

        # Full sorted label list (must match label_encoder in generation)
        ESC50_LABELS = sorted([
            "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat", "chainsaw", "chirping_birds",
            "church_bells", "clapping", "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire", "crickets",
            "crow", "crying_baby", "dog", "door_wood_creaks", "door_wood_knock", "drinking_sipping", "engine",
            "fireworks",
            "footsteps", "frog", "glass_breaking", "hand_saw", "helicopter", "hen", "insects", "keyboard_typing",
            "laughing", "mouse_click", "pig", "pouring_water", "rain", "rooster", "sea_waves", "sheep", "siren",
            "sneezing",
            "snoring", "thunderstorm", "toilet_flush", "train", "vacuum_cleaner", "washing_machine", "water_drops",
            "wind"
        ])

        # Assert to catch potential encoder mismatch
        assert y_raw.max() < len(ESC50_LABELS), "Label index out of bounds for ESC-50"

        # Define artificial label set
        ARTIFICIAL_LABELS = {
            "airplane", "brushing_teeth", "can_opening", "car_horn", "chainsaw", "church_bells", "clapping",
            "clock_alarm",
            "clock_tick", "coughing", "crying_baby", "door_wood_creaks", "door_wood_knock", "drinking_sipping",
            "engine",
            "fireworks", "footsteps", "glass_breaking", "hand_saw", "helicopter", "keyboard_typing", "laughing",
            "mouse_click", "siren", "sneezing", "snoring", "toilet_flush", "train", "vacuum_cleaner", "washing_machine"
        }

        # Decode integer labels to strings
        label_names = [ESC50_LABELS[i] for i in y_raw]

        # Convert to boolean labels: True = artificial, False = natural
        y = np.array([label in ARTIFICIAL_LABELS for label in label_names], dtype=np.int8)
    else:
        y = data["labels"].astype(np.int8)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    if max_samples is not None:
        indices = indices[:max_samples]

    split_idx = int(len(indices) * train_size)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    train = Dataset(X[train_idx], y[train_idx], original_indices=train_idx)
    test = Dataset(X[test_idx], y[test_idx], original_indices=test_idx)

    return SplitDataset(train=train, test=test)
