import numpy as np


class RemovalSetGenerator:
    """
    Base class for generating sets of sample removals.
    """

    def __init__(self, dataset_size: int, num_sizes: int = 100, min_percent: float = 0.1, max_percent: float = 5.0):
        """
        Initialize the RemovalSetGenerator.

        Parameters:
            dataset_size (int): Total number of samples in the dataset.
            num_sizes (int): Number of removal sizes to consider.
            min_percent (float): Minimum size of removal set as a percentage of the dataset.
            max_percent (float): Maximum size of removal set as a percentage of the dataset.
        """
        self.name = ""
        self.dataset_size = dataset_size
        self.removal_sizes = np.linspace(
            min_percent / 100 * dataset_size,
            max_percent / 100 * dataset_size,
            num_sizes,
            dtype=int
        )

    def generate_removal_set(self, size: int, **kwargs) -> np.ndarray:
        """
        Generate a single removal set. To be implemented in subclasses.

        Parameters:
            size (int): The size of the removal set.

        Returns:
            np.ndarray: Array of integers representing indices of the removal set.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_all_removal_sets(self, **kwargs) -> list:
        """
        Generate removal sets for all specified sizes.

        Returns:
            list: A list of np.ndarray objects, each representing a removal set.
        """
        return [self.generate_removal_set(size, **kwargs) for size in self.removal_sizes]


class RandomRemovalSetGenerator(RemovalSetGenerator):
    """
    Generate removal sets by randomly sampling indices.
    """

    def __init__(self, dataset_size: int, **kwargs):
        super().__init__(dataset_size, **kwargs)
        self.name = "Random Subsets"

    def generate_removal_set(self, size: int) -> np.ndarray:
        return np.random.choice(self.dataset_size, size=size, replace=False)


class ClusterRemovalSetGenerator(RemovalSetGenerator):
    """
    Generate removal sets based on feature-based closeness.
    """

    def __init__(self, dataset_size: int, features: np.ndarray, subset_selection_method: str = "Random Feature",
                 **kwargs):
        """
        Initialize the FeatureBasedRemovalSetGenerator.

        Parameters:
            features (np.ndarray): Array of features with shape (n_samples, n_features).
        """
        super().__init__(dataset_size, **kwargs)
        self.features = features
        self.subset_selection_method = subset_selection_method
        self.name = "Cluster by " + self.subset_selection_method

    def generate_removal_set(self, size: int) -> np.ndarray:
        """
        Generate a removal set based on the specified method.

        Parameters:
            size (int): The size of the removal set.
            method (str): The method for selecting samples. Options:
                - "random_feature_closeness": Select k samples closest to a random sample in a random feature.
                - "random_point_l2": Select k samples closest to a random sample in L2 distance.

        Returns:
            np.ndarray: Array of indices representing the removal set.
        """
        if self.subset_selection_method == "Random Feature":
            # Pick a random feature and random sample
            random_feature = np.random.choice(self.features.shape[1])
            random_sample = np.random.choice(self.dataset_size)
            distances = np.abs(self.features[:, random_feature] - self.features[random_sample, random_feature])
        elif self.subset_selection_method == "L2 Distance":
            # Pick a random sample and compute L2 distances
            random_sample = np.random.choice(self.dataset_size)
            distances = np.linalg.norm(self.features - self.features[random_sample], axis=1)
        else:
            raise ValueError(f"Unknown method: {self.subset_selection_method}")

        # Get indices of the k smallest distances
        return np.argsort(distances)[:size]


class TopPercentileRemovalSetGenerator(RemovalSetGenerator):
    """
    Generate removal sets from the top percentage of samples based on a metric.
    """

    def __init__(self, dataset_size: int, metric: np.ndarray, metric_name: str = "Loss", **kwargs):
        """
        Initialize the TopPercentageRemovalSetGenerator.

        Parameters:
            metric (np.ndarray): Array of metric values (e.g., loss) for each sample.
        """
        super().__init__(dataset_size, **kwargs)
        self.metric = metric
        self.name = "High " + metric_name

    def generate_removal_set(self, size: int) -> np.ndarray:
        """
        Generate a removal set from the top percentage of samples based on the metric.

        Parameters:
            size (int): The size of the removal set.

        Returns:
            np.ndarray: Array of indices representing the removal set.
        """
        # Use partition to find the threshold value for the top-k values
        threshold_index = -2 * size  # Get the top 2*size (negative indexing for descending values)
        partitioned_indices = np.argpartition(-self.metric, kth=threshold_index)[:2 * size]

        # Filter the top 2*size indices and their metric values
        top_indices = partitioned_indices[np.argsort(-self.metric[partitioned_indices])[:size]]

        # Randomly select a group of the desired size from the top indices
        return np.random.choice(top_indices, size=size, replace=False)
