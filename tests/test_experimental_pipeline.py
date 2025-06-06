import unittest
from datasets.load_datasets import load_dataset


class TestExperimentalPipeline(unittest.TestCase):
    def test_load_dataset(self):
        # Expected dimensions for each dataset
        expected_dimensions = {
            "DogFish": (1800, 2048),
            "Enron": (4137, 5407),
            "CDR": (24177, 328),
            "Diabetes": (20000, 125),
        }

        for dataset_name, expected_dim in expected_dimensions.items():
            with self.subTest(dataset=dataset_name):
                # Load the dataset
                experiment = load_dataset(dataset_name)

                # Check dimensions of train features
                self.assertEqual(experiment.train.features.shape, expected_dim)


if __name__ == "__main__":
    unittest.main()
