import argparse
import time
from collections import defaultdict as dd
from datetime import timedelta
from pathlib import Path
from typing import List, Dict
from typing import Union

import numpy as np

from datasets.common import SplitDataset, Dataset
from datasets.all_datasets import get_all_dataset_names, select_datasets
from datasets.frozen_embeddings.cifar10 import (
    get_random_cifar_image_by_class,
    load_cifar_image, CIFAR10_CLASSES, map_filtered_to_global_index,
)
from datasets.frozen_embeddings.embeddings.vision import embed_images_vision_models, ImageListDataset
from datasets.frozen_embeddings.loader import EmbeddedDataset
from experiments.argument_parsing import extract_kwargs_from_args, parse_unknown_args, \
    get_function_param_names
from experiments.experimental_pipeline import DEFAULT_RESULTS, get_default_regularization
from experiments.image_augmentations import (
    augment_image_embedding,
)
from src.influence_functions import compute_model_influences
from datasets.load_datasets import load_dataset
from src.logistic_regression import LogisticRegression, RegularizationType
from src.logistic_regression import LogisticRegressionModel

from typing import Optional


def evaluate_counterfactual(
        original_model: LogisticRegressionModel,
        counterfactual_weights: np.ndarray,
        test_feature: np.ndarray,
        test_label: np.ndarray,
        reg_strength: float = 0.0,
        reg_type=None,
) -> dict:
    """Evaluate the difference between original and counterfactual models on a test sample."""
    delta_w = counterfactual_weights - original_model.weights

    # True gradient from original model (mean gradient, but only 1 sample so same)
    grad = original_model.get_gradient(test_feature, test_label, regularization=reg_strength,
                                       reg_type=reg_type).reshape(-1)

    # Change in loss: forward pass
    def sigmoid(z): return 1 / (1 + np.exp(-z))

    def loss_fn(w):  # Binary cross-entropy
        z = test_feature @ w
        p = sigmoid(z)
        label = test_label.item()
        return -label * np.log(p + 1e-8) - (1 - label) * np.log(1 - p + 1e-8)

    loss_orig = loss_fn(original_model.weights)
    loss_counterfactual = loss_fn(counterfactual_weights)

    # Compute metrics
    metrics = {
        "loss_change": float(loss_counterfactual - loss_orig),
        "first_order_approx": float(grad @ delta_w),
        "logit_change": float(test_feature @ delta_w),
    }
    return metrics


#
# def save_results(results: list, output_path: Union[str, Path]):
#     """
#     Save the results of poisoned sample experiments to a .npz file.
#     Each item in results is a dict for one test sample.
#     """
#     indices = []
#     features = []
#     labels = []
#     models = []
#     deltas = []
#     metrics = []
#
#     for res in results:
#         indices.append(res["test_index"])
#         features.append(res["feature"])
#         labels.append(res["label"])
#         models.append(res["poisoned_model_weights"])
#         deltas.append({
#             "IF": res["if_weights"],
#             "RIF": res["rif_weights"],
#             "original": res["original_weights"],
#         })
#         metrics.append(res["metrics"])  # Dict of dicts: {"IF": {...}, "RIF": {...}, "original": {...}}
#
#     np.savez_compressed(
#         output_path,
#         test_indices=np.array(indices),
#         features=np.array(features),
#         labels=np.array(labels),
#         poisoned_models=np.array(models),
#         counterfactual_weights=np.array(deltas, dtype=object),
#         metrics=np.array(metrics, dtype=object)
#     )


def split_into_words(text: str, vocabulary: list[str]) -> list[str]:
    """
    Efficiently splits a compound string into known words from a vocabulary using greedy longest-prefix matching.

    Parameters:
        text (str): Compound string (e.g., 'catdogfrog').
        vocabulary (list of str): List of allowed words.

    Returns:
        list of str: List of vocabulary words found in the string.

    Raises:
        ValueError: If text cannot be fully segmented using the vocabulary.
    """
    vocab = sorted(vocabulary, key=lambda w: -len(w))  # Sort by length descending for greedy matching
    tokens = []
    i = 0
    while i < len(text):
        match = None
        for word in vocab:
            if text.startswith(word, i):
                match = word
                break
        if match is None:
            raise ValueError(f"Cannot segment '{text}' at position {i}")
        tokens.append(match)
        i += len(match)
    return tokens


def save_results(results: dict, output_path: Union[str, Path]):
    """
    Save multi-experiment results to a .npz file.
    Each key in `results` is an experiment name, and each value is a list of result dicts.
    The .npz will contain one key per experiment name, storing the list of dicts as an object array.
    """
    structured = {
        exp_name: np.array(res_list, dtype=object)
        for exp_name, res_list in results.items()
    }

    np.savez_compressed(output_path, **structured)


def generate_augmented_embeddings(
        dataset_name: EmbeddedDataset,
        dataset: SplitDataset,
        index: int,
        base_regression: Optional[LogisticRegression] = None
) -> List[Dict]:
    """
    Given a sample index in an embedded dataset, generate embeddings of augmented versions of the original image.

    Returns a list of dictionaries with keys:
        - "embedding": np.ndarray
        - "augmentation": str
        - "augmentation_params": dict
        - "original_index": int
    """

    # === Step 1: Find corresponding raw CIFAR index
    classes = split_into_words("".join(dataset_name.name.value.split("_")[1:]).replace("auto", "automobile"),
                               vocabulary=CIFAR10_CLASSES)
    print(f"{classes=}")
    class1, class2 = classes
    excluded_classes = [class1, class2]
    raw_idx = int(dataset.test.original_indices[index])
    orig_emb = dataset.test.features[index]
    global_idx = map_filtered_to_global_index(raw_idx, [class1, class2])
    orig_img, orig_label = load_cifar_image(global_idx)

    # === Step 2: Identify class names
    # print(f"{dataset_name=}")
    # print(f"{dataset_name.name=}")
    # print(f"{dataset_name.name.value=}")
    # parts = dataset_name.name.value.split("_")[1:3]
    # print(f"{parts=}")

    # === Step 3: Get a donor image from a different class
    donor_img, donor_class, donor_idx = get_random_cifar_image_by_class(exclude=excluded_classes)

    high_freq_seed = np.random.randint(0, 2147483648)
    low_rank_seed = np.random.randint(0, 2147483648)
    if base_regression is None:
        H_inv = np.eye(513)
    else:
        H_inv = base_regression.compute_hessian_inv()
        H_inv = H_inv

    # === Step 4: Create augmentations
    embedding_augmentation = augment_image_embedding(
        model_name="resnet18",
        img=orig_img,
        Q=H_inv[:-1, :-1],  # identity Q
        l2_norm=0.001 * np.linalg.norm(orig_img.ravel()),
        mode="max"
    )

    delta_tensor = embedding_augmentation["delta_max"].detach().cpu()
    if delta_tensor.ndim == 4:
        delta_tensor = delta_tensor.squeeze(0)
    delta_img = delta_tensor.numpy()

    # Step 4: Apply perturbation to image and clip to valid range
    augmented_img_max = orig_img.astype(np.float32) + delta_img * 255.0  # rescale back
    augmented_img_max = np.clip(augmented_img_max, 0, 255).astype(np.uint8)

    augmentations = [
        ("identity", orig_img, {}),
        ("embedding_augmentation", augmented_img_max, {"augmented_image": augmented_img_max})
        # ("tennis_ball", add_tennis_ball(orig_img, portion=0.5), {"portion": 0.5}),
        # ("merge_top_right", merge_top_right_patch(orig_img, donor_img, 0.5), {"donor_index": donor_idx}),
        # ("average_with", average_with_image(orig_img, donor_img, 0.5), {"alpha": 0.5, "donor_index": donor_idx}),
        # ("high_freq_noise", add_high_freq_noise(orig_img, epsilon=250, seed=high_freq_seed),
        #  {"epsilon": 250, "seed": high_freq_seed}),
        # ("low_rank", add_low_rank_perturbation(orig_img, rank=5, scale=10.0, seed=low_rank_seed),
        #  {"rank": 5, "scale": 10.0, "seed": low_rank_seed}),
    ]

    # === Step 5: Apply transformation and embed
    augmented_imgs = [img for _, img, _ in augmentations]
    # stacked_tensor = torch.stack(transformed_imgs)

    # Note: embed_images_vision_models expects a dataset-like object; we mimic it using Torch tensors + dummy labels
    embeddings, _ = embed_images_vision_models(
        image_dataset=ImageListDataset(augmented_imgs),  # labels don't matter for embedding
        target_labels=[],
        model_name=dataset_name.model,
        embed_all_labels=True,
        max_samples=None,
        classes=CIFAR10_CLASSES,
        all_labels=None
    )
    embeddings = np.hstack((embeddings, np.ones((embeddings.shape[0], 1))))

    # === Step 6: Return list of metadata
    output = []
    for (aug_name, _, params), emb in zip(augmentations, embeddings):
        output.append({
            "embedding": emb,
            "augmentation": aug_name,
            "augmentation_params": {
                **params,
                "original_index": raw_idx,
                "donor_class": donor_class if "donor_index" in params else None
            },
            "original_index": raw_idx,
        })

    return output


def compute_poisoning_effects(
        base_regression: LogisticRegression,
        base_dataset: SplitDataset,
        poisoned_feature: np.ndarray,
        poisoned_label: np.ndarray,
        poisoned_metadata: dict,
        test_index: int,
        test_feature: np.ndarray,
        test_label: np.ndarray,
        regularization: float,
        verbosity: int = 1,
) -> dict:
    """
    Evaluate the effect of injecting one poisoned point into the training set.
    Returns a dict with new weights, IF/RIF, and evaluation metrics.
    """

    # === Prepare poisoned training data ===
    poisoned_X = np.vstack([base_dataset.train.features, poisoned_feature])
    poisoned_y = np.concatenate([base_dataset.train.labels, poisoned_label])

    poisoned_dataset = SplitDataset(
        train=Dataset(
            features=poisoned_X,
            labels=poisoned_y
        )
    )

    # === Fit poisoned model ===
    poisoned_regression = LogisticRegression(
        features=poisoned_X,
        labels=poisoned_y,
        regularization=regularization,
        reg_type=RegularizationType.L2
    )

    poisoned_regression.fit(verbose=(verbosity >= 3), warm_start=False)

    # === Influence function approximation ===
    influences = compute_model_influences(
        regression=poisoned_regression,
        experiment=poisoned_dataset,
        verbose=(verbosity >= 2),
        compute_gram_matrix=False
    )

    # === Counterfactual weights ===
    w_orig = base_regression.model.weights
    w_poisoned = poisoned_regression.model.weights
    w_if = w_poisoned + influences.influence_scores[-1]
    w_rif = w_poisoned + influences.rescaled_influence_scores[-1]

    # === Evaluate effect on test sample ===
    eval_orig = evaluate_counterfactual(poisoned_regression.model, w_orig, test_feature, test_label,
                                        regularization, RegularizationType.L2)
    eval_if = evaluate_counterfactual(poisoned_regression.model, w_if, test_feature, test_label,
                                      regularization, RegularizationType.L2)
    eval_rif = evaluate_counterfactual(poisoned_regression.model, w_rif, test_feature, test_label,
                                       regularization, RegularizationType.L2)

    if verbosity >= 2:
        print(f"Test idx {test_index}:")
        print(f"  original={eval_orig}")
        print(f"  IF={eval_if}")
        print(f"  RIF={eval_rif}")

    return {
        "test_index": test_index,
        "original_index": getattr(base_dataset.test, "original_indices", [None] * len(base_dataset.test.features))[
            test_index],
        "feature": test_feature,
        "label": test_label,
        "original_weights": w_orig,
        "poisoned_model_weights": w_poisoned,
        "if_weights": w_if,
        "rif_weights": w_rif,
        "poisoned_metadata": poisoned_metadata,
        "metrics": {
            "original": eval_orig,
            "IF": eval_if,
            "RIF": eval_rif,
        }
    }


def run_experiment_poisoned_regression(
        dataset_name: Union[str, Dataset, EmbeddedDataset],
        num_test: int = None,
        regularization: float = 0.1,
        seed: int = 42,
        verbosity: int = 2,
        max_train: int = None,
        max_test: int = None,
        output_dir: Path = DEFAULT_RESULTS,
        use_augmentations: bool = False,
        experiment_name: Optional[str] = None

):
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Original shapes: train={dataset.train.features.shape}, test={dataset.test.features.shape}")
    dataset = dataset.subsample(max_train=max_train, max_test=max_test, seed=seed)

    print("Fitting original logistic regression model...")
    base_regression = LogisticRegression(
        features=dataset.train.features,
        labels=dataset.train.labels,
        regularization=regularization,
        reg_type=RegularizationType.L2
    )
    base_regression.fit(verbose=(verbosity >= 1))

    X_test, y_test = dataset.test.features, dataset.test.labels
    num_test_samples = num_test or len(X_test)
    selected_indices = np.random.choice(len(X_test), size=num_test_samples, replace=False)

    results = dd(list)
    n, d = dataset.train.features.shape
    start_time = time.time()

    for i, idx in enumerate(selected_indices):
        x = X_test[idx:idx + 1]
        y = y_test[idx:idx + 1]
        flipped_y = 1 - y

        if verbosity >= 1:
            print(
                f"[{i + 1}/{num_test_samples}] Test sample idx={idx}, label={y[0]}, elapsed time: {timedelta(seconds=time.time() - start_time)}")

        # === 1. Label Flip Poisoning ===
        result = compute_poisoning_effects(
            base_regression=base_regression,
            base_dataset=dataset,
            poisoned_feature=x,
            poisoned_label=flipped_y,
            poisoned_metadata={"type": "label_flip"},
            test_index=idx,
            test_feature=x,
            test_label=y,
            regularization=regularization,
            verbosity=verbosity
        )
        results["label_flip"].append(result)

        if not use_augmentations:
            continue

        # === 2. Augmented Poisoning Variants ===
        if verbosity >= 1:
            print("Generating embeddings...", end=" ")
            t0 = time.time()
        augmentations = generate_augmented_embeddings(
            dataset_name=dataset_name,
            index=idx,
            dataset=dataset,
            base_regression=base_regression
        )
        if verbosity >= 1:
            print(f"Done. Took {time.time() - t0:.1f} seconds.")

        # First embedding is the identity augmentation, which is used to debug the embedding procedure.
        assert np.allclose(augmentations[0]["embedding"], x.ravel(), atol=1e-4), \
            f"Embeddings differ more than expected.\n" \
            f"Max abs diff: {np.max(np.abs(augmentations[0]['embedding'] - x.ravel()))}"

        for aug in augmentations[1:]:
            if verbosity >= 2:
                print("Augmentation:", aug.get("augmentation"))
            aug_embedding = aug["embedding"]  # embedded augmented image
            aug_label = np.array([flipped_y[0]])  # keep same label

            result = compute_poisoning_effects(
                base_regression=base_regression,
                base_dataset=dataset,
                poisoned_feature=aug_embedding[None, :],
                poisoned_label=aug_label,
                poisoned_metadata={"type": aug["augmentation"], **aug["augmentation_params"]},
                test_index=idx,
                test_feature=x,
                test_label=y,
                regularization=regularization,
                verbosity=verbosity
            )
            results[aug["augmentation"]].append(result)

    # Save
    # suffix = f"test{num_test_samples}" if num_test else "all_test"
    if experiment_name is None:
        experiment_name = f"{dataset_name}_poisoned_counterfactuals"
    output_file = output_dir / (experiment_name + ".npz")
    print(f"Saving results to {output_file}")
    save_results(dict(results), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run poisoned counterfactual experiments.")
    parser.add_argument("--dataset", type=str, help="Dataset name or pattern")
    parser.add_argument("--regex", action="store_true", help="Interpret dataset pattern as regex")
    parser.add_argument("--list", action="store_true", help="List all dataset names and exit")
    parser.add_argument("--num_test", type=int, default=None, help="Number of test samples to poison")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbosity", type=int, default=2, help="Verbosity level")
    parser.add_argument("--max_train", type=int, default=None, help="Maximum number of training samples to use")
    parser.add_argument("--max_test", type=int, default=None, help="Maximum number of test samples to use")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS, help="Directory to save results")
    parser.add_argument("--use_augmentations", action="store_true",
                        help="Compute data poisonings of augmented samples as well")
    # parser, added_args = generate_argparser_from_func(
    #     run_experiment_poisoned_regression,
    #     parser
    # )

    args, unknown_args = parser.parse_known_args()
    # Parse extra keyword arguments
    additional_params = get_function_param_names(run_experiment_poisoned_regression)
    extra_args = parse_unknown_args(unknown_args)
    kw_data = extract_kwargs_from_args(
        extra_args, additional_params
    )

    if args.list:
        print("Available datasets:")
        all_datasets = get_all_dataset_names()
        for name in all_datasets:
            print(" -", name)
        exit(0)

    if not args.dataset:
        parser.error("the following arguments are required: --dataset (unless using --list)")

    selected_datasets = select_datasets(args.dataset, args.regex)

    if not selected_datasets:
        print(f"No datasets matched pattern: {args.dataset}")
        exit(1)
    else:
        print(f"{extra_args=}, {additional_params=}, {selected_datasets=}")

    for dataset_name in selected_datasets:
        print("\n" + "=" * 80)
        print(f"Starting poisoned experiment for dataset: {dataset_name}")
        print("=" * 80)
        t0 = time.time()
        regularization, _ = get_default_regularization(dataset_name)
        # regularization = 1E-6
        print(f"{regularization=}")
        run_experiment_poisoned_regression(
            dataset_name=dataset_name,
            num_test=args.num_test,
            regularization=regularization,
            seed=args.seed,
            verbosity=args.verbosity,
            max_train=args.max_train,
            max_test=args.max_test,
            output_dir=DEFAULT_RESULTS,
            use_augmentations=args.use_augmentations,
            **kw_data
        )
        print(f"Runtime: {time.time() - t0:.1f} seconds")
