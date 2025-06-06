import argparse
import fnmatch
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets.frozen_embeddings.cifar10 import map_filtered_to_global_index, load_cifar_image
from experiments.experimental_pipeline import DEFAULT_RESULTS


#
# def plot_logit_scatter_from_structured_file(result_file: Path, save_dir: Path, verbosity: int = 1):
#     data = np.load(result_file, allow_pickle=True)
#
#     markers = ['o', 's', 'D', '^', 'v', 'P', 'X']  # Enough for multiple augmentations
#     augmentations = list(data.keys())
#     marker_cycle = iter(markers)
#
#     # === Plot for label_flip ===
#     if "label_flip" in data:
#         metrics_list = data["label_flip"]
#         gt_logit_changes = [m["metrics"]["original"]["logit_change"] for m in metrics_list]
#         if_logit_changes = [m["metrics"]["IF"]["logit_change"] for m in metrics_list]
#         rif_logit_changes = [m["metrics"]["RIF"]["logit_change"] for m in metrics_list]
#
#         plt.figure(figsize=(8, 6))
#         plt.scatter(gt_logit_changes, if_logit_changes, label="Influence Functions", color="green", marker="*", s=40)
#         plt.scatter(gt_logit_changes, rif_logit_changes, label="Rescaled Influence Functions", color="blue", marker="*",
#                     s=40)
#         plt.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
#         plt.xlabel("Ground Truth Logit Change")
#         plt.ylabel("Estimated Logit Change")
#         plt.title("Logit Change: label_flip")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#
#         save_path = save_dir / (result_file.stem + "_label_flip_logit_scatter.png")
#         plt.savefig(save_path)
#         plt.close()
#
#         if verbosity >= 1:
#             print(f"Saved label_flip plot to: {save_path}")
#
#     # === Plot for all other augmentations ===
#     plt.figure(figsize=(10, 8))
#     for aug in augmentations:
#         if aug == "label_flip":
#             continue
#
#         try:
#             metrics_list = data[aug]
#             gt_logit_changes = [m["metrics"]["original"]["logit_change"] for m in metrics_list]
#             if_logit_changes = [m["metrics"]["IF"]["logit_change"] for m in metrics_list]
#             rif_logit_changes = [m["metrics"]["RIF"]["logit_change"] for m in metrics_list]
#
#             marker = next(marker_cycle)
#
#             plt.scatter(gt_logit_changes, if_logit_changes, label=f"{aug} - IF", color="green", marker=marker,
#                         alpha=0.7, s=40)
#             plt.scatter(gt_logit_changes, rif_logit_changes, label=f"{aug} - RIF", color="blue", marker=marker,
#                         alpha=0.7, s=40)
#         except Exception as e:
#             if verbosity >= 1:
#                 print(f"Skipping {aug} due to error: {e}")
#
#     plt.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
#     plt.xlabel("Ground Truth Logit Change")
#     plt.ylabel("Estimated Logit Change")
#     plt.title("Logit Change: All Augmentations")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     save_path = save_dir / (result_file.stem + "_augmentations_logit_scatter.png")
#     plt.savefig(save_path)
#     plt.close()
#
#     if verbosity >= 1:
#         print(f"Saved combined augmentations plot to: {save_path}")


def plot_logit_scatter_label_flip_only(result_file: Path, save_dir: Path, verbosity: int = 1):
    """
    Plots a scatter comparing actual effect vs IF/RIF estimates for 'label_flip' only.
    """
    data = np.load(result_file, allow_pickle=True)

    if "label_flip" not in data:
        raise ValueError(f"No 'label_flip' data found in result file{result_file}.")

    metrics_list = data["label_flip"]
    gt_logit_changes = [m["metrics"]["original"]["logit_change"] for m in metrics_list]
    if_logit_changes = [m["metrics"]["IF"]["logit_change"] for m in metrics_list]
    rif_logit_changes = [m["metrics"]["RIF"]["logit_change"] for m in metrics_list]

    plt.figure(figsize=(14, 11))
    plt.scatter(gt_logit_changes, if_logit_changes, label="IF", color="green", marker="*", s=350)
    plt.scatter(gt_logit_changes, rif_logit_changes, label="RIF", color="blue", marker="*", s=350)

    plt.axline((0, 0), slope=1, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Actual Effect", fontsize=32)
    # plt.ylabel("Influence", fontsize=32)
    plt.tick_params(axis='both', labelsize=24)
    plt.locator_params(axis='both', nbins=7)
    plt.legend(fontsize=26)
    plt.grid(True)
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()

    FORMATS = ["pdf", "png", "svg"]
    for fmt in FORMATS:
        save_path = save_dir / (result_file.stem + f"_data_poisoning_plot.{fmt}")
        plt.savefig(save_path)
    plt.close()

    if verbosity >= 1:
        print(f"Saved label_flip plot to: {save_path}")

        # === After closing the scatter plot ===

    # Find left-most data point (lowest x: actual effect on test logit)
    min_idx = np.argmin(gt_logit_changes)
    sample = metrics_list[min_idx]

    original_index = sample.get("original_index", None)
    print(f"Raw original index: {original_index}")

    if True:
        class_subset = ["automobile", "truck"]
        global_idx = map_filtered_to_global_index(original_index, class_subset)
        raw_img, label = load_cifar_image(global_idx)  # raw_img is 32x32x3 numpy array

        fig_width = 7
        fig_height = 7.4
        plt.figure(figsize=(fig_width, fig_height))

        # Plot image in square covering bottom 6/7 of the figure
        image_size_frac = 6 / 7
        bottom = 0
        height = image_size_frac
        left = (1 - image_size_frac) / 2  # center image
        width = image_size_frac

        ax_img = plt.axes([left, bottom, width, height])
        ax_img.imshow(raw_img)
        ax_img.axis('off')

        # Decide poisoned label: whichever class it isn't
        # true_class = class_subset[label]
        poisoned_class = [c for c in class_subset if c != label][0]

        plt.suptitle(f"Poisoned Label: {poisoned_class}", y=0.95, fontsize=28)

        for fmt in FORMATS:
            save_path = save_dir / (result_file.stem + f"_sample.{fmt}")
            plt.savefig(save_path, bbox_inches="tight")

        plt.close()

        if verbosity >= 1:
            print(f"Saved poisoned image plot to: {save_path}")

    # except Exception as e:
    #     if verbosity >= 1:
    #         print(f"Error generating image plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot IF vs RIF logit change estimates")
    parser.add_argument("--dataset", required=True, help="Dataset pattern (e.g., 'cifar*dog*18')")
    parser.add_argument("--output_name", required=False, default=None, help="Optional output PNG name")
    parser.add_argument("--verbosity", required=False, type=int, default=1, help="Verbosity level")

    args = parser.parse_args()

    # Find matching result file
    pattern = re.compile(fnmatch.translate(args.dataset), re.IGNORECASE)
    matched_files = [f for f in DEFAULT_RESULTS.glob("*_poisoned_counterfactuals.npz") if pattern.match(str(f.name))]

    if not matched_files:
        print(f"No result files matched pattern: {args.dataset}")
        return

    result_file = matched_files[0]
    print(f"Found result file: {result_file}")

    # Determine output name
    output_name = args.output_name or (result_file.stem + "_logit_scatter.png")
    save_path = DEFAULT_RESULTS / output_name

    # Plot
    plot_logit_scatter_label_flip_only(result_file, DEFAULT_RESULTS, verbosity=args.verbosity)


if __name__ == "__main__":
    main()
