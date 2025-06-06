from enum import Enum
from typing import List, Optional, Dict, Iterable, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np

from datasets.common import DatasetName
from datasets.frozen_embeddings.loader import EmbeddedDataset
from experiments.experiment import ExperimentResult


class EstimationMethod(Enum):
    INFLUENCE = "influence_function_estimate"
    RESCALED_INFLUENCE = "rescaled_influence_function_estimate"
    NEWTON_STEP = "newton_step_estimate"


ALL_STRATEGIES = [
    'Random Subsets', 'Cluster by Random Feature', 'Cluster by L2 Distance', "High Loss", 'High Positive Test-Loss Inf',
    'High Negative Test-Loss Inf', 'High Positive Test Inf', 'High Negative Test Inf'
]
ALL_ESTIMATION_METHODS = (EstimationMethod.INFLUENCE, EstimationMethod.RESCALED_INFLUENCE, EstimationMethod.NEWTON_STEP)
DATASET_ORDER = ["CDR", "Enron", "DogFish", "Diabetes", "MNIST"]

METRIC_LABELS = {
    "influence_on_test_predictions": "Effect on\ntest prediction",
    "influence_on_test_fixed_loss": "Effect on\ntest loss",
    "influence_on_test_total_loss": "Effect on\ntotal test loss",
    "influence_on_total_loss": "Effect on\nself-loss",
}
DEFAULT_METRICS = list(METRIC_LABELS.keys())
MARKERS = ['h', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'o']

def load_experiment_results(npz_file: str):
    print(f"Loading from {npz_file=}")
    data = np.load(npz_file, allow_pickle=True)
    results = {}
    for strategy_name in data.files:
        if strategy_name == "original_model":
            continue
        results[strategy_name] = [
            ExperimentResult.from_dict(result_dict.items())
            for result_dict in data[strategy_name]
        ]
    return results


from collections import OrderedDict
import os


def load_data_multiple_experiments(directory: str, selected_experiments: Optional[List[str]]) -> dict:
    """
    Load experiment results from .npz files, preserving the order of `selected_experiments`.

    Parameters:
    -----------
    directory : str
        Path to directory containing .npz experiment result files.
    selected_experiments : list of str, optional
        List of dataset base names (without .npz extension) to load, in desired display order.

    Returns:
    --------
    dict
        Mapping from filename base (e.g., 'DogFish') to loaded experiment data,
        ordered according to selected_experiments.
    """
    all_files = [f for f in os.listdir(directory) if f.endswith(".npz")]
    print(f"{all_files=}")

    if not selected_experiments:
        # Default: load all
        return {os.path.splitext(f)[0]: load_experiment_results(os.path.join(directory, f)) for f in all_files}

    selected_clean = [str(e).strip().lower() for e in selected_experiments]
    experiment_map = {os.path.splitext(f)[0].strip().lower(): f for f in all_files}

    ordered_results = OrderedDict()
    for name in selected_clean:
        if name not in experiment_map:
            raise ValueError(f"Dataset '{name}' not found in directory: {directory}")
        file_path = os.path.join(directory, experiment_map[name])
        ordered_results[name] = load_experiment_results(file_path)

    return ordered_results


def plot_metric_vs_prediction(
        ax,
        strategy: str,
        exact: np.ndarray,
        predictions: Dict[EstimationMethod, np.ndarray],
        estimate_methods: Iterable[EstimationMethod],
        colors: Dict[EstimationMethod, str],
        markers: List[str],
        strategy_idx: int,
        z_orders: Optional[Dict[EstimationMethod, int]] = None
):
    """
    Plot predicted vs actual values for a given strategy using preprocessed data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    strategy : str
        Name of the removal strategy (used for legend or internal tracking).
    exact : np.ndarray
        Ground truth values (x-axis).
    predictions : dict
        Dictionary of EstimationMethod → predicted values (same length as `exact`).
    estimate_methods : list of EstimationMethod
        The estimation methods to plot.
    colors : dict
        Mapping from estimation method to plot color.
    markers : list of str
        Marker styles to differentiate strategies.
    strategy_idx : int
        Index of the strategy (for marker cycling).
    z_orders : dict, optional
        Z-order control for estimation methods.
    """
    for method in estimate_methods:
        ax.scatter(
            exact,
            predictions[method],
            facecolors=colors[method],
            edgecolors=colors[method],
            marker=markers[strategy_idx % len(markers)],
            alpha=0.6,
            linewidths=1.2,
            s=60,
            label=f"{method.name}_{strategy_idx}",  # internal label
            zorder=z_orders[method] if z_orders and method in z_orders else 2
        )


def add_identity_line(ax, overlap_threshold=0.8, **kwargs):
    """
    Add a y = x line and set symmetric limits if x and y ranges overlap enough.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    overlap_threshold : float
        Minimum overlap ratio (0 to 1) of the x/y ranges (relative to the smaller range)
        required to enforce equal axis limits.
    kwargs : dict
        Additional line style options for the identity line.
    """
    kwargs.setdefault('linestyle', '--')
    kwargs.setdefault('color', 'gray')

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Compute ranges
    x_range = x1 - x0
    y_range = y1 - y0

    # Compute overlap
    overlap_min = max(x0, y0)
    overlap_max = min(x1, y1)
    overlap = max(0, overlap_max - overlap_min)

    max_range = max(x_range, y_range)
    overlap_ratio = overlap / max_range if max_range > 0 else 0

    # print(f"{(x0, x1)=}, {(y0, y1)=}, {overlap_min=}, {overlap_max=}, {overlap=}, {max_range=}, {overlap_ratio=}")

    # Print for debugging if needed
    # print(f"overlap: {overlap:.3f}, ratio: {overlap_ratio:.2f}")

    # Draw identity line
    lim_min = min(x0, y0)
    lim_max = max(x1, y1)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], **kwargs)

    # Sync limits only if overlap is strong
    if overlap_ratio >= overlap_threshold:
        ax.set_xlim((lim_min, lim_max))
        ax.set_ylim((lim_min, lim_max))
    else:
        ax.set_xlim((x0, x1))
        ax.set_ylim((y0, y1))


# def make_legend(ax, estimation_methods, strategies, colors, markers):
#     """
#     Create a legend that embeds into the given Axes object.
#
#     Parameters:
#     -----------
#     ax : matplotlib.axes.Axes
#         The axes to draw the legend into. Will be used solely for legend display.
#     estimation_methods : list
#         Estimation methods, used to define color-coded entries.
#     strategies : list
#         Strategies, used to define marker shape entries. If empty, legend is one-column.
#     colors : dict
#         Mapping from estimation method to color.
#     markers : list
#         List of marker styles.
#     """
#
#     ax.clear()
#
#     # Step 2: Add a solid white background to completely block any visuals
#     from matplotlib.patches import Rectangle
#     ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
#                            color='white', zorder=0))
#
#     ax.set_axis_off()  # Hide ticks/spines
#
#     # Step 3: Build legend handles
#     legend_items = []
#
#     for method in estimation_methods:
#         legend_items.append(
#             plt.Line2D([], [], color=colors[method], marker='o', linestyle='None', label=method.name)
#         )
#
#     for i, strategy in enumerate(strategies):
#         legend_items.append(
#             plt.Line2D([], [], color='gray', marker=markers[i % len(markers)],
#                        linestyle='None', label=strategy)
#         )
#
#     # Step 4: Draw legend (inside the same ax)
#     ncol = 2 if strategies else 1
#     legend = ax.legend(
#         handles=legend_items,
#         loc="center",
#         ncol=ncol,
#         fontsize=14,
#         title="Legend",
#         title_fontsize=16,
#         frameon=True,
#         fancybox=False,
#         edgecolor="black"
#     )
#
#     # ✅ Force the legend to be clipped to the axes bounding box
#     legend.set_clip_box(ax.get_window_extent())
#     legend.set_bbox_to_anchor((0.5, 0.5))  # Ensure it's centered
#     legend.set_transform(ax.transAxes)  # Stay in axes coords


def draw_full_axes_legend(ax, estimation_methods, strategies, colors, markers, preserve_title: bool = True):
    """
    Fill the given axes with a custom legend:
    - Left column: estimation methods (color-coded)
    - Right column: data removal strategies (marker shape)
    - Optionally preserves the axes title.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to fill with the legend.
    estimation_methods : list
        Estimation methods (e.g., INFLUENCE, RIF, etc.)
    strategies : list
        Strategy names to show in the right column.
    colors : dict
        Mapping of estimation method to color.
    markers : list
        Marker styles for strategies.
    preserve_title : bool, default=True
        If True, retains and redraws the title of this axes after clearing.
    """
    from matplotlib.patches import Rectangle
    import textwrap

    # Optionally preserve axes title
    current_title = ax.get_title() if preserve_title else None
    title_fontsize = ax.title.get_fontsize()

    ax.clear()
    ax.set_axis_off()

    # Background + black border
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor='white', edgecolor='white', linewidth=2, zorder=0))

    # Layout constants
    top_margin = 0.025
    bottom_margin_right = 0.0125
    bottom_margin_left = 0.2
    title_spacing_right = 0.09
    title_spacing_left = 0.25

    n_left = len(estimation_methods)
    n_right = len(strategies)

    # Column layout (independent height allocations)
    usable_height_left = 1 - top_margin - title_spacing_left - bottom_margin_left
    usable_height_right = 1 - top_margin - title_spacing_right - bottom_margin_right
    row_height_left = usable_height_left / max(n_left, 1)
    row_height_right = usable_height_right / max(n_right, 1)

    text_x1 = 0.15  # Left column (colors)
    text_x2 = 0.525  # Right column (shapes)

    # Titles (larger font, centered)
    # ax.text(0.5, 1 - top_margin, "Legend",
    #         fontsize=30, ha='center', va='top', transform=ax.transAxes)
    # column_title_fontsize = 20
    # ax.text(0.225, 1 - top_margin, "Data\nAttribution",
    #         fontsize=column_title_fontsize, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    # if n_right > 0:
    #     ax.text(0.725, 1 - top_margin, "Sample\nSelection",
    #             fontsize=column_title_fontsize, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    # Label mapping
    short_names = {
        "INFLUENCE": "IF",
        "RESCALED_INFLUENCE": "RIF",
        "NEWTON_STEP": "NS"
    }

    short_strategy_names = {
        "Random Subsets": "Random",
        "Cluster by Random Feature": "Cluster by Feature"
    }

    # Left column: IF, RIF, NS
    for i in range(n_left):
        y = 1 - top_margin - title_spacing_left - ((i + 0.5) * row_height_left)
        method = estimation_methods[i]
        label = short_names.get(method.name, method.name)
        ax.plot([text_x1 - 0.06], [y], marker='o', color=colors[method],
                markersize=12, linestyle='None', transform=ax.transAxes)
        ax.text(text_x1, y, label, fontsize=26, va='center', ha='left',
                transform=ax.transAxes)

    # Right column: strategies
    for i in range(n_right):
        y = 1 - top_margin - title_spacing_right - ((i + 0.5) * row_height_right)
        strategy = strategies[i]
        # strategy = short_strategy_names.get(strategy, strategy).replace("High ", "").replace(" Inf", "")
        strategy = short_strategy_names.get(strategy, strategy).replace("Inf", "Infl.")
        marker = markers[i % len(markers)]
        wrapped_label = textwrap.fill(strategy, width=16)
        ax.plot([text_x2 - 0.04], [y], marker=marker, color='black',
                markersize=12, linestyle='None', transform=ax.transAxes)
        ax.text(text_x2, y, wrapped_label, fontsize=16, va='center', ha='left',
                transform=ax.transAxes)

    if preserve_title and current_title:
        if preserve_title and current_title:
            ax.set_title(current_title, fontsize=title_fontsize)


def extract_strategy_metric_data(
        exp_results,
        metric_name,
        estimate_methods,
        use_linear=False,
        excluded_strategies=None
):
    """
    Converts a results dict into arrays of exact/predicted values for each strategy.
    Skips any strategies in `excluded_strategies`.

    Returns:
    --------
    dict[strategy_name] = {
        "exact": np.ndarray,
        EstimationMethod: np.ndarray,
        ...
    }
    """
    from collections import defaultdict
    import numpy as np

    linear = "_linear" if use_linear else ""
    strategy_data = {}
    excluded_set = set(excluded_strategies or [])

    for strategy, results in exp_results.items():
        if strategy in excluded_set:
            continue

        exact = []
        predictions = defaultdict(list)

        for result in results:
            if result.metrics and metric_name in result.metrics:
                m = result.metrics[metric_name]
                exact.append(getattr(m, "ground_truth" + linear))
                for method in estimate_methods:
                    predictions[method].append(getattr(m, method.value + linear))

        strategy_data[strategy] = {
            "exact": np.array(exact),
            **{method: np.array(predictions[method]) for method in estimate_methods}
        }
    return strategy_data


def get_outlier_mask(strategy_data, k=4, delta_fraction=0.1) -> Tuple[set, bool, bool]:
    # Combine all exact x values
    all_x = np.concatenate([v["exact"] for v in strategy_data.values()])
    x_sorted = np.sort(all_x)
    x_range = x_sorted[-1] - x_sorted[0]
    delta = delta_fraction * x_range

    # Greedy search: right
    right_indices = []
    for i in range(len(x_sorted) - 1, max(0, len(x_sorted) - 1 - k), -1):
        if x_sorted[i] - x_sorted[i - 1] >= delta:
            right_indices.append(i)

    # Greedy search: left
    left_indices = []
    for i in range(min(len(x_sorted) - 1, k)):
        if x_sorted[i + 1] - x_sorted[i] >= delta:
            left_indices.append(i)

    # If no outliers on either side, return empty
    if not right_indices and not left_indices:
        print("Would remove 0 x-values (no sufficient gaps found)")
        return set(), False, False

    # Determine x cutoff based on span removed
    right_cutoff = x_sorted[max(right_indices)] if right_indices else None
    left_cutoff = x_sorted[min(left_indices)] if left_indices else None

    right_span = x_sorted[-1] - right_cutoff if right_cutoff is not None else 0
    left_span = left_cutoff - x_sorted[0] if left_cutoff is not None else 0

    remove_right = False
    remove_left = False

    # Decide which side to remove from
    if right_cutoff is not None and right_span >= left_span:
        remove_set = set(all_x[all_x >= right_cutoff])
        remove_right = True
        print(f"Would remove {len(remove_set)} x-values (right tail, span: {right_span:.4f})")
    elif left_cutoff is not None:
        remove_set = set(all_x[all_x <= left_cutoff])
        remove_left = True
        print(f"Would remove {len(remove_set)} x-values (left tail, span: {left_span:.4f})")

    return remove_set, remove_left, remove_right


#
# def plot_single_estimation_method(
#         directory: str,
#         estimate_method: EstimationMethod = EstimationMethod.INFLUENCE,
#         selected_experiments: Optional[List[str]] = None,
#         selected_metrics: Optional[List[str]] = None,
#         selected_strategies: Optional[List[str]] = None,
#         save_path: Optional[str] = None
# ):
#     experiment_data = load_data_multiple_experiments(directory, selected_experiments)
#     metrics_to_plot = selected_metrics if selected_metrics else DEFAULT_METRICS
#     num_metrics, num_experiments = len(metrics_to_plot), len(experiment_data)
#
#     fig, axes = plt.subplots(num_metrics, num_experiments, figsize=(5 * num_experiments, 5 * num_metrics), dpi=400)
#     # if num_metrics == 1 and num_experiments == 1:
#     #     axes = np.array([[axes]])
#
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
#     colors = {estimate_method: 'blue'}
#
#     for r, metric in enumerate(metrics_to_plot):
#         for c, (exp_name, exp_results) in enumerate(experiment_data.items()):
#             ax = axes[r, c] if num_metrics > 1 and num_experiments > 1 else axes[max(r, c)]
#             ax.set_title(os.path.splitext(exp_name)[0], fontsize=22) if r == 0 else None
#             if r == num_metrics - 1:
#                 ax.set_xlabel("Actual Effect", fontsize=18)
#             if c == 0:
#                 ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=18)
#
#             for idx, (strategy, results) in enumerate(exp_results.items()):
#                 if selected_strategies and strategy not in selected_strategies:
#                     continue
#                 plot_metric_vs_prediction(ax, results, metric, [estimate_method], colors, markers, idx)
#
#             add_identity_line(ax)
#             ax.tick_params(axis='both', labelsize=12)
#
#     make_legend(axes[0, -1] if num_experiments > 1 else axes[-1], [estimate_method],
#                 list(experiment_data.values())[0].keys(), colors, markers)
#
#     plt.tight_layout(rect=[0, 0, 1, 1])
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()
#

def plot_multiple_methods(
        directory: str,
        estimate_methods: Iterable[EstimationMethod] = ALL_ESTIMATION_METHODS,
        dataset_names: Optional[List[Union[str, DatasetName, EmbeddedDataset]]] = None,
        dataset_titles: Optional[List[str]] = None,
        selected_metrics: Optional[List[str]] = None,
        selected_strategies: Optional[List[str]] = None,
        excluded_strategies: Optional[List[str]] = None,
        include_strategy_legend: bool = False,
        save_path: Optional[str] = None,
        use_linear: bool = False
):
    """
    Plot predicted vs. actual influence metrics across multiple datasets and methods.

    Parameters:
    -----------
    directory : str
        Path to directory containing experiment results.
    estimate_methods : Iterable[EstimationMethod], default=ALL_ESTIMATION_METHODS
        Estimation methods to include in the plot (e.g., IF, RIF, NS).
    dataset_names : list of str or DatasetName or EmbeddedDataset, optional
        Identifiers of datasets to include. If None, include all available datasets.
    dataset_titles : list of str, optional
        Custom subplot titles for the datasets. Must match length of `dataset_names`.
    selected_metrics : list of str, optional
        Influence metrics to plot (e.g., 'self_loss', 'test_prediction').
    selected_strategies : list of str, optional
        If provided, only these data removal strategies will be plotted.
    excluded_strategies : list of str, optional
        Strategies to skip in both the plot and legend.
    include_strategy_legend : bool, default=False
        Whether to include strategy names in the legend (alongside methods).
    save_path : str, optional
        If provided, save the figure to this file path (e.g., 'fig.pdf').
    use_linear : bool, default=False
        If True, use linear scale for axes (instead of default log scale).

    Notes:
    ------
    - The y = x reference line is always plotted on top (highest z-order).
    - Font sizes and tick density are adjusted for readability in publication contexts.
    """
    # Load experiment results into a nested dict: {dataset_name: {strategy: result_data}}
    experiment_data = load_data_multiple_experiments(directory, dataset_names)

    # Set up metrics to plot
    metrics_to_plot = selected_metrics if selected_metrics else DEFAULT_METRICS
    num_metrics, num_experiments = len(metrics_to_plot), len(experiment_data)

    # Create a grid of subplots sized proportionally to the number of datasets/metrics
    fig, axes = plt.subplots(num_metrics, num_experiments,
                             figsize=(5 * num_experiments, 5 * num_metrics),
                             dpi=300)
    # if num_metrics == 1 and num_experiments == 1:
    #     axes = np.array([[axes]])

    # Visual styles: marker shapes, colors, and plotting order (z-order)

    colors = {
        EstimationMethod.INFLUENCE: "g",
        EstimationMethod.RESCALED_INFLUENCE: "b",
        EstimationMethod.NEWTON_STEP: "c"
    }
    z_orders = {
        EstimationMethod.NEWTON_STEP: 1,
        EstimationMethod.INFLUENCE: 2,
        EstimationMethod.RESCALED_INFLUENCE: 3
    }

    # Iterate through rows (metrics) and columns (datasets)
    for r, metric in enumerate(metrics_to_plot):
        for c, (exp_name, exp_results) in enumerate(experiment_data.items()):
            ax = axes[r, c] if num_metrics > 1 and num_experiments > 1 else axes[max(r, c)]

            # Set title using user-defined dataset title or default to filename
            title = dataset_titles[c] if dataset_titles else os.path.splitext(exp_name)[0]
            if r == 0:
                ax.set_title(title, fontsize=28)
            # if r == num_metrics - 1:
            #     ax.set_xlabel("Actual Effect", fontsize=18)
            if c == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=28)

            strategy_data = extract_strategy_metric_data(
                exp_results, metric, estimate_methods, use_linear=use_linear, excluded_strategies=excluded_strategies
            )
            x_outliers, remove_left, remove_right = get_outlier_mask(strategy_data, k=3, delta_fraction=0.1)

            for idx, (strategy, data) in enumerate(strategy_data.items()):
                is_outlier = np.isin(data["exact"], list(x_outliers))
                is_kept = ~is_outlier

                # # Step 1: plot outliers (grayed out)
                # for method in estimate_methods:
                #     ax.scatter(
                #         data["exact"][is_outlier],
                #         data[method][is_outlier],
                #         facecolors='lightgray',
                #         edgecolors='gray',
                #         marker=markers[idx % len(markers)],
                #         alpha=0.3,
                #         linewidths=0.8,
                #         s=40,
                #         zorder=0,
                #     )
                # Step 2: plot regular (kept) points as usual
                plot_metric_vs_prediction(
                    ax=ax,
                    strategy=strategy,
                    exact=data["exact"][is_kept],
                    predictions={m: data[m][is_kept] for m in estimate_methods},
                    estimate_methods=estimate_methods,
                    colors=colors,
                    markers=MARKERS,
                    strategy_idx=idx,
                    z_orders=z_orders
                )
                if r == num_metrics - 1 and c == 2:
                    print(f"{ax.get_xlim()=}, {ax.get_ylim()=}")
            # Add text annotation once per subplot
            if remove_right:
                ax.text(
                    1.015, 0.5,
                    f"{len(x_outliers) * len(estimate_methods)} hidden",
                    transform=ax.transAxes,
                    color='red',
                    rotation='vertical',
                    fontsize=20,
                    va='center',
                    ha='left',
                    clip_on=False
                )
            elif remove_left:
                ax.text(
                    -0.01, 0.5,
                    f"{len(x_outliers) * len(estimate_methods)} hidden",
                    transform=ax.transAxes,
                    color='red',
                    rotation='vertical',
                    fontsize=20,
                    va='center',
                    ha='right',
                    clip_on=False
                )
            if r == num_metrics - 1 and c == 2:
                print(f"{ax.get_xlim()=}, {ax.get_ylim()=}")
            # Add y=x reference line on top of everything
            add_identity_line(ax, zorder=10, color='k', linestyle='-', linewidth=1.5)
            if r == num_metrics - 1 and c == 2:
                print(f"{ax.get_xlim()=}, {ax.get_ylim()=}")
            # Make x-axis ticks large and sparse
            ax.tick_params(axis='x', labelsize=20)  # Larger font size
            ax.locator_params(axis='x', nbins=4)  # Limit number of ticks

            # Remove y-axis ticks entirely
            ax.tick_params(axis='y', left=False, right=False, labelleft=False)

    # Add legend to top-right or bottom-most plot
    legend_ax = axes[0, -1]
    if excluded_strategies is None:
        excluded_strategies = []
    strategies_for_legend = [x for x in list(experiment_data.values())[-1].keys() if
                             x not in excluded_strategies] if include_strategy_legend else []
    print(f"{strategies_for_legend=}")
    draw_full_axes_legend(
        legend_ax,
        list(estimate_methods),
        strategies_for_legend,
        colors, MARKERS
    )

    # Final layout adjustment and saving
    # plt.tight_layout(rect=[0, 0, 1, 1])
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.075)  # ✅ shrink subplot area
    fig.text(0.5, 0.02, "Actual Effect", ha='center', va='center', fontsize=32)  # ✅ safe label position
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_reg_vs_sample_scaling(
        results_dir: str,
        regularization_dataset: str = "DogFish",
        num_samples_dataset: Union[str, EmbeddedDataset] = EmbeddedDataset(DatasetName.IMDB),
        influence_metric: str = "influence_on_test_predictions",
        estimate_methods: Iterable[EstimationMethod] = ALL_ESTIMATION_METHODS,
        use_linear: bool = False,
        save_path: Optional[str] = None,
):
    """
    Compare influence estimation across varying sample sizes and regularization strengths.

    Top row: sample scaling (e.g., IMDB, n = d, 2d, ..., 16d)
    Bottom row: regularization sweep (e.g., DogFish, lambda/n = 1e-4, ..., 1)

    Parameters:
    -----------
    results_dir : str
        Path to directory containing experiment `.npz` result files.
    regularization_dataset : str
        Name of dataset for regularization sweep.
    num_samples_dataset : str or EmbeddedDataset
        Dataset for sample size scaling.
    influence_metric : str
        Influence metric to visualize (e.g., 'test_prediction').
    estimate_methods : Iterable[EstimationMethod]
        Influence estimation methods to compare.
    use_linear : bool, default=False
        Use linear axes if True; otherwise log-log.
    save_path : str, optional
        File path to save the figure.
    """

    # Sample size experiment setup
    sample_suffixes = ["1d", "2d", "4d", "8d", "16d"]
    sample_titles = [r"$n=d$", r"$n=2d$", r"$n=4d$", r"$n=8d$", r"$n=16d$"]
    sample_exp_names = [f"{str(num_samples_dataset)}_{suffix}".lower() for suffix in sample_suffixes]

    # Regularization experiment setup
    reg_suffixes = ["1E_4", "1E_3", "1E_2", "1E_1", "1"]
    reg_titles = [
        r"$\lambda/n = 10^{-4}$", r"$\lambda/n = 10^{-3}$", r"$\lambda/n = 10^{-2}$",
        r"$\lambda/n = 10^{-1}$", r"$\lambda/n = 1$"
    ]
    reg_exp_names = [f"{regularization_dataset}_{suffix}".lower() for suffix in reg_suffixes]

    # Plot setup
    num_cols = max(len(sample_exp_names), len(reg_exp_names))
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), dpi=300)

    # Marker/color/zorder configs
    colors = {
        EstimationMethod.INFLUENCE: "g",
        EstimationMethod.RESCALED_INFLUENCE: "b",
        EstimationMethod.NEWTON_STEP: "c"
    }
    z_orders = {
        EstimationMethod.NEWTON_STEP: 1,
        EstimationMethod.INFLUENCE: 2,
        EstimationMethod.RESCALED_INFLUENCE: 3
    }

    def plot_row(ax_row, exp_data, exp_names, titles, row_title):
        renames = {
            "imdb": "IMDB"
        }
        row_title = renames.get(row_title, row_title)
        for col, (exp_name, title) in enumerate(zip(exp_names, titles)):
            ax = ax_row[col]
            if col == 0:
                ax.set_ylabel(row_title, fontsize=32)
            if exp_name not in exp_data:
                ax.set_title(f"Missing: {title}", fontsize=18)
                ax.axis("off")
                continue

            exp_results = exp_data[exp_name]
            strategy_data = extract_strategy_metric_data(
                exp_results, influence_metric,
                estimate_methods=estimate_methods,
                use_linear=use_linear
            )
            # x_outliers, remove_left, remove_right = get_outlier_mask(strategy_data, k=3, delta_fraction=0.1)

            for idx, (strategy, data) in enumerate(strategy_data.items()):
                # is_outlier = np.isin(data["exact"], list(x_outliers))
                # is_kept = ~is_outlier

                plot_metric_vs_prediction(
                    ax=ax,
                    strategy=strategy,
                    exact=data["exact"],
                    predictions={m: data[m] for m in estimate_methods},
                    estimate_methods=estimate_methods,
                    colors=colors,
                    markers=MARKERS,
                    strategy_idx=idx,
                    z_orders=z_orders
                )

            # Identity line and styling
            add_identity_line(ax, zorder=10, color='k', linestyle='-', linewidth=1.5)
            ax.set_title(title, fontsize=22)
            # ax.set_xlabel("Actual Effect", fontsize=18)
            ax.tick_params(axis='x', labelsize=16)
            ax.locator_params(axis='x', nbins=4)
            ax.tick_params(axis='y', left=False, labelleft=False)

    # Plot top (samples) and bottom (reg) rows
    # Load results
    exp_data = load_data_multiple_experiments(results_dir, sample_exp_names + reg_exp_names)
    print(f"{list(exp_data.keys())}")
    plot_row(axes[0], exp_data, sample_exp_names, sample_titles, row_title=str(num_samples_dataset))
    plot_row(axes[1], exp_data, reg_exp_names, reg_titles, row_title=str(regularization_dataset))

    # Legend and final layout
    # legend_ax = axes[0, -1]
    # draw_full_axes_legend(legend_ax, list(estimate_methods), ALL_STRATEGIES, colors, markers)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)

    fig.text(0.5, 0.02, "Actual Effect", ha='center', va='center', fontsize=32)

    if save_path:
        plt.savefig(save_path)
    plt.show()
