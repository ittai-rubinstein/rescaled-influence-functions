import argparse
import time
from typing import List

from datasets.all_datasets import get_all_dataset_names, select_datasets
from experiments.argument_parsing import parse_unknown_args, get_function_param_names, extract_kwargs_from_args
from experiments.experimental_pipeline import DEFAULT_RESULTS
from paper_plots.influence_plot_helper import plot_multiple_methods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot influence estimation comparisons across datasets.")

    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="List of dataset name patterns (exact or glob style). Each must match exactly one dataset."
    )
    parser.add_argument(
        "--titles", nargs="+", default=None,
        help="Optional list of custom titles, same length as --datasets."
    )
    parser.add_argument(
        "--exclude_strategies", nargs="+", default=None,
        help="Strategies to exclude from the plots (e.g., High_Loss)."
    )
    parser.add_argument(
        "--include_strategy_legend", action="store_true",
        help="If set, includes strategies in the plot legend."
    )
    parser.add_argument(
        "--output", type=str, default="influence_plot.pdf",
        help="Path to save the output plot."
    )
    parser.add_argument(
        "--use_linear", action="store_true",
        help="Use linear axis scaling instead of default log scale."
    )
    parser.add_argument(
        "--selected_metrics", nargs="+", default=None,
        help="List of influence metrics to plot (e.g., influence_on_test_loss)."
    )

    args, unknown_args = parser.parse_known_args()
    # Parse extra keyword arguments
    additional_params = get_function_param_names(plot_multiple_methods)
    extra_args = parse_unknown_args(unknown_args)
    kw_data = extract_kwargs_from_args(
        extra_args, additional_params
    )

    resolved_names: List[str] = []

    # Match each pattern to exactly one dataset
    for pattern in args.datasets:
        matches = select_datasets(pattern, regex=False)
        if len(matches) == 0:
            raise ValueError(f"No dataset matched pattern: {pattern}")
        elif len(matches) > 1:
            raise ValueError(f"Pattern '{pattern}' matched multiple datasets: {matches}")
        resolved_names.append(matches[0])

    # Validate titles if provided
    if args.titles and len(args.titles) != len(resolved_names):
        raise ValueError("Length of --titles must match number of resolved --datasets.")

    print("Resolved datasets:")
    for name in resolved_names:
        print(f" - {name}")
    print(f"Output will be saved to: {args.output}")

    t0 = time.time()
    plot_multiple_methods(
        directory=DEFAULT_RESULTS,
        dataset_names=resolved_names,
        dataset_titles=args.titles,
        selected_metrics=args.selected_metrics,
        excluded_strategies=args.exclude_strategies,
        include_strategy_legend=args.include_strategy_legend,
        save_path=args.output,
        use_linear=args.use_linear,
    )

    print(f"Finished in {time.time() - t0:.1f} seconds")
