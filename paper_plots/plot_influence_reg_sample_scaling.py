import argparse
import time

from datasets.all_datasets import select_datasets
from experiments.argument_parsing import (
    parse_unknown_args,
    get_function_param_names,
    extract_kwargs_from_args
)
from experiments.experimental_pipeline import DEFAULT_RESULTS
from paper_plots.influence_plot_helper import plot_reg_vs_sample_scaling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot influence estimation vs. regularization and sample size.")

    parser.add_argument(
        "--regularization_dataset", required=True,
        help="Name of dataset used for regularization experiments (e.g., DogFish)."
    )
    parser.add_argument(
        "--sample_dataset", required=True,
        help="Name or pattern for the dataset used in sample scaling (e.g., IMDB)."
    )
    parser.add_argument(
        "--output", type=str, default="sample_vs_reg_plot.pdf",
        help="Path to save the output plot."
    )
    parser.add_argument(
        "--use_linear", action="store_true",
        help="Use linear axes instead of log-log."
    )
    parser.add_argument(
        "--metric", type=str, default="test_prediction",
        help="Influence metric to use (default: test_prediction)."
    )

    args, unknown_args = parser.parse_known_args()

    # Parse extra kwargs for plot_reg_vs_sample_scaling
    fn_param_names = get_function_param_names(plot_reg_vs_sample_scaling)
    extra_args = parse_unknown_args(unknown_args)
    plot_kwargs = extract_kwargs_from_args(extra_args, fn_param_names)

    # Match sample dataset name exactly
    sample_matches = select_datasets(args.sample_dataset, regex=False)
    if len(sample_matches) == 0:
        raise ValueError(f"No dataset matched sample pattern: {args.sample_dataset}")
    elif len(sample_matches) > 1:
        raise ValueError(f"Pattern '{args.sample_dataset}' matched multiple datasets: {sample_matches}")
    sample_dataset_clean = sample_matches[0]

    print(f"Regularization dataset: {args.regularization_dataset}")
    print(f"Sample dataset: {sample_dataset_clean.name}")
    print(f"Output will be saved to: {args.output}")

    t0 = time.time()
    plot_reg_vs_sample_scaling(
        results_dir=DEFAULT_RESULTS,
        regularization_dataset=args.regularization_dataset,
        num_samples_dataset=sample_dataset_clean,
        influence_metric=args.metric,
        use_linear=args.use_linear,
        save_path=args.output,
        **plot_kwargs,
    )
    print(f"Finished in {time.time() - t0:.1f} seconds")
