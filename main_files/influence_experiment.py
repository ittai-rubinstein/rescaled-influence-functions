import argparse
import os
import sys
import traceback

from datasets.all_datasets import get_all_dataset_names, select_datasets
from experiments.argument_parsing import extract_kwargs_from_args, parse_unknown_args, get_function_param_names
from experiments.experimental_pipeline import ExperimentalPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for selected datasets.", add_help=True)
    parser.add_argument("--dataset", type=str, help="Dataset name or regex pattern to select which experiment to run")
    parser.add_argument("--regex", action="store_true",
                        help="Interpret dataset pattern as a full regular expression instead of a glob")
    parser.add_argument("--list", action="store_true", help="List all dataset names and exit")

    # parser, added_args = generate_argparser_from_func(
    #     ExperimentalPipeline,
    #     parser
    # )
    # Parse known and unknown args
    additional_params = get_function_param_names(ExperimentalPipeline)
    args, unknown_args = parser.parse_known_args()
    # Parse extra keyword arguments
    extra_args = parse_unknown_args(unknown_args)

    # Filter for valid ExperimentalPipeline arguments
    kw_args = extract_kwargs_from_args(
        extra_args, additional_params
    )

    if args.list:
        all_datasets = get_all_dataset_names()
        print("Available datasets:")
        for name in all_datasets:
            print(" -", name)
        sys.exit(0)

    if not args.dataset:
        parser.error("the following arguments are required: --dataset (unless using --list)")

    selected_datasets = select_datasets(args.dataset, args.regex)
    if not selected_datasets:
        print(f"No datasets matched pattern: {args.dataset}")
        sys.exit(1)

    print(f"{extra_args=}, {additional_params=}, {kw_args=}, {selected_datasets=}")

    for dataset_name in selected_datasets:
        print("\n" + "=" * 80)
        print(f"Starting experiment for dataset: {dataset_name}")
        print("=" * 80)

        dataloader_flags = {"load_legacy": True} if dataset_name == "Enron" else {}

        try:
            experiment = ExperimentalPipeline(
                dataset_name=dataset_name,
                dataloader_flags=dataloader_flags,
                storage_allowance_mb=1024 if os.getenv("DATA_DIRECTORY") else 25,
                **kw_args
            )
            experiment.run()
        except KeyboardInterrupt:
            print("\nExperiment interrupted. Exiting.")
            raise
        except Exception as e:
            print(f"\nAn error occurred while running the experiment for {dataset_name}: {e}")
            traceback.print_exc()
            continue
