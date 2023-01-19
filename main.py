import argparse

from src.data_processing.run import run_data_processing
from src.model_training.run import run_model_training
from src.inference.run import run_inference

if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser(description="Battery-SoC-Estimation")
    parser.add_argument(
        "-r",
        "--run",
        help="What application do you want to run?",
        choices=[
            "data_processing",
            "model_training",
            "inference",
        ],
        required=True,
    )
    parser.add_argument(
        "-dpc",
        "--data_processing_config",
        help="What is the name of the data processing config file?",
    )
    parser.add_argument(
        "-mtc",
        "--model_training_config",
        help="What is the name of the model training config file?",
    )
    parser.add_argument(
        "-ic",
        "--inference_config",
        help="What is the name of the inference config file?",
    )

    args = parser.parse_args()

    # Call specified process
    if args.run == "data_processing":
        run_data_processing(dpc=args.data_processing_config)
    elif args.run == "model_training":
        run_model_training(
            dpc=args.data_processing_config,
            mtc=args.model_training_config,
        )
    elif args.run == "inference":
        run_inference(
            dpc=args.data_processing_config,
            mtc=args.model_training_config,
        )
    else:
        raise Exception(
            "Error occurred while parsing command line arguments. Check for typos."
        )
