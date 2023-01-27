import argparse

from src.data_processing.run import run_data_processing
from src.model_training.run import run_model_training
from src.inference.run import run_inference
from src.hyperparameter_tuning.run import run_hyperparameter_tuning

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
            "hyperparameter_tuning",
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
    parser.add_argument(
        "-htc",
        "--hyperparameter_tuning_config",
        help="What is the name of the hyperparameter tuning config file?",
    )
    parser.add_argument(
        "-wu",
        "--wandb_user",
        help="What is the name of the Weights&Biases user?",
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
    elif args.run == "hyperparameter_tuning":
        run_hyperparameter_tuning(
            dpc=args.data_processing_config,
            mtc=args.model_training_config,
            htc=args.hyperparameter_tuning_config,
            wandb_user=args.wandb_user,
        )
    else:
        raise Exception(
            "Error occurred while parsing command line arguments. Check for typos."
        )
