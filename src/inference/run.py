import pandas as pd
import os
import json
import logging

from tensorflow.keras.models import load_model
from configparser import ConfigParser

from src.inference.analysis import get_stats

logging.basicConfig()
logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)

LABEL_FEATURE = "SOC [%]"
PREDICTED_FEATURE = "SOC_PREDICTED [%]"


def run_inference(dpc: str, mtc: str):
    # Parse data processing config
    config_object = ConfigParser()
    config_file_name = os.path.join("src", "data_processing", "configs", dpc)
    config_object.read(config_file_name)
    dpc_values = config_object["VALUES"]

    # Get data
    test_df = pd.read_csv(
        os.path.join(
            "src",
            "data_processing",
            "processed_data",
            f"{dpc[:-4]}_test.csv",
        )
    )

    # Get model
    model = load_model(
        os.path.join(
            "src",
            "model_training",
            "trained_models",
            f"{dpc[:-4]}_{mtc[:-4]}",
        )
    )

    # Get predictions
    logger.info(f"Running inference")
    features = json.loads(dpc_values["features"])
    X_test = test_df[features]
    test_df[PREDICTED_FEATURE] = model.predict(X_test).flatten()

    # Save results
    logger.info(f"Saving results")
    results_path = os.path.join("src", "inference", "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    test_df.to_csv(
        os.path.join(
            results_path,
            f"{dpc[:-4]}_{mtc[:-4]}.csv",
        )
    )

    # Calculate performance statistics
    stats = get_stats(test_df, LABEL_FEATURE, PREDICTED_FEATURE)
    logger.info(
        f"Mean Abs Error: {stats['mae']} %, Max Abs Error: {stats['max_abs']} %, Root Mean Sq Error {stats['rmse']} %"
    )
