import os
import pandas as pd
import json
import logging

from configparser import ConfigParser

from src.data_processing.filters import *

logging.basicConfig()
logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)


def run_data_processing(dpc: str):
    # Parse data processing config
    config_object = ConfigParser()
    config_file_name = os.path.join("src", "data_processing", "configs", dpc)
    config_object.read(config_file_name)
    dpc_values = config_object["VALUES"]

    cycles = {}
    for process in ["train", "val", "test"]:
        cycles[process] = {
            "start": json.loads(dpc_values[f"{process}_cycles"])[0],
            "end": json.loads(dpc_values[f"{process}_cycles"])[1],
        }

    # Split data on tran, val, and test
    dfs = {"train": [], "test": [], "val": []}
    for file in sorted(
        os.listdir("data"),
        key=lambda x: (int(x[5]), int(x.split("Cycle")[-1][:-4])),
    ):
        path_to_file = os.path.join("data", file)
        df = pd.read_csv(path_to_file)
        cycle_num = int(file.split("Cycle")[-1][:-4])

        for process in ["train", "val", "test"]:
            if (
                cycle_num >= cycles[process]["start"]
                and cycle_num <= cycles[process]["end"]
            ):
                dfs[process].append(df)

    # Merge each process' data into one dataset
    for process in ["train", "val", "test"]:
        dfs[process] = pd.concat(dfs[process]).reset_index(drop=True)

    # Add filtered features if necessary
    for feature in dpc_values["features"]:
        if "_MA" in feature:
            logger.info(f"Applying Moving Average filter")
            for process in ["train", "val", "test"]:
                col_values_to_filter = dfs[process][feature.replace("_MA", "")]
                window_size = dpc_values["ma_window_size"][feature]
                dfs[process][feature] = get_ma(col_values_to_filter, window_size)
        if "_BWF" in feature:
            logger.info(f"Applying Butterworth filter")
            for process in ["train", "val", "test"]:
                col_values_to_filter = dfs[process][feature.replace("_MA", "")]
                cutoff_fs = dpc_values["bwf_cutoff_fs"][feature]
                order = dpc_values["bwf_order"][feature]
                dfs[process][feature] = get_bwf(col_values_to_filter, cutoff_fs, order)
        if "_H" in feature:
            logger.info(f"Applying Hampel filter")
            for process in ["train", "val", "test"]:
                col_values_to_filter = dfs[process][feature.replace("_MA", "")]
                window_size = dpc_values["h_window_size"][feature]
                n = dpc_values["h_n"][feature]
                dfs[process][feature] = get_bwf(col_values_to_filter, window_size, n)

    # Save processed data
    logger.info(f"Saving train, val, and test datasets")
    for process in ["train", "val", "test"]:
        dfs[process].to_csv(
            os.path.join(
                "src",
                "data_processing",
                "processed_data",
                f"{dpc[:-4]}_{process}.csv",
            ),
            index=False,
        )
