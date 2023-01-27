import json
import logging

from src.data_processing.filters import *

logging.basicConfig()
logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)


def fill_in_filters_parameters(parameters: dict, dpc_values: str, htc_values: str):
    if "h_window_size" in htc_values.keys():
        h_features = list(
            filter(
                lambda feature: "_H" in feature,
                json.loads(dpc_values["features"]),
            )
        )
        for h_feature in h_features:
            parameters[h_feature + "_window_size"] = {
                "values": json.loads(htc_values["h_window_size"])
            }
            parameters[h_feature + "_n"] = {"values": json.loads(htc_values["h_n"])}
    if "ma_window_size" in htc_values.keys():
        ma_features = list(
            filter(
                lambda feature: "_MA" in feature,
                json.loads(dpc_values["features"]),
            )
        )
        for ma_feature in ma_features:
            parameters[ma_feature + "_window_size"] = {
                "values": json.loads(htc_values["ma_window_size"])
            }
    if "bwf_cutoff_fs" in htc_values.keys():
        bwf_features = list(
            filter(
                lambda feature: "_BWF" in feature,
                json.loads(dpc_values["features"]),
            )
        )
        for bwf_feature in bwf_features:
            parameters[bwf_feature + "_cutoff_fs"] = {
                "values": json.loads(htc_values["bwf_cutoff_fs"])
            }
            parameters[bwf_feature + "_order"] = {
                "values": json.loads(htc_values["bwf_order"])
            }

    return parameters


def add_all_possible_filtered_features(df, dpc_values, htc_values):
    if "h_window_size" in htc_values.keys():
        h_features = list(
            filter(
                lambda feature: "_H" in feature,
                json.loads(dpc_values["features"]),
            )
        )
        for h_feature in h_features:
            for h_window_size in json.loads(htc_values["h_window_size"]):
                for h_n in json.loads(htc_values["h_n"]):
                    logger.info(
                        f"Applying Hampel filter with window size {h_window_size} and n {h_n} on {h_feature}"
                    )
                    df[f"{h_feature}_{h_window_size}_{h_n}"] = get_hampel(
                        df[h_feature], h_window_size, h_n
                    )
    if "ma_window_size" in htc_values.keys():
        ma_features = list(
            filter(
                lambda feature: "_MA" in feature,
                json.loads(dpc_values["features"]),
            )
        )
        for ma_feature in ma_features:
            for ma_window_size in json.loads(htc_values["ma_window_size"]):
                logger.info(
                    f"Applying Moving Average filter with window size {ma_window_size}on {ma_feature}"
                )
                df[f"{ma_feature}_{ma_window_size}"] = get_ma(
                    df[ma_feature], ma_window_size
                )
    if "bwf_cutoff_fs" in htc_values.keys():
        bwf_features = list(
            filter(
                lambda feature: "_BWF" in feature,
                json.loads(dpc_values["features"]),
            )
        )
        for bwf_feature in bwf_features:
            for bwf_cutoff_fs in json.loads(htc_values["bwf_cutoff_fs"]):
                for bwf_order in json.loads(htc_values["bwf_order"]):
                    logger.info(
                        f"Applying Butterworth filter with cutoff fs {bwf_cutoff_fs} and order {bwf_order} on {bwf_feature}"
                    )
                    df[f"{bwf_feature}_{bwf_cutoff_fs}_{bwf_order}"] = get_bwf(
                        df[bwf_feature], bwf_cutoff_fs, bwf_order
                    )

    return df
