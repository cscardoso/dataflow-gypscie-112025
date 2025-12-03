import logging

import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree


def find_nearest_grid_point(
    df_radar: pd.DataFrame, df_grid: pd.DataFrame
) -> pd.DataFrame:
    tree = KDTree(df_grid[["latitude", "longitude"]].values)
    df_radar["coord"] = list(zip(df_radar["latitude"], df_radar["longitude"]))
    radar_lat_lon = np.array(df_radar["coord"].tolist())
    dist, idx = tree.query(radar_lat_lon)
    df_radar["nearest_latitude"] = df_grid.iloc[idx]["latitude"].values
    df_radar["nearest_longitude"] = df_grid.iloc[idx]["longitude"].values
    df_radar["nearest_point"] = list(
        zip(df_radar["nearest_latitude"], df_radar["nearest_longitude"])
    )
    df_radar.drop(columns=["nearest_latitude", "nearest_longitude"], inplace=True)
    return df_radar


def aggregate_radar_data(df_radar: pd.DataFrame) -> pd.DataFrame:
    df_grouped = (
        df_radar.groupby(["datetime", "nearest_point"])
        .agg({"horizontal_reflectivity_mean": "max"})
        .reset_index()
    )
    df_grouped[["latitude", "longitude"]] = df_grouped["nearest_point"].tolist()
    df_grouped = df_grouped.drop(columns=["nearest_point"])
    return df_grouped


def aggregate_rain_gauge_data(df_rain_gauge: pd.DataFrame) -> pd.DataFrame:
    df_rain_gauge["coord"] = list(
        zip(df_rain_gauge["latitude"], df_rain_gauge["longitude"])
    )
    df_rain_gauge["datetime"] = df_rain_gauge["datetime"].dt.ceil("h")
    df_rain_gauge = (
        df_rain_gauge.groupby(["coord", "datetime"])
        .agg({"precipitation": "sum"})
        .reset_index()
    )
    df_rain_gauge[["latitude", "longitude"]] = df_rain_gauge["coord"].tolist()
    return df_rain_gauge


def idw_interpolation_with_nearest_points(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    values_obs: np.ndarray,
    x_interp: np.ndarray,
    y_interp: np.ndarray,
    n_neighbors: int = 3,
    power: float = 2.0,
) -> np.ndarray:
    """
    Interpolation by IDW (Inverse Distance Weighting) with limit of number of nearest points.

    Parâmetros:
        x_obs (np.ndarray): x coordinates of observed points.
        y_obs (np.ndarray): y coordinates of observed points.
        values_obs (np.ndarray): Values observed in the points.
        x_interp (np.ndarray): x coordinates of points to be interpolated.
        y_interp (np.ndarray): y coordinates of points to be interpolated.
        n_neighbors (int, optional): Number of nearest points to be considered (default is 3).
        power (float, optional): Power for the weights calculation (default is 2.0).

    Returns:
        array: Interpolated values in the specified points.
    """
    tree = KDTree(np.column_stack((x_obs, y_obs)))
    distances, indices = tree.query(
        np.column_stack((x_interp, y_interp)), k=n_neighbors
    )
    values_interp = np.zeros(len(x_interp))
    for i in range(len(x_interp)):
        weights = 1 / distances[i] ** power
        values_interp[i] = np.sum(weights * values_obs[indices[i]]) / np.sum(weights)
    return values_interp


def apply_idw_interpolation(
    df_rain_gauge: pd.DataFrame, df_grid: pd.DataFrame
) -> pd.DataFrame:
    df_interpolated = pd.DataFrame()
    lat_grid = df_grid["latitude"].values
    lon_grid = df_grid["longitude"].values
    datetimes = df_rain_gauge["datetime"].unique()
    for datetime in datetimes:
        df_tmp = df_rain_gauge.loc[df_rain_gauge["datetime"] == datetime]
        lat_obs = df_tmp["latitude"].values
        lon_obs = df_tmp["longitude"].values
        precipitation_obs = df_tmp["precipitation"].values
        precipitation_interpolated = idw_interpolation_with_nearest_points(
            lat_obs, lon_obs, precipitation_obs, lat_grid, lon_grid, n_neighbors=2
        )
        df_tmp = pd.DataFrame(
            {
                "datetime": datetime,
                "latitude": lat_grid,
                "longitude": lon_grid,
                "precipitation": precipitation_interpolated,
            }
        )
        df_interpolated = pd.concat([df_interpolated, df_tmp])
    df_interpolated = df_interpolated.reset_index(drop=True)
    return df_interpolated


def merge_dataframes(
    df_radar: pd.DataFrame, df_rain_gauge: pd.DataFrame
) -> pd.DataFrame:
    df = df_radar.merge(
        df_rain_gauge, on=["datetime", "latitude", "longitude"], how="inner"
    )
    df = df.reset_index(drop=True)
    return df


def minmax_scale(data, data_min, data_max, feature_range=(0, 1)):
    min_, max_ = feature_range
    scale = (max_ - min_) / (data_max - data_min)
    min_adjusted = min_ - data_min * scale
    scaled_data = data * scale + min_adjusted
    return scaled_data


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: load scaler from file
    scaling_params = {
        "hour_sin": {"min": -1.0, "max": 1.0, "range": (0, 1)},
        "hour_cos": {"min": -1.0, "max": 1.0, "range": (0, 1)},
        "month_sin": {"min": -1.0, "max": 1.0, "range": (0, 1)},
        "month_cos": {"min": -1.0, "max": 1.0, "range": (0, 1)},
        "horizontal_reflectivity_mean": {"min": 0.0, "max": 157.09375, "range": (0, 1)},
        "precipitation": {"min": 0.0, "max": 81.770131684971, "range": (0, 1)},
    }
    for feature, params in scaling_params.items():
        if feature in df.columns:
            df[feature] = minmax_scale(
                df[feature], params["min"], params["max"], params["range"]
            )
    return df


def cyclic_datetime_encoding(df: pd.DataFrame) -> pd.DataFrame:
    s = df["datetime"].copy()
    s_time = s.dt.hour + s.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * s_time / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * s_time / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * s.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * s.dt.month / 12.0)
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        by=["datetime", "latitude", "longitude"], ascending=[True, False, True]
    )
    df = df.reset_index(drop=True)
    return df


def build_input_time_series(df: pd.DataFrame) -> np.ndarray:
    features = [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "horizontal_reflectivity_mean",
        "precipitation",
    ]
    num_channels = len(features)
    window_size = len(df["datetime"].unique())
    grid_height, grid_width = df["latitude"].nunique(), df["longitude"].nunique()
    X = np.zeros((1, num_channels, window_size, grid_height, grid_width))
    for i, feature in enumerate(features):
        feature_data = (
            df[feature].to_numpy().reshape(window_size, grid_height, grid_width)
        )
        X[0, i, :, :, :] = feature_data
    return X


# dataflow function
def load(radar_data_path: str, rain_gauge_data_path: str, grid_data_path: str):
    # TODO: get start and end datetime dynamically according to the current time
    datetime_start = pd.Timestamp("2023-01-12 19:00:00", tz="UTC")
    datetime_end = pd.Timestamp("2023-01-13 00:00:00", tz="UTC")
    shared_columns = ["datetime", "latitude", "longitude"]
    radar_columns = shared_columns + ["altitude", "horizontal_reflectivity_mean"]
    rain_gauge_columns = shared_columns + ["precipitation"]
    df_radar = pd.read_parquet(radar_data_path).query(
        "datetime >= @datetime_start & datetime <= @datetime_end"
    )[radar_columns]
    df_radar = df_radar.query("altitude < 2").drop(columns=["altitude"])
    df_rain_gauge = pd.read_parquet(rain_gauge_data_path).query(
        "datetime >= @datetime_start & datetime <= @datetime_end"
    )[rain_gauge_columns]
    df_grid = pd.read_parquet(grid_data_path)
    return df_radar, df_rain_gauge, df_grid


# dataflow function
def preprocessing(
    df_radar: pd.DataFrame, df_rain_gauge: pd.DataFrame, df_grid: pd.DataFrame
):
    df_grid["coord"] = list(zip(df_grid["latitude"], df_grid["longitude"]))
    df_radar = find_nearest_grid_point(df_radar, df_grid)
    df_radar = aggregate_radar_data(df_radar)
    df_rain_gauge = aggregate_rain_gauge_data(df_rain_gauge)
    df_rain_gauge = apply_idw_interpolation(df_rain_gauge, df_grid)
    df = merge_dataframes(df_radar, df_rain_gauge)
    assert not df.isna().any().any(), "There are NaN values in the dataframe."
    df = cyclic_datetime_encoding(df)
    df = normalize(df)
    df = sort_data(df)
    X = build_input_time_series(df)
    return X


def predict(X: np.ndarray, model):
    output = model(torch.from_numpy(X).float())
    output = output.detach().numpy()
    output_path = "output.npy"
    np.save(output_path, output)
    mlflow.log_artifact(output_path)
    return output


def load_model(model_path):
    model = torch.jit.load(model_path)
    model.eval()
    return model


def test_dataflow(
    radar_data_path: str,
    rain_gauge_data_path: str,
    grid_data_path: str,
    model_path: str,
):
    logger.info("Loading data")
    df_radar, df_rain_gauge, df_grid = load(
        radar_data_path, rain_gauge_data_path, grid_data_path
    )
    logger.info("Building input time series")
    X = preprocessing(df_radar, df_rain_gauge, df_grid)
    logger.info(f"X shape: {X.shape}")
    logger.info("Predicting")
    model = load_model(model_path)
    output = predict(X, model)
    logger.info(f"Output type: {type(output)}")
    logger.info(f"Output shape: {output.shape}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    test_dataflow(
        "data/radar.parquet",
        "data/rain_gauge.parquet",
        "data/grid_points.parquet",
        "data/convlstm_scripted.pt",
    )
