from typing import Self
from pathlib import Path
from polars import LazyFrame, scan_parquet, read_parquet_schema

DATA_PATH: Path = Path("data/all_data.parquet")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH=} does not exists")

def get_available_features() -> list[str]:
    """
    Helper function that returns a list of available features
    """
    available_features: list[str] = list(read_parquet_schema(DATA_PATH))
    # TODO: Remove unrelated column names
    return available_features

def get_available_predictions() -> list[str]:
    """
    Helper function that returns a list of available predictions
    """
    available_predictions: list[str] = list(read_parquet_schema(DATA_PATH))
    # TODO: Remove unrelated column names
    return available_predictions

def get_available_interventions() -> list[str]:
    """
    Helper function that returns a list of available interventions
    """
    available_interventions: list[str] = list(read_parquet_schema(DATA_PATH))
    # TODO: Remove unrelated column names
    return available_interventions

def check_feature_availability(features: str | list[str]) -> None:
    """
    Helper function to check if requested features are available
    """
    if isinstance(features, str):
        features = [features]

    available_features = get_available_features()
    invalid_features: list[str] = [
        feature for feature in features if feature not in available_features
    ]

    if len(invalid_features) > 0:
        raise ValueError(
            f"Features {(*invalid_features,)} not available. Avaliable choices are:\n{(*available_features,)}"
        )

def check_prediction_availability(prediction: str) -> None:
    """
    Helper function to check if requested prediction is available
    """
    available_predictions = get_available_predictions()
    if prediction not in available_predictions:
        raise ValueError(
            f"Prediction target {prediction} not available. Avaliable choices are:\n{(*available_predictions,)}"
        )

def check_intervention_availability(interventions: str | list[str]) -> None:
    """
    Helper function to check if requested interventions are available
    """
    if isinstance(interventions, str):
        interventions = [interventions]

    available_interventions = get_available_interventions()
    invalid_interventions: list[str] = [
        intervention for intervention in interventions if intervention not in available_interventions
    ]

    if len(invalid_interventions) > 0:
        raise ValueError(
            f"Interventions {(*invalid_interventions,)} not available. Avaliable choices are:\n{(*available_interventions,)}"
        )


class Data:
    """
    Class that loads data and provides context information for
    training, prediction and intervention strategies.

    Parameters
    ----------
    features : list[str], list of features to be used for training a model.
    interventions : list[str], list of interventions to be simulated.
    prediction : str, what the model should predict, default will use the opioid
        related mortality rate.

    Attributes
    ----------
    df : polars.LazyFrame, lazy frame with data.
    features : list[str], list of features to be used for training a model.
    interventions : list[str], list of interventions to be simulated.
    prediction : str, what the model should predict, default will use the opioid
        related mortality rate.

    Methods
    -------
    load_data() -> polars.LazyFrame : Loads dataset if not loaded yet and returns self.df.

    Examples
    --------
    Create Data object to train a model to predict opioid related mortality using
    unemployment and prescription rates as features and implement interventions that
    regulate prescription rates

    >>> features = ["unemployment", "prescription_rates"]
    >>> interventions = ["prescription_rates"]
    >>> prediction = "opioid_related_mortality_rate"
    >>> data = Data(features, interventions, prediction)
    >>> data.df.collect()
    shape: (X, X)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘
    """

    df: LazyFrame | None
    features: list[str]
    interventions: list[str] | None
    prediction: str

    def __init__(
        self: Self,
        features: list[str],
        interventions: list[str] | None = None,
        prediction: str | None = None,
    ) -> None:
        if prediction is None:
            self.prediction = "opioid_related_mortality_rate"
        else:
            check_prediction_availability(prediction)
            self.prediction = prediction

        check_feature_availability(features)
        self.features = features

        if interventions is not None:
            check_intervention_availability(interventions)

        self.interventions = interventions

    def get_data(self: Self) -> LazyFrame:
        if self.df is None:
            data: LazyFrame = scan_parquet(DATA_PATH)
            self.df = data.select(
                ["fips", "year"] + self.features + [self.prediction]
            )
        return self.df
