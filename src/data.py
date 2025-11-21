from typing import Self, Sequence, Literal
from pathlib import Path
from polars import LazyFrame, scan_parquet, read_parquet_schema

VALID_FEATURES: list[str] = ["unemployment", "prescription_rate"]
VALID_INTERVENTIONS: list[str] = ["prescription_rate"]
VALID_PREDICTIONS: list[str] = ["opioid_related_mortality"]
DATA_PATH: Path = Path("data/all_data.parquet")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} does not exists, run create_dataset.py first")

type Feature = Literal["unemployment", "prescription_rate"]
type Intervention = Literal["prescription_rate"]
type Prediction = Literal["opioid_related_mortality"]

def get_schema() -> list[str]:
    """
    Helper function that returns the schema of the dataset
    """
    schema: list[str] = list(read_parquet_schema(DATA_PATH))
    return schema

def validate_feature(feature: Feature) -> bool:
    if feature in VALID_FEATURES:
        return True
    else:
        return False

def validate_features(features: Sequence[Feature]) -> None:
    """
    Helper function to check if requested features are valid
    """
    invalid_features: list[str] = [
        feature for feature in features if validate_feature(feature) is False
    ]

    if len(invalid_features) > 0:
        raise ValueError(
            f"Features {(*invalid_features,)} not valid. Valid choices are:\n{(*VALID_FEATURES,)}"
        )

def validate_prediction(prediction: Prediction) -> None:
    """
    Helper function to check if requested prediction is valid
    """
    if prediction not in VALID_PREDICTIONS:
        raise ValueError(
            f"Prediction target {prediction} not valid. Valid choices are:\n{(*VALID_PREDICTIONS,)}"
        )

def validate_intervention(intervention: Intervention) -> bool:
    if intervention in VALID_INTERVENTIONS:
        return True
    else:
        return False

def validate_interventions(interventions: Sequence[Intervention]) -> None:
    """
    Helper function to check if requested interventions are available
    """
    invalid_interventions: list[str] = [
        intervention for intervention in interventions if validate_intervention(intervention) is False
    ]

    if len(invalid_interventions) > 0:
        raise ValueError(
            f"Interventions {(*invalid_interventions,)} not valid. Valid choices are:\n{(*VALID_INTERVENTIONS,)}"
        )


class Data:
    """
    Class that loads data and provides context information for
    training, prediction and intervention strategies.

    Parameters
    ----------
    fixed_factors : list[str], list of fixed_factors to be used for training a model.
    interventions : list[str], list of interventions to be simulated.
    prediction : str, what the model should predict, default will use the opioid
        related mortality rate.

    Attributes
    ----------
    df : polars.LazyFrame, lazy frame with data.
    fixed_factors : list[str], list of fixed_factors to be used for training a model.
    interventions : list[str], list of interventions to be simulated.
    prediction : str, what the model should predict, default will use the opioid
        related mortality rate.

    Methods
    -------
    load_data() -> polars.LazyFrame : Loads dataset if not loaded yet and returns self.df.

    Examples
    --------
    Create Data object to train a model to predict opioid related mortality using
    unemployment as a fixed_factor and implement interventions that
    regulate prescription rates

    >>> fixed_factors = ["unemployment"]
    >>> interventions = ["prescription_rates"]
    >>> prediction = "opioid_related_mortality_rate"
    >>> data = Data(fixed_factors, interventions, prediction)
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
    fixed_factors: Sequence[Feature]
    interventions: Sequence[Intervention] | None
    prediction: Prediction

    def __init__(
        self: Self,
        fixed_factors: Sequence[Feature],
        interventions: Sequence[Intervention] | None,
        prediction: Prediction
    ) -> None:
        validate_prediction(prediction)
        validate_features(fixed_factors)
        if interventions is not None:
            validate_interventions(interventions)

        self.prediction = prediction
        self.fixed_factors = fixed_factors
        self.interventions = interventions

    def get_data(self: Self) -> LazyFrame:
        if self.df is None:
            data: LazyFrame = scan_parquet(DATA_PATH)
            features = list(self.fixed_factors) if self.interventions is None else list(self.fixed_factors) + list(self.interventions)
            self.df = data.select(
                ["fips", "year"] + features + [self.prediction]
            )
        return self.df
