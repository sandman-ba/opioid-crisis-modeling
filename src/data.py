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
        intervention
        for intervention in interventions
        if validate_intervention(intervention) is False
    ]

    if len(invalid_interventions) > 0:
        raise ValueError(
            f"Interventions {(*invalid_interventions,)} not valid. Valid choices are:\n{(*VALID_INTERVENTIONS,)}"
        )


def get_data(
    fixed_factors: Sequence[Feature],
    interventions: Sequence[Intervention] | None,
    prediction: Prediction,
) -> LazyFrame:
    data: LazyFrame = scan_parquet(DATA_PATH)
    features = (
        list(fixed_factors)
        if interventions is None
        else list(fixed_factors) + list(interventions)
    )
    df = data.select(["id", "fips", "year"] + features + [prediction])
    return df


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
    >>> data.get_data()
    >>> data.df.head(5).collect()
    shape: (5, 5)
    ┌──────────┬──────┬──────┬───────────────────┬──────────────────────────┐
    │ id       ┆ fips ┆ year ┆ prescription_rate ┆ opioid_related_mortality │
    │ ---      ┆ ---  ┆ ---  ┆ ---               ┆ ---                      │
    │ u32      ┆ u16  ┆ u16  ┆ f32               ┆ f32                      │
    ╞══════════╪══════╪══════╪═══════════════════╪══════════════════════════╡
    │ 51172014 ┆ 5117 ┆ 2014 ┆ 91.300003         ┆ 0.0                      │
    │ 51192014 ┆ 5119 ┆ 2014 ┆ 117.5             ┆ 13.75                    │
    │ 51232014 ┆ 5123 ┆ 2014 ┆ 96.400002         ┆ 0.0                      │
    │ 51332014 ┆ 5133 ┆ 2014 ┆ 82.900002         ┆ 0.0                      │
    │ 51492014 ┆ 5149 ┆ 2014 ┆ 72.300003         ┆ 8.41                     │
    └──────────┴──────┴──────┴───────────────────┴──────────────────────────┘
    """

    df: LazyFrame
    fixed_factors: Sequence[Feature]
    interventions: Sequence[Intervention] | None
    prediction: Prediction

    def __init__(
        self: Self,
        fixed_factors: Sequence[Feature],
        interventions: Sequence[Intervention] | None,
        prediction: Prediction,
    ) -> None:
        validate_prediction(prediction)
        validate_features(fixed_factors)
        if interventions is not None:
            validate_interventions(interventions)

        self.prediction = prediction
        self.fixed_factors = fixed_factors
        self.interventions = interventions
        self.df = get_data(fixed_factors, interventions, prediction)

    def __repr__(self: Self) -> str:
        return f"Data(fixed_factors={self.fixed_factors}, interventions={self.interventions}, prediction={self.prediction})"

    def __str__(self: Self) -> str:
        return f"Data object with attributes\nfixed_factors: {self.fixed_factors}\ninterventions: {self.interventions}\nprediction: {self.prediction}\ndf: {self.df.head(5).collect()}\n\n"
