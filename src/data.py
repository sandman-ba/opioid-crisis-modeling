from typing import Self, Sequence, Literal
from pathlib import Path
from polars import LazyFrame, scan_parquet, read_parquet_schema, col, concat

VALID_FEATURES: list[str] = ["unemployment", "prescription_rate"]
VALID_INTERVENTIONS: list[str] = ["prescription_rate"]
VALID_PREDICTIONS: list[str] = ["opioid_related_mortality"]
DATA_PATH: Path = Path("data/all_data.parquet")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} does not exists, run create_dataset.py first")

type Feature = Literal["unemployment", "prescription_rate"]
type Intervention = Literal["prescription_rate"]
type Prediction = Literal["opioid_related_mortality"]


def validate_feature(feature: Feature) -> bool:
    return True if feature in VALID_FEATURES else False


def validate_features(features: Sequence[Feature]) -> None:
    invalid_features: list[str] = [
        feature for feature in features if validate_feature(feature) is False
    ]

    if len(invalid_features) > 0:
        raise ValueError(
            f"Features {(*invalid_features,)} not valid. Valid choices are:\n{(*VALID_FEATURES,)}"
        )


def validate_prediction(prediction: Prediction) -> None:
    if prediction not in VALID_PREDICTIONS:
        raise ValueError(
            f"Prediction target {prediction} not valid. Valid choices are:\n{(*VALID_PREDICTIONS,)}"
        )


def validate_intervention(intervention: Intervention) -> bool:
    return True if intervention in VALID_INTERVENTIONS else False


def validate_interventions(interventions: Sequence[Intervention]) -> None:
    invalid_interventions: list[str] = [
        intervention
        for intervention in interventions
        if validate_intervention(intervention) is False
    ]

    if len(invalid_interventions) > 0:
        raise ValueError(
            f"Interventions {(*invalid_interventions,)} not valid. Valid choices are:\n{(*VALID_INTERVENTIONS,)}"
        )


def get_schema() -> list[str]:
    return list(read_parquet_schema(DATA_PATH))


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
    lazy_frame = data.select(["id", "fips", "year"] + features + [prediction])
    return lazy_frame


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
    lazy_frame : polars.LazyFrame, lazy frame with data.
    fixed_factors : list[str], list of fixed_factors to be used for training a model.
    interventions : list[str], list of interventions to be simulated.
    prediction : str, what the model should predict, default will use the opioid
        related mortality rate.

    Methods
    -------
    get_training_years() -> polars.LazyFrame : Returns lazy frame with one column named "year" listing the years that have data available for training a model.

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
    >>> data.lazy_frame.head(5).collect()
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

    lazy_frame: LazyFrame
    fixed_factors: Sequence[Feature]
    interventions: Sequence[Intervention] | None
    prediction: Prediction

    def __init__(
        self: Self,
        fixed_factors: Sequence[Feature],
        interventions: Sequence[Intervention] | None = None,
        prediction: Prediction | None = None,
    ) -> None:
        validate_features(fixed_factors)
        self.fixed_factors = fixed_factors

        if prediction is None:
            self.prediction = "opioid_related_mortality"
        else:
            validate_prediction(prediction)
            self.prediction = prediction

        if interventions is not None:
            validate_interventions(interventions)

        self.interventions = interventions
        self.lazy_frame = get_data(fixed_factors, interventions, self.prediction)

    def __repr__(self: Self) -> str:
        return f"Data(fixed_factors={self.fixed_factors}, interventions={self.interventions}, prediction={self.prediction})"

    def __str__(self: Self) -> str:
        return f"Data object with attributes\nfixed_factors: {self.fixed_factors}\ninterventions: {self.interventions}\nprediction: {self.prediction}\nlazy_frame: {self.lazy_frame.head(5).collect()}\n\n"

    def get_training_years(self: Self) -> LazyFrame:
        return self.lazy_frame.select("year").unique().sort("year").head(-1)

    def get_fixed_factors(self: Self, training_year: int) -> LazyFrame:
        return (
            self.lazy_frame.filter(col("year") == training_year)
            .sort("fips")
            .select(self.fixed_factors)
        )

    def get_interventions(self: Self, training_year: int) -> LazyFrame | None:
        if self.interventions is None:
            return None
        return (
            self.lazy_frame.filter(col("year") == training_year)
            .sort("fips")
            .select(self.interventions)
        )

    def get_prediction(self: Self, training_year: int) -> LazyFrame:
        return (
            self.lazy_frame.filter(col("year") == training_year + 1)
            .sort("fips")
            .select(self.prediction)
        )

    def get_training_data(
        self: Self, training_year: int
    ) -> tuple[LazyFrame, LazyFrame]:
        features: LazyFrame
        if self.interventions is None:
            features = self.get_fixed_factors(training_year)
        else:
            features = concat(
                [
                    self.get_fixed_factors(training_year),
                    self.get_interventions(training_year),
                ],
                how="horizontal",
            )  # type: ignore[type-var, assignment]
        prediction: LazyFrame = self.get_prediction(training_year)
        return features, prediction
