from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Self
from numpy.typing import NDArray, ArrayLike
from polars import LazyFrame
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from shap import TreeExplainer, KernelExplainer, Explanation, Cohorts
from src.data import Data


type Regressor = XGBRegressor | RandomForestRegressor | MLPRegressor
type Explainer = TreeExplainer | KernelExplainer
type ShapValues = Explanation | Cohorts | dict[Any, Explanation]
type ModelName = Literal["xgboost", "random_forest", "mlp"]

VALID_MODELS: list[ModelName] = ["xgboost", "random_forest", "mlp"]


@dataclass
class Results:
    metrics: dict | None = None
    predictions: NDArray | None = None
    risk_scores: dict | None = None
    shap_values: ShapValues | None = None


@dataclass
class XGBoostConfig:
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    objective: str = "reg:squarederror"
    n_jobs: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    eval_metric: str | list[str] | None = "rmse"


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = (
        "squared_error"
    )
    max_depth: int = 15
    min_samples_split: int | float = 4
    min_samples_leaf: int | float = 2
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = -1
    random_state: int = 42
    warm_start: bool = False


@dataclass
class MLPConfig:
    hidden_layer_sizes: ArrayLike = (128, 64, 32)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    solver: Literal["lbfgs", "sgd", "adam"] = "adam"
    alpha: float = 0.0001
    batch_size: int = 128
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "adaptive"
    learning_rate_init: float = 0.001
    max_iter: int = 500
    random_state: int = 42
    verbose: bool = False
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20


type ModelConfig = XGBoostConfig | RandomForestConfig | MLPConfig


def get_default_config(model_name: ModelName) -> ModelConfig:
    match model_name:
        case "xgboost":
            return XGBoostConfig()
        case "random_forest":
            return RandomForestConfig()
        case "mlp":
            return MLPConfig()
        case _:
            raise ValueError(
                f"Model not valid. Please choose from:\n{(*VALID_MODELS,)}"
            )


def initialize_model(model_name: ModelName, config: ModelConfig) -> Regressor:
    match model_name:
        case "xgboost":
            return XGBRegressor(**asdict(config))
        case "random_forest":
            return RandomForestRegressor(**asdict(config))
        case "mlp":
            return MLPRegressor(**asdict(config))
        case _:
            raise ValueError(
                f"Model not valid. Please choose from:\n{(*VALID_MODELS,)}"
            )


class PredictionModel:
    name: ModelName
    model: Regressor
    explainer: Explainer | None
    results: Results | None

    def __init__(
        self: Self, name: ModelName, config: ModelConfig | None = None
    ) -> None:
        self.name = name
        if config is None:
            config = get_default_config(name)
        self.model = initialize_model(name, config)
        self.results = Results()
        self.explainer = None

    def __str__(self: Self) -> str:
        return f"PredictionModel object:\n{self.name=}\n{self.model=}\n{self.explainer=}\n{self.results=}\n\n"

    def train(
        self: Self,
        data: Data,
        training_year: int,
        pretrained_model: Path | Regressor | None = None,
    ) -> None:
        X: LazyFrame
        y: LazyFrame
        X, y = data.get_training_data(training_year)
        match self.name:
            case "xgboost":
                self.model.fit(X, y, xgb_model=pretrained_model, eval_set=[(X,y)], verbose=False)
                self.results.metrics = self.model.evals_result()['validation_0']
            case _:
                raise NotImplementedError(
                    f"Training for model {self.name} not yet implemented"
                )

    def save(self: Self, save_path: Path) -> None:
        file_name: Path = Path("trained_model.ubj")
        match self.name:
            case "xgboost":
                self.model.save_model(save_path / file_name)
            case _:
                raise NotImplementedError(
                    f"Saving for model {self.name} not yet implemented"
                )

    def explain(self: Self) -> None:
        raise NotImplementedError("Method not yet implemented")

    def risk(self: Self) -> None:
        # self.results.predictions = self.model.predict()
        raise NotImplementedError("Method not yet implemented")
