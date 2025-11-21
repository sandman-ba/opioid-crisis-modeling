from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence, Self
from numpy.typing import NDArray, ArrayLike
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from shap import TreeExplainer, KernelExplainer, Explanation, Cohorts
from src.data import Data, Feature, Intervention, Prediction

type SKLearnModel = XGBRegressor | RandomForestRegressor | MLPRegressor
type Explainer = TreeExplainer | KernelExplainer
type ShapValues = Explanation | Cohorts | dict[Any, Explanation]
type ModelName = Literal["xgboost", "random_forest", "mlp"]
VALID_MODELS: list[str] = ["xgboost", "random_forest", "mlp"]


@dataclass
class Results:
    metrics: dict
    predictions: NDArray
    risk_scores: dict
    shap_values: ShapValues


@dataclass
class TaskContext:
    fixed_factors: Sequence[Feature]  # TODO: Find better name
    interventions: Sequence[Intervention] | None = None
    prediction: Prediction = "opioid_related_mortality"
    model_name: ModelName = "xgboost"


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
            raise ValueError(f"Model not valid. Please choose from:\n{(*VALID_MODELS,)}")

def get_pretrained_model(model_name: ModelName, config: ModelConfig) -> SKLearnModel:
    match model_name:
        case "xgboost":
            return XGBRegressor(**asdict(config))
        case "random_forest":
            return RandomForestRegressor(**asdict(config))
        case "mlp":
            return MLPRegressor(**asdict(config))
        case _:
            raise ValueError(f"Model not valid. Please choose from:\n{(*VALID_MODELS,)}")


class PredictionModel:
    data: Data
    name: ModelName
    model: SKLearnModel | None
    explainer: Explainer | None
    results: Results | None

    def __init__(self: Self, context: TaskContext, config: ModelConfig | None) -> None:
        self.data = Data(
            fixed_factors=context.fixed_factors,
            interventions=context.interventions,
            prediction=context.prediction,
        )
        self.name = context.model_name
        if config is None:
            config = get_default_config(context.model_name)
        self.model = get_pretrained_model(context.model_name, config)
        self.explainer = None
        self.results = None
