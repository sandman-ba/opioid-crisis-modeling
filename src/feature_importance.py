from typing import Any
from dataclasses import dataclass
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from shap import TreeExplainer, KernelExplainer, Explanation, Cohorts
from shap.plots import bar as plot_bar
from shap.utils import hclust
from numpy import ndarray
from pandas import DataFrame
from matplotlib.pyplot import Axes

TREE_MODELS: list[str] = ["xgboost", "random_forest"]

type Explainer = TreeExplainer | KernelExplainer
type ShapValues = Explanation | Cohorts | dict[Any, Explanation]
type Model = XGBRegressor | RandomForestRegressor | MLPRegressor
type Data = ndarray | DataFrame
type TwoDArrayLike = Any
type OneDArrayLike = Any

### TODO: Move this elsewhere
@dataclass
class ModelContext:
    name: str
    model: Model

def get_shap_explainer(model_context: ModelContext) -> Explainer:
    match model_context.name:
        case tree if tree in TREE_MODELS:
            return TreeExplainer(model_context.model)
        ### TODO: Add data, could be all training data or summary with kmeans
        # case "mlp":
        #     return KernelExplainer(model_context.model.predict, data)
        case _:
            raise NotImplementedError(f"No feature importance available for {model_context.name} yet")

def get_shap_values(explainer: Explainer, X: Data) -> ndarray | list[ndarray]:
    return explainer.shap_values(X)

def get_dendogram_plot(shap_values: ShapValues, X: TwoDArrayLike, y: OneDArrayLike, ax: Axes) -> Axes:
    clust = hclust(X, y)
    return plot_bar(shap_values, ax=ax, clustering=clust, clustering_cutoff=1, show=False)
