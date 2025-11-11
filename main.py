### 11/09/25, EB: Runs the risk score modelling pipeline, and produces maps of the risk scores for each predicted year.

from src.model_training import yearly_mortality_prediction_polars
from src.data_processing import CountyDataLoader
from src.visualizations import plot_county_metric_maps, plot_yearly_feature_importances
from src.metrics import compute_all_risk_scores
from src.models.xgboost import xgb_model
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Run opioid risk modeling pipeline."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost"],#, "random_forest"],
        help="Which model to use for training."
    )

    parser.add_argument(
        "--plot",
        type=str,
        default="risk",
        choices=["risk", "features", "mortality"],
        help="Which plot to generate after training."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save plots (if not provided, plots are displayed interactively)."
    )

    return parser.parse_args()

MODEL_REGISTRY = {
    "xgboost": lambda: xgb_model
    # "random_forest": lambda: rf_model(),
}

def main():
    args = get_args()

    data = CountyDataLoader()
    df = data.load()

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[args.model]()  # dynamically pick model

    metrics, feature_importances, predictions, all_errors, save_dir = (
        yearly_mortality_prediction_polars(df, model, save_path=args.save_dir)
    )

    risk_scores = compute_all_risk_scores(predictions)

    PLOT_DISPATCH = {
        "risk": lambda: plot_county_metric_maps(risk_scores, "AbsError_Risk", save_dir=save_dir),
        "features": lambda: plot_yearly_feature_importances(feature_importances, save_dir=save_dir),
        "mortality": lambda: plot_county_metric_maps(df, "mortality_rate", save_dir=save_dir),
    }

    if args.plot not in PLOT_DISPATCH:
        raise ValueError(f"Unknown plot type: {args.plot}. "
                         f"Available: {list(PLOT_DISPATCH.keys())}")

    PLOT_DISPATCH[args.plot]()  # run selected plotting function


if __name__ == "__main__":
    main()