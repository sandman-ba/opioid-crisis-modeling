from src.models import PredictionModel, ModelName, XGBoostConfig
from src.data import Data, Feature


def train_loop(data: Data, prediction_model: PredictionModel) -> None:
    training_years: list[int] = (
        data.get_training_years().collect().get_column("year").to_list()
    )
    for year in training_years:
        prediction_model.train(training_year=year)
        prediction_model.explain()
        prediction_model.risk()
        prediction_model.save()


def main() -> None:
    fixed_factors: list[Feature] = ["prescription_rate"]
    model_name: ModelName = "xgboost"
    model_config: XGBoostConfig = XGBoostConfig(eval_metric=["rmse", "mae"])
    data = Data(fixed_factors)
    prediction_model = PredictionModel(model_name, model_config)
    print(prediction_model)
    print(data)
    prediction_model.train(data, 2019)
    print(prediction_model.results.metrics)


if __name__ == "__main__":
    main()
