from src.models import PredictionModel, ModelName
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
    model_name: ModelName = "mlp"
    data = Data(fixed_factors)
    prediction_model = PredictionModel(model_name)
    print(prediction_model)
    print(data)


if __name__ == "__main__":
    main()
