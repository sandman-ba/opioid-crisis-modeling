from src.models import PredictionModel, TaskContext

def main() -> None:
    context = TaskContext(fixed_factors=["prescription_rate"], model_name="mlp")
    prediction_model = PredictionModel(context)
    print(prediction_model)
    print(prediction_model.data)


if __name__ == "__main__":
    main()
