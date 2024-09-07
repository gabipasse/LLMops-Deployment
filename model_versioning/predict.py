import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    PreTrainedModel,
)
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)
from transformers.models.distilbert.tokenization_distilbert_fast import (
    DistilBertTokenizerFast,
    PreTrainedTokenizerFast,
)
import torch
from torch import device
import os
from dotenv import load_dotenv


def predict(
    sample_texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    avaible_device: device,
) -> list[str]:
    """Generate predictions based on the inputs.

    Parameters
    ----------
    sample_texts: List[str]
        List of inputs to be inferenced from.
    model: PreTrainedModel
        Model which will be used to inference.
    tokenizer: PreTrainedTokenizerFast
        Tokenizer which will be used to tokenize the string input into tokenized indexed values.
    avaible_device: device
        Device (CPU or GPU) on which the model's predictions and tokenization will be performed.

    Return
    ------
    List[str]
        The list which contains the predicted labels based on the sample_texts input list.
    """
    inputs = tokenizer(
        sample_texts, padding=True, truncation=True, return_tensors="pt"
    ).to(avaible_device)
    model.eval()
    with torch.no_grad():
        outputs: SequenceClassifierOutput = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    predictions_cpu = predictions.cpu().detach().numpy()
    predictions_as_text_labels = [
        model.config.id2label[prediction] for prediction in predictions_cpu
    ]

    return predictions_as_text_labels


def main():
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    model_name = "agnews_pt_classifier"
    model_version = "1"
    model_uri = f"models:/{model_name}/{model_version}"
    model: PreTrainedModel = mlflow.pytorch.load_model(model_uri)
    avaible_device = device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(avaible_device)

    # Mimicking user inputs
    sample_texts = [
        "The local high school soccer team triumphed in the state championship, securing victory with a last-second winning goal.",
        "DataCore is set to acquire startup InnovativeAI for $2 billion, aiming to enchance its position in the artificial intelligence market.",
    ]
    # Attention because this tokenizer is the Fast version (implemented in Rust)
    tokenizer: DistilBertTokenizerFast = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    predictions_from_torch_model = predict(
        sample_texts, model, tokenizer, avaible_device
    )
    print()
    print(f"Predictions from base torch model: {predictions_from_torch_model}")
    print()
    custom_model_name = "agnews_transformer"
    custom_model_version = "1"
    custom_model_version_details = client.get_model_version(
        name=custom_model_name, version=custom_model_version
    )
    run_id = custom_model_version_details.run_id
    artifact_path_original: str = custom_model_version_details.source
    artifact_path_modified = artifact_path_original.replace("file:///", "")
    custom_model_local_path = "models/agnews_transformer"
    os.makedirs(custom_model_local_path, exist_ok=True)

    client.download_artifacts(
        run_id,
        artifact_path_modified,
        dst_path=custom_model_local_path,
    )

    custom_model: DistilBertForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(
            f"{custom_model_local_path}/custom_model"
        )
    )
    custom_tokenizer = AutoTokenizer.from_pretrained(
        f"{custom_model_local_path}/custom_model"
    )

    custom_model.to(device)
    predictions_from_custom_model = predict(
        sample_texts, custom_model, custom_tokenizer
    )
    print()
    print(f"Predictions from custom model: {predictions_from_custom_model}")
    print()

    mlflow.set_experiment("sequence_classification")
    with mlflow.start_run(run_name="iteration3") as run:
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.pytorch.log_model(model, "model")
        registered_model_details = mlflow.register_model(model_uri, model_name)

    model_versions = client.search_model_versions(f"name='{model_name}'")

    for version in model_versions:
        print(
            f"\nVersion {version.version}\nDescription {version.description}\nStatus {version.status}"
        )

    registered_model_actual_version = int(registered_model_details.version)
    registed_model_older_version = registered_model_actual_version - 1
    client.delete_model_version(name=model_name, version=registed_model_older_version)
    client.delete_registered_model(name=model_name)


if __name__ == "__main__":
    main()
