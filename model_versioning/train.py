import mlflow
import mlflow.experiments
import torch
from datasets import load_dataset
from datasets.formatting.formatting import LazyBatch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
from typing import Union
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from dotenv import load_dotenv
from transformers.modeling_outputs import SequenceClassifierOutput
from numpy import float64

load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
params = {
    "model_name": "distilbert-base-uncased",
    "learning_rate": 5e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "dataset_name": "ag_news",
    "task_name": "sequence_classification",
    "log_steps": 100,
    "max_seq_length": 128,
    "output_dir": "models/distilbert-base-uncased-ag_news-sequence_classification",
}

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(params["task_name"])
run = mlflow.start_run(run_name=f"{params['model_name']}-{params['dataset_name']}")
mlflow.log_params(params)

dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset] = (
    load_dataset(params["dataset_name"])
)
tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(
    params["model_name"],
)


def tokenize(batch: LazyBatch) -> BatchEncoding:
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=params["max_seq_length"],
    )


# TODO MODIFICAR PARA 20k e 2k
train_dataset = dataset["train"].shuffle().select(range(20)).map(tokenize, batched=True)
test_dataset = dataset["test"].shuffle().select(range(20)).map(tokenize, batched=True)

train_dataset.to_parquet("data/train.parquet")
test_dataset.to_parquet("data/test.parquet")

mlflow.log_artifact("data/train.parquet", "datasets")
mlflow.log_artifact("data/test.parquet", "datasets")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)
labels = dataset["train"].features["label"].names

model: PreTrainedModel = DistilBertForSequenceClassification.from_pretrained(
    params["model_name"], num_labels=len(labels)
)

model.config.id2label = {i: label for i, label in enumerate(labels)}

avaible_device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.to(avaible_device)

optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])


def evaluate_model(
    model: PreTrainedModel, dataloader: DataLoader, avaible_device: torch.device
) -> tuple[float, float64, float64, float64]:
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs, masks, labels = (
                batch["input_ids"].to(avaible_device),
                batch["attention_mask"].to(avaible_device),
                batch["label"].to(avaible_device),
            )

            # Using forward to enable the insertion of attention masks
            outputs: SequenceClassifierOutput = model.forward(
                inputs, attention_mask=masks, labels=labels
            )

            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)

            predictions.extend(predicted_labels.cpu().detach().numpy())
            true_labels.extend(labels.cpu().detach().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="macro"
        )

        return accuracy, precision, recall, f1


# Using tqdm for better visualizations of the training process
with tqdm(
    total=params["num_epochs"] * len(train_loader),
) as pbar:
    for epoch in range(params["num_epochs"]):
        epoch_counting_from_1 = epoch + 1
        pbar.set_description(
            f"Epoch [{epoch_counting_from_1}/{params['num_epochs']}] - (Loss: N/A) - Steps"
        )
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            inputs, masks, labels = (
                batch["input_ids"].to(avaible_device),
                batch["attention_mask"].to(avaible_device),
                batch["label"].to(avaible_device),
            )

            optimizer.zero_grad()

            outputs: SequenceClassifierOutput = model.forward(
                inputs, attention_mask=masks, labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % params["log_steps"] == 0:
                avg_loss = running_loss / params["log_steps"]
                mlflow.log_metric(
                    "loss", avg_loss, step=(epoch * len(train_loader) + i)
                )

            pbar.update(1)

        accuracy, precision, recall, f1 = evaluate_model(
            model, test_loader, avaible_device
        )
        print(
            "Epoch: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(
                epoch_counting_from_1, accuracy, precision, recall, f1
            )
        )
        mlflow.log_metrics(
            {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1},
            step=epoch_counting_from_1,
        )

# Input example is not set because we are using a Dataloader as input
mlflow.pytorch.log_model(model, "model")
model_uri = f"runs:/{run.info.run_id}/model"
model_name = "agnews_pt_classifier"
mlflow.register_model(model_uri, model_name)

# This approach is for registering custom models
os.makedirs(params["output_dir"], exist_ok=True)
model.save_pretrained(params["output_dir"])
tokenizer.save_pretrained(params["output_dir"])
mlflow.log_artifacts(params["output_dir"], artifact_path="custom_model")

custom_model_uri = f"runs:/{run.info.run_id}/custom_model"
custom_model_name = "agnews_transformer"
mlflow.register_model(custom_model_uri, custom_model_name)

mlflow.end_run()
