# %%capture
# !pip install peft datasets wandb
# !pip install --upgrade transformers
# !pip install -q peft bitsandbytes accelerate

import torch
import wandb
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
import re
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

# Ensure GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")

# Login to Weights & Biases (W&B) - Required in Kaggle
wandb.login(key="REDACTED")  # Replace with your actual API key
wandb.init(project="Spring2025")


# Define preprocessing function
def preprocess_text(text):
    """
    Preprocess Supreme Court case text for classification.
    - Lowercasing
    - Removing special characters
    - Removing extra spaces
    - Normalizing whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(text.split())
    return text

def tokenize_text(texts, max_length=512):
    """
    Tokenizes a list of texts and returns input tensors.
    """
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokens

# Load dataset from CSV
# df = pd.read_csv("/kaggle/input/labels-web-of-law/15_labels_data.csv")  # Adjust file path if needed
# df = pd.read_csv("archive/15_labels_data.csv")  # Adjust file path if needed

import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "15_labels_data.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "dhruvjoshi892/labels-web-of-law",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)


# Apply preprocessing
df = df[:10]
df["text"] = df["text"].apply(preprocess_text)


# Load tokenizer
model_name = "bert-base-uncased"  # Modify as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
tokens = tokenize_text(list(df["text"]))
input_ids, attention_masks = tokens["input_ids"], tokens["attention_mask"]
labels = torch.tensor(df["label"].values)

# Split dataset into train and validation
train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(
    input_ids, labels, attention_masks, test_size=0.1, random_state=42
)


# Convert to PyTorch Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

train_dataset = CustomDataset(train_inp, train_mask, train_label)
eval_dataset = CustomDataset(val_inp, val_mask, val_label)


# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=15).to(device)

# Apply LoRA Configuration (PEFT technique)
lora_config = LoraConfig(
    r=32,  # Higher rank to improve adaptation
    lora_alpha=64,  # Adjusted scaling factor
    target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],  # Apply LoRA to more layers
    lora_dropout=0.01,  # Lower dropout for stable learning on large data
    bias="none")

# Integrate LoRA with the model
model = get_peft_model(model, lora_config).to(device)
model.print_trainable_parameters()

# Define metrics
def compute_metrics(eval_pred):
    print("compute_metrics function called!")  # Debugging statement

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")  # Debug shape info
    print(f"Unique predicted labels: {np.unique(preds)}")  # Debugging predictions
    print(f"Unique actual labels: {np.unique(labels)}")  # Debugging actual labels

    accuracy = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)  # Adjust for imbalanced labels
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    print(f"Metrics computed: {metrics}")  # Debugging computed metrics

    return metrics


# Define training arguments
training_args = TrainingArguments(
    output_dir="./werk/outputs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./werk/logs",
    logging_steps=10,
    num_train_epochs=10,  # More epochs for better legal text learning
    per_device_train_batch_size=8,  # Lower batch size for stability
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Helps with larger texts
    learning_rate=3e-5,  # Reduce LR for better fine-tuning
    warmup_ratio=0.1,  # Stabilize early training
    weight_decay=0.01,
    metric_for_best_model="balanced_accuracy",  # Handle imbalanced classes
    load_best_model_at_end=True,
    report_to=["wandb"],  # Enable W&B logging
    fp16=torch.cuda.is_available(),  # Use FP16 if GPU is available
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Finish W&B run
wandb.finish()

results = trainer.evaluate()
print("For 15 label classification:"+results)

