import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import matplotlib.pyplot as plt

# Load intents.json data
with open("intents.json") as file:
    intents = json.load(file)

# Extract text and labels
texts = []
labels = []
tag_to_id = {intent["tag"]: idx for idx, intent in enumerate(intents["intents"])}
id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(tag_to_id[intent["tag"]])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Split data into training and evaluation sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenize the training and evaluation data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=128)

# Custom dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IntentDataset(train_encodings, train_labels)
eval_dataset = IntentDataset(eval_encodings, eval_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(tag_to_id)
)

# Define metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=75,  # Set epochs for training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=25,
    save_steps=300,
    evaluation_strategy="steps",  # Evaluate at each epoch
    save_strategy="steps",  # Save model at each epoch
    report_to=None,  # Disable HuggingFace logging
    save_total_limit = 5
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
train_results = trainer.train()

# Evaluate the model
metrics = trainer.evaluate(eval_dataset=eval_dataset)
print(f"Final Accuracy: {metrics['eval_accuracy']:.4f}")

# Save the model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Plot training and evaluation metrics
training_logs = trainer.state.log_history

# Extract epochs, accuracy, and loss
epochs = []
accuracy = []
loss = []

for log in training_logs:
    if "epoch" in log:
        epochs.append(log["epoch"])
        if "eval_accuracy" in log:
            accuracy.append(log["eval_accuracy"])
        if "eval_loss" in log:
            loss.append(log["eval_loss"])

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs[:len(accuracy)], accuracy, label="Evaluation Accuracy")
plt.title("Epochs vs Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs[:len(loss)], loss, label="Evaluation Loss", color="red")
plt.title("Epochs vs Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
