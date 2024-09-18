Step-by-step guide with code snippets to train bert language model using the **PROMISE_exp1.csv** dataset and then use it to label the reviews in **reviews.csv**. This guide is designed to be run in Google Colab.

### **Step 1: Setup and Install Dependencies**
Run this code to install necessary packages.
```python
# Install required packages
!pip install transformers
!pip install datasets
!pip install torch
!pip install pandas
!pip install sklearn
!pip install tqdm
```

### **Step 2: Import Libraries**
Import all the required libraries.
```python
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
```

### **Step 3: Load and Prepare the PROMISE Dataset**
Load the labeled dataset and prepare it for training.
```python
# Load PROMISE_exp1.csv from your Google Drive
from google.colab import files
uploaded = files.upload()

# Load the PROMISE dataset
promise_df = pd.read_csv('PROMISE_exp1.csv')

# Map labels to integers (F -> 0, NF -> 1)
promise_df['_class_'] = promise_df['_class_'].map({'F': 0, 'NF': 1})

# Split the data into training and validation sets
train_df, val_df = train_test_split(promise_df, test_size=0.2, random_state=42)

# Create Dataset objects
train_dataset = Dataset.from_pandas(train_df[['RequirementText', '_class_']])
val_dataset = Dataset.from_pandas(val_df[['RequirementText', '_class_']])
```

### **Step 4: Load Pre-trained Model and Tokenizer**
Load a pre-trained BERT model and tokenizer.
```python
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['RequirementText'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', '_class_'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', '_class_'])
```

### **Step 5: Fine-tune the Model**
Fine-tune the model on the labeled dataset.
```python
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()
```

### **Step 6: Save the Fine-tuned Model**
Save the trained model for later use.
```python
# Save the model and tokenizer
model.save_pretrained('fine-tuned-bert')
tokenizer.save_pretrained('fine-tuned-bert')
```

### **Step 7: Load and Label the Reviews Dataset**
Load the **reviews.csv** dataset and use the fine-tuned model to classify each review.
```python
# Load reviews.csv from your Google Drive
uploaded = files.upload()

# Load the reviews dataset
reviews_df = pd.read_csv('reviews.csv')

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('fine-tuned-bert')
tokenizer = BertTokenizer.from_pretrained('fine-tuned-bert')

# Function to classify reviews
def classify_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return 'F' if predicted_class == 0 else 'NF'

# Apply classification to the reviews
tqdm.pandas()
reviews_df['RequirementType'] = reviews_df['content'].progress_apply(classify_review)

# Save the labeled reviews
reviews_df.to_csv('labeled_reviews.csv', index=False)
print("Labeled reviews saved to 'labeled_reviews.csv'.")
```

### **Step 8: Download the Labeled Reviews**
Run this code to download the labeled reviews file.
```python
from google.colab import files
files.download('labeled_reviews.csv')
```

### Notes:
- Ensure to upload both **PROMISE_exp1.csv** and **reviews.csv** datasets when prompted by the `files.upload()` function.
- The **reviews.csv** dataset will be labeled with functional ('F') or non-functional ('NF') requirements in the new column `RequirementType`.
- This guide fine-tunes a BERT model, which can be computationally intensive. Adjust batch sizes and epochs if you face memory issues.

