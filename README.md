# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Problem Statement
Developing a Deep Learning Model for Named Entity Recognition (NER) using LSTM

In Natural Language Processing (NLP), understanding and extracting meaningful information from text is essential. One such task is Named Entity Recognition (NER), which involves identifying and classifying entities in text into predefined categories such as person names, locations, organizations, dates, etc.

The goal of this project is to develop a deep learning model using Long Short-Term Memory (LSTM) networks to automatically recognize and classify named entities in a given text sequence.
## DESIGN STEPS
### STEP 1:
Load data, create word/tag mappings, and group sentences.

### STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3:
Define dataset and DataLoader for batching.

### STEP 4:
Build a bidirectional LSTM model for sequence tagging.

### STEP 5:
Train the model over multiple epochs, tracking loss.

### STEP 6:
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.






## PROGRAM

### Name: SAHITH M

### Register Number: 212224230236

```python

# Model definition
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")

    return train_losses, val_losses




```

### OUTPUT

## Loss Vs Epoch Plot


<img width="677" height="547" alt="image" src="https://github.com/user-attachments/assets/1b5c4142-28f0-428f-858b-0249530f21d8" />


### Sample Text Prediction
<img width="341" height="336" alt="image" src="https://github.com/user-attachments/assets/47c232ca-5164-4742-b36d-c5c41ccf4223" />



## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
