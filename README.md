# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset


## DESIGN STEPS
### STEP 1
Import required libraries such as PyTorch, NumPy, and Matplotlib, and prepare the dataset by converting words and tags into numerical indices.

### STEP 2
Build a BiLSTM neural network model consisting of an embedding layer, LSTM layer, and a fully connected layer to predict entity tags.

### STEP 3
Train the model using a loss function and optimizer, evaluate the model performance on validation data, and visualize the training and validation loss.



## PROGRAM
### Name: Shivasri 
### Register Number: 212224220098
```python
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size=len(word2idx)+1, tagset_size=len(tag2idx), embedding_dim=128, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        lstm_out, _ = self.lstm(x)

        logits = self.fc(lstm_out)

        return logits
        


model = BiLSTMTagger().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        total_train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)

            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)

                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot



### Sample Text Prediction
Include your sample text prediction here.

![NER Output](https://raw.githubusercontent.com/shivu1405/NER-using-LSTM/0d15426f5605d7c82f42c2c4dc46a95348e49693/Screenshot%202026-03-15%20191427.png)

## RESULT
