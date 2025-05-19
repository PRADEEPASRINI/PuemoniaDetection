# main.py
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import classification_report
from preprocess import get_dataloaders
from models.dual_cnn import DualCNN
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders()
model = DualCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
def train(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")

def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

train(epochs=10)
evaluate(test_loader)
