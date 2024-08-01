from model import FashionMNISTModel
from torch import nn
import torch
from timeit import default_timer as timer
from dataloader import train_dataloader, test_dataloader
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

def accuracy_fn(y_preds, y_true):
    return (y_preds == y_true).sum().item() / len(y_preds)

def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)
        loss = loss_fn(y_logits, y)

        y_pred_probs = y_logits.argmax(dim=1)

        train_loss += loss.item()
        acc = accuracy_fn(y_pred_probs, y)
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_preds_logits = model(X)
            y_pred_probs = y_preds_logits.argmax(dim=1)

            loss = loss_fn(y_preds_logits, y)
            test_loss += loss.item()
            acc = accuracy_fn(y_pred_probs, y)
            test_acc += acc

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    return test_loss, test_acc

def train_save_model():
    model = FashionMNISTModel(input_shape=1, hidden_units=64, output_shape=10).to(device)

    epochs = 30
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)

    start_time = timer()

    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n------------")

        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

    end_time = timer()
    print(f"Training time on GPU: {end_time - start_time:.3f} seconds")

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "FashionMNISTModelPyTorchV6.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    history_df = pd.DataFrame(history)
    history_df.to_csv("model_history.csv", index=False)

train_save_model()