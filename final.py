import pandas as pd
import matplotlib.pyplot as plt

model_history = pd.read_csv("model_history.csv")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model_history['epoch'], model_history['train_loss'], label='Train Loss')
plt.plot(model_history['epoch'], model_history['test_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(model_history['epoch'], model_history['train_acc'], label='Train Accuracy')
plt.plot(model_history['epoch'], model_history['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()