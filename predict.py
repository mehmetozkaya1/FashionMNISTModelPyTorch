## Load saved model
import torch
from model import FashionMNISTModel
from dataset import train_dataset, test_dataset
import matplotlib.pyplot as plt

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_model = FashionMNISTModel(
    input_shape = 1,
    hidden_units = 64,
    output_shape = 10
)

loaded_model.load_state_dict(torch.load("models/FashionMNISTModelPyTorchV6.pth"))
loaded_model.to(device)

def make_predictions(model : torch.nn.Module,
                     data : list,
                     device : torch.device = device):
    
    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim = 0).to(device)

            pred_logits = model(sample)

            prob = torch.softmax(pred_logits.squeeze(), dim = 0)

            pred_probs.append(prob.cpu())
    
    return torch.stack(pred_probs)

import random
random.seed(42)

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_dataset), k = 9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model = loaded_model, data = test_samples)

pred_classes = pred_probs.argmax(dim=1)
print("Expected values : ",test_labels)
print("Predicted Values : ", pred_classes)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3

class_names = train_dataset.classes

for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i + 1)

  plt.imshow(sample.squeeze(), cmap="gray")

  pred_label = class_names[pred_classes[i]]

  truth_label = class_names[test_labels[i]]

  title_text = f"Pred: {pred_label} | Truth: {truth_label}"

  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c = "g")
  else:
    plt.title(title_text, fontsize=10, c = "r")

  plt.axis(False)
  
plt.show()