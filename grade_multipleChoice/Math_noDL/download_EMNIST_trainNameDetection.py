# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn

import torchvision

from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch.optim as optim

import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date
import time


# %%
import multiple_choice_core as mg

import importlib;
importlib.reload(mg);

# %%

# %%
download_path = "/Users/hn/Documents/01_research_data/"

# %%

# Download EMNIST letters dataset
train_dataset = EMNIST(root=download_path, split="letters",
                       train=True, download=True, transform=transforms.ToTensor())

# %%
# Transform - For training, let us transform more robustly
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = EMNIST(root=download_path, split='letters', 
                       download=True, train=True, transform=transform)

# %%

# %%
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# %%
# Get one batch
images, labels = next(iter(train_loader))

# Show first 6 images
fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for i in range(6):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(chr(labels[i] + 64))  # EMNIST letters labels start at 1 (A)
    axes[i].axis('off')
plt.show()

# %%
# Validation dataset
val_dataset = EMNIST(root=download_path, split='letters', train=False,
                     download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# %% [markdown]
# ### Define model stuff

# %%
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mg.EMNIST_Model_CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# ## Train model

# %%
# %%time

epochs = 15
train_losses = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)-1  # shift 1–26 → 0–25
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
          f"Accuracy: {100*correct/total:.2f}%")


# %% [markdown]
# ### Validation

# %%
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)-1
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100*correct/total:.2f}%")

# %%
tick_legend_FontSize = 4
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 10,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 1.2), sharey=False, sharex=False, dpi=600)

ax.plot(train_losses, c ='dodgerblue', label = 'training losses', linewidth = .71)
ax.legend(loc = "upper right");

ax.set_ylabel(r'loss'); ax.set_xlabel(r'epochs');
ax.tick_params(axis='both', which='major', width=.25, length=2)
plt.yscale('log')

# %%
## Save Model
model_dir = download_path + "multiple_choices/models/"


checkpoint = {'epoch': epochs,
              'model_state_dict': model.state_dict(),
              # 'train_losses' : train_losses,
              'optimizer_state_dict': optimizer.state_dict(),
              "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# torch.save(checkpoint, '../../model_checkpoint.pth')
torch.save(model.state_dict(), model_dir + "emnist_letters_Aadam.pth")

checkpoint = {'epoch': epochs,
              'model_state_dict': model.state_dict(),
              # 'train_losses' : train_losses,
              'optimizer_state_dict': optimizer.state_dict(),
              "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
torch.save(checkpoint, model_dir + "emnist_letters_Aadam_checkPoint.pth")

# %%
len(images)

# %%

# %%

# %%

# %%

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


val_dataset = EMNIST(root=download_path, split='letters', train=False,
                     download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# %%
idx = 0  # pick any index you want
image, label = val_dataset[idx]

image = image.unsqueeze(0).to(device)  # [1, 1, 28, 28]
label = label - 1 

DL_name_reading_model.eval()

with torch.no_grad():
    outputs = DL_name_reading_model(image)
    _, predicted = torch.max(outputs, 1)
    
pred_char = chr(predicted.item() + ord('A'))
true_char = chr(label + ord('A'))

print(f"Prediction: {pred_char}")
print(f"Ground truth: {true_char}")




## predict all batches in validation set
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)-1
        outputs = DL_name_reading_model(images)
        _, predicted = torch.max(outputs, 1)
        predicted_letters = [chr(p.item() + ord('A')) for p in predicted]

        
        
## predict first batche in validation set

images, labels = next(iter(val_loader))
images = images.to(device)
labels = labels.to(device) - 1

with torch.no_grad():
    outputs = DL_name_reading_model(images)
    _, predicted = torch.max(outputs, 1)

# move to CPU for plotting
images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()
predicted_letters = [chr(p.item() + ord('A')) for p in predicted]


def emnist_fix(img):
    img = img.squeeze(0)          # [1,28,28] → [28,28]
    img = img.numpy()
    img = img.T                   # transpose
    img = img[:, ::-1]            # horizontal flip
    return img


plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


# fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=150)
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     img = emnist_fix(images[i])

#     pred_char = chr(predicted[i].item() + ord('A'))
#     true_char = chr(labels[i].item() + ord('A'))

#     ax.imshow(img, cmap='gray')
#     ax.set_title(
#         f"Pred: {pred_char} | True: {true_char}",
#         fontsize=12,
#         fontweight='bold',
#         color='green' if pred_char == true_char else 'crimson'
#     )
#     ax.axis('off')

# plt.tight_layout(pad=2.0)
# plt.show()

# ########

# # find mistake indices
# mistake_idx = (predicted != labels).nonzero(as_tuple=True)[0]

# if len(mistake_idx) == 0:
#     print("No mistakes in this batch 🎉")
# else:
#     fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=150)
#     axes = axes.flatten()

#     for ax, idx in zip(axes, mistake_idx[:len(axes)]):
#         img = emnist_fix(images[idx])

#         pred_char = chr(predicted[idx].item() + ord('A'))
#         true_char = chr(labels[idx].item() + ord('A'))

#         ax.imshow(img, cmap='gray')
#         ax.set_title(
#             f"Pred: {pred_char} | True: {true_char}",
#             fontsize=12,
#             fontweight='bold',
#             color='crimson'
#         )
#         ax.axis('off')

#     # hide unused axes if < 16 mistakes
#     for ax in axes[len(mistake_idx):]:
#         ax.axis('off')

#     plt.tight_layout(pad=2.0)
#     plt.show()

