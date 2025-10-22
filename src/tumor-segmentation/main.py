#! /usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from itertools import chain
import wandb
import os

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
# Initialize your Weights & Biases environment
wandb.login(key=WANDB_API_KEY)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set: Dataset = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)
test_set: Dataset = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
)

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(train_set, batch_size=100)


class GarmentClassifier(nn.Module):
    """
    The LeNet-5 architecture for classifying FashionMNIST garments.
    """

    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()
model.to(device)

# Weights & Biases setup
wandb.init(
    project="fashion-mnist-cnn",
    name="fashion-mnist-cnn",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 100,
        "model": "GarmentClassifier",
    },
)

wandb.watch(model, log="all", log_freq=50)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


num_epochs = 50
step = 0
# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        # Forward pass
        outputs = model(train)
        loss = error(outputs, labels)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        step += 1
        wandb.log({"train/loss": loss.item()}, step=step)

        # Testing the model
        if step % 50 == 0:  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images.view(100, 1, 28, 28))

                with torch.no_grad():
                    outputs = model(test)

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()

                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(step)
            accuracy_list.append(accuracy)
            wandb.log(
                {
                    "test/loss": loss.item(),
                    "test/accuracy": accuracy,
                },
                step=step,
            )

        if not (step % 500):
            print(f"Iteration: {step}, Loss: {loss.data}, Accuracy: {accuracy}%")


predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

flat_preds = sum(predictions_l, [])
flat_labels = sum(labels_l, [])

# compute
cm = confusion_matrix(flat_labels, flat_preds)
report = metrics.classification_report(flat_labels, flat_preds, output_dict=True)
print("Classification report for CNN :\n%s\n" % report)
wandb.log(
    {
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=flat_labels,
            preds=flat_preds,
            class_names=[
                "T-shirt/Top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle Boot",
            ],
        ),
        "classification_report": report,
    }
)

print("Done! Confusion matrix and classification report logged to WandB.")
