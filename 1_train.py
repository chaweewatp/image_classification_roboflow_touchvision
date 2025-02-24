import os
from roboflow import Roboflow
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize,
    CenterCrop,
)
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorama

from PIL import Image

rf = Roboflow(api_key="XXXXXXX")
project = rf.workspace("XXXXXX").project("XXXXX")
version = project.version(1)
dataset = version.download("folder")


os.environ["DATASET_DIRECTORY"] = "/content/XXXXX"
print({dataset.location})
train_data = ImageFolder(
    os.path.join(os.getcwd(), "/content/XXXXX/" , "train"),
    transform=Compose([
        Resize((288,288)),
        ToTensor()]
    ),
)
valid_data = ImageFolder(
    os.path.join(os.getcwd(),"/content/XXXXX/" , "valid"),
    transform=Compose([
        Resize((288,288)),
        ToTensor()
    ]),
)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True)

class_names = train_data.classes
print(class_names)

# Take one batch from the train loader
data, labels = next(iter(train_loader))
data, labels = data[0:5], labels[0:5]

# Plot the images
fig = plt.figure(figsize=(16, 9))
for i in range(0, 5):
    fig.add_subplot(1, 5, i + 1)
    plt.imshow(data[i].permute(1, 2, 0))
    plt.xlabel(class_names[labels[i]])


model_ft = torchvision.models.efficientnet_b2(pretrained=True)
print(model_ft)

# model_ft.fc
model_ft.classifier[1] = nn.Linear(in_features=1408, out_features=256, bias=True)

model_ft.classifier.append(nn.Dropout(0.3,inplace=True))
# model_ft.classifier.append(nn.functional.leaky_relu(inplace=True))
model_ft.classifier.append(nn.Linear(in_features=256, out_features=9, bias=True))


print(model_ft.classifier)

model_ft.requires_grad_(False)
model_ft.classifier.requires_grad_(True)


def accuracy(model, data_loader, device):
    model.eval()

    num_correct = 0
    num_samples = 0
    with torch.no_grad():  # deactivates autograd, reduces memory usage and speeds up computations
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            predictions = torch.argmax(model(data), 1)  # find the class number with the largest output
            num_correct += (predictions == labels).sum().item()
            num_samples += len(predictions)

    return num_correct / num_samples

def train(
    model,
    train_loader,
    valid_loader,
    device,
    num_epochs=3,
    learning_rate=0.1,
    decay_learning_rate=False,
):
    # Some models behave differently in training and testing mode (Dropout, BatchNorm)
    # so it is good practice to specify which behavior you want.
    model.train()

    # We will use the Adam with Cross Entropy loss
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.85)

    max_loss=10000
    count = 0
    # We make multiple passes over the dataset
    for epoch in range(num_epochs):
        print("=" * 40, "Starting epoch %d" % (epoch + 1), "=" * 40)

        if decay_learning_rate:
            scheduler.step()

        total_epoch_loss = 0.0
        # Make one pass in batches
        for batch_number, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            # if batch_number % 5 == 0:
            #     print("Batch %d/%d" % (batch_number, len(train_loader)))

        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, valid_loader, device)

        print(
            colorama.Fore.GREEN
            + "\nEpoch %d/%d, Loss=%.4f, Train-Acc=%d%%, Valid-Acc=%d%%"
            % (
                epoch + 1,
                num_epochs,
                total_epoch_loss / len(train_data),
                100 * train_acc,
                100 * test_acc,
            ),
            colorama.Fore.RESET,
        )
        # early stop
        if (total_epoch_loss / len(train_data)) < max_loss:
            max_loss = (total_epoch_loss / len(train_data))
            count = 0
            torch.save(model_ft.state_dict(), 'model-epoch{}.pt'.format(epoch+1))
            print(max_loss)
            print('model saved')
        else:
            count += 1

        if count >3:
            print('exceed counting')
            break



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)

train(model_ft, train_loader, valid_loader, device, num_epochs=20)

model_ft.eval()



tfms = transforms.Compose([
        transforms.Resize((288,288)),
        transforms.ToTensor()
      ])

# img = Image.open('/content/Demo-animal-1/test/shrimp/download-29-_jpeg.rf.a79181c07a34775bb67290b535ae7300.jpg')
# img_tensor = tfms(img).to(device).unsqueeze(0)
# output = model_ft(img_tensor)
# _, predicted = torch.max(output, 1)

# answer = class_names[predicted[0].tolist()]
# plt.imshow(img)
# plt.title(answer)