"""
โค้ดนี้ใช้ PyTorch และ torchvision สำหรับสร้างและฝึกโมเดล CNN โดยใช้ EfficientNet-B2
เพื่อนำมาทำการจำแนกภาพจากชุดข้อมูลที่โหลดมาผ่าน DataLoader
"""

import os
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
)
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import colorama

# โหลดชุดข้อมูลสำหรับการฝึก (Train) และการตรวจสอบความถูกต้อง (Validation)
train_data = ImageFolder(
    os.path.join(os.getcwd(), "./Cats,-dogs-and-birds-3" , "train"),  # ระบุโฟลเดอร์ที่เก็บข้อมูลฝึก
    transform=Compose([
        Resize((288,288)),  # ปรับขนาดภาพเป็น 288x288 พิกเซล
        ToTensor()  # แปลงภาพเป็น Tensor เพื่อให้ใช้งานกับ PyTorch ได้
    ]),
)

valid_data = ImageFolder(
    os.path.join(os.getcwd(), "./Cats,-dogs-and-birds-3" , "valid"),  # ระบุโฟลเดอร์ที่เก็บข้อมูล validation
    transform=Compose([
        Resize((288,288)),  # ปรับขนาดภาพให้เท่ากันกับข้อมูลฝึก
        ToTensor()
    ]),
)

# สร้าง DataLoader สำหรับโหลดข้อมูลทีละ batch
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True)

# ดึงชื่อของคลาส (ประเภทของภาพที่โมเดลต้องเรียนรู้)
class_names = train_data.classes
print(class_names)

# # โหลดโมเดล EfficientNet-B2 และกำหนดจำนวนคลาส
# model_ft = torchvision.models.efficientnet_b2(pretrained=False, num_classes=len(class_names))
# print(model_ft)
# model_ft.requires_grad_(True)
# model_ft.classifier.requires_grad_(True)


model_ft = torchvision.models.efficientnet_b2(pretrained=True)
print(model_ft)
model_ft.classifier[1] = nn.Linear(in_features=1408, out_features=256, bias=True)
model_ft.classifier.append(nn.Dropout(0.3,inplace=True))
model_ft.classifier.append(nn.Linear(in_features=256, out_features=9, bias=True))
print(model_ft.classifier)
model_ft.requires_grad_(False)
model_ft.classifier.requires_grad_(True)


# ฟังก์ชันคำนวณความแม่นยำของโมเดล
def accuracy(model, data_loader, device):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():  # ปิดการคำนวณ autograd เพื่อลดการใช้หน่วยความจำ
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            predictions = torch.argmax(model(data), 1)  # เลือกค่าที่มีค่ามากที่สุดเป็นคลาสที่คาดการณ์
            num_correct += (predictions == labels).sum().item()
            num_samples += len(predictions)
    return num_correct / num_samples

# ฟังก์ชันสำหรับฝึกโมเดล
def train(
    model,
    train_loader,
    valid_loader,
    device,
    num_epochs=3,
    learning_rate=0.1,
    decay_learning_rate=False,
):
    model.train()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.85)

    max_loss = 10000
    count = 0
    for epoch in range(num_epochs):
        print("=" * 40, f"Starting epoch {epoch + 1}", "=" * 40)

        if decay_learning_rate:
            scheduler.step()

        total_epoch_loss = 0.0
        for batch_number, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, valid_loader, device)

        print(
            colorama.Fore.GREEN
            + f"\nEpoch {epoch + 1}/{num_epochs}, Loss={total_epoch_loss / len(train_data):.4f}, Train-Acc={100 * train_acc:.0f}%, Valid-Acc={100 * test_acc:.0f}%",
            colorama.Fore.RESET,
        )

        # Early stopping และบันทึกโมเดลหาก loss ลดลง
        if (total_epoch_loss / len(train_data)) < max_loss:
            max_loss = (total_epoch_loss / len(train_data))
            count = 0
            torch.save(model_ft.state_dict(), f'model-epoch{epoch+1}.pt')
            print(max_loss)
            print('model saved')
        else:
            count += 1

        if count > 3:
            print('exceed counting')
            break

# กำหนดอุปกรณ์ในการประมวลผล (ใช้ GPU หากมี)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)

# เรียกใช้ฟังก์ชันฝึกโมเดล
train(model_ft, train_loader, valid_loader, device, num_epochs=20)
