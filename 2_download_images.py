"""
โค้ดนี้ใช้ Roboflow สำหรับดาวน์โหลดชุดข้อมูลภาพ และใช้ PyTorch เพื่อโหลดข้อมูลเข้าสู่ DataLoader
และเตรียมข้อมูลสำหรับการฝึกโมเดล CNN
"""

from roboflow import Roboflow
rf = Roboflow(api_key="QfdxwHQSxLUWGhZYHRzG")  # ใช้ API Key ของ Roboflow เพื่อเข้าถึงข้อมูล
project = rf.workspace("ai-peas1").project("meter-czrjc")  # กำหนดโปรเจกต์ที่ต้องการใช้งาน
version = project.version(1)  # ใช้เวอร์ชันที่ 1 ของโปรเจกต์
dataset = version.download("folder")  # ดาวน์โหลดชุดข้อมูลและบันทึกลงในโฟลเดอร์

from torchvision.datasets import ImageFolder
import os
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# โหลดชุดข้อมูลสำหรับการฝึก (Train) และการตรวจสอบความถูกต้อง (Validation)
train_data = ImageFolder(
    os.path.join(os.getcwd(), "./meter-1" , "train"),  # ระบุโฟลเดอร์ที่เก็บข้อมูลฝึก
    transform=Compose([
        Resize((288,288)),  # ปรับขนาดภาพเป็น 288x288 พิกเซล
        ToTensor()  # แปลงภาพเป็น Tensor เพื่อให้ใช้งานกับ PyTorch ได้
    ]),
)

valid_data = ImageFolder(
    os.path.join(os.getcwd(), "./meter-1" , "valid"),  # ระบุโฟลเดอร์ที่เก็บข้อมูล validation
    transform=Compose([
        Resize((288,288)),  # ปรับขนาดภาพให้เท่ากันกับข้อมูลฝึก
        ToTensor()
    ]),
)

# สร้าง DataLoader สำหรับโหลดข้อมูลทีละ batch
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)  # โหลดข้อมูลฝึกครั้งละ 8 ภาพ และสลับข้อมูลทุกครั้ง
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True)  # โหลดข้อมูล validation ครั้งละ 8 ภาพ

# ดึงชื่อของคลาส (เช่น ประเภทของภาพที่โมเดลต้องเรียนรู้)
class_names = train_data.classes
print(class_names)  # แสดงรายการคลาสทั้งหมดที่มีในชุดข้อมูล

# ดึง batch ข้อมูลตัวอย่างจาก train_loader
data, labels = next(iter(train_loader))  # ดึงภาพและป้ายกำกับของ batch แรก
data, labels = data[0:5], labels[0:5]  # เลือกเฉพาะ 5 ภาพแรกของ batch

# แสดงภาพตัวอย่างจากข้อมูลฝึก
fig = plt.figure(figsize=(16, 9))
for i in range(0, 5):
    fig.add_subplot(1, 5, i + 1)  # สร้าง subplot สำหรับแสดงแต่ละภาพ
    plt.imshow(data[i].permute(1, 2, 0))  # แปลงข้อมูล Tensor ให้สามารถแสดงเป็นภาพได้
    plt.xlabel(class_names[labels[i]])  # กำกับชื่อคลาสของแต่ละภาพ
plt.show()  # แสดงภาพทั้งหมดในกราฟ
