###
# โค้ดนี้ใช้สำหรับโหลดโมเดลที่ฝึกไว้แล้ว และทดสอบกับข้อมูลภาพใหม่
# โดยใช้ EfficientNet-B2 เพื่อทำการจำแนกประเภทของภาพ

import torch
import torchvision
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms

# กำหนดการแปลงภาพ (resize และแปลงเป็น tensor)
tfms = transforms.Compose([
        transforms.Resize((288,288)),  # ปรับขนาดภาพให้ตรงกับที่ใช้ตอนฝึกโมเดล
        transforms.ToTensor()
      ])

# โหลดโมเดลที่ถูกฝึกไว้แล้ว
PATH_model = 'model-epoch4.pt' 
weights = torch.load(PATH_model)
class_names = ['Cat', 'bird', 'dog']  # รายการคลาสที่โมเดลต้องจำแนก
model_test = torchvision.models.efficientnet_b2(pretrained=False, num_classes=len(class_names))

# กำหนดอุปกรณ์ที่ใช้ในการประมวลผล (ใช้ GPU ถ้ามี)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลดค่าถ่วงน้ำหนักของโมเดลที่ฝึกไว้แล้ว
model_test.load_state_dict(weights, strict=False)
model_test.eval()
model_test.to(device)

# วนลูปอ่านไฟล์ภาพจากโฟลเดอร์ test และทำการทำนายผลลัพธ์
for root, dirs, files in os.walk("./Cats,-dogs-and-birds-3/test/", topdown=False):
   for name in files:
      print(os.path.join(root, name))  # แสดงชื่อไฟล์ภาพที่กำลังทดสอบ
      
      img = Image.open(os.path.join(root, name))  # เปิดไฟล์ภาพ
      img_tensor = tfms(img).to(device).unsqueeze(0)  # แปลงภาพเป็น tensor และเพิ่ม batch dimension
      
      output = model_test(img_tensor)  # ส่งข้อมูลภาพเข้าโมเดลเพื่อทำการทำนาย
      _, predicted = torch.max(output, 1)  # หาค่าที่โมเดลให้คะแนนสูงสุด
      
      try:
          answer = class_names[predicted[0].tolist()]  # แปลงค่าทำนายเป็นชื่อคลาส
      except:
          answer = '--'  # กรณีที่เกิดข้อผิดพลาดในการแมปคลาส
      
      print(answer)  # แสดงผลลัพธ์ที่ทำนายได้
###
