# 📌 RESTful API with Flask – First Project

โปรเจกต์นี้เป็นตัวอย่างการสร้าง RESTful API ครั้งแรกด้วย **Flask** และ **Flask-RESTful**  
Credit : https://www.youtube.com/watch?v=AZfJ8buL5II 
โดยเชื่อมต่อกับโมเดล Machine Learning (Linear Regression) ที่เทรนไว้แล้ว (`simple_linear_regression.pkl`)  
เพื่อทำการทำนายยอดขายจากงบประมาณการตลาด (Marketing Budget)

---

## ⚙️ Features
- API สำหรับดึงข้อมูล (`/api`) → อ่านข้อมูลจาก `data.xlsx`
- API สำหรับทำ **prediction** (`/prediction/<budget>`) → ส่งค่า Marketing Budget แล้วจะได้ผลการทำนายยอดขายกลับมา
- ใช้ **pickle** โหลดโมเดลที่เทรนไว้แล้ว
- รองรับ **CORS** (สำหรับเชื่อมต่อกับ Frontend/React หรืออื่น ๆ)

---

## 📂 Project Structure
```plaintext
├── app.py # main API file
├── simple_linear_regression.pkl # โมเดล Linear Regression ที่บันทึกไว้
├── data.xlsx # ไฟล์ข้อมูลตัวอย่าง
├── requirements.txt # dependencies ที่ใช้ (แนะนำให้เพิ่ม)
└── README.md
```

## ▶️ Run Server
```bash
python app.py
```
## 🔗 API Endpoints
1) ดึงข้อมูลทั้งหมด

GET /api
คืนค่า JSON records จาก data.xlsx

2) ทำการทำนาย

GET /prediction/<budget>
ส่งค่า budget (เช่น 150) เพื่อให้โมเดลทำนาย Actual Sales

Request
```bash
http://127.0.0.1:5000/prediction/150
```
Response
```bash
"11"
```
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)