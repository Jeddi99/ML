# Telco Customer Churn — Deployment

เว็บแอปทำนายโอกาสที่ลูกค้าจะยกเลิกบริการ สร้างจากโมเดลใน `project/telco_jed.ipynb`

## โครงสร้างไฟล์

```
telcoDemo/
├── features.py          ฟังก์ชันสร้างฟีเจอร์ (ใช้ร่วมกันทั้งตอนเทรนและตอนรัน)
├── train_telco.py       เทรนโมเดลแล้วบันทึกเป็น Pipeline ไฟล์เดียว
├── app.py               เว็บเซิร์ฟเวอร์ Flask
├── templates/
│   └── index.html       หน้าเว็บกรอกข้อมูลและแสดงผล
├── model/
│   ├── churn_pipeline.pkl   โมเดลที่เทรนแล้ว (เตรียมข้อมูล + ทำนาย รวมอยู่ในไฟล์เดียว)
│   └── model_info.json      คะแนนของโมเดล ใช้แสดงท้ายหน้าเว็บ
└── requirements.txt
```

## รันบนเครื่องตัวเอง

```bash
pip install -r requirements.txt
python train_telco.py     # ครั้งแรกครั้งเดียว สร้าง model/churn_pipeline.pkl
python app.py             # เปิด http://127.0.0.1:5001
```

## ตัวอย่าง Input → Output

ลูกค้าสัญญารายเดือน ใช้งานมา 2 เดือน ไฟเบอร์ออปติก ไม่มีบริการช่วยเหลือทางเทคนิค
จ่ายด้วยเช็คอิเล็กทรอนิกส์ ค่าบริการเดือนละ 95.5

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No",
       "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
       "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":190}'
```

ผลลัพธ์

```json
{
  "churn_probability": 0.6258,
  "churn_percent": 62.6,
  "risk_level": "สูง",
  "recommended_action": "จัดเข้าคิวโทรหาภายในสัปดาห์นี้",
  "model": "Logistic Regression"
}
```

ลูกค้าสัญญา 2 ปี ใช้งานมา 65 เดือน มีบริการเสริมครบ ได้ผล 3.3% (ความเสี่ยงต่ำ)

## นำขึ้นอินเทอร์เน็ตด้วย PythonAnywhere (ฟรี)

ใช้เวลาประมาณ 10 นาที ต้องสมัครบัญชีเองที่ https://www.pythonanywhere.com
เลือกแพ็กเกจ **Beginner** ซึ่งฟรีตลอดชีพ จะได้เว็บ 1 เว็บที่ชื่อ
`ชื่อผู้ใช้.pythonanywhere.com`

### ขั้นที่ 1 อัปโหลดไฟล์

ไฟล์ `telcoDemo.zip` เตรียมไว้ให้แล้วที่โฟลเดอร์แม่ของโปรเจกต์

ไปที่แท็บ **Files** กด **Upload a file** แล้วเลือก `telcoDemo.zip`
อัปโหลดไว้ที่โฟลเดอร์แรกสุด (`/home/ชื่อผู้ใช้`)

### ขั้นที่ 2 แตกไฟล์และติดตั้งไลบรารี

ไปที่แท็บ **Consoles** กด **Bash** แล้วพิมพ์ทีละบรรทัด

```bash
unzip telcoDemo.zip
cd telcoDemo
pip install --user flask scikit-learn pandas joblib
```

บรรทัดสุดท้ายใช้เวลาสักครู่ รอจนขึ้นบรรทัดคำสั่งใหม่

### ขั้นที่ 3 เทรนโมเดลบนเซิร์ฟเวอร์

```bash
python3 train_telco.py
```

**ทำไมต้องเทรนใหม่บนเซิร์ฟเวอร์** ไฟล์ `.pkl` ที่เทรนจากเครื่องเรา มักโหลดไม่ขึ้น
บนเซิร์ฟเวอร์ที่เวอร์ชัน scikit-learn ไม่ตรงกัน การเทรนใหม่ตรงนี้ใช้เวลาไม่ถึงนาที
และตัดปัญหานี้ทิ้งไปเลย ข้อมูล CSV แนบมาในโฟลเดอร์ `data/` แล้ว
จึงไม่ต้องต่อ Kaggle ซึ่งบัญชีฟรีเข้าไม่ได้อยู่แล้ว

ต้องเห็นข้อความ `บันทึกโมเดลไว้ที่ model/churn_pipeline.pkl เรียบร้อยแล้ว`

### ขั้นที่ 4 สร้างเว็บแอป

ไปที่แท็บ **Web** แล้วทำตามลำดับ

1. กด **Add a new web app** แล้วกด **Next**
2. เลือก **Flask**
3. เลือก **Python 3.13**
   ต้องตรงกับเวอร์ชันที่ติดตั้งไลบรารีไว้ในขั้นที่ 3
   ดูได้จากข้อความตอน pip install ว่าลงไว้ที่ `/usr/local/lib/python3.13/site-packages`
   ถ้าเลือกผิดเวอร์ชัน เว็บจะพังเพราะหา scikit-learn ไม่เจอ
4. ช่อง path ให้กด **Next** ผ่านไปเลย เดี๋ยวเราแก้ทีหลัง

### ขั้นที่ 5 ชี้ไฟล์ WSGI มาที่โปรเจกต์

ยังอยู่ที่แท็บ **Web** เลื่อนลงหาหัวข้อ **Code** กดลิงก์ข้าง
**WSGI configuration file** จะเปิดหน้าแก้ไขไฟล์ขึ้นมา

**ลบข้อความเดิมทั้งหมดทิ้ง** แล้ววางข้อความนี้แทน
(เปลี่ยน `YOURUSERNAME` เป็นชื่อผู้ใช้ของคุณ)

```python
import sys

project_home = '/home/YOURUSERNAME/telcoDemo'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application
```

กด **Save** ที่มุมขวาบน

### ขั้นที่ 6 เปิดใช้งาน

กลับไปแท็บ **Web** กดปุ่มเขียว **Reload** แล้วเปิดลิงก์
`https://ชื่อผู้ใช้.pythonanywhere.com`

### ถ้าเว็บขึ้น Error

ที่แท็บ **Web** เลื่อนลงไปหาหัวข้อ **Log files** กด **Error log**
แล้วดูบรรทัดล่างสุด สาเหตุที่พบบ่อยมีสามอย่าง

| ข้อความที่เห็น | สาเหตุ | วิธีแก้ |
|---|---|---|
| `ModuleNotFoundError: No module named 'features'` | ลืมใส่ `sys.path.insert` หรือสะกดชื่อผู้ใช้ผิด | กลับไปแก้ไฟล์ WSGI ในขั้นที่ 5 |
| `FileNotFoundError: churn_pipeline.pkl` | ยังไม่ได้เทรนโมเดล | กลับไปทำขั้นที่ 3 |
| `ModuleNotFoundError: No module named 'sklearn'` | เลือก Python คนละเวอร์ชันกับที่ติดตั้งไลบรารีไว้ | ลบเว็บแอปทิ้งแล้วสร้างใหม่ เลือก Python 3.13 |

แก้เสร็จทุกครั้งต้องกด **Reload** ที่แท็บ Web ไม่งั้นโค้ดเก่ายังทำงานอยู่

## ข้อจำกัดของระบบ

- โมเดลเทรนจากข้อมูลบริษัทโทรคมนาคมในสหรัฐอเมริกา ถ้านำมาใช้กับลูกค้าไทยต้องเทรนใหม่
- ทำนายจากข้อมูล ณ จุดเดียว ไม่ได้ดูพฤติกรรมย้อนหลัง เช่น จำนวนครั้งที่โทรร้องเรียน
- ที่เกณฑ์ 50% โมเดลจับลูกค้าที่จะยกเลิกได้ 78% แต่ในกลุ่มที่เตือนมีคนที่ไม่ยกเลิกปนอยู่ประมาณครึ่งหนึ่ง
  จึงเหมาะกับการใช้จัดลำดับว่าควรโทรหาใครก่อน มากกว่าใช้ตัดสินขาด
