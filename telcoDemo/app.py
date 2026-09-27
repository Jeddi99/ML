"""
เว็บแอปทำนายการยกเลิกบริการของลูกค้า (Customer Churn Prediction)
โครงสร้างเดียวกับ irisDemo: Flask + joblib + templates
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ต้อง import ไฟล์นี้ไว้ ถึงจะไม่ได้เรียกใช้ตรง ๆ ก็ตาม
# เพราะ .pkl อ้างถึงฟังก์ชัน add_engineered_features ที่อยู่ในไฟล์นี้
from features import RAW_COLUMNS, add_engineered_features  # noqa: F401

app = Flask(__name__)

# ระบุ path แบบเต็ม เพื่อให้รันได้ทั้งบนเครื่องเราและบน PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'churn_pipeline.pkl')
INFO_PATH = os.path.join(BASE_DIR, 'model', 'model_info.json')

# โหลดโมเดลครั้งเดียวตอนเปิดเซิร์ฟเวอร์ (ห้ามโหลดใหม่ทุกครั้งที่มีคนกด เพราะจะช้ามาก)
model = joblib.load(MODEL_PATH)

with open(INFO_PATH, encoding='utf-8') as f:
    MODEL_INFO = json.load(f)

# ตัวเลือกของแต่ละช่อง ต้องสะกดตรงกับข้อมูลตอนเทรนเป๊ะ ๆ
# (ค่าที่ส่งให้โมเดล, ข้อความที่แสดงบนหน้าเว็บ)
CHOICES = {
    'gender': [('Female', 'หญิง'), ('Male', 'ชาย')],
    'SeniorCitizen': [('0', 'ไม่ใช่'), ('1', 'ใช่')],
    'Partner': [('No', 'ไม่มี'), ('Yes', 'มี')],
    'Dependents': [('No', 'ไม่มี'), ('Yes', 'มี')],
    'PhoneService': [('Yes', 'มี'), ('No', 'ไม่มี')],
    'MultipleLines': [('No', 'ไม่มี'), ('Yes', 'มี'),
                      ('No phone service', 'ไม่ได้ใช้บริการโทรศัพท์')],
    'InternetService': [('DSL', 'DSL'), ('Fiber optic', 'ไฟเบอร์ออปติก'),
                        ('No', 'ไม่ใช้อินเทอร์เน็ต')],
    'OnlineSecurity': [('No', 'ไม่มี'), ('Yes', 'มี'),
                       ('No internet service', 'ไม่ได้ใช้อินเทอร์เน็ต')],
    'OnlineBackup': [('No', 'ไม่มี'), ('Yes', 'มี'),
                     ('No internet service', 'ไม่ได้ใช้อินเทอร์เน็ต')],
    'DeviceProtection': [('No', 'ไม่มี'), ('Yes', 'มี'),
                         ('No internet service', 'ไม่ได้ใช้อินเทอร์เน็ต')],
    'TechSupport': [('No', 'ไม่มี'), ('Yes', 'มี'),
                    ('No internet service', 'ไม่ได้ใช้อินเทอร์เน็ต')],
    'StreamingTV': [('No', 'ไม่มี'), ('Yes', 'มี'),
                    ('No internet service', 'ไม่ได้ใช้อินเทอร์เน็ต')],
    'StreamingMovies': [('No', 'ไม่มี'), ('Yes', 'มี'),
                        ('No internet service', 'ไม่ได้ใช้อินเทอร์เน็ต')],
    'Contract': [('Month-to-month', 'รายเดือน'), ('One year', 'สัญญา 1 ปี'),
                 ('Two year', 'สัญญา 2 ปี')],
    'PaperlessBilling': [('Yes', 'ใช่'), ('No', 'ไม่ใช่')],
    'PaymentMethod': [('Electronic check', 'เช็คอิเล็กทรอนิกส์'),
                      ('Mailed check', 'เช็คทางไปรษณีย์'),
                      ('Bank transfer (automatic)', 'ตัดบัญชีธนาคารอัตโนมัติ'),
                      ('Credit card (automatic)', 'ตัดบัตรเครดิตอัตโนมัติ')],
}

# ป้ายชื่อภาษาไทยของแต่ละช่อง ใช้แสดงบนฟอร์ม
LABELS = {
    'gender': 'เพศ', 'SeniorCitizen': 'เป็นผู้สูงอายุ', 'Partner': 'มีคู่สมรส',
    'Dependents': 'มีผู้อยู่ในอุปการะ', 'tenure': 'อายุการใช้งาน (เดือน)',
    'PhoneService': 'บริการโทรศัพท์', 'MultipleLines': 'หลายเลขหมาย',
    'InternetService': 'ประเภทอินเทอร์เน็ต', 'OnlineSecurity': 'ระบบความปลอดภัยออนไลน์',
    'OnlineBackup': 'สำรองข้อมูลออนไลน์', 'DeviceProtection': 'ประกันอุปกรณ์',
    'TechSupport': 'บริการช่วยเหลือทางเทคนิค', 'StreamingTV': 'ดูทีวีออนไลน์',
    'StreamingMovies': 'ดูหนังออนไลน์', 'Contract': 'รูปแบบสัญญา',
    'PaperlessBilling': 'รับบิลออนไลน์', 'PaymentMethod': 'วิธีชำระเงิน',
    'MonthlyCharges': 'ค่าบริการต่อเดือน (ดอลลาร์)', 'TotalCharges': 'ค่าบริการสะสม (ดอลลาร์)',
}

# ค่าตั้งต้นที่ใส่ให้ในฟอร์ม เป็นลูกค้าความเสี่ยงปานกลาง
DEFAULTS = {
    'gender': 'Female', 'SeniorCitizen': '0', 'Partner': 'No', 'Dependents': 'No',
    'tenure': '5', 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': '79.5',
    'TotalCharges': '400.0',
}

# ช่องที่เป็นตัวเลข ต้องแปลงชนิดข้อมูลก่อนส่งให้โมเดล
NUMERIC_INPUTS = {'tenure': int, 'MonthlyCharges': float,
                  'TotalCharges': float, 'SeniorCitizen': int}


def build_dataframe(form):
    """แปลงข้อมูลจากฟอร์ม HTML ให้เป็นตาราง 1 แถว ที่โมเดลรับได้"""
    row = {}
    for col in RAW_COLUMNS:
        value = form.get(col, DEFAULTS[col])
        caster = NUMERIC_INPUTS.get(col)
        row[col] = caster(value) if caster else value
    # เรียงคอลัมน์ตาม RAW_COLUMNS ให้ตรงกับตอนเทรน
    return pd.DataFrame([row], columns=RAW_COLUMNS)


def risk_level(prob):
    """แปลงความน่าจะเป็นเป็นระดับความเสี่ยง เพื่อให้ทีมดูแลลูกค้าใช้งานง่าย"""
    if prob >= 0.70:
        return 'สูงมาก', 'ติดต่อกลับทันทีวันนี้', 'danger'
    if prob >= 0.50:
        return 'สูง', 'จัดเข้าคิวโทรหาภายในสัปดาห์นี้', 'warn'
    if prob >= 0.30:
        return 'ปานกลาง', 'ส่งข้อเสนอโปรโมชันทางอีเมล', 'watch'
    return 'ต่ำ', 'ยังไม่ต้องดำเนินการเป็นพิเศษ', 'safe'


@app.route('/')
def home():
    return render_template('index.html', choices=CHOICES, labels=LABELS,
                           values=DEFAULTS, info=MODEL_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        X = build_dataframe(request.form)

        # predict_proba คืนความน่าจะเป็นของทั้ง 2 คลาส เราเอาคอลัมน์ที่ 1 (โอกาสยกเลิก)
        prob = float(model.predict_proba(X)[0][1])
        level, action, tone = risk_level(prob)

        return render_template(
            'index.html', choices=CHOICES, labels=LABELS,
            values=request.form.to_dict(), info=MODEL_INFO,
            probability=round(prob * 100, 1), level=level,
            action=action, tone=tone,
        )
    except Exception as e:
        return render_template('index.html', choices=CHOICES, labels=LABELS,
                               values=request.form.to_dict(), info=MODEL_INFO,
                               error_text=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """ช่องทางเรียกใช้แบบ JSON สำหรับต่อกับระบบอื่น (ใช้เป็นตัวอย่างในรายงาน)"""
    try:
        payload = request.get_json(force=True)
        row = {}
        for col in RAW_COLUMNS:
            value = payload.get(col, DEFAULTS[col])
            caster = NUMERIC_INPUTS.get(col)
            row[col] = caster(value) if caster else value
        X = pd.DataFrame([row], columns=RAW_COLUMNS)

        prob = float(model.predict_proba(X)[0][1])
        level, action, _ = risk_level(prob)
        return jsonify({
            'churn_probability': round(prob, 4),
            'churn_percent': round(prob * 100, 1),
            'risk_level': level,
            'recommended_action': action,
            'model': MODEL_INFO['best_model'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)
