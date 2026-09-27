"""
ฟีเจอร์ที่สร้างเพิ่ม (Feature Engineering)

สำคัญมาก: ไฟล์นี้ต้องถูก import ทั้งตอนเทรน (train_telco.py) และตอนใช้งานจริง (app.py)
เพราะ joblib จะไม่ได้เก็บ "โค้ดของฟังก์ชัน" ไว้ในไฟล์ .pkl
แต่เก็บแค่ "ชื่อและที่อยู่" ของฟังก์ชันไว้เท่านั้น
ถ้าตอนโหลดโมเดลหาไฟล์นี้ไม่เจอ โปรแกรมจะพังทันที
"""
import pandas as pd

# คอลัมน์บริการเสริมทั้ง 6 อย่าง ใช้นับว่าลูกค้าสมัครกี่บริการ
ADDON_COLS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

# คอลัมน์ดิบ 19 ตัวที่ผู้ใช้ต้องกรอกผ่านหน้าเว็บ (ลำดับต้องตรงกับตอนเทรน)
RAW_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges',
]

# คอลัมน์ตัวเลข (หลังสร้างฟีเจอร์เพิ่มแล้ว)
NUM_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'Total_Addons', 'AvgMonthlySpend']

# คอลัมน์หมวดหมู่ = คอลัมน์ดิบทั้งหมดที่ไม่ใช่ตัวเลข
CAT_FEATURES = [c for c in RAW_COLUMNS if c not in NUM_FEATURES]


def add_engineered_features(df):
    """รับตารางข้อมูลดิบ คืนตารางที่ทำความสะอาดและเพิ่มฟีเจอร์ใหม่แล้ว

    ทำ 3 อย่าง
    1. แปลง TotalCharges จากข้อความเป็นตัวเลข (ในไฟล์ต้นฉบับมี 11 แถวที่เป็นช่องว่าง)
    2. Total_Addons  = จำนวนบริการเสริมที่ลูกค้าสมัคร (0-6)
    3. AvgMonthlySpend = ค่าใช้จ่ายรวม หารด้วยอายุการใช้งาน (+1 กันหารศูนย์)
    """
    df = df.copy()

    # 1. ทำความสะอาด TotalCharges ให้เป็นตัวเลขเสมอ
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'].astype(str).str.strip(), errors='coerce'
    ).fillna(0)

    # 2. นับบริการเสริม
    df['Total_Addons'] = (df[ADDON_COLS] == 'Yes').sum(axis=1)

    # 3. ค่าใช้จ่ายเฉลี่ยต่อเดือนจริง
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

    return df
