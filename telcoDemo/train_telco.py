"""
เทรนโมเดลทำนายการยกเลิกบริการ (Churn) แล้วบันทึกเป็นไฟล์เดียวพร้อมใช้งาน

ต่างจากใน notebook ตรงที่: รวม "การเตรียมข้อมูล + โมเดล" เข้าเป็น Pipeline เดียว
เพื่อให้เว็บส่งข้อมูลดิบ (เช่น 'Month-to-month', 'Yes') เข้ามาได้ตรง ๆ
"""
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import (CAT_FEATURES, NUM_FEATURES, RAW_COLUMNS,
                      add_engineered_features)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


# ---------- 1. โหลดข้อมูล ----------
# หาไฟล์ในโฟลเดอร์ data/ ก่อน (แนบไปกับโปรเจกต์ ใช้ได้แม้เซิร์ฟเวอร์ต่อ Kaggle ไม่ได้)
# ถ้าไม่เจอค่อยโหลดจาก Kaggle
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
local_csv = os.path.join(BASE_DIR, 'data', CSV_NAME)

if os.path.exists(local_csv):
    csv_path = local_csv
    print("ใช้ข้อมูลจากโฟลเดอร์ data/")
else:
    import kagglehub
    csv_path = os.path.join(
        kagglehub.dataset_download("blastchar/telco-customer-churn"), CSV_NAME)
    print("ดาวน์โหลดข้อมูลจาก Kaggle")

df = pd.read_csv(csv_path)
print(f"โหลดข้อมูลแล้ว: {df.shape[0]} แถว {df.shape[1]} คอลัมน์")

# แปลงเฉลย Churn จาก Yes/No เป็น 1/0
y = df['Churn'].map({'No': 0, 'Yes': 1})
# เอาเฉพาะ 19 คอลัมน์ดิบที่หน้าเว็บจะกรอกได้จริง (ตัด customerID และ Churn ทิ้ง)
X = df[RAW_COLUMNS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"แบ่งข้อมูล: train {len(X_train)} แถว / test {len(X_test)} แถว")


# ---------- 2. สร้างท่อเตรียมข้อมูล ----------
# ขั้นแรกของท่อ: สร้างฟีเจอร์เพิ่มจากข้อมูลดิบ
feature_builder = FunctionTransformer(add_engineered_features)

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, NUM_FEATURES),
    ('cat', cat_transformer, CAT_FEATURES),
])


# ---------- 3. เทรนและเปรียบเทียบโมเดล ----------
# ใช้ค่าพารามิเตอร์ที่ดีที่สุดจากการทำ GridSearchCV ใน notebook
candidates = {
    "Logistic Regression": LogisticRegression(
        C=10.0, solver='lbfgs', max_iter=1000,
        class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5, min_samples_leaf=4,
        class_weight='balanced', random_state=42),
}

scoreboard = []
fitted = {}

for name, clf in candidates.items():
    pipe = Pipeline([
        ('features', feature_builder),
        ('prep', preprocessor),
        ('model', clf),
    ])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    scoreboard.append({
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
        "F1 (Churn)": round(f1_score(y_test, y_pred), 4),
    })
    fitted[name] = pipe

board = pd.DataFrame(scoreboard)
print("\n=== เปรียบเทียบโมเดล (บนชุด Test) ===")
print(board.to_string(index=False))

# เลือกผู้ชนะด้วย ROC-AUC เพราะข้อมูลไม่สมดุล (Churn มีแค่ 26.5%)
# ถ้าวัดด้วย Accuracy จะหลอกตัวเอง เพราะเดา "ไม่หนี" ทุกคนก็ได้ 73.5% แล้ว
best_name = board.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
best_pipe = fitted[best_name]
print(f"\nโมเดลที่เลือกใช้งานจริง: {best_name}")
print("\n=== Classification Report ===")
print(classification_report(y_test, best_pipe.predict(X_test),
                            target_names=['ไม่ยกเลิก (0)', 'ยกเลิก (1)']))


# ---------- 4. บันทึกโมเดล ----------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_pipe, os.path.join(MODEL_DIR, 'churn_pipeline.pkl'))

meta = {
    "best_model": best_name,
    "scoreboard": scoreboard,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "churn_rate": round(float(y.mean()) * 100, 2),
}
with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nบันทึกโมเดลไว้ที่ model/churn_pipeline.pkl เรียบร้อยแล้ว")
