"""
ไฟล์ WSGI สำหรับบัญชี PythonAnywhere ชื่อ Jeddilok

วิธีใช้: คัดลอกทั้งไฟล์นี้ ไปวางทับไฟล์ WSGI configuration
ในแท็บ Web ของ PythonAnywhere (ลบของเดิมออกให้หมดก่อน)
"""
import sys

# Linux แยกตัวพิมพ์เล็กพิมพ์ใหญ่ ต้องเป็น Jeddilok ตัว J ใหญ่เท่านั้น
project_home = '/home/Jeddilok/telcoDemo'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# PythonAnywhere มองหาตัวแปรชื่อ application เท่านั้น จึงต้องเปลี่ยนชื่อจาก app
from app import app as application  # noqa: E402
