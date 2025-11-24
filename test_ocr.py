"""
Test OCR functionality
Run this to test if OCR is working properly
"""

import sys
import os

print("=" * 70)
print("HealthLens AI - OCR Test Script")
print("=" * 70)

# Test 1: Check if modules exist
print("\n1. Checking if modules exist...")
if os.path.exists('modules/ocr_module.py'):
    print("   ✅ ocr_module.py found")
else:
    print("   ❌ ocr_module.py not found")
    print("   Create modules/ocr_module.py first!")
    sys.exit(1)

# Test 2: Import modules
print("\n2. Testing module imports...")
try:
    sys.path.insert(0, 'modules')
    from ocr_module import MedicalReportOCR, MedicalTextParser
    print("   ✅ Modules imported successfully")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("   Install required packages: pip install pytesseract opencv-python pillow")
    sys.exit(1)

# Test 3: Check Tesseract
print("\n3. Checking Tesseract OCR...")
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"   ✅ Tesseract version: {version}")
except Exception as e:
    print(f"   ❌ Tesseract error: {e}")
    print("   Install Tesseract OCR:")
    print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Linux: sudo apt-get install tesseract-ocr")
    print("   Mac: brew install tesseract")
    sys.exit(1)

# Test 4: Initialize OCR
print("\n4. Initializing OCR engine...")
try:
    ocr = MedicalReportOCR()
    parser = MedicalTextParser()
    print("   ✅ OCR engine initialized")
except Exception as e:
    print(f"   ❌ Initialization error: {e}")
    sys.exit(1)

# Test 5: Test with sample text
print("\n5. Testing text parsing...")
sample_text = """
PATIENT NAME: John Doe
AGE: 45 Years
GENDER: Male
DATE: 15/10/2024

COMPLETE BLOOD COUNT (CBC)

Hemoglobin: 12.5 g/dL (13.0 - 17.0)
WBC Count: 8.5 ×10³/µL (4.0 - 11.0)
RBC Count: 4.8 ×10⁶/µL (4.5 - 5.5)
Platelets: 250 ×10³/µL (150 - 400)

BLOOD GLUCOSE
Fasting Blood Sugar: 110 mg/dL (70 - 100)

LIPID PROFILE
Total Cholesterol: 220 mg/dL (<200)
HDL Cholesterol: 45 mg/dL (>40)
LDL Cholesterol: 140 mg/dL (<100)
Triglycerides: 175 mg/dL (<150)
"""

try:
    # Parse the sample text
    parsed_data = parser.parse_report(sample_text)
    patient_info = parser.extract_patient_info(sample_text)
    
    print(f"   ✅ Parsed {len(parsed_data)} test values")
    print(f"   ✅ Extracted patient info: {patient_info.get('name', 'N/A')}")
    
    print("\n   Extracted values:")
    for test, data in list(parsed_data.items())[:5]:
        if isinstance(data, dict):
            print(f"     • {test}: {data['value']} {data['unit']}")
    
except Exception as e:
    print(f"   ❌ Parsing error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test with image (if available)
print("\n6. Testing image processing...")
if os.path.exists('uploads') and any(os.scandir('uploads')):
    print("   Upload folder contains files - ready for image processing")
else:
    print("   ℹ️  No test images found in uploads/ folder")
    print("   Upload an image to test OCR extraction")

print("\n" + "=" * 70)
print("OCR Test Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. If all tests passed, run: streamlit run app.py")
print("2. Upload a medical report to test full OCR pipeline")
print("3. Check the 'Extract Data from Report' button works")