"""
HealthLens AI - OCR & Text Extraction Module
Extracts text from medical lab reports (PDF/Images)
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
from pdf2image import convert_from_path
import PyPDF2
import io

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class MedicalReportOCR:
    """OCR module for extracting text from medical lab reports"""
    
    def __init__(self):
        pass
    
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        processed = cv2.erode(processed, kernel, iterations=1)
        
        return processed
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text as string
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Preprocess
            processed = self.preprocess_image(image)
            
            # OCR with custom config
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            # Try direct text extraction first
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # If no text found, use OCR
            if not text.strip():
                images = convert_from_path(pdf_path)
                text = ""
                for image in images:
                    processed = self.preprocess_image(image)
                    text += pytesseract.image_to_string(processed)
            
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text(self, file_path):
        """
        Extract text from file (auto-detect PDF or Image)
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text as string
        """
        file_path = str(file_path).lower()
        
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return self.extract_text_from_image(file_path)
        else:
            raise ValueError("Unsupported file format")


class MedicalTextParser:
    """NLP-based parser for medical lab reports"""
    
    def __init__(self):
        # Define test patterns
        self.test_patterns = {
            'hemoglobin': [
                r'hemoglobin[:\s]*(\d+\.?\d*)\s*(g/dl|gm/dl)?',
                r'hb[:\s]*(\d+\.?\d*)\s*(g/dl|gm/dl)?',
                r'hgb[:\s]*(\d+\.?\d*)\s*(g/dl|gm/dl)?'
            ],
            'wbc': [
                r'wbc[:\s]*(\d+\.?\d*)\s*(×10³/µl|x10\^3/ul|thou/cumm)?',
                r'white blood cell[:\s]*(\d+\.?\d*)',
                r'total\s*leucocyte\s*count[:\s]*(\d+\.?\d*)'
            ],
            'rbc': [
                r'rbc[:\s]*(\d+\.?\d*)\s*(mill/cumm|×10⁶/µl)?',
                r'red blood cell[:\s]*(\d+\.?\d*)',
                r'total\s*rbc\s*count[:\s]*(\d+\.?\d*)'
            ],
            'platelets': [
                r'platelet[s]?[:\s]*(\d+\.?\d*)\s*(×10³/µl|thou/cumm)?',
                r'plt[:\s]*(\d+\.?\d*)'
            ],
            'glucose': [
                r'glucose[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'blood\s*sugar[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'fbs[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'random\s*blood\s*sugar[:\s]*(\d+\.?\d*)'
            ],
            'cholesterol': [
                r'cholesterol[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'total\s*cholesterol[:\s]*(\d+\.?\d*)\s*(mg/dl)?'
            ],
            'hdl': [
                r'hdl[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'hdl\s*cholesterol[:\s]*(\d+\.?\d*)'
            ],
            'ldl': [
                r'ldl[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'ldl\s*cholesterol[:\s]*(\d+\.?\d*)'
            ],
            'triglycerides': [
                r'triglyceride[s]?[:\s]*(\d+\.?\d*)\s*(mg/dl)?',
                r'tg[:\s]*(\d+\.?\d*)\s*(mg/dl)?'
            ],
            'hba1c': [
                r'hba1c[:\s]*(\d+\.?\d*)\s*(%)?',
                r'glycated\s*hemoglobin[:\s]*(\d+\.?\d*)'
            ]
        }
        
        # Reference ranges
        self.reference_ranges = {
            'hemoglobin': {'min': 13.0, 'max': 17.0, 'unit': 'g/dL'},
            'wbc': {'min': 4.0, 'max': 11.0, 'unit': '×10³/µL'},
            'rbc': {'min': 4.5, 'max': 5.5, 'unit': '×10⁶/µL'},
            'platelets': {'min': 150, 'max': 400, 'unit': '×10³/µL'},
            'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
            'cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL'},
            'hdl': {'min': 40, 'max': 60, 'unit': 'mg/dL'},
            'ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
            'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'},
            'hba1c': {'min': 4.0, 'max': 5.7, 'unit': '%'}
        }
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\-:/()×µ]', '', text)
        
        return text
    
    def extract_test_value(self, text, test_name):
        """
        Extract a specific test value from text
        
        Args:
            text: Cleaned text from report
            test_name: Name of the test to extract
            
        Returns:
            Extracted value as float or None
        """
        patterns = self.test_patterns.get(test_name, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def parse_report(self, text):
        """
        Parse complete medical report and extract all test values
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Dictionary with test names and values
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract all test values
        results = {}
        for test_name in self.test_patterns.keys():
            value = self.extract_test_value(cleaned_text, test_name)
            if value is not None:
                results[test_name] = {
                    'value': value,
                    'unit': self.reference_ranges[test_name]['unit'],
                    'reference_min': self.reference_ranges[test_name]['min'],
                    'reference_max': self.reference_ranges[test_name]['max'],
                    'status': self.classify_value(test_name, value)
                }
        
        return results
    
    def classify_value(self, test_name, value):
        """
        Classify test value as normal/abnormal
        
        Args:
            test_name: Name of the test
            value: Test value
            
        Returns:
            'normal', 'low', or 'high'
        """
        ref = self.reference_ranges.get(test_name)
        if not ref:
            return 'unknown'
        
        if value < ref['min']:
            return 'low'
        elif value > ref['max']:
            return 'high'
        else:
            return 'normal'
    
    def extract_patient_info(self, text):
        """
        Extract patient information from report
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Dictionary with patient info
        """
        info = {}
        
        # Extract name
        name_pattern = r'patient\s*name[:\s]*([a-z\s]+)'
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            info['name'] = name_match.group(1).strip()
        
        # Extract age
        age_pattern = r'age[:\s]*(\d+)'
        age_match = re.search(age_pattern, text, re.IGNORECASE)
        if age_match:
            info['age'] = int(age_match.group(1))
        
        # Extract gender
        gender_pattern = r'gender[:\s]*(male|female|m|f)'
        gender_match = re.search(gender_pattern, text, re.IGNORECASE)
        if gender_match:
            info['gender'] = gender_match.group(1).strip()
        
        # Extract date
        date_pattern = r'date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            info['date'] = date_match.group(1)
        
        return info


# Example usage
if __name__ == "__main__":
    # Initialize OCR
    ocr = MedicalReportOCR()
    parser = MedicalTextParser()
    
    # Example text (simulated OCR output)
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
    
    # Parse report
    patient_info = parser.extract_patient_info(sample_text)
    test_results = parser.parse_report(sample_text)
    
    print("Patient Information:")
    print(patient_info)
    print("\nTest Results:")
    for test, data in test_results.items():
        print(f"{test}: {data['value']} {data['unit']} - {data['status'].upper()}")