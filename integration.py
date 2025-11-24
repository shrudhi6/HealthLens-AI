"""
HealthLens AI - Complete Integration Script
End-to-end pipeline: OCR → NLP → ML → Recommendations

This script demonstrates the complete workflow of the HealthLens AI system.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules (ensure they're in the same directory or in PYTHONPATH)
# from modules.ocr_module import MedicalReportOCR, MedicalTextParser
# from modules.ml_classifier import HealthConditionClassifier, RecommendationEngine
# from modules.data_loader import MedicalDatasetLoader


class HealthLensIntegration:
    """
    Complete integration of all HealthLens AI components
    """
    
    def __init__(self):
        """Initialize all components"""
        print("🔬 Initializing HealthLens AI System...")
        
        # Initialize components
        self.ocr_engine = None  # MedicalReportOCR()
        self.text_parser = None  # MedicalTextParser()
        self.ml_classifier = None  # HealthConditionClassifier()
        self.rec_engine = None  # RecommendationEngine()
        
        # For demonstration, we'll use mock components
        self._init_mock_components()
        
        print("✅ All components initialized successfully!")
    
    def _init_mock_components(self):
        """Initialize mock components for demonstration"""
        # These would be replaced with actual imports in production
        self.ocr_engine = MockOCR()
        self.text_parser = MockParser()
        self.ml_classifier = MockClassifier()
        self.rec_engine = MockRecommendations()
    
    def process_report(self, file_path, patient_info=None):
        """
        Complete end-to-end processing of a medical report
        
        Args:
            file_path: Path to medical report (PDF/Image)
            patient_info: Optional patient information dictionary
            
        Returns:
            Complete analysis results with recommendations
        """
        print("\n" + "=" * 70)
        print("HEALTHLENS AI - MEDICAL REPORT ANALYSIS")
        print("=" * 70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'patient_info': patient_info or {},
            'processing_steps': []
        }
        
        # Step 1: OCR Text Extraction
        print("\n📄 Step 1: Extracting text from report...")
        try:
            extracted_text = self.ocr_engine.extract_text(file_path)
            results['extracted_text'] = extracted_text
            results['processing_steps'].append({
                'step': 'OCR',
                'status': 'success',
                'details': f'Extracted {len(extracted_text)} characters'
            })
            print(f"✅ Text extraction complete ({len(extracted_text)} characters)")
        except Exception as e:
            results['processing_steps'].append({
                'step': 'OCR',
                'status': 'error',
                'error': str(e)
            })
            print(f"❌ Error in text extraction: {e}")
            return results
        
        # Step 2: NLP Parsing
        print("\n🔍 Step 2: Parsing medical values...")
        try:
            parsed_data = self.text_parser.parse_report(extracted_text)
            patient_data = self.text_parser.extract_patient_info(extracted_text)
            
            results['parsed_values'] = parsed_data
            results['patient_info'].update(patient_data)
            results['processing_steps'].append({
                'step': 'NLP_Parsing',
                'status': 'success',
                'details': f'Extracted {len(parsed_data)} test values'
            })
            print(f"✅ Parsing complete ({len(parsed_data)} values extracted)")
        except Exception as e:
            results['processing_steps'].append({
                'step': 'NLP_Parsing',
                'status': 'error',
                'error': str(e)
            })
            print(f"❌ Error in parsing: {e}")
            return results
        
        # Step 3: ML Classification
        print("\n🤖 Step 3: Analyzing health conditions...")
        try:
            predictions = self.ml_classifier.predict_conditions(parsed_data)
            results['health_analysis'] = predictions
            results['processing_steps'].append({
                'step': 'ML_Classification',
                'status': 'success',
                'details': f'Analyzed {len(predictions)} conditions'
            })
            print(f"✅ Analysis complete")
            
            # Print detected conditions
            detected = [cond for cond, data in predictions.items() 
                       if data.get('prediction', False)]
            if detected:
                print(f"   ⚠️  Detected conditions: {', '.join(detected)}")
            else:
                print(f"   ✅ All parameters within normal range")
                
        except Exception as e:
            results['processing_steps'].append({
                'step': 'ML_Classification',
                'status': 'error',
                'error': str(e)
            })
            print(f"❌ Error in classification: {e}")
            return results
        
        # Step 4: Generate Recommendations
        print("\n💡 Step 4: Generating personalized recommendations...")
        try:
            recommendations = self.rec_engine.generate_recommendations(
                predictions, parsed_data
            )
            results['recommendations'] = recommendations
            results['processing_steps'].append({
                'step': 'Recommendations',
                'status': 'success',
                'details': 'Generated complete wellness plan'
            })
            print(f"✅ Recommendations generated")
        except Exception as e:
            results['processing_steps'].append({
                'step': 'Recommendations',
                'status': 'error',
                'error': str(e)
            })
            print(f"❌ Error in generating recommendations: {e}")
        
        # Step 5: Calculate Health Score
        print("\n📊 Step 5: Calculating overall health score...")
        try:
            health_score = self._calculate_health_score(parsed_data, predictions)
            results['health_score'] = health_score
            print(f"✅ Health Score: {health_score}/100")
        except Exception as e:
            print(f"⚠️  Could not calculate health score: {e}")
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        return results
    
    def _calculate_health_score(self, test_values, predictions):
        """Calculate overall health score (0-100)"""
        score = 100
        
        # Deduct points for abnormal values
        for test, data in test_values.items():
            if isinstance(data, dict) and data.get('status') in ['low', 'high']:
                score -= 10
        
        # Deduct points for detected conditions
        for condition, data in predictions.items():
            if data.get('prediction', False):
                risk_level = data.get('risk_level', 'Low')
                if risk_level == 'High':
                    score -= 20
                elif risk_level == 'Moderate':
                    score -= 10
                else:
                    score -= 5
        
        return max(0, min(100, score))
    
    def generate_report(self, analysis_results, output_format='json'):
        """
        Generate formatted report from analysis results
        
        Args:
            analysis_results: Results from process_report()
            output_format: 'json', 'html', or 'pdf'
            
        Returns:
            Formatted report
        """
        if output_format == 'json':
            return self._generate_json_report(analysis_results)
        elif output_format == 'html':
            return self._generate_html_report(analysis_results)
        elif output_format == 'pdf':
            return self._generate_pdf_report(analysis_results)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _generate_json_report(self, results):
        """Generate JSON report"""
        return json.dumps(results, indent=2, default=str)
    
    def _generate_html_report(self, results):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HealthLens AI - Medical Report Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #1f77b4; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .normal {{ color: green; }}
                .abnormal {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
                th {{ background: #f0f0f0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 HealthLens AI</h1>
                <p>Medical Report Analysis - {results.get('timestamp', '')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Health Score</h2>
                <h1>{results.get('health_score', 'N/A')}/100</h1>
            </div>
            
            <div class="section">
                <h2>🔬 Test Results</h2>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
        """
        
        # Add test results
        for test, data in results.get('parsed_values', {}).items():
            if isinstance(data, dict):
                status_class = 'normal' if data.get('status') == 'normal' else 'abnormal'
                html += f"""
                    <tr>
                        <td>{test.title()}</td>
                        <td>{data.get('value')} {data.get('unit', '')}</td>
                        <td class="{status_class}">{data.get('status', 'Unknown').upper()}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
        """
        
        # Add recommendations
        recommendations = results.get('recommendations', {})
        if recommendations.get('dietary_plan'):
            html += "<h3>🍎 Dietary Recommendations</h3><ul>"
            for condition, diet in recommendations['dietary_plan'].items():
                html += f"<li><strong>{condition.title()}:</strong><ul>"
                for item in diet.get('include', [])[:3]:
                    html += f"<li>Include: {item}</li>"
                html += "</ul></li>"
            html += "</ul>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_pdf_report(self, results):
        """Generate PDF report (placeholder)"""
        # Would require reportlab or similar library
        return "PDF generation requires additional setup"
    
    def batch_process(self, file_paths):
        """
        Process multiple reports in batch
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of analysis results
        """
        results = []
        
        print(f"\n🔄 Processing {len(file_paths)} reports...")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n--- Processing Report {i}/{len(file_paths)} ---")
            result = self.process_report(file_path)
            results.append(result)
        
        print(f"\n✅ Batch processing complete: {len(results)} reports processed")
        
        return results


# Mock components for demonstration
class MockOCR:
    def extract_text(self, file_path):
        return """
        PATIENT NAME: John Doe
        AGE: 45 Years
        GENDER: Male
        DATE: 15/10/2024
        
        COMPLETE BLOOD COUNT
        Hemoglobin: 12.5 g/dL
        WBC Count: 8.5 ×10³/µL
        Platelets: 250 ×10³/µL
        
        BLOOD GLUCOSE
        Fasting Blood Sugar: 110 mg/dL
        
        LIPID PROFILE
        Total Cholesterol: 220 mg/dL
        Triglycerides: 175 mg/dL
        """


class MockParser:
    def parse_report(self, text):
        return {
            'hemoglobin': {'value': 12.5, 'unit': 'g/dL', 'status': 'low'},
            'wbc': {'value': 8.5, 'unit': '×10³/µL', 'status': 'normal'},
            'platelets': {'value': 250, 'unit': '×10³/µL', 'status': 'normal'},
            'glucose': {'value': 110, 'unit': 'mg/dL', 'status': 'high'},
            'cholesterol': {'value': 220, 'unit': 'mg/dL', 'status': 'high'},
            'triglycerides': {'value': 175, 'unit': 'mg/dL', 'status': 'high'}
        }
    
    def extract_patient_info(self, text):
        return {
            'name': 'John Doe',
            'age': 45,
            'gender': 'Male',
            'date': '15/10/2024'
        }


class MockClassifier:
    def predict_conditions(self, test_results):
        return {
            'anemia': {
                'prediction': True,
                'probability': 0.75,
                'risk_level': 'Moderate'
            },
            'diabetes': {
                'prediction': True,
                'probability': 0.65,
                'risk_level': 'Moderate'
            },
            'cholesterol': {
                'prediction': True,
                'probability': 0.80,
                'risk_level': 'High'
            }
        }


class MockRecommendations:
    def generate_recommendations(self, predictions, test_results):
        return {
            'dietary_plan': {
                'anemia': {
                    'include': ['Iron-rich foods', 'Vitamin C sources', 'Lean meats'],
                    'avoid': ['Excessive coffee', 'Calcium with iron meals']
                },
                'diabetes': {
                    'include': ['Whole grains', 'Vegetables', 'Lean proteins'],
                    'avoid': ['Refined sugars', 'Processed foods']
                }
            },
            'exercise_plan': {
                'anemia': {'activities': ['Light walking', 'Yoga'], 'frequency': '3-4 times/week'},
                'diabetes': {'activities': ['Brisk walking', 'Cycling'], 'frequency': '5 days/week'}
            },
            'lifestyle_changes': [
                'Maintain regular sleep schedule',
                'Stay hydrated',
                'Manage stress'
            ],
            'follow_up': [
                'Repeat tests in 1 month',
                'Consult healthcare provider'
            ]
        }


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHLENS AI - COMPLETE INTEGRATION DEMO")
    print("=" * 70)
    
    # Initialize system
    healthlens = HealthLensIntegration()
    
    # Process single report
    print("\n### Single Report Processing ###")
    results = healthlens.process_report(
        file_path="sample_report.pdf",
        patient_info={'mrn': '12345', 'visit_id': 'V001'}
    )
    
    # Generate JSON report
    print("\n### Generating JSON Report ###")
    json_report = healthlens.generate_report(results, output_format='json')
    print("\nJSON Report Preview:")
    print(json_report[:500] + "...")
    
    # Generate HTML report
    print("\n### Generating HTML Report ###")
    html_report = healthlens.generate_report(results, output_format='html')
    with open('healthlens_report.html', 'w') as f:
        f.write(html_report)
    print("✅ HTML report saved to 'healthlens_report.html'")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Patient: {results['patient_info'].get('name', 'N/A')}")
    print(f"Health Score: {results.get('health_score', 'N/A')}/100")
    print(f"Tests Analyzed: {len(results.get('parsed_values', {}))}")
    print(f"Conditions Detected: {sum(1 for p in results.get('health_analysis', {}).values() if p