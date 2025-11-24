HealthLens AI 🩺

AI-powered lab report analyzer using Machine Learning, OCR, and Google Gemini for personalized diet recommendations.

🚀 Features

📄 OCR Extraction: Reads PDF/image lab reports (CBC, Lipid Panel, Glucose).

🤖 ML Predictions: Detects

Anemia

Diabetes

High Cholesterol

🍎 Gemini Diet Plans: Generates personalized nutrition suggestions.

📊 Streamlit Dashboard: Health score, test summary, predictions.

🧠 Machine Learning

Algorithm: Random Forest Classifier

Trained on real Kaggle datasets

Preprocessing: Missing value handling, feature selection, scaling

Metrics: Accuracy, AUC-ROC, F1 Score

📂 Project Structure
HealthLens-AI/
│── app.py
│── requirements.txt
│── modules/
│     ├── ml_classifier.py
│     ├── ocr_module.py
│     ├── ai_recommendation.py
│     ├── download_datasets.py
│── models/       # trained models (ignored)
│── data/         # datasets (ignored)
│── uploads/      # user reports (ignored)

⚙️ Setup
git clone https://github.com/shrudhi6/HealthLens-AI.git
cd HealthLens-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


Create .env:

GOOGLE_API_KEY=your_key_here


Run:

streamlit run app.py

📜 Datasets

CBC (Anemia)

PIMA Diabetes

Cleveland Heart Disease
