"""
HealthLens AI - Production ML Classifier with REAL Medical Datasets
Uses verified Kaggle datasets for Anemia, Diabetes, and Cholesterol detection

Required Datasets:
1. Anemia: diagnosed_cbc_data_v4.csv (ehababoelnaga/anemia-types-classification)
2. Diabetes: diabetes.csv (uciml/pima-indians-diabetes-database)
3. Heart Disease: heart.csv (johnsmith88/heart-disease-dataset)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HealthConditionClassifier:
    """
    Production ML Classifier for health condition detection
    Uses ONLY real medical datasets - NO synthetic data
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}

        # Define condition-specific features
        self.condition_features = {
            'anemia': ['Hemoglobin', 'MCH', 'MCHC', 'MCV'],
            'diabetes': ['Glucose', 'BMI', 'Age', 'BloodPressure'],
            'cholesterol': ['chol', 'thalach', 'age', 'trestbps']
        }

        # Dataset information
        self.datasets_info = {
            'anemia': {
                'filename': 'diagnosed_cbc_data_v4.csv',
                'target_column': 'Result'
            },
            'diabetes': {
                'filename': 'diabetes.csv',
                'target_column': 'Outcome'
            },
            'cholesterol': {
                'filename': 'heart.csv',
                'target_column': 'target'
            }
        }

    def load_anemia_dataset(self):
        """Load REAL anemia dataset (diagnosed_cbc_data_v4.csv)"""
        print("\n" + "=" * 70)
        print("📊 LOADING REAL ANEMIA DATASET (CBC DATA)")
        print("=" * 70)

        data_dir = Path('data')

        possible_files = [
            'diagnosed_cbc_data_v4.csv',  # ✅ actual anemia dataset
            'anemia_data.csv',
            'CBCDataset.csv',
            'anemia.csv'
        ]

        csv_path = None
        for filename in possible_files:
            full_path = data_dir / filename
            if full_path.exists():
                csv_path = full_path
                break

        if not csv_path:
            raise FileNotFoundError("❌ Anemia dataset not found in data/")

        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} samples from {csv_path.name}")
            print(f"   Columns: {list(df.columns)}")

            # Clean column names
            df.columns = df.columns.str.strip()

            # ✅ Rename columns to standard form
            df.rename(columns={'HGB': 'Hemoglobin', 'Diagnosis': 'Result'}, inplace=True)

            # Select relevant CBC features
            feature_candidates = ['Hemoglobin', 'MCH', 'MCHC', 'MCV', 'RBC', 'WBC']
            available_features = [f for f in feature_candidates if f in df.columns]

            if len(available_features) < 3:
                raise ValueError(f"Insufficient features found. Found only: {available_features}")

            print(f"   Using features: {available_features}")

            # Fill missing values with column mean
            df[available_features] = df[available_features].fillna(df[available_features].mean())

            # Extract features
            X = df[available_features].values

            # ✅ Use Diagnosis → Result column as target
            if 'Result' in df.columns:
                # Normalize case and map to binary
                df['Result'] = df['Result'].astype(str).str.lower()
                y = df['Result'].map({
                    'anemia': 1,
                    'anemic': 1,
                    'iron deficiency anemia': 1,
                    'sickle cell anemia': 1,
                    'thalassemia': 1,
                    'normal': 0,
                    'healthy': 0
                })
                # Fallback rule if unmapped
                y = y.fillna((df['Hemoglobin'] < 12.5).astype(int))
            else:
                # If no target, use rule-based anemia detection
                y = (df['Hemoglobin'] < 12.5).astype(int)

            # Clean NaN targets
            valid_idx = ~pd.isna(y)
            X = X[valid_idx]
            y = y[valid_idx].values

            print(f"   Anemia cases: {y.sum()} / {len(y)} ({y.mean():.1%})")
            print(f"   Features shape: {X.shape}")

            # Store feature names
            self.condition_features['anemia'] = available_features
            return X, y

        except Exception as e:
            print(f"❌ Error loading anemia dataset: {e}")
            raise


    def load_diabetes_dataset(self):
        """Load REAL Pima Indians Diabetes Dataset"""
        print("\n" + "=" * 70)
        print("📊 LOADING REAL DIABETES DATASET (PIMA INDIANS)")
        print("=" * 70)

        data_dir = Path('data')
        csv_path = data_dir / 'diabetes.csv'
        if not csv_path.exists():
            raise FileNotFoundError("❌ diabetes.csv not found in data/")

        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} samples from {csv_path.name}")
        print(f"   Columns: {list(df.columns)}")

        features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        for col in features:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

        X = df[features].values
        y = df['Outcome'].values

        print(f"   Diabetes cases: {y.sum()} / {len(y)} ({y.mean():.1%})")
        print(f"   Features shape: {X.shape}")

        self.condition_features['diabetes'] = features
        return X, y

    def load_cholesterol_dataset(self):
        """Load REAL Heart Disease Dataset (Cleveland)"""
        print("\n" + "=" * 70)
        print("📊 LOADING REAL HEART DISEASE DATASET (CLEVELAND)")
        print("=" * 70)

        data_dir = Path('data')
        csv_path = data_dir / 'heart.csv'
        if not csv_path.exists():
            raise FileNotFoundError("❌ heart.csv not found in data/")

        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} samples from {csv_path.name}")
        print(f"   Columns: {list(df.columns)}")

        features = ['chol', 'thalach', 'age', 'trestbps']
        available_features = [f for f in features if f in df.columns]

        df[available_features] = df[available_features].fillna(df[available_features].median())
        X = df[available_features].values
        y = (df['chol'] > 200).astype(int)

        print(f"   High cholesterol cases: {y.sum()} / {len(y)} ({y.mean():.1%})")
        print(f"   Features shape: {X.shape}")

        self.condition_features['cholesterol'] = available_features
        return X, y

    def train_models(self, test_size=0.2):
        """Train Random Forest models"""
        print("\n" + "=" * 70)
        print("🌲 TRAINING MODELS ON REAL MEDICAL DATA")
        print("=" * 70)

        conditions = ['anemia', 'diabetes', 'cholesterol']

        for condition in conditions:
            print(f"\n{'='*70}")
            print(f"Training {condition.upper()} Model")
            print(f"{'='*70}")

            try:
                if condition == 'anemia':
                    X, y = self.load_anemia_dataset()
                elif condition == 'diabetes':
                    X, y = self.load_diabetes_dataset()
                elif condition == 'cholesterol':
                    X, y = self.load_cholesterol_dataset()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[condition] = scaler

                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba)
                cm = confusion_matrix(y_test, y_pred)

                print(f"\n✅ MODEL PERFORMANCE for {condition.upper()}")
                print(f"   Accuracy       : {accuracy:.4f}")
                print(f"   Precision      : {precision:.4f}")
                print(f"   Recall (Sensitivity): {recall:.4f}")
                print(f"   F1 Score       : {f1:.4f}")
                print(f"   AUC-ROC        : {auc:.4f}")
                print(f"   Confusion Matrix:\n{cm}")

                print(f"\n✅ MODEL PERFORMANCE for {condition.upper()}")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   AUC-ROC: {auc:.4f}")
                print(f"   Confusion Matrix:\n{cm}")

                feature_names = self.condition_features[condition]
                importances = model.feature_importances_
                for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
                    print(f"   {feat}: {imp:.4f}")

                self.models[condition] = model
                self.feature_importances[condition] = dict(zip(feature_names, importances))
                print(f"✅ {condition.upper()} model trained successfully!")

            except FileNotFoundError as e:
                print(f"❌ {condition} dataset not found: {e}")
                continue

        print("\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE! {len(self.models)} models trained")
        print("=" * 70)

    def save_models(self, path='models/'):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'{path}{name}_rf_model.pkl')
            print(f"✅ Saved {name}_rf_model.pkl")
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{path}{name}_scaler.pkl')
            print(f"✅ Saved {name}_scaler.pkl")
        metadata = {
            'condition_features': self.condition_features,
            'feature_importances': self.feature_importances
        }
        with open(f'{path}model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ All models and metadata saved!")

    def load_models(self, path="models/"):
        """Load all pre-trained models and scalers"""
        import os

        base_path = os.path.join(os.getcwd(), path)
        print(f"\n📂 Loading models from: {base_path}")

        if not os.path.exists(base_path):
            raise FileNotFoundError("Models directory not found!")

        for condition in ["anemia", "diabetes", "cholesterol"]:
            model_path = os.path.join(base_path, f"{condition}_rf_model.pkl")
            scaler_path = os.path.join(base_path, f"{condition}_scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[condition] = joblib.load(model_path)
                self.scalers[condition] = joblib.load(scaler_path)
                print(f"✅ Loaded {condition} model and scaler")
            else:
                print(f"⚠️ Missing files for {condition}")

        metadata_path = os.path.join(base_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.condition_features = metadata.get("condition_features", self.condition_features)
            self.feature_importances = metadata.get("feature_importances", {})
            print("✅ Loaded model metadata")

        if len(self.models) == 0:
            raise FileNotFoundError("No trained models found in 'models/'")

        print("\n✅ All pre-trained models loaded successfully!")

    def predict_conditions(self, test_results):
        """
        Predicts health conditions based on input test results.
        Accepts a dictionary of lab values and returns predictions for each condition.
        """
        if not self.models:
            raise RuntimeError("Models not loaded. Please call load_models() first.")

        predictions = {}

        for condition, model in self.models.items():
            features = self.condition_features.get(condition, [])
            if not features:
                print(f"⚠️ No features defined for {condition}. Skipping.")
                continue

            # Collect input values for each feature
            try:
                input_data = np.array([[float(test_results.get(f.lower(), test_results.get(f, 0))) for f in features]])
            except Exception as e:
                print(f"⚠️ Error preparing features for {condition}: {e}")
                continue

            # Scale the input
            scaler = self.scalers.get(condition)
            if not scaler:
                print(f"⚠️ Missing scaler for {condition}")
                continue

            input_scaled = scaler.transform(input_data)

            # Make prediction
            prob = model.predict_proba(input_scaled)[0, 1]
            pred = int(prob >= 0.5)

            predictions[condition] = {
                "prediction": pred,
                "probability": prob
            }

            print(f"🩺 {condition.upper()} → {'Detected' if pred else 'Normal'} ({prob:.2%})")

        return predictions



# Main execution
if __name__ == "__main__":
    classifier = HealthConditionClassifier()
    classifier.train_models()
    classifier.save_models()
