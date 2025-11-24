"""
HealthLens AI - Dataset Loader & Preprocessor
Handles loading and preprocessing of medical datasets (Kaggle, MIMIC-IV, etc.)

Requirements:
pip install pandas numpy scikit-learn kaggle
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json


class MedicalDatasetLoader:
    """
    Load and preprocess medical datasets from various sources
    """
    
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_kaggle_cbc_dataset(self, file_path):
        """
        Load CBC (Complete Blood Count) dataset from Kaggle
        
        Common Kaggle datasets:
        - Blood Test Analysis Dataset
        - Medical Lab Test Results
        - Healthcare Analytics Dataset
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"\nColumns: {list(df.columns)}")
            
            self.data = df
            return df
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def load_mimic_dataset(self, file_path):
        """
        Load MIMIC-IV dataset (PhysioNet)
        
        MIMIC-IV is a large, freely-available database comprising 
        de-identified health data from patients admitted to ICUs
        
        Args:
            file_path: Path to MIMIC CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            # MIMIC datasets are usually large, so we might want to read in chunks
            df = pd.read_csv(file_path, low_memory=False)
            print(f"✅ MIMIC Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            self.data = df
            return df
        except Exception as e:
            print(f"❌ Error loading MIMIC dataset: {e}")
            return None
    
    def create_sample_dataset(self, n_samples=5000, save_path=None):
        """
        Create a synthetic medical dataset for testing
        
        Args:
            n_samples: Number of samples to generate
            save_path: Optional path to save the dataset
            
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(42)
        
        # Generate synthetic patient data
        data = {
            'patient_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'hemoglobin': np.random.normal(14, 2.5, n_samples),
            'wbc': np.random.normal(7, 2.5, n_samples),
            'rbc': np.random.normal(5, 0.7, n_samples),
            'platelets': np.random.normal(250, 70, n_samples),
            'mcv': np.random.normal(88, 8, n_samples),
            'mch': np.random.normal(30, 3, n_samples),
            'mchc': np.random.normal(34, 2, n_samples),
            'glucose': np.random.normal(100, 25, n_samples),
            'cholesterol': np.random.normal(200, 45, n_samples),
            'hdl': np.random.normal(50, 15, n_samples),
            'ldl': np.random.normal(120, 35, n_samples),
            'triglycerides': np.random.normal(150, 50, n_samples),
            'hba1c': np.random.normal(5.5, 1.2, n_samples),
            'creatinine': np.random.normal(1.0, 0.3, n_samples),
            'urea': np.random.normal(30, 10, n_samples),
            'alt': np.random.normal(25, 15, n_samples),
            'ast': np.random.normal(28, 12, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target labels
        df['has_anemia'] = (df['hemoglobin'] < 13.0).astype(int)
        df['has_diabetes'] = ((df['glucose'] > 126) | (df['hba1c'] > 6.5)).astype(int)
        df['has_high_cholesterol'] = ((df['cholesterol'] > 240) | (df['ldl'] > 160)).astype(int)
        df['has_kidney_issue'] = (df['creatinine'] > 1.3).astype(int)
        df['has_liver_issue'] = ((df['alt'] > 40) | (df['ast'] > 40)).astype(int)
        
        # Ensure realistic bounds
        df['hemoglobin'] = df['hemoglobin'].clip(5, 20)
        df['wbc'] = df['wbc'].clip(2, 20)
        df['platelets'] = df['platelets'].clip(50, 500)
        df['glucose'] = df['glucose'].clip(50, 300)
        
        self.data = df
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✅ Synthetic dataset saved to {save_path}")
        
        print(f"✅ Generated synthetic dataset: {df.shape[0]} samples")
        print(f"\nLabel distribution:")
        print(f"  Anemia: {df['has_anemia'].sum()} ({df['has_anemia'].mean():.1%})")
        print(f"  Diabetes: {df['has_diabetes'].sum()} ({df['has_diabetes'].mean():.1%})")
        print(f"  High Cholesterol: {df['has_high_cholesterol'].sum()} ({df['has_high_cholesterol'].mean():.1%})")
        
        return df
    
    def preprocess_data(self, df=None, target_columns=None):
        """
        Preprocess medical data for ML training
        
        Args:
            df: DataFrame to preprocess (uses self.data if None)
            target_columns: List of target column names
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data to preprocess. Load a dataset first.")
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in target_columns if target_columns else []:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Separate features and targets
        if target_columns is None:
            target_columns = ['has_anemia', 'has_diabetes', 'has_high_cholesterol']
        
        # Select numeric feature columns (exclude IDs and targets)
        exclude_cols = ['patient_id'] + target_columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_columns] if len(target_columns) > 1 else df[target_columns[0]]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(target_columns) == 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Data preprocessed:")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        print(f"  Testing samples: {X_test_scaled.shape[0]}")
        print(f"  Features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_statistics(self):
        """Get statistical summary of features"""
        if self.data is None:
            print("❌ No data loaded")
            return None
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = self.data[numeric_cols].describe()
        
        return stats
    
    def check_data_quality(self):
        """Check data quality and report issues"""
        if self.data is None:
            print("❌ No data loaded")
            return
        
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        # Missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️  Missing Values:")
            print(missing[missing > 0])
        else:
            print("\n✅ No missing values")
        
        # Duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\n{'⚠️' if duplicates > 0 else '✅'}  Duplicate rows: {duplicates}")
        
        # Outliers (using IQR method)
        print("\n📊 Outlier Detection (IQR method):")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | 
                       (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"  {col}: {outliers} outliers ({outliers/len(self.data)*100:.1f}%)")
        
        print("\n" + "=" * 60)
    
    def export_processed_data(self, output_path='processed_data.csv'):
        """Export processed data"""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"✅ Data exported to {output_path}")
        else:
            print("❌ No data to export")


class DataAugmentation:
    """
    Data augmentation techniques for medical datasets
    """
    
    @staticmethod
    def add_noise(data, noise_level=0.05):
        """
        Add Gaussian noise to numeric columns
        
        Args:
            data: DataFrame
            noise_level: Standard deviation of noise (as fraction of data std)
            
        Returns:
            Augmented DataFrame
        """
        augmented = data.copy()
        numeric_cols = augmented.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            noise = np.random.normal(0, data[col].std() * noise_level, len(data))
            augmented[col] = augmented[col] + noise
        
        return augmented
    
    @staticmethod
    def oversample_minority_class(X, y, target_ratio=0.5):
        """
        Oversample minority class using SMOTE-like technique
        
        Args:
            X: Feature matrix
            y: Target labels
            target_ratio: Desired ratio of minority class
            
        Returns:
            Balanced X and y
        """
        from sklearn.utils import resample
        
        # Identify minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        
        # Separate classes
        X_minority = X[y == minority_class]
        X_majority = X[y != minority_class]
        y_minority = y[y == minority_class]
        y_majority = y[y != minority_class]
        
        # Calculate number of samples needed
        n_minority_samples = int(len(X_majority) * target_ratio / (1 - target_ratio))
        
        # Oversample minority class
        X_minority_upsampled, y_minority_upsampled = resample(
            X_minority, y_minority,
            n_samples=n_minority_samples,
            random_state=42
        )
        
        # Combine
        X_balanced = np.vstack([X_majority, X_minority_upsampled])
        y_balanced = np.concatenate([y_majority, y_minority_upsampled])
        
        return X_balanced, y_balanced


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("HealthLens AI - Dataset Loader & Preprocessor")
    print("=" * 60)
    
    # Initialize loader
    loader = MedicalDatasetLoader()
    
    # Create synthetic dataset
    print("\n1. Creating Synthetic Dataset...")
    df = loader.create_sample_dataset(n_samples=2000, save_path='synthetic_medical_data.csv')
    
    # Check data quality
    print("\n2. Checking Data Quality...")
    loader.check_data_quality()
    
    # Get statistics
    print("\n3. Feature Statistics:")
    stats = loader.get_feature_statistics()
    print(stats[['hemoglobin', 'glucose', 'cholesterol']].round(2))
    
    # Preprocess data
    print("\n4. Preprocessing Data...")
    X_train, X_test, y_train, y_test = loader.preprocess_data(
        target_columns=['has_anemia', 'has_diabetes', 'has_high_cholesterol']
    )
    
    print(f"\n✅ Data ready for training!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Data augmentation example
    print("\n5. Data Augmentation Example...")
    augmenter = DataAugmentation()
    df_augmented = augmenter.add_noise(df.head(100), noise_level=0.03)
    print(f"   Original data: {len(df.head(100))} samples")
    print(f"   Augmented data: {len(df_augmented)} samples")
    
    print("\n" + "=" * 60)
    print("Dataset processing complete!")
    print("=" * 60)