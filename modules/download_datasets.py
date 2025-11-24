"""
Automatic Dataset Downloader for HealthLens AI
Downloads real medical datasets from Kaggle

Usage:
    python download_datasets.py --all
    python download_datasets.py --anemia
    python download_datasets.py --diabetes
    python download_datasets.py --cholesterol
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


class DatasetDownloader:
    """
    Automated downloader for medical datasets
    """
    
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        self.datasets = {
            'anemia': {
                'kaggle_id': 'ehababoelnaga/anemia-types-classification',
                'files': ['anemia_data.csv', 'CBCDataset.csv'],
                'description': 'Anemia Types Classification (CBC Data)'
            },
            'diabetes': {
                'kaggle_id': 'uciml/pima-indians-diabetes-database',
                'files': ['diabetes.csv'],
                'description': 'Pima Indians Diabetes Database'
            },
            'cholesterol': {
                'kaggle_id': 'johnsmith88/heart-disease-dataset',
                'files': ['heart.csv'],
                'description': 'Heart Disease Dataset (Cleveland)'
            }
        }
    
    def check_installed_packages(self):
        """Check if required packages are installed"""
        print("\n🔍 Checking dependencies...")
        
        required_packages = ['kaggle', 'opendatasets']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - Not installed")
                missing.append(package)
        
        if missing:
            print(f"\n📦 Installing missing packages: {', '.join(missing)}")
            for package in missing:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"   ✅ Installed {package}")
                except:
                    print(f"   ⚠️  Failed to install {package}")
        
        print("✅ All dependencies ready!\n")
    
    def check_kaggle_setup(self):
        """Check if Kaggle API is configured"""
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        
        if kaggle_config.exists():
            print("✅ Kaggle API configured")
            return True
        else:
            print("\n⚠️  Kaggle API not configured!")
            print("\n📝 Setup Instructions:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. This downloads kaggle.json")
            print("5. Move it to:", kaggle_config.parent)
            print("\n   mkdir -p ~/.kaggle")
            print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("   chmod 600 ~/.kaggle/kaggle.json")
            print("\n💡 Alternative: Use opendatasets (no API token needed)")
            return False
    
    def check_dataset_exists(self, condition):
        """Check if dataset already exists"""
        dataset_info = self.datasets[condition]
        
        for filename in dataset_info['files']:
            file_path = self.data_dir / filename
            if file_path.exists():
                print(f"   ✅ {filename} already exists")
                return True
        
        return False
    
    def download_with_kaggle(self, condition):
        """Download using Kaggle CLI"""
        dataset_info = self.datasets[condition]
        kaggle_id = dataset_info['kaggle_id']
        
        print(f"\n📥 Downloading via Kaggle CLI...")
        print(f"   Dataset: {kaggle_id}")
        
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', kaggle_id,
            '-p', str(self.data_dir),
            '--unzip'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Download successful!")
                return True
            else:
                print(f"   ❌ Download failed")
                print(f"   Error: {result.stderr}")
                return False
        except FileNotFoundError:
            print("   ❌ Kaggle CLI not found")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def download_with_opendatasets(self, condition):
        """Download using opendatasets (no API token needed)"""
        try:
            import opendatasets as od
        except ImportError:
            print("   ❌ opendatasets not installed")
            return False
        
        dataset_info = self.datasets[condition]
        kaggle_id = dataset_info['kaggle_id']
        dataset_url = f'https://www.kaggle.com/datasets/{kaggle_id}'
        
        print(f"\n📥 Downloading via opendatasets...")
        print(f"   URL: {dataset_url}")
        print("   Note: You may be prompted for Kaggle credentials")
        
        try:
            # Download to data directory
            od.download(dataset_url, data_dir=str(self.data_dir))
            print(f"   ✅ Download successful!")
            return True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def download_dataset(self, condition):
        """Download a specific dataset"""
        if condition not in self.datasets:
            print(f"❌ Unknown dataset: {condition}")
            return False
        
        dataset_info = self.datasets[condition]
        
        print("\n" + "="*70)
        print(f"📊 {dataset_info['description'].upper()}")
        print("="*70)
        
        # Check if already exists
        if self.check_dataset_exists(condition):
            print(f"✅ Dataset already downloaded!")
            return True
        
        # Try downloading
        success = False
        
        # Try Kaggle CLI first
        if self.check_kaggle_setup():
            success = self.download_with_kaggle(condition)
        
        # Fall back to opendatasets
        if not success:
            print("\n💡 Trying opendatasets method...")
            success = self.download_with_opendatasets(condition)
        
        if success:
            print(f"\n✅ {condition.upper()} dataset ready!")
        else:
            print(f"\n❌ Failed to download {condition} dataset")
            print("\n📝 Manual Download:")
            print(f"   1. Go to: https://www.kaggle.com/datasets/{dataset_info['kaggle_id']}")
            print(f"   2. Click 'Download'")
            print(f"   3. Extract to: {self.data_dir.absolute()}")
        
        return success
    
    def download_all(self):
        """Download all datasets"""
        print("\n" + "="*70)
        print("🚀 DOWNLOADING ALL MEDICAL DATASETS")
        print("="*70)
        
        results = {}
        
        for condition in ['anemia', 'diabetes', 'cholesterol']:
            results[condition] = self.download_dataset(condition)
        
        # Summary
        print("\n" + "="*70)
        print("📊 DOWNLOAD SUMMARY")
        print("="*70)
        
        for condition, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {condition.capitalize()}: {status}")
        
        total_success = sum(results.values())
        print(f"\n🎉 Downloaded {total_success}/{len(results)} datasets successfully!")
        
        if total_success == len(results):
            print("\n✅ All datasets ready! You can now train models:")
            print("   python ml_classifier.py")
        else:
            print("\n⚠️  Some downloads failed. Check manual download instructions above.")
        
        return total_success == len(results)
    
    def verify_datasets(self):
        """Verify all datasets are present"""
        print("\n" + "="*70)
        print("🔍 VERIFYING DATASETS")
        print("="*70)
        
        all_present = True
        
        for condition, info in self.datasets.items():
            print(f"\n{condition.upper()}:")
            found = False
            
            for filename in info['files']:
                file_path = self.data_dir / filename
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {filename} ({size_mb:.2f} MB)")
                    found = True
                    break
            
            if not found:
                print(f"   ❌ Not found")
                all_present = False
        
        print("\n" + "="*70)
        
        if all_present:
            print("✅ All datasets present and ready!")
        else:
            print("⚠️  Some datasets missing. Run download script:")
            print("   python download_datasets.py --all")
        
        return all_present


def main():
    parser = argparse.ArgumentParser(description='Download HealthLens AI datasets')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--anemia', action='store_true', help='Download anemia dataset')
    parser.add_argument('--diabetes', action='store_true', help='Download diabetes dataset')
    parser.add_argument('--cholesterol', action='store_true', help='Download cholesterol dataset')
    parser.add_argument('--verify', action='store_true', help='Verify datasets exist')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    # Check dependencies
    downloader.check_installed_packages()
    
    if args.verify:
        downloader.verify_datasets()
        return
    
    # Download datasets
    if args.all:
        downloader.download_all()
    else:
        downloaded_any = False
        
        if args.anemia:
            downloader.download_dataset('anemia')
            downloaded_any = True
        
        if args.diabetes:
            downloader.download_dataset('diabetes')
            downloaded_any = True
        
        if args.cholesterol:
            downloader.download_dataset('cholesterol')
            downloaded_any = True
        
        if not downloaded_any:
            print("\n📚 HealthLens AI Dataset Downloader")
            print("\nUsage:")
            print("  python download_datasets.py --all              # Download all datasets")
            print("  python download_datasets.py --anemia           # Download anemia only")
            print("  python download_datasets.py --diabetes         # Download diabetes only")
            print("  python download_datasets.py --cholesterol      # Download cholesterol only")
            print("  python download_datasets.py --verify           # Check if datasets exist")
            print("\nExample:")
            print("  python download_datasets.py --all")


if __name__ == "__main__":
    main()