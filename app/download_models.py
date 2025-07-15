# download_models.py - Run this script separately to pre-download models

import os
import ssl
import requests
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL verification for downloads
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def download_model_safely(model_name):
    """Download model with SSL fixes"""
    try:
        print(f"📥 Downloading {model_name}...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=False,
            trust_remote_code=False
        )
        print(f"✅ Downloaded tokenizer for {model_name}")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=False,
            trust_remote_code=False
        )
        print(f"✅ Downloaded model for {model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def main():
    """Download all models"""
    models = [
        "distilgpt2",
        "microsoft/DialoGPT-small"
    ]
    
    for model in models:
        success = download_model_safely(model)
        if success:
            print(f"✅ {model} ready for offline use")
        else:
            print(f"❌ Failed to download {model}")
        print("-" * 50)

if __name__ == "__main__":
    main()