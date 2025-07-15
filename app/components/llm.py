from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
import os
import ssl
import certifi

# Corporate network SSL fix
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()

# Option 1: Small conversational model (DialoGPT-small ~117MB)
def setup_dialogpt_small():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )
    
    return HuggingFacePipeline(pipeline=pipe)



# Option 3: TinyLlama (very small ~1.1GB but more capable)
def setup_tinyllama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",
        max_length=200,
        temperature=0.7,
        do_sample=True,
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Choose your model
llm = setup_dialogpt_small()  # Change this to your preferred option

# Alternative: Direct HuggingFace model without pipeline wrapper
def setup_direct_hf():
    from langchain_community.llms import HuggingFaceHub
    
    # This uses HuggingFace's hosted inference API (requires HF token)
    return HuggingFaceHub(
        repo_id="microsoft/DialoGPT-small",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 200
        }
    )

# If you want to use HuggingFace Hub instead of local model
# llm = setup_direct_hf()

