"""
Optimized final LLM configuration with better response generation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
import os
import ssl
import certifi
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Corporate network SSL fix
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

def get_hf_token():
    """Get HuggingFace token from environment"""
    token_names = ['HUGGINGFACE_TOKEN', 'HF_TOKEN', 'HUGGINGFACE_API_KEY', 'HF_API_KEY']
    for name in token_names:
        token = os.getenv(name)
        if token and token != 'your_huggingface_token_here':
            return token.strip().strip('"').strip("'")
    return None

def setup_biomistral_7b():
    """Setup BioMistral with optimized parameters for RTX A6000"""
    model_name = "BioMistral/BioMistral-7B"
    
    try:
        print(f"🔄 Loading BioMistral-7B...")
        
        # Check for GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            device = "cuda"
        else:
            print("💻 Using CPU")
            device = "cpu"
        
        # Get HuggingFace token
        token = get_hf_token()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=token,
            trust_remote_code=False
        )
        
        # Load model with optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            trust_remote_code=False,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Move model to device
        if device == "cuda":
            model = model.to(device)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try to use the new HuggingFacePipeline
        try:
            from langchain_huggingface import HuggingFacePipeline
            print("Using updated langchain-huggingface package")
        except ImportError:
            from langchain_community.llms import HuggingFacePipeline
            print("Using langchain-community package (consider upgrading)")
        
        # Create optimized pipeline for better responses
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,  # Generate up to 100 new tokens
            min_new_tokens=10,   # Ensure at least 10 tokens
            temperature=0.8,     # Slightly higher for more creativity
            do_sample=True,
            top_p=0.9,          # Nucleus sampling
            top_k=50,           # Top-k sampling
            repetition_penalty=1.1,  # Reduce repetition
            device=0 if device == "cuda" else -1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print(f"✅ BioMistral loaded successfully on {device.upper()}")
        return HuggingFacePipeline(pipeline=pipe)
        
    except Exception as e:
        print(f"❌ Failed to load BioMistral: {e}")
        return None

def get_llm():
    """Get optimized BioMistral LLM"""
    
    print("🔧 Initializing optimized BioMistral LLM...")
    
    # Load the local model
    llm = setup_biomistral_7b()
    if llm is not None:
        print("✅ Using local BioMistral model")
        return llm
    
    print("❌ Failed to load BioMistral!")
    return None

def test_llm(llm):
    """Test the LLM with various prompts"""
    if llm is None:
        print("❌ No LLM to test")
        return False
    
    try:
        print("🧪 Testing LLM with optimized parameters...")
        
        test_prompts = [
            "Hello! How are you doing today?",
            "What's your favorite programming language?",
            "Can you tell me about artificial intelligence?",
            "What do you think about machine learning?",
            "How can I improve my coding skills?"
        ]
        
        successful_tests = 0
        
        for i, prompt in enumerate(test_prompts):
            try:
                print(f"\n📝 Test {i+1}: {prompt}")
                response = llm.invoke(prompt)
                
                # Check if response is meaningful
                if response and len(response.strip()) > 1:
                    print(f"✅ Response: {response}")
                    successful_tests += 1
                else:
                    print(f"⚠️ Short response: '{response}'")
                    
            except Exception as e:
                print(f"❌ Test {i+1} failed: {e}")
        
        success_rate = successful_tests / len(test_prompts)
        print(f"\n📊 Success rate: {successful_tests}/{len(test_prompts)} ({success_rate:.1%})")
        
        return success_rate >= 0.6  # At least 60% meaningful responses
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def benchmark_performance(llm, num_tests=5):
    """Benchmark LLM performance"""
    if llm is None:
        return
    
    print(f"\n🏃 Performance Benchmark ({num_tests} tests)...")
    
    import time
    
    prompt = "Tell me about the benefits of using Python for data science and machine learning projects."
    times = []
    
    for i in range(num_tests):
        start_time = time.time()
        try:
            response = llm.invoke(prompt)
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            print(f"Test {i+1}: {duration:.2f}s - {len(response)} chars")
        except Exception as e:
            print(f"Test {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Results:")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Min: {min_time:.2f}s")
        print(f"   Max: {max_time:.2f}s")
        
        # GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

# Initialize the LLM
if __name__ == "__main__":
    print("🚀 BioMistral LLM")
    print("=" * 40)
    
    llm = get_llm()
    if llm:
        if test_llm(llm):
            print("\n🎉 LLM is working great!")
            benchmark_performance(llm)
        else:
            print("\n⚠️ LLM working but responses could be better")
    else:
        print("❌ Failed to load LLM")
else:
    llm = get_llm()

# Export the LLM instance
__all__ = ['llm', 'get_llm', 'test_llm', 'benchmark_performance']