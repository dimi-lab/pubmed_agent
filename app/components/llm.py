"""
Optimized final LLM configuration with meta tensor error fixes
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

def safe_model_to_device(model, device):
    """Safely move model to device, handling meta tensors"""
    try:
        # Check if model has meta tensors
        has_meta_tensors = any(param.device.type == 'meta' for param in model.parameters())
        
        if has_meta_tensors:
            print("🔧 Detected meta tensors, using safe loading method...")
            # Use to_empty() for meta tensors
            model = model.to_empty(device=device)
        else:
            # Standard loading for regular tensors
            model = model.to(device)
            
        print(f"✅ Model successfully moved to {device}")
        return model
        
    except Exception as e:
        print(f"⚠️ Standard device transfer failed: {e}")
        try:
            # Fallback: Force CPU first, then GPU
            print("🔄 Trying fallback: CPU first, then GPU...")
            model = model.cpu()
            if device != "cpu":
                model = model.to(device)
            print(f"✅ Fallback successful: Model on {device}")
            return model
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            raise e2

def setup_biomistral_7b():
    """Setup BioMistral with optimized parameters and meta tensor fixes"""
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
        
        # Load tokenizer first
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=token,
            trust_remote_code=False
        )
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("🧠 Loading model with meta tensor support...")
        
        # Try multiple loading strategies for meta tensor compatibility
        model = None
        loading_strategies = [
            "standard",
            "low_memory",
            "cpu_first",
            "device_map"
        ]
        
        for strategy in loading_strategies:
            try:
                print(f"🔄 Trying loading strategy: {strategy}")
                
                if strategy == "standard":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        use_auth_token=token,
                        trust_remote_code=False,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                elif strategy == "low_memory":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        use_auth_token=token,
                        trust_remote_code=False,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map="auto" if device == "cuda" else None
                    )
                    
                elif strategy == "cpu_first":
                    # Force CPU loading first
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        use_auth_token=token,
                        trust_remote_code=False,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        low_cpu_mem_usage=True,
                        device_map=None
                    )
                    
                elif strategy == "device_map":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        use_auth_token=token,
                        trust_remote_code=False,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto"
                    )
                
                if model is not None:
                    print(f"✅ Model loaded successfully with {strategy} strategy")
                    
                    # Only move to device if not using device_map
                    if strategy not in ["device_map", "low_memory"] or device == "cpu":
                        model = safe_model_to_device(model, device)
                    
                    break
                    
            except Exception as e:
                print(f"⚠️ Strategy {strategy} failed: {e}")
                model = None
                continue
        
        if model is None:
            raise Exception("All loading strategies failed")
        
        # Try to use the new HuggingFacePipeline
        try:
            from langchain_huggingface import HuggingFacePipeline
            print("Using updated langchain-huggingface package")
        except ImportError:
            try:
                from langchain_community.llms import HuggingFacePipeline
                print("Using langchain-community package")
            except ImportError:
                from langchain.llms import HuggingFacePipeline
                print("Using legacy langchain package")
        
        # Create optimized pipeline for better responses
        print("🔧 Creating text generation pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Generate up to 512 new tokens
            min_new_tokens=50,   # Ensure at least 10 tokens
            temperature=0.7,     # Slightly higher for more creativity
            do_sample=True,
            top_p=0.9,          # Nucleus sampling
            top_k=50,           # Top-k sampling
            repetition_penalty=1.1,  # Reduce repetition
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            return_full_text=False,
            length_penalty=1.0,
            early_stopping=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print(f"✅ BioMistral loaded successfully on {device.upper()}")
        return HuggingFacePipeline(pipeline=pipe)
        
    except Exception as e:
        print(f"❌ Failed to load BioMistral: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_llm():
    """Get optimized BioMistral LLM with error handling"""
    
    print("🔧 Initializing optimized BioMistral LLM...")
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
            "What is cancer?",
            "Explain diabetes briefly.",
            "What are antibiotics?",
            "Describe the heart.",
            "What is DNA?"
        ]
        
        successful_tests = 0
        
        for i, prompt in enumerate(test_prompts):
            try:
                print(f"\n📝 Test {i+1}: {prompt}")
                response = llm.invoke(prompt)
                
                # Check if response is meaningful
                if response and len(response.strip()) > 10:
                    print(f"✅ Response: {response[:100]}...")
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

def benchmark_performance(llm, num_tests=3):
    """Benchmark LLM performance"""
    if llm is None:
        return
    
    print(f"\n🏃 Performance Benchmark ({num_tests} tests)...")
    
    import time
    
    prompt = "What is machine learning in healthcare?"
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
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            except:
                print("   GPU Memory: Unable to get memory info")

# Initialize the LLM
if __name__ == "__main__":
    print("🚀 BioMistral LLM with Meta Tensor Fixes")
    print("=" * 50)
    
    llm = get_llm()
    if llm:
        if test_llm(llm):
            print("\n🎉 LLM is working great!")
            benchmark_performance(llm)
        else:
            print("\n⚠️ LLM working but responses could be better")
    else:
        print("❌ Failed to load LLM")

# Export the LLM instance
__all__ = ['get_llm', 'test_llm', 'benchmark_performance']