#!/usr/bin/env python3
"""
Simple test script for DialoGPT-small LLM
"""

import time

def test_dialogpt():
    """Test DialoGPT-small LLM"""
    print("🧪 Testing DialoGPT-small LLM")
    print("=" * 40)
    
    try:
        # Import the LLM
        from llm import get_llm, test_llm
        
        # Load LLM
        print("🔄 Loading LLM...")
        start_time = time.time()
        
        llm = get_llm()
        
        load_time = time.time() - start_time
        print(f"⏱️ Loading time: {load_time:.2f}s")
        
        if llm is None:
            print("❌ Failed to load LLM")
            return False
        
        # Test basic functionality
        success = test_llm(llm)
        if not success:
            return False
        
        # Test multiple queries
        print("\n🔍 Testing multiple queries...")
        
        test_queries = [
            "Hello, how are you?",
            "What is Python?",
            "Tell me about AI.",
            "How does machine learning work?"
        ]
        
        for i, query in enumerate(test_queries):
            try:
                print(f"\n📝 Query {i+1}: {query}")
                
                start_time = time.time()
                response = llm.invoke(query)
                query_time = time.time() - start_time
                
                print(f"⏱️ Response time: {query_time:.2f}s")
                print(f"💬 Response: {response[:100]}...")
                
            except Exception as e:
                print(f"❌ Query {i+1} failed: {e}")
                return False
        
        print(f"\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run DialoGPT test"""
    print("🚀 DialoGPT-small Test Suite")
    print("=" * 50)
    
    success = test_dialogpt()
    
    if success:
        print(f"\n🎉 DialoGPT-small is working perfectly!")
        print("✅ LLM loaded successfully")
        print("✅ Queries processed correctly")
        print("✅ Ready for use in RAG system")
    else:
        print(f"\n❌ DialoGPT-small test failed")
        print("💡 Check your .env file for HUGGINGFACE_TOKEN")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)