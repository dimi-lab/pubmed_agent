import streamlit as st
import os
import sys
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PUBMED_EMAIL="gnanaolivu.rohandavid@mayo.com"
PUBMED_API_KEY="f9ecce69a73048b82ca4747f51d6dec61408"
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['POSTHOG_HOST'] = ''

def initialize_system_components() -> Dict[str, Any]:
    """Initialize all system components with comprehensive error handling"""
    components = {
        'ai_model': None,
        'retriever': None,
        'rag_system': None,
        'status': {},
        'errors': {}
    }

    # 1. Initialize BioMistral AI Model
    try:
        # Import the optimized BioMistral LLM
        from components.llm import get_llm, test_llm
        
        st.info("🔄 Loading BioMistral-7B model...")
        ai_model = get_llm()
        
        if ai_model is not None:
            # Test the model with a longer response
            try:
                test_prompt = "Explain what cancer is in simple terms."
                test_response = ai_model.invoke(test_prompt)
                
                if test_response and len(test_response.strip()) > 5:
                    components['ai_model'] = ai_model
                    components['status']['ai'] = "✅ AI Assistant: Ready (BioMistral-7B on RTX A6000)"
                    logger.info(f"BioMistral-7B model initialized and tested successfully. Test response: {test_response[:100]}...")
                else:
                    components['errors']['ai'] = f"Model loaded but test response was too short: '{test_response}'"
                    components['status']['ai'] = "⚠️ AI Assistant: Loaded but giving short responses"
                    components['ai_model'] = ai_model  # Still use it
                    logger.warning(f"BioMistral-7B giving short responses: '{test_response}'")
            except Exception as test_error:
                components['errors']['ai'] = f"Model test failed: {str(test_error)}"
                components['status']['ai'] = f"⚠️ AI Assistant: Loaded but untested - {str(test_error)}"
                components['ai_model'] = ai_model  # Still use it
        else:
            components['errors']['ai'] = "Failed to load BioMistral-7B model"
            components['status']['ai'] = "❌ AI Assistant: Failed to load BioMistral-7B model"
            
    except ImportError as e:
        components['errors']['ai'] = f"Import error: {str(e)}"
        components['status']['ai'] = f"❌ AI Assistant: Module not found - {str(e)}"
        logger.error(f"Failed to import BioMistral-7B model: {e}")
    except Exception as e:
        components['errors']['ai'] = str(e)
        components['status']['ai'] = f"❌ AI Assistant: Failed - {str(e)}"
        logger.error(f"Failed to initialize BioMistral-7B model: {e}")

    # 2. Initialize PubMed Retriever (Optional)
    try:
        from backend.abstract_retrieval.pubmed_retriever import create_configured_pubmed_retriever
        components['retriever'] = create_configured_pubmed_retriever(max_results=20)
        components['status']['pubmed'] = "✅ PubMed Retriever: Ready"
        logger.info("PubMed retriever initialized successfully")
    except ImportError as e:
        components['errors']['pubmed'] = f"Import error: {str(e)}"
        components['status']['pubmed'] = "⚠️ PubMed Retriever: Module not found (using AI-only mode)"
        logger.warning(f"PubMed retriever not available: {e}")
    except Exception as e:
        components['errors']['pubmed'] = str(e)
        components['status']['pubmed'] = f"❌ PubMed Retriever: Failed - {str(e)}"
        logger.error(f"Failed to initialize PubMed retriever: {e}")

    # 3. Initialize RAG System (Optional)
    try:
        if components['ai_model']:  # Only need AI model for basic RAG
            from backend.rag_pipeline.chromadb_rag import ChromaDbRag
            from backend.rag_pipeline.embeddings import embeddings
            from backend.data_repository.local_storage import LocalJSONStore

            # Check if embeddings are available
            if embeddings is not None:
                # Initialize storage and RAG
                storage = LocalJSONStore("./data/storage")
                rag_system = ChromaDbRag("./data/vector_store", embeddings)

                components['storage'] = storage
                components['rag_system'] = rag_system
                components['status']['rag'] = "✅ RAG System: Ready (BioMistral-7B + Embeddings)"
                logger.info("RAG system initialized successfully with BioMistral")
            else:
                components['errors']['rag'] = "Embeddings not available"
                components['status']['rag'] = "⚠️ RAG System: Embeddings not available"
    except Exception as e:
        components['errors']['rag'] = str(e)
        components['status']['rag'] = f"⚠️ RAG System: Not available - {str(e)}"
        logger.warning(f"RAG system not available: {e}")

    return components


def enhance_prompt_for_biomistral(question: str, length_preference: str = "Medium") -> str:
    """Enhanced prompting specifically optimized for BioMistral"""
    
    # BioMistral works better with conversational, medical context
    if length_preference == "Short":
        length_instruction = "Provide a concise but informative medical answer (2-3 sentences)."
    elif length_preference == "Long":
        length_instruction = "Provide a comprehensive medical explanation with details, mechanisms, and clinical relevance."
    else:
        length_instruction = "Provide a clear, informative medical answer with appropriate detail."
    
    # Enhanced prompt structure for BioMistral
    enhanced_prompt = f"""You are BioMistral, an expert medical AI assistant with deep knowledge of biomedical research, clinical medicine, and life sciences. {length_instruction}

Medical Question: {question}

Please provide an evidence-based response that includes:
- Clear explanation of relevant concepts
- Scientific mechanisms when applicable  
- Clinical significance or applications
- Current research status if relevant

Medical Response:"""
    
    return enhanced_prompt


def get_BioMistral_response(ai_model, question: str, length_preference: str = "Medium", max_retries: int = 3) -> str:
    """Enhanced BioMistral response generation with better prompting"""
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                # First attempt: Enhanced medical prompt
                prompt = enhance_prompt_for_biomistral(question, length_preference)
            elif attempt == 1:
                # Second attempt: Simple medical conversation
                prompt = f"""As a medical AI assistant, please answer this question about {question}

Provide a detailed medical explanation:"""
            elif attempt == 2:
                # Third attempt: Direct biomedical prompt
                prompt = f"""Question: {question}

Medical Answer:"""
            else:
                # Final attempt: Very simple
                prompt = question
            
            logger.info(f"BioMistral attempt {attempt + 1} with enhanced prompt...")
            
            # Use the model
            response = ai_model.invoke(prompt)
            
            # Clean up response
            if response:
                response = response.strip()
                
                # Remove common prompt artifacts
                if response.startswith(("Medical Response:", "Medical Answer:", "Answer:")):
                    response = response.split(":", 1)[1].strip()
                
                # Remove conversation formatting
                if "Human:" in response or "Assistant:" in response:
                    lines = response.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        if not line.strip().startswith(('Human:', 'Assistant:', 'Question:')):
                            cleaned_lines.append(line)
                    response = '\n'.join(cleaned_lines).strip()
                
                # Check if response is meaningful
                if len(response) > 10 and response not in [".", "?", "!", "...", "N/A"]:
                    logger.info(f"BioMistral successful response (attempt {attempt + 1}): {response[:100]}...")
                    return response
                else:
                    logger.warning(f"BioMistral short response (attempt {attempt + 1}): '{response}'")
            
        except Exception as e:
            logger.error(f"BioMistral attempt {attempt + 1} failed: {e}")
    
    # If all attempts fail, return a helpful message
    return f"""I apologize, but I'm experiencing difficulty generating a proper response to your question: "{question}"

This may be due to:
- Model configuration issues
- Prompt formatting problems  
- Resource constraints

Please try:
1. Rephrasing your question more simply
2. Using shorter, more specific queries
3. Checking the system status
4. Using a different research mode

If the problem persists, please check the system logs for more details."""


# STEP 3B: Add better error handling to your initialization
def initialize_system_components_enhanced() -> Dict[str, Any]:
    """Enhanced system initialization with better error handling"""
    components = {
        'ai_model': None,
        'retriever': None,
        'rag_system': None,
        'storage': None,
        'status': {},
        'errors': {}
    }

    # 1. Initialize BioMistral AI Model with enhanced testing
    try:
        from components.llm import get_llm
        
        st.info("🔄 Loading BioMistral model...")
        ai_model = get_llm()
        
        if ai_model is not None:
            # Enhanced model testing with better prompts
            try:
                test_questions = [
                    "What is cancer?",
                    "Explain diabetes briefly.",
                    "What are antibiotics?"
                ]
                
                success_count = 0
                for test_q in test_questions:
                    test_response = get_BioMistral_response(ai_model, test_q, "Short", max_retries=1)
                    if test_response and len(test_response.strip()) > 15:
                        success_count += 1
                
                if success_count >= 2:  # At least 2 out of 3 tests should pass
                    components['ai_model'] = ai_model
                    components['status']['ai'] = f"✅ AI Assistant: Ready (BioMistral - {success_count}/3 tests passed)"
                    logger.info(f"BioMistral model initialized successfully ({success_count}/3 tests passed)")
                else:
                    components['errors']['ai'] = f"Model giving poor responses ({success_count}/3 tests passed)"
                    components['status']['ai'] = f"⚠️ AI Assistant: Working but responses may be short ({success_count}/3)"
                    components['ai_model'] = ai_model  # Still use it
                    
            except Exception as test_error:
                components['errors']['ai'] = f"Model test failed: {str(test_error)}"
                components['status']['ai'] = f"⚠️ AI Assistant: Loaded but untested - {str(test_error)}"
                components['ai_model'] = ai_model  # Still use it
        else:
            components['errors']['ai'] = "Failed to load BioMistral model"
            components['status']['ai'] = "❌ AI Assistant: Failed to load"
            
    except Exception as e:
        components['errors']['ai'] = str(e)
        components['status']['ai'] = f"❌ AI Assistant: Failed - {str(e)}"
        logger.error(f"Failed to initialize BioMistral: {e}")

    # 2. Initialize PubMed Retriever
    try:
        from backend.abstract_retrieval.pubmed_retriever import create_configured_pubmed_retriever
        components['retriever'] = create_configured_pubmed_retriever(max_results=20)
        components['status']['pubmed'] = "✅ PubMed Retriever: Ready"
        logger.info("PubMed retriever initialized successfully")
    except Exception as e:
        components['errors']['pubmed'] = str(e)
        components['status']['pubmed'] = f"⚠️ PubMed Retriever: {str(e)}"
        logger.warning(f"PubMed retriever not available: {e}")

    # 3. Initialize RAG System (Enhanced)
    try:
        if components['ai_model']:
            from backend.rag_pipeline.chromadb_rag import ChromaDbRag
            from backend.rag_pipeline.embeddings import embeddings
            from backend.data_repository.local_storage import LocalJSONStore

            # Check if embeddings are available
            if embeddings is not None:
                # Initialize storage and RAG
                storage = LocalJSONStore("./data/storage")
                rag_system = ChromaDbRag("./data/vector_store", embeddings)

                # Test RAG system
                test_working, test_message = test_rag_integration({
                    'rag_system': rag_system,
                    'storage': storage,
                    'ai_model': components['ai_model']
                })

                if test_working:
                    components['storage'] = storage
                    components['rag_system'] = rag_system
                    components['status']['rag'] = "✅ RAG System: Ready and tested"
                    logger.info("RAG system initialized and tested successfully")
                else:
                    components['errors']['rag'] = f"RAG test failed: {test_message}"
                    components['status']['rag'] = f"⚠️ RAG System: Available but test failed"
                    # Still provide the components for manual testing
                    components['storage'] = storage
                    components['rag_system'] = rag_system
            else:
                components['errors']['rag'] = "Embeddings not available"
                components['status']['rag'] = "⚠️ RAG System: Embeddings not available"
    except Exception as e:
        components['errors']['rag'] = str(e)
        components['status']['rag'] = f"⚠️ RAG System: Not available - {str(e)}"
        logger.warning(f"RAG system not available: {e}")

    return components


def render_system_status(components: Dict[str, Any]):
    """Render system status with expandable details"""
    has_errors = any(components['errors'].values())
    has_warnings = any("⚠️" in status for status in components['status'].values())
    
    # Show expanded if there are errors, collapsed if just warnings
    expanded = has_errors

    with st.expander("🔍 System Status", expanded=expanded):
        # Core status
        for component, status in components['status'].items():
            if "✅" in status:
                st.success(status)
            elif "⚠️" in status:
                st.warning(status)
            else:
                st.error(status)

        # Show GPU status if AI model is loaded
        if components['ai_model']:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    st.info(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB total, {memory_used:.2f}GB used)")
                else:
                    st.info("💻 Running on CPU")
            except:
                pass

        # Show environment info
        st.markdown("**Environment Configuration:**")
        env_status = []

        if PUBMED_EMAIL:
            env_status.append("✅ PUBMED_EMAIL: Configured")
        else:
            env_status.append("⚠️ PUBMED_EMAIL: Not set")

        if PUBMED_API_KEY:
            env_status.append("✅ PUBMED_API_KEY: Configured")
        else:
            env_status.append("⚠️ PUBMED_API_KEY: Not set")

        # Check for HuggingFace token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            env_status.append("✅ HUGGINGFACE_TOKEN: Configured")
        else:
            env_status.append("⚠️ HUGGINGFACE_TOKEN: Not set")

        for status in env_status:
            if "✅" in status:
                st.success(status)
            else:
                st.warning(status)

        # Show errors if any
        if components['errors']:
            st.markdown("**Error Details:**")
            for component, error in components['errors'].items():
                st.error(f"{component.upper()}: {error}")

        # Show recommendations
        st.markdown("**Recommendations:**")
        recommendations = []

        if not components['ai_model']:
            recommendations.append("🤖 Check BioMistral model installation and dependencies")
            recommendations.append("🔧 Ensure PyTorch and transformers are properly installed")

        if not components['retriever']:
            recommendations.append("📖 Install metapub: `pip install metapub`")
            recommendations.append("🔑 Set PUBMED_API_KEY for PubMed access")

        if not hf_token:
            recommendations.append("🤗 Set HUGGINGFACE_TOKEN for model downloads")

        # Add BioMistral specific recommendations
        if components['ai_model'] and "short responses" in str(components.get('errors', {}).get('ai', '')):
            recommendations.append("🔧 BioMistral may need parameter tuning for longer responses")
            recommendations.append("💡 Try rephrasing questions if responses are too short")

        if not recommendations:
            recommendations.append("✅ All systems optimal!")

        for rec in recommendations:
            st.info(rec)


def process_pubmed_query(question: str, retriever, ai_model, max_results: int = 20, length_preference: str = "Medium") -> str:
    """Process a question using PubMed retrieval + BioMistral analysis"""
    try:
        # Step 1: Retrieve abstracts from PubMed
        st.info(f"📖 Searching PubMed for: '{question}'")
        abstracts = retriever.get_abstract_data(question)

        if not abstracts:
            fallback_response = get_BioMistral_response(ai_model, question, length_preference)
            return f"❌ No relevant abstracts found in PubMed for: '{question}'\n\nUsing BioMistral knowledge instead:\n\n{fallback_response}"

        st.success(f"📋 Retrieved {len(abstracts)} relevant abstracts")

        # Step 2: Format abstracts for BioMistral analysis
        formatted_abstracts = "\n\n".join([
            f"Paper {i+1}:\n"
            f"Title: {abstract.title}\n"
            f"Authors: {', '.join(abstract.authors[:2])}{'...' if len(abstract.authors) > 2 else ''}\n"
            f"Year: {abstract.year}\n"
            f"Abstract: {abstract.abstract_content[:500]}{'...' if len(abstract.abstract_content) > 500 else ''}\n"
            for i, abstract in enumerate(abstracts[:5])  # Limit to first 5 for BioMistral processing
        ])

        # Step 3: Create enhanced prompt for BioMistral
        enhanced_question = f"Based on these research papers, {question}"
        
        # Step 4: Get BioMistral analysis
        st.info("🧠 Analyzing retrieved literature with BioMistral...")
        ai_response = get_BioMistral_response(ai_model, enhanced_question, length_preference)

        # Step 5: Format final response
        final_response = f"""## Literature-Based Analysis

{ai_response}

---
## Source Summary
Retrieved {len(abstracts)} papers from PubMed
Analyzed top {min(5, len(abstracts))} papers for this response

**Key Papers:**
"""

        for i, abstract in enumerate(abstracts[:3]):  # Show top 3 papers
            final_response += f"\n{i+1}. **{abstract.title}** ({abstract.year})\n   {', '.join(abstract.authors[:2])}{'...' if len(abstract.authors) > 2 else ''}\n"

        return final_response

    except Exception as e:
        logger.error(f"Error in PubMed query processing: {e}")
        fallback_response = get_BioMistral_response(ai_model, question, length_preference)
        return f"❌ Error retrieving from PubMed: {str(e)}\n\nFalling back to BioMistral knowledge:\n\n{fallback_response}"

def render_chat_interface(ai_model):
    """Render interactive chat interface with BioMistral"""
    st.markdown("---")
    st.subheader("💬 Interactive Chat with BioMistral")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm BioMistral, your biomedical research assistant. Ask me anything about medicine, biology, or health research!"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask BioMistral about biomedical research..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate BioMistral response
        with st.chat_message("assistant"):
            with st.spinner("BioMistral is thinking..."):
                try:
                    response = get_BioMistral_response(ai_model, prompt, "Medium", max_retries=2)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Chat error: {e}")

# STEP 2: Replace your render_research_interface function in app.py

def render_research_interface(components: Dict[str, Any]):
    """Render the main research interface with RAG support"""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Research Mode")

        # Mode selection with RAG option
        if components['retriever'] and components['ai_model'] and components['rag_system']:
            mode = st.radio(
                "Choose research mode:",
                [
                    "🧠 BioMistral Knowledge Only",
                    "📬 PubMed + BioMistral (Simple)",
                    "🚀 PubMed + RAG + BioMistral (Enhanced)"  # NEW RAG MODE!
                ],
                help="""
                - Knowledge Only: Uses BioMistral's training knowledge
                - Simple: Retrieves papers and feeds them directly to BioMistral  
                - Enhanced: Uses semantic search (RAG) to find most relevant paper sections
                """
            )
            use_rag = "RAG" in mode
            use_pubmed = "PubMed" in mode
        elif components['retriever'] and components['ai_model']:
            mode = st.radio(
                "Choose research mode:",
                ["🧠 BioMistral Knowledge Only", "📬 PubMed + BioMistral (Simple)"],
                help="RAG system not available"
            )
            use_rag = False
            use_pubmed = "PubMed" in mode
        elif components['ai_model']:
            st.info("🧠 BioMistral Knowledge Mode Only (PubMed and RAG not available)")
            use_rag = False
            use_pubmed = False
        else:
            st.error("❌ No AI model available")
            return

        # Show what mode is selected
        if use_rag:
            st.success("🚀 Enhanced RAG Mode: Semantic search + AI analysis")
        elif use_pubmed:
            st.info("📬 Simple Mode: Direct paper analysis")
        else:
            st.info("🧠 Knowledge Mode: AI model only")

        # Example questions tailored for BioMistral
        st.markdown("**Try these examples:**")
        examples = [
            "How do cancer cells develop drug resistance?",
            "What role does gut microbiota play in diabetes?",
            "Explain the mechanism of CRISPR gene editing",
            "How does machine learning help in drug discovery?",
            "What are biomarkers for early Alzheimer's detection?",
            "Describe immunotherapy mechanisms in cancer treatment",
            "How do mRNA vaccines work?",
            "What causes antibiotic resistance?",
            "Explain the blood-brain barrier function",
            "How do stem cells differentiate?"
        ]

        selected_example = None
        for i, example in enumerate(examples):
            if st.button(f"💡 {example}", key=f"ex_{i}"):
                selected_example = example

        # Custom question input
        st.markdown("**Or ask your own question:**")
        custom_question = st.text_area(
            "Enter your biomedical research question:",
            value=selected_example or "",
            placeholder="Ask about biology, medicine, diseases, treatments, etc...",
            height=100
        )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            if use_pubmed:
                max_papers = st.slider("Max papers to retrieve:", 5, 50, 20)
            
            # BioMistral specific options
            response_length = st.select_slider(
                "Response length preference:",
                options=["Short", "Medium", "Long"],
                value="Medium"
            )
            
            # Add retry option
            max_retries = st.slider("Max retry attempts for better responses:", 1, 5, 3)

        ask_button = st.button("🧠 Analyze Question", type="primary")

    with col2:
        st.subheader("🤖 BioMistral Analysis")

        if ask_button and custom_question.strip():
            with st.spinner("🤔 Processing your question..."):
                try:
                    if use_rag and components['rag_system']:
                        # Use RAG-enhanced processing
                        st.info("🚀 Using RAG-enhanced analysis...")
                        response = process_pubmed_query_with_rag(
                            custom_question,
                            components['retriever'],
                            components['ai_model'],
                            components['rag_system'],
                            components['storage'],
                            max_results=locals().get('max_papers', 20),
                            length_preference=locals().get('response_length', 'Medium')
                        )
                    elif use_pubmed and components['retriever']:
                        # Use simple PubMed + BioMistral analysis
                        st.info("📬 Using simple PubMed analysis...")
                        response = process_pubmed_query(
                            custom_question,
                            components['retriever'],
                            components['ai_model'],
                            max_results=locals().get('max_papers', 20),
                            length_preference=locals().get('response_length', 'Medium')
                        )
                    else:
                        # BioMistral knowledge base only
                        st.info("🧠 Using BioMistral knowledge only...")
                        response = get_BioMistral_response(
                            components['ai_model'], 
                            custom_question, 
                            length_preference=locals().get('response_length', 'Medium'),
                            max_retries=locals().get('max_retries', 3)
                        )

                    st.markdown("### 📋 Analysis Results:")
                    
                    # Display response with better formatting
                    if response and len(response.strip()) > 5:
                        st.write(response)
                        st.success("✅ Analysis complete!")
                        
                        # Show mode used
                        if use_rag:
                            st.info("🚀 Used: RAG-enhanced semantic analysis")
                        elif use_pubmed:
                            st.info("📬 Used: Simple PubMed analysis")
                        else:
                            st.info("🧠 Used: BioMistral knowledge only")
                    else:
                        st.warning(f"⚠️ BioMistral returned a short response: '{response}'")
                        st.info("💡 Try rephrasing your question or using a different mode.")

                    # Add to session state for reference
                    if 'research_history' not in st.session_state:
                        st.session_state.research_history = []

                    st.session_state.research_history.append({
                        'question': custom_question,
                        'response': response,
                        'timestamp': datetime.now(),
                        'mode': mode
                    })

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {e}")
                    
                    # Try a simpler approach if the enhanced one fails
                    try:
                        st.info("🔄 Trying simplified approach...")
                        simple_response = components['ai_model'].invoke(custom_question)
                        st.markdown("### 📋 Simplified Analysis:")
                        st.write(simple_response if simple_response else "No response generated")
                    except Exception as e2:
                        st.error(f"❌ Simplified approach also failed: {str(e2)}")
                        
        elif ask_button:
            st.warning("⚠️ Please enter a question first.")
        else:
            st.info("👈 Select an example or enter your own question to get started!")
            
            # Show what each mode does
            st.markdown("### 🎯 Mode Explanations:")
            if components['rag_system']:
                st.markdown("""
                **🚀 RAG Enhanced Mode:**
                - Searches PubMed for relevant papers
                - Creates semantic embeddings of paper content
                - Uses AI to find most relevant sections
                - Provides context-aware analysis
                
                **📬 Simple Mode:**
                - Searches PubMed for relevant papers
                - Feeds papers directly to BioMistral
                - Good for straightforward questions
                
                **🧠 Knowledge Mode:**
                - Uses BioMistral's training knowledge
                - No external paper retrieval
                - Fast responses
                """)
            else:
                st.markdown("""
                **📬 Simple Mode:**
                - Searches PubMed for relevant papers
                - Feeds papers directly to BioMistral
                
                **🧠 Knowledge Mode:**
                - Uses BioMistral's training knowledge
                - No external paper retrieval
                """)

# Add this to your render_sidebar function
def render_sidebar_with_rag_test(components: Dict[str, Any]):
    """Enhanced sidebar with RAG testing"""
    with st.sidebar:
        st.header("🛠️ Tools")

        # RAG Test Button (NEW!)
        if components['rag_system'] and st.button("🧪 Test RAG Integration"):
            with st.spinner("Testing RAG system..."):
                working, message = test_rag_integration(components)
                if working:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")

        # Test BioMistral button
        if components['ai_model'] and st.button("🧪 Test BioMistral Response"):
            with st.spinner("Testing BioMistral..."):
                test_question = "What is diabetes?"
                test_response = get_BioMistral_response(components['ai_model'], test_question, "Medium")
                st.write(f"**Test Question:** {test_question}")
                st.write(f"**Response:** {test_response}")

        # Rest of your existing sidebar code...
        # (Chat controls, model info, etc.)

def render_sidebar(components: Dict[str, Any]):
    """Render sidebar with tools and information"""
    with st.sidebar:
        st.header("🛠️ Tools")

        # Test BioMistral button
        if components['ai_model'] and st.button("🧪 Test BioMistral Response"):
            with st.spinner("Testing BioMistral..."):
                test_question = "What is diabetes?"
                test_response = get_BioMistral_response(components['ai_model'], test_question, "Medium")
                st.write(f"**Test Question:** {test_question}")
                st.write(f"**Response:** {test_response}")

        # Chat controls
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared! I'm BioMistral, ready to help with your biomedical questions!"}
            ]
            st.rerun()

        if st.button("📋 Clear Research History"):
            if 'research_history' in st.session_state:
                del st.session_state.research_history
            st.success("Research history cleared!")

        # Model information
        if components['ai_model']:
            st.markdown("---")
            st.subheader("🤖 Model Info")
            st.write("**Current Model:** BioMistral-small")
            st.write("**Platform:** Microsoft/HuggingFace")
            st.write("**Specialization:** Conversational AI")
            
            # Show GPU info if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    st.write(f"**Hardware:** {gpu_name}")
                    memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    st.write(f"**GPU Memory:** {memory_used:.2f}GB used")
            except:
                pass

        # Research history
        if 'research_history' in st.session_state and st.session_state.research_history:
            st.markdown("---")
            st.subheader("📚 Recent Research")
            for i, item in enumerate(reversed(st.session_state.research_history[-5:])):  # Last 5
                with st.expander(f"Q{len(st.session_state.research_history)-i}: {item['question'][:50]}..."):
                    st.write(f"**Mode:** {item['mode']}")
                    st.write(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Question:** {item['question']}")
                    st.write(f"**Response:** {item['response'][:200]}...")

        st.markdown("---")
        st.subheader("📊 Current Capabilities")

        capabilities = []
        if components['ai_model']:
            capabilities.append("✅ BioMistral AI Assistant")
        if components['retriever']:
            capabilities.append("✅ PubMed Search")
        if components['rag_system']:
            capabilities.append("✅ RAG System")

        if not capabilities:
            capabilities.append("❌ Limited functionality")

        for cap in capabilities:
            st.write(cap)

        st.markdown("---")
        st.subheader("🎯 Usage Tips for BioMistral")
        st.markdown("""
        **For best results with BioMistral:**
        - Ask clear, specific questions
        - Use conversational language
        - Break complex questions into parts
        - If you get short responses, try rephrasing
        - Use the test button to check BioMistral status
        """)

        st.markdown("---")
        st.subheader("🏥 Knowledge Areas")
        st.markdown("""
        - **Cancer Research** 🎗️
        - **Neuroscience** 🧠 
        - **Immunology** 🦠
        - **Genetics** 🧬
        - **Drug Development** 💊
        - **Clinical Trials** 📊
        - **Biomarkers** 🔬
        - **Medical Devices** ⚕️
        """)
# STEP 1: Replace this function in your app.py

def process_pubmed_query_with_rag(question: str, retriever, ai_model, rag_system, storage, 
                                  max_results: int = 20, length_preference: str = "Medium") -> str:
    """
    Process a question using PubMed retrieval + RAG + BioMistral analysis
    This replaces your old process_pubmed_query function
    """
    try:
        # Step 1: Retrieve abstracts from PubMed
        st.info(f"📖 Searching PubMed for: '{question}'")
        abstracts = retriever.get_abstract_data(question)

        if not abstracts:
            fallback_response = get_BioMistral_response(ai_model, question, length_preference)
            return f"❌ No relevant abstracts found in PubMed for: '{question}'\n\nUsing BioMistral knowledge instead:\n\n{fallback_response}"

        st.success(f"📋 Retrieved {len(abstracts)} relevant abstracts")

        # Step 2: Save abstracts and create RAG index
        st.info("🔧 Creating RAG knowledge base from retrieved papers...")
        
        # Save abstracts to storage
        query_id = storage.save_dataset(abstracts, question)
        
        # Convert abstracts to documents for RAG
        documents = storage.create_document_list(abstracts)
        
        # Create vector index for semantic search
        vector_store = rag_system.create_vector_index_for_user_query(documents, query_id)
        
        st.success(f"✅ RAG knowledge base created with {len(documents)} document chunks")

        # Step 3: Use RAG to get relevant context
        st.info("🔍 Finding most relevant paper sections using semantic search...")
        
        # Create retriever from vector store
        retriever_chain = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Get top 5 most relevant chunks
        )
        
        # Get relevant documents using semantic search
        relevant_docs = retriever_chain.get_relevant_documents(question)
        
        # Step 4: Format context for BioMistral
        context_sections = []
        for i, doc in enumerate(relevant_docs):
            context_sections.append(f"""
Paper Section {i+1}:
Source: {doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('year_of_publication', 'N/A')})
Authors: {doc.metadata.get('authors', 'Unknown')}
Content: {doc.page_content}
""")
        
        formatted_context = "\n".join(context_sections)
        
        # Step 5: Create enhanced prompt for BioMistral with RAG context
        rag_prompt = f"""You are BioMistral, a medical research AI assistant. Based on the following scientific research papers, please answer this question: {question}

RESEARCH CONTEXT FROM PUBMED PAPERS:
{formatted_context}

Please provide a comprehensive analysis that:
1. Synthesizes information from the research papers
2. Identifies key findings and trends
3. Notes any contradictions or debates in the literature
4. Provides evidence-based conclusions

Your detailed response:"""

        # Step 6: Get BioMistral analysis with RAG context
        st.info("🧠 Analyzing with BioMistral using RAG-enhanced context...")
        ai_response = get_BioMistral_response(ai_model, rag_prompt, length_preference)

        # Step 7: Format final response
        final_response = f"""## 🔬 RAG-Enhanced Literature Analysis

{ai_response}

---
## 📊 Analysis Methodology
- **Papers Retrieved:** {len(abstracts)} from PubMed
- **RAG Processing:** {len(documents)} document chunks created
- **Semantic Search:** Top {len(relevant_docs)} most relevant sections analyzed
- **AI Model:** BioMistral-7B with RAG enhancement

## 📚 Key Papers Analyzed:
"""

        # Add paper citations
        for i, abstract in enumerate(abstracts[:5]):  # Show top 5 papers
            final_response += f"\n{i+1}. **{abstract.title}** ({abstract.year})\n"
            final_response += f"   {', '.join(abstract.authors[:2])}{'...' if len(abstract.authors) > 2 else ''}\n"
            if abstract.pmid:
                final_response += f"   PMID: {abstract.pmid}\n"

        # Clean up - optionally delete the temporary RAG index
        # rag_system.delete_vector_index(query_id)  # Uncomment if you want to clean up

        return final_response

    except Exception as e:
        logger.error(f"Error in RAG-enhanced PubMed query processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simple BioMistral response
        fallback_response = get_BioMistral_response(ai_model, question, length_preference)
        return f"❌ Error in RAG processing: {str(e)}\n\nFalling back to BioMistral knowledge:\n\n{fallback_response}"


def test_rag_integration(components):
    """Test if RAG integration is working properly"""
    if not all([components['rag_system'], components['storage'], components['ai_model']]):
        return False, "Missing required components"
    
    try:
        # Create test documents
        from langchain.schema import Document
        test_docs = [
            Document(
                page_content="Cancer immunotherapy uses the body's immune system to fight cancer cells. T-cells are engineered to better recognize and attack tumors.",
                metadata={"title": "Cancer Immunotherapy Review", "year": 2023, "authors": "Smith et al."}
            ),
            Document(
                page_content="CRISPR gene editing allows precise modification of DNA sequences. This technology has applications in treating genetic diseases.",
                metadata={"title": "CRISPR Technology in Medicine", "year": 2023, "authors": "Johnson et al."}
            )
        ]
        
        # Test vector store creation
        test_query_id = "test_rag_integration"
        vector_store = components['rag_system'].create_vector_index_for_user_query(test_docs, test_query_id)
        
        # Test retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        relevant_docs = retriever.get_relevant_documents("What is cancer immunotherapy?")
        
        # Cleanup
        components['rag_system'].delete_vector_index(test_query_id)
        
        if relevant_docs and "immunotherapy" in relevant_docs[0].page_content.lower():
            return True, f"RAG working correctly - retrieved {len(relevant_docs)} relevant documents"
        else:
            return False, "RAG retrieval not working as expected"
            
    except Exception as e:
        return False, f"RAG test failed: {str(e)}"
    
# STEP 4: Update your main() function in app.py

def main():
    """Enhanced main application function with RAG support"""
    st.set_page_config(
        page_title="PubMed Abstract Screener with BioMistral + RAG",
        page_icon='🔬',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Header
    st.title("🔬 PubMed Abstract Screener")
    st.markdown("**AI-Powered Biomedical Research Assistant with BioMistral + RAG**")

    # Initialize system components with enhanced initialization
    with st.spinner("🚀 Initializing BioMistral, RAG system, and components..."):
        # Use enhanced initialization if you implemented Step 3B, otherwise use your existing one
        try:
            components = initialize_system_components_enhanced()
        except:
            # Fallback to your existing initialization
            components = initialize_system_components()

    # Render system status
    render_system_status(components)

    # Check if core AI is available
    if not components['ai_model']:
        st.error("❌ BioMistral AI assistant not available. Please check the system status above.")
        st.info("💡 Make sure your BioMistral model is properly installed and configured.")
        st.stop()

    # Show RAG availability status
    if components['rag_system']:
        st.success("🚀 RAG-enhanced analysis available! Try the 'RAG + BioMistral' mode for best results.")
    elif components['retriever']:
        st.info("📬 PubMed analysis available. RAG system not fully ready - using simple mode.")
    else:
        st.warning("🧠 Only BioMistral knowledge mode available. PubMed and RAG systems not ready.")

    # Main interface (updated)
    render_research_interface(components)

    # Chat interface
    render_chat_interface(components['ai_model'])

    # Enhanced sidebar with RAG test
    render_sidebar_with_rag_test(components)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🔬 PubMed Abstract Screener | Powered by BioMistral + RAG | Built with Streamlit |
        <a href='https://www.ncbi.nlm.nih.gov/pubmed/' target='_blank'>PubMed</a> Integration
    </div>
    """, unsafe_allow_html=True)





if __name__ == "__main__":
    main()