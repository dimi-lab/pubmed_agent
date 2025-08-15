import streamlit as st
import os
import sys
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Then your CONFIG can read from environment
CONFIG = {
    'PUBMED_EMAIL': os.getenv('PUBMED_EMAIL', "gnanaolivu.rohandavid@mayo.com"),
    'PUBMED_API_KEY': os.getenv('PUBMED_API_KEY', "f9ecce69a73048b82ca4747f51d6dec61408"),
    'MAX_PAPERS_DEFAULT': 20,
    'RESPONSE_LENGTH_DEFAULT': "Medium",
    'MAX_RETRIES_DEFAULT': 3
}

# Set environment variables (this will now use loaded values)
os.environ.update({
    'PUBMED_EMAIL': CONFIG['PUBMED_EMAIL'],
    'PUBMED_API_KEY': CONFIG['PUBMED_API_KEY'],
    'NCBI_API_KEY': os.getenv('NCBI_API_KEY', CONFIG['PUBMED_API_KEY']),
    'HUGGINGFACE_KEY': os.getenv('HUGGINGFACE_KEY', ''),
    'ANONYMIZED_TELEMETRY': 'False',
    'CHROMA_TELEMETRY': 'False',
    'POSTHOG_HOST': ''
})
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_custom_css():
    """Load clean white theme CSS"""
    st.markdown("""
    <style>
    .stApp { background: #ffffff; color: #000000 !important; }
    * { color: #000000 !important; }
    .css-1d391kg { background: #f0f2f6; border-right: 1px solid #e6e6e6; }
    h1, h2, h3, h4, h5, h6 { color: #0066cc !important; font-weight: 600; }
    .stButton button { 
        background-color: #0066cc; color: #ffffff !important; border: none; 
        border-radius: 8px; padding: 8px 16px; font-weight: 500; 
    }
    .stButton button:hover { background-color: #0052a3; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ffffff; color: #000000; border: 2px solid #e6e6e6; border-radius: 8px;
    }
    .stTextArea textarea { 
        background-color: #f8f9fa !important; color: #000000 !important; 
        border: 2px solid #e6e6e6 !important; border-radius: 8px;
    }
    .stSuccess { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724 !important; }
    .stInfo { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460 !important; }
    .stWarning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404 !important; }
    .stError { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24 !important; }
    .metric-container {
        background: #ffffff; padding: 20px; border-radius: 8px; 
        border: 1px solid #e6e6e6; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

class SystemInitializer:
    """Handle system component initialization"""
    
    @staticmethod
    def init_ai_model():
        """Initialize BioMistral AI model with multiple strategies"""
        try:
            from components.llm import get_llm
            
            strategies = ["default", "cpu_first", "low_memory", "torch_dtype_fix"]
            
            for strategy in strategies:
                try:
                    st.info(f"🔄 Trying strategy: {strategy}")
                    
                    if strategy == "cpu_first":
                        import torch
                        with torch.device('cpu'):
                            model = get_llm()
                    elif strategy == "low_memory":
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                        model = get_llm()
                    else:
                        model = get_llm()
                    
                    if model and SystemInitializer._test_model(model):
                        st.success(f"✅ Model loaded with {strategy}")
                        return model, "✅ AI Assistant: Ready (BioMistral-7B)"
                        
                except Exception as e:
                    st.warning(f"⚠️ {strategy} failed: {str(e)}")
                    continue
            
            return None, "❌ AI Assistant: Failed to load"
            
        except Exception as e:
            return None, f"❌ AI Assistant: {str(e)}"
    
    @staticmethod
    def _test_model(model):
        """Test if model works"""
        try:
            response = model.invoke("What is cancer?")
            return response and len(response.strip()) > 5
        except:
            return False
    
    @staticmethod
    def init_pubmed():
        """Initialize PubMed retriever"""
        try:
            from backend.abstract_retrieval.pubmed_retriever import create_configured_pubmed_retriever
            retriever = create_configured_pubmed_retriever(max_results=CONFIG['MAX_PAPERS_DEFAULT'])
            return retriever, "✅ PubMed Retriever: Ready"
        except Exception as e:
            return None, f"⚠️ PubMed Retriever: {str(e)}"
    
    @staticmethod
    def init_rag(ai_model):
        """Initialize RAG system"""
        if not ai_model:
            return None, None, "⚠️ RAG System: No AI model"
        
        try:
            from backend.rag_pipeline.chromadb_rag import ChromaDbRag
            from backend.rag_pipeline.embeddings import embeddings
            from backend.data_repository.local_storage import LocalJSONStore
            
            if embeddings:
                storage = LocalJSONStore("./data/storage")
                rag_system = ChromaDbRag("./data/vector_store", embeddings)
                return storage, rag_system, "✅ RAG System: Ready"
            else:
                return None, None, "⚠️ RAG System: No embeddings"
                
        except Exception as e:
            return None, None, f"⚠️ RAG System: {str(e)}"

def initialize_system_components():
    """Initialize all system components"""
    st.info("🔄 Loading system components...")
    
    # Initialize AI model
    ai_model, ai_status = SystemInitializer.init_ai_model()
    
    # Initialize PubMed
    retriever, pubmed_status = SystemInitializer.init_pubmed()
    
    # Initialize RAG
    storage, rag_system, rag_status = SystemInitializer.init_rag(ai_model)
    
    return {
        'ai_model': ai_model,
        'retriever': retriever,
        'rag_system': rag_system,
        'storage': storage,
        'status': {
            'ai': ai_status,
            'pubmed': pubmed_status,
            'rag': rag_status
        }
    }

class PromptEngine:
    """Handle prompt generation and response processing"""
    
    @staticmethod
    def create_biomistral_prompt(question: str, length_preference: str = "Medium", context: str = None):
        """Create optimized BioMistral prompt"""
        length_map = {
            "Short": "Provide a concise medical answer (2-3 sentences).",
            "Long": "Provide a comprehensive medical explanation with details and mechanisms (300-500 words).",
            "Medium": "Provide a clear, informative medical answer with appropriate detail (150-300 words)."
        }
        
        base_prompt = f"""You are BioMistral, an expert medical AI assistant. {length_map.get(length_preference, length_map['Medium'])}

Medical Question: {question}"""
        
        if context:
            base_prompt += f"\n\nRelevant Research Context:\n{context}"
        
        base_prompt += "\n\nProvide a complete and thorough medical response:"
        return base_prompt
    
    @staticmethod
    def get_biomistral_response(ai_model, question: str, length_preference: str = "Medium", 
                               context: str = None, max_retries: int = 3):
        """Get response from BioMistral with retries and better parameters"""
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    prompt = PromptEngine.create_biomistral_prompt(question, length_preference, context)
                elif attempt == 1:
                    prompt = f"As a medical AI assistant, please provide a complete answer to: {question}"
                else:
                    prompt = f"Please provide a detailed medical explanation for: {question}"
                
                # Get response with explicit request for completion
                response = ai_model.invoke(prompt)
                
                if response and len(response.strip()) > 10:
                    # Clean response
                    response = response.strip()
                    if response.startswith(("Medical Response:", "Medical Answer:", "Answer:")):
                        response = response.split(":", 1)[1].strip()
                    
                    # Check if response seems complete (doesn't end mid-sentence)
                    if PromptEngine._is_complete_response(response):
                        return response
                    else:
                        logger.warning(f"Response appears truncated (attempt {attempt + 1}): ...{response[-50:]}")
                        # Try again with different approach
                        continue
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        return f"Unable to generate complete response for: '{question}'. Please try rephrasing or check system status."
    
    @staticmethod
    def _is_complete_response(response: str) -> bool:
        """Check if response appears complete"""
        # Check if response ends with proper punctuation
        if response.endswith(('.', '!', '?', ':', ';')):
            return True
        
        # Check if response seems to end mid-word or mid-sentence
        last_words = response.split()[-3:] if len(response.split()) >= 3 else response.split()
        
        # If last few words seem incomplete
        incomplete_indicators = ['the', 'a', 'an', 'and', 'or', 'but', 'that', 'this', 'with', 'from', 'by']
        if any(word.lower() in incomplete_indicators for word in last_words):
            return False
            
        return True

class PubMedProcessor:
    """Handle PubMed queries and analysis"""
    
    @staticmethod
    def process_simple_query(question: str, retriever, ai_model, max_results: int = 20, 
                           length_preference: str = "Medium"):
        """Process PubMed query with simple analysis"""
        try:
            st.info(f"📖 Searching PubMed for: '{question}'")
            abstracts = retriever.get_abstract_data(question)
            
            if not abstracts:
                fallback = PromptEngine.get_biomistral_response(ai_model, question, length_preference)
                return f"❌ No PubMed results for: '{question}'\n\n**BioMistral Knowledge:**\n{fallback}"
            
            st.success(f"📋 Retrieved {len(abstracts)} papers")
            
            # Format abstracts for analysis
            context = PubMedProcessor._format_abstracts(abstracts[:5])
            
            # Get analysis
            st.info("🧠 Analyzing papers with BioMistral...")
            analysis = PromptEngine.get_biomistral_response(
                ai_model, question, length_preference, context
            )
            
            return PubMedProcessor._format_response(question, analysis, abstracts)
            
        except Exception as e:
            logger.error(f"PubMed processing error: {e}")
            fallback = PromptEngine.get_biomistral_response(ai_model, question, length_preference)
            return f"❌ PubMed error: {str(e)}\n\n**Fallback Analysis:**\n{fallback}"
    
    @staticmethod
    def process_rag_query(pubmed_query: str, general_question: str, retriever, ai_model, rag_system, storage, 
                         max_results: int = 20, length_preference: str = "Medium"):
        """Process RAG query: Use PubMed query for literature retrieval, general question for RAG analysis"""
        try:
            st.info(f"📖 Searching PubMed with: '{pubmed_query}'")
            abstracts = retriever.get_abstract_data(pubmed_query)
            
            if not abstracts:
                fallback = PromptEngine.get_biomistral_response(ai_model, general_question, length_preference)
                return f"❌ No PubMed results for: '{pubmed_query}'\n\n**BioMistral Knowledge for '{general_question}':**\n{fallback}"
            
            st.success(f"📋 Retrieved {len(abstracts)} papers using PubMed query")
            st.info("🔧 Creating RAG knowledge base from retrieved papers...")
            
            # Create RAG index using PubMed results
            query_id = storage.save_dataset(abstracts, pubmed_query)
            documents = storage.create_document_list(abstracts)
            vector_store = rag_system.create_vector_index_for_user_query(documents, query_id)
            
            st.info(f"🔍 Querying RAG with: '{general_question}'")
            retriever_chain = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            # Use GENERAL QUESTION to query the RAG system
            relevant_docs = retriever_chain.get_relevant_documents(general_question)
            
            # Format context from RAG results
            context = "\n\n".join([
                f"Relevant Section {i+1}: {doc.page_content}" 
                for i, doc in enumerate(relevant_docs)
            ])
            
            st.info("🧠 Generating RAG-enhanced analysis...")
            
            # Create enhanced prompt that shows the separation
            rag_prompt = f"""Based on the following research papers retrieved from PubMed using the search "{pubmed_query}", please answer the question: "{general_question}"

RELEVANT LITERATURE SECTIONS:
{context}

Please provide a comprehensive analysis that:
1. Directly answers the question: "{general_question}"
2. Uses evidence from the retrieved literature about "{pubmed_query}"
3. Synthesizes the most relevant information
4. Provides clinical insights and implications

Your detailed response:"""
            
            analysis = PromptEngine.get_biomistral_response(
                ai_model, rag_prompt, length_preference
            )
            
            return PubMedProcessor._format_rag_response_enhanced(
                pubmed_query, general_question, analysis, abstracts, len(documents), len(relevant_docs)
            )
            
        except Exception as e:
            logger.error(f"RAG processing error: {e}")
            fallback = PromptEngine.get_biomistral_response(ai_model, general_question, length_preference)
            return f"❌ RAG error: {str(e)}\n\n**Fallback Analysis for '{general_question}':**\n{fallback}"
    
    @staticmethod
    def _format_abstracts(abstracts: List) -> str:
        """Format abstracts for AI analysis"""
        return "\n\n".join([
            f"Paper {i+1}:\nTitle: {abs.title}\nAuthors: {', '.join(abs.authors[:3])}\n"
            f"Year: {abs.year}\nAbstract: {abs.abstract_content}"
            for i, abs in enumerate(abstracts)
        ])
    
    @staticmethod
    def _format_response(question: str, analysis: str, abstracts: List) -> str:
        """Format simple PubMed response"""
        response = f"""## 📚 Literature Analysis

**Question:** {question}

### 🔬 BioMistral's Analysis of {len(abstracts)} PubMed Papers:

{analysis}

---

### 📖 Key Papers Analyzed:
"""
        for i, abs in enumerate(abstracts[:3]):
            response += f"\n**{i+1}.** {abs.title} ({abs.year})\n"
        
        return response
    
    @staticmethod
    def _format_rag_response_enhanced(pubmed_query: str, general_question: str, analysis: str, abstracts: List, doc_count: int, relevant_sections: int) -> str:
        """Format RAG-enhanced response with clear separation of queries"""
        return f"""## 🚀 RAG-Enhanced Literature Analysis

**PubMed Search Query:** "{pubmed_query}"
**Research Question:** "{general_question}"

### 🔬 Analysis:

{analysis}

---

### 📊 RAG Methodology:
- **Literature Search:** Used "{pubmed_query}" to retrieve {len(abstracts)} papers from PubMed
- **Question Analysis:** Used "{general_question}" to query the RAG knowledge base
- **Document Chunks:** {doc_count} created from retrieved papers
- **Relevant Sections:** {relevant_sections} most relevant sections identified
- **Method:** Semantic search + BioMistral analysis

### 📖 Key Papers Retrieved:
""" + "\n".join([f"**{i+1}.** {abs.title} ({abs.year})" for i, abs in enumerate(abstracts[:5])])
    
    @staticmethod
    def _format_rag_response(question: str, analysis: str, abstracts: List, doc_count: int) -> str:
        """Format RAG-enhanced response (legacy method for compatibility)"""
        return f"""## 🚀 RAG-Enhanced Literature Analysis

**Question:** {question}

### 🔬 Analysis:

{analysis}

---

### 📊 Methodology:
- **Papers Retrieved:** {len(abstracts)} from PubMed
- **Document Chunks:** {doc_count} created
- **Method:** RAG + BioMistral analysis

### 📖 Key Papers:
""" + "\n".join([f"**{i+1}.** {abs.title} ({abs.year})" for i, abs in enumerate(abstracts[:5])])

class UIComponents:
    """Handle UI component rendering"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #ffffff;'>
            <h1 style='color: #0066cc; font-weight: 700; margin-bottom: 10px;'>
                🔬 CIM PubMed Research Agent
            </h1>
            <h3 style='color: #333333; margin-top: 0px; font-weight: 400;'>
                AI-Powered Biomedical Research with BioMistral + RAG
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_system_status(components: Dict[str, Any]):
        """Render system status"""
        has_errors = any("❌" in status for status in components['status'].values())
        
        with st.expander("🔍 System Status", expanded=has_errors):
            for component, status in components['status'].items():
                if "✅" in status:
                    st.success(status)
                elif "⚠️" in status:
                    st.warning(status)
                else:
                    st.error(status)
    
    @staticmethod
    def render_mode_selector(components: Dict[str, Any]):
        """Render research mode selector with better debugging"""
        # Show component status for debugging
        st.markdown("**🔍 Component Status:**")
        col_debug1, col_debug2, col_debug3 = st.columns(3)
        
        with col_debug1:
            ai_status = "✅" if components.get('ai_model') else "❌"
            st.markdown(f"**AI Model:** {ai_status}")
            
        with col_debug2: 
            retriever_status = "✅" if components.get('retriever') else "❌"
            st.markdown(f"**PubMed:** {retriever_status}")
            
        with col_debug3:
            rag_status = "✅" if components.get('rag_system') else "❌"
            st.markdown(f"**RAG System:** {rag_status}")
        
        st.markdown("---")
        
        # Determine available options based on components
        if components.get('retriever') and components.get('ai_model') and components.get('rag_system'):
            options = [
                "🧠 BioMistral Knowledge Only",
                "📬 PubMed + BioMistral (Simple)",
                "🚀 PubMed + RAG + BioMistral (Enhanced)"
            ]
            st.success("🚀 **All modes available!** Enhanced RAG mode is ready.")
        elif components.get('retriever') and components.get('ai_model'):
            options = [
                "🧠 BioMistral Knowledge Only", 
                "📬 PubMed + BioMistral (Simple)"
            ]
            st.warning("⚠️ **RAG system not available.** Only simple modes available.")
        else:
            options = ["🧠 BioMistral Knowledge Only"]
            st.info("ℹ️ **Limited mode.** Only AI knowledge available.")
        
        mode = st.radio("**Choose research mode:**", options)
        
        # Show what each mode does
        with st.expander("ℹ️ What do these modes do?"):
            st.markdown("""
            **🧠 BioMistral Knowledge Only:**
            - Uses AI's training knowledge
            - Fast responses
            - No external data retrieval
            
            **📬 PubMed + BioMistral (Simple):**
            - Searches PubMed for papers
            - Feeds abstracts directly to AI
            - Literature-based analysis
            
            **🚀 PubMed + RAG + BioMistral (Enhanced):**
            - Searches PubMed for papers
            - Creates semantic embeddings
            - Uses vector search for relevance
            - Most comprehensive analysis
            """)
        
        return mode, "RAG" in mode, "PubMed" in mode
    
    @staticmethod
    def render_input_fields():
        """Render input fields"""
        st.subheader("📝 Input Fields")
        
        general_question = st.text_area(
            "🧠 General Medical/Research Question:",
            placeholder="e.g., How do cancer cells develop drug resistance?",
            height=80
        )
        
        pubmed_query = st.text_input(
            "📚 PubMed Literature Search Query:",
            placeholder="e.g., cancer immunotherapy, CRISPR Cas9",
            help="Use specific medical terms for better results"
        )
        
        return general_question, pubmed_query
    
    @staticmethod
    def render_advanced_options(use_pubmed: bool):
        """Render advanced options"""
        with st.expander("⚙️ Advanced Options"):
            options = {}
            if use_pubmed:
                options['max_papers'] = st.slider("Max papers:", 5, 50, CONFIG['MAX_PAPERS_DEFAULT'])
            options['response_length'] = st.select_slider(
                "Response length:", ["Short", "Medium", "Long"], value=CONFIG['RESPONSE_LENGTH_DEFAULT']
            )
            options['max_retries'] = st.slider("Max retries:", 1, 5, CONFIG['MAX_RETRIES_DEFAULT'])
            return options
    
    @staticmethod
    def render_action_buttons(use_pubmed: bool, use_rag: bool):
        """Render action buttons with clear RAG indication"""
        st.markdown("---")
        st.markdown("**🎯 Available Actions:**")
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if use_pubmed:
                button_text = "📚 Search Literature"
                if use_rag:
                    button_text += " (RAG Enhanced)"
                pubmed_button = st.button(button_text, type="primary", use_container_width=True)
                
                # Show what this button will do
                if use_rag:
                    st.caption("🚀 Will use semantic search + AI analysis")
                else:
                    st.caption("📬 Will use simple literature analysis")
            else:
                pubmed_button = False
                st.button("📚 Literature Search", disabled=True, use_container_width=True)
                st.caption("⚠️ Requires PubMed mode")
                
        with btn_col2:
            general_button = st.button("🧠 Ask BioMistral", type="secondary", use_container_width=True)
            st.caption("💭 Uses AI knowledge only")
        
        # Show instructions for RAG
        if use_rag:
            st.success("🚀 **RAG Enhanced Mode Active!**")
            st.info("📋 **How RAG works:** Enter PubMed terms to retrieve papers, then ask your research question to analyze them.")
        elif use_pubmed:
            st.info("📬 **Simple PubMed Mode Active!** Will analyze papers directly.")
        
    @staticmethod
    def render_response(response: str, response_type: str = "general"):
        """Render AI response with appropriate formatting"""
        st.markdown("### 📋 Analysis Results:")
        
        with st.container():
            if len(response) > 500:
                st.text_area(
                    "Full Response:",
                    value=response,
                    height=400 if response_type == "pubmed" else 300,
                    disabled=True,
                    key=f"{response_type}_response_display"
                )
            else:
                st.write(response)
        
        st.success("✅ Analysis complete!")
        
        if len(response) > 100:
            with st.expander("📋 Copy Response Text"):
                st.code(response, language="text")

def render_research_interface(components: Dict[str, Any]):
    """Main research interface"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Research Configuration")
        
        # Mode selection
        mode, use_rag, use_pubmed = UIComponents.render_mode_selector(components)
        
        # Show mode status
        if use_rag:
            st.success("🚀 Enhanced RAG Mode")
        elif use_pubmed:
            st.info("📬 Simple PubMed Mode")
        else:
            st.info("🧠 Knowledge Mode")
        
        st.markdown("---")
        
        # Input fields
        general_question, pubmed_query = UIComponents.render_input_fields()
        
        # Advanced options
        options = UIComponents.render_advanced_options(use_pubmed)
        
        # Action buttons
        st.markdown("---")
        st.markdown("**🎯 Available Actions:**")
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if use_pubmed:
                button_text = "📚 Search Literature"
                if use_rag:
                    button_text += " (RAG Enhanced)"
                pubmed_button = st.button(button_text, type="primary", use_container_width=True)
                
                # Show what this button will do
                if use_rag:
                    st.caption(" Will use semantic search + AI analysis")
                else:
                    st.caption(" Will use simple literature analysis")
            else:
                pubmed_button = False
                st.button("📚 Literature Search", disabled=True, use_container_width=True)
                st.caption("⚠️ Requires PubMed mode")
                
        with btn_col2:
            general_button = st.button("🧠 Ask BioMistral", type="secondary", use_container_width=True)
            st.caption("💭 Uses AI knowledge only")
        
        # Show instructions for RAG
        if use_rag:
            st.success("🚀 **RAG Enhanced Mode Active!**")
            st.info("📋 **How RAG works:** Enter PubMed terms to retrieve papers, then ask your research question to analyze them.")
        elif use_pubmed:
            st.info("📬 **Simple PubMed Mode Active!** Will analyze papers directly.")
    
    with col2:
        st.subheader("🤖 Analysis Results")
        
        # Handle PubMed search
        if pubmed_button and pubmed_query.strip() and use_pubmed:
            with st.spinner("🔍 Processing literature..."):
                if use_rag and general_question.strip():
                    # RAG mode: Use PubMed query for retrieval, general question for analysis
                    st.info("🚀 RAG Mode: Using PubMed query for literature retrieval and general question for analysis")
                    response = PubMedProcessor.process_rag_query(
                        pubmed_query, general_question,  # Separate queries
                        components['retriever'], components['ai_model'],
                        components['rag_system'], components['storage'],
                        options.get('max_papers', CONFIG['MAX_PAPERS_DEFAULT']),
                        options.get('response_length', CONFIG['RESPONSE_LENGTH_DEFAULT'])
                    )
                elif use_rag and not general_question.strip():
                    # RAG mode but no general question - use PubMed query for both
                    st.warning("⚠️ RAG mode selected but no general question provided. Using PubMed query for analysis.")
                    response = PubMedProcessor.process_rag_query(
                        pubmed_query, pubmed_query,  # Use same query for both
                        components['retriever'], components['ai_model'],
                        components['rag_system'], components['storage'],
                        options.get('max_papers', CONFIG['MAX_PAPERS_DEFAULT']),
                        options.get('response_length', CONFIG['RESPONSE_LENGTH_DEFAULT'])
                    )
                else:
                    # Simple PubMed mode
                    response = PubMedProcessor.process_simple_query(
                        pubmed_query, components['retriever'], components['ai_model'],
                        options.get('max_papers', CONFIG['MAX_PAPERS_DEFAULT']),
                        options.get('response_length', CONFIG['RESPONSE_LENGTH_DEFAULT'])
                    )
                
                UIComponents.render_response(response, "pubmed")
        
        # Handle general question
        elif general_button and general_question.strip():
            with st.spinner("🤔 Processing question..."):
                if use_pubmed and not pubmed_query.strip():
                    st.warning("⚠️ PubMed mode selected but no query provided! Using knowledge mode.")
                
                response = PromptEngine.get_biomistral_response(
                    components['ai_model'], general_question,
                    options.get('response_length', CONFIG['RESPONSE_LENGTH_DEFAULT']),
                    max_retries=options.get('max_retries', CONFIG['MAX_RETRIES_DEFAULT'])
                )
                
                UIComponents.render_response(response, "general")
        
        # Show instructions
        else:
            if not (pubmed_button or general_button):
                st.info("👈 Choose your research approach")
                st.markdown("""
                **🧠 General Questions:** Knowledge-based answers from BioMistral
                **📚 Literature Research:** Evidence-based analysis from PubMed papers
                """)

def main():
    """Main application"""
    st.set_page_config(
        page_title="CIM PubMed Research Agent",
        page_icon='🔬',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Load theme
    load_custom_css()
    
    # Render header
    UIComponents.render_header()
    
    # Initialize components
    with st.spinner("🚀 Initializing system..."):
        components = initialize_system_components()
    
    # Show system status
    UIComponents.render_system_status(components)
    
    # Check if AI model is available
    if not components['ai_model']:
        st.error("❌ BioMistral not available. Please check system status.")
        with st.expander("🔧 Troubleshooting", expanded=True):
            st.markdown("""
            **Try these fixes:**
            1. Update PyTorch: `pip install torch --upgrade`
            2. Check model files
            3. Use CPU mode: `export CUDA_VISIBLE_DEVICES=""`
            """)
        st.stop()
    
    # Show capabilities
    if components['rag_system']:
        st.success("🚀 Full capabilities available!")
    elif components['retriever']:
        st.info("📬 PubMed + AI available")
    else:
        st.warning("🧠 AI-only mode")
    
    # Main interface
    render_research_interface(components)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; font-size: 0.9em; padding: 20px;'>
        🔬 <strong>CIM PubMed Research Agent</strong> | 
        Powered by BioMistral + RAG | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()