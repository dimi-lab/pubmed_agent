import streamlit as st
import os
from typing import List, Optional
from datetime import datetime
import ssl
import urllib3

def apply_ssl_fixes():
    """Apply SSL certificate fixes for corporate networks"""
    try:
        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Disable urllib3 warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set environment variables
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_VERIFY'] = 'false'
        
        return True
    except Exception as e:
        print(f"Could not apply SSL fixes: {e}")
        return False

# Apply fixes when module loads
apply_ssl_fixes()

def initialize_system():
    """Initialize and test system components with better error handling"""
    status = {
        'pubmed_available': False,
        'ai_model_available': False,
        'pubmed_retriever': None,
        'ai_model': None,
        'error_messages': [],
        'connection_details': {}
    }
    
    # Test basic internet connectivity first
    internet_status = test_basic_connectivity()
    status['connection_details'].update(internet_status)
    
    # Test PubMed connectivity with multiple approaches
    if internet_status.get('basic_internet', False):
        pubmed_status = test_pubmed_connectivity()
        status['connection_details'].update(pubmed_status)
        
        if pubmed_status.get('pubmed_accessible', False):
            try:
                # Initialize PubMed retriever
                retriever = create_pubmed_retriever()
                if retriever:
                    status['pubmed_available'] = True
                    status['pubmed_retriever'] = retriever
                    status['connection_details']['pubmed_retriever'] = 'Successfully created'
                else:
                    status['error_messages'].append("PubMed retriever creation failed")
                    
            except Exception as e:
                status['error_messages'].append(f"PubMed retriever error: {str(e)}")
    else:
        status['error_messages'].append("No internet connection detected")
    
    # Test AI model (this should always work)
    try:
        from components.llm_offline import OfflineAIModel
        ai_model = OfflineAIModel()
        # Quick test
        test_response = ai_model.invoke("Hello")
        if test_response:
            status['ai_model_available'] = True
            status['ai_model'] = ai_model
            status['connection_details']['ai_model'] = 'Successfully loaded'
        
    except Exception as e:
        status['error_messages'].append(f"AI model error: {str(e)}")
    
    return status

def test_basic_connectivity():
    """Test basic internet connectivity"""
    results = {
        'basic_internet': False,
        'dns_resolution': False,
        'https_access': False
    }
    
    try:
        import requests
        
        # Test 1: Basic HTTP request
        try:
            response = requests.get("http://httpbin.org/get", timeout=10, verify=False)
            if response.status_code == 200:
                results['basic_internet'] = True
        except:
            pass
        
        # Test 2: DNS resolution
        try:
            import socket
            socket.gethostbyname("google.com")
            results['dns_resolution'] = True
        except:
            pass
        
        # Test 3: HTTPS access
        try:
            response = requests.get("https://httpbin.org/get", timeout=10, verify=False)
            if response.status_code == 200:
                results['https_access'] = True
        except:
            pass
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

def test_pubmed_connectivity():
    """Test PubMed specific connectivity"""
    results = {
        'pubmed_accessible': False,
        'ncbi_eutils': False,
        'metapub_working': False
    }
    
    try:
        import requests
        
        # Test 1: NCBI E-utilities access
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': 'cancer',
                'retmax': 1,
                'retmode': 'xml'
            }
            response = requests.get(url, params=params, timeout=15, verify=False)
            if response.status_code == 200 and 'xml' in response.headers.get('content-type', ''):
                results['ncbi_eutils'] = True
                results['pubmed_accessible'] = True
        except Exception as e:
            results['ncbi_error'] = str(e)
        
        # Test 2: Metapub library
        try:
            from metapub import PubMedFetcher
            fetcher = PubMedFetcher()
            
            # Try a simple search
            pmids = fetcher.pmids_for_query("cancer", retmax=1)
            if pmids and len(pmids) > 0:
                results['metapub_working'] = True
                results['pubmed_accessible'] = True
        except Exception as e:
            results['metapub_error'] = str(e)
            
    except Exception as e:
        results['general_error'] = str(e)
    
    return results

def create_pubmed_retriever():
    """Create PubMed retriever with error handling"""
    try:
        from metapub import PubMedFetcher
        from backend.abstract_retrieval.pubmed_retriever import PubMedAbstractRetriever
        
        # Create fetcher with retries
        fetcher = PubMedFetcher()
        
        # Test the fetcher
        test_pmids = fetcher.pmids_for_query("test", retmax=1)
        
        # Create retriever
        retriever = PubMedAbstractRetriever(fetcher, max_results=20)
        
        return retriever
        
    except Exception as e:
        print(f"Error creating PubMed retriever: {e}")
        return None


def main():
    st.set_page_config(
        page_title="PubMed Agent",
        page_icon='🔬',
        layout='wide'
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .mode-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 1rem;
        color: white;
    }
    .online-mode { background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); }
    .offline-mode { background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%); }
    .feature-highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .abstract-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .abstract-title {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .abstract-meta {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .abstract-content {
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize system components
    system_status = initialize_system()
    
    # Header section
    col1, col2 = st.columns([1, 3])
    
    with col1:
        logo_path = 'assets/pubmed-screener-logo.jpg'
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
        else:
            st.markdown("### 🔬")
            st.markdown("**PubMed**")
            st.markdown("**Screener**")
    
    with col2:
        mode_class = "online-mode" if system_status['pubmed_available'] else "offline-mode"
        mode_text = "🌐 ONLINE MODE" if system_status['pubmed_available'] else "🔒 OFFLINE MODE"
        
        st.markdown(f"""
        <div class="main-header">
            <h1>🔬 PubMed Abstract Screener</h1>
            <p>AI-Powered Biomedical Research Assistant</p>
            <span class="mode-badge {mode_class}">{mode_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights based on available modes
    if system_status['pubmed_available']:
        st.markdown("""
        <div class="feature-highlight">
        <strong>🌐 Online Features Available:</strong><br>
        ✅ Real-time PubMed literature search &nbsp;&nbsp;
        ✅ Latest research abstracts &nbsp;&nbsp;
        ✅ AI-powered query simplification &nbsp;&nbsp;
        ✅ Comprehensive biomedical database access
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="feature-highlight">
        <strong>🔒 Offline Features:</strong><br>
        ✅ Built-in biomedical knowledge base &nbsp;&nbsp;
        ✅ Expert responses on research topics &nbsp;&nbsp;
        ✅ No internet required &nbsp;&nbsp;
        ✅ Privacy-focused local processing
        </div>
        """, unsafe_allow_html=True)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Literature Search", "💬 AI Assistant", "⚙️ Settings"])
    
    with tab1:
        literature_search_interface(system_status)
    
    with tab2:
        ai_assistant_interface(system_status)
    
    with tab3:
        settings_interface(system_status)

def initialize_system():
    """Initialize and test system components"""
    status = {
        'pubmed_available': False,
        'ai_model_available': False,
        'pubmed_retriever': None,
        'ai_model': None,
        'error_messages': []
    }
    
    # Test PubMed connectivity
    try:
        from metapub import PubMedFetcher
        from backend.abstract_retrieval.pubmed_retriever import PubMedAbstractRetriever
        
        # Test with a simple query
        pubmed_fetcher = PubMedFetcher()
        retriever = PubMedAbstractRetriever(pubmed_fetcher, max_results=5)
        
        # Quick connectivity test
        test_pmids = retriever._get_abstract_list("cancer", simplify_query=False)
        if test_pmids:
            status['pubmed_available'] = True
            status['pubmed_retriever'] = retriever
        
    except Exception as e:
        status['error_messages'].append(f"PubMed unavailable: {str(e)}")
    
    # Test AI model
    try:
        from components.llm_offline import OfflineAIModel
        ai_model = OfflineAIModel()
        status['ai_model_available'] = True
        status['ai_model'] = ai_model
        
    except Exception as e:
        status['error_messages'].append(f"AI model unavailable: {str(e)}")
    
    return status

def literature_search_interface(system_status):
    """Interface for literature search functionality"""
    st.subheader("📚 Scientific Literature Search")
    
    if not system_status['pubmed_available']:
        st.warning("⚠️ PubMed search is currently unavailable. Please check your internet connection.")
        st.info("You can still use the AI Assistant tab for biomedical questions using our offline knowledge base.")
        return
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., What are the latest treatments for Alzheimer's disease?",
            help="Enter your research question in natural language. The system will automatically optimize it for PubMed search."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        search_button = st.button("🔍 Search Literature", type="primary")
    
    # Advanced options
    with st.expander("🔧 Advanced Search Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_results = st.slider("Maximum results", 5, 100, 20)
        
        with col2:
            simplify_query = st.checkbox("Auto-simplify query", value=True, 
                                       help="Automatically optimize your query for better PubMed results")
        
        with col3:
            year_filter = st.selectbox("Publication year", 
                                     ["All years", "Last 5 years", "Last 10 years", "Custom range"])
    
    # Perform search
    if search_button and search_query:
        with st.spinner("🔍 Searching PubMed database..."):
            try:
                # Update retriever with new max_results
                retriever = system_status['pubmed_retriever']
                retriever.max_results = max_results
                
                # Perform search
                abstracts = retriever.get_abstract_data(search_query, simplify_query=simplify_query)
                
                # Apply year filtering if specified
                if year_filter == "Last 5 years":
                    current_year = datetime.now().year
                    abstracts = [a for a in abstracts if a.year and a.year >= current_year - 5]
                elif year_filter == "Last 10 years":
                    current_year = datetime.now().year
                    abstracts = [a for a in abstracts if a.year and a.year >= current_year - 10]
                
                # Store results in session state
                st.session_state.search_results = abstracts
                st.session_state.search_query = search_query
                
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                return
    
    # Display search results
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
        display_search_results(st.session_state.search_results, st.session_state.search_query)

def display_search_results(abstracts, query):
    """Display search results in a formatted way"""
    st.markdown("---")
    st.subheader(f"📄 Search Results ({len(abstracts)} abstracts found)")
    st.caption(f"Query: {query}")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Results", len(abstracts))
    with col2:
        years = [a.year for a in abstracts if a.year]
        avg_year = sum(years) / len(years) if years else 0
        st.metric("Average Year", f"{avg_year:.0f}" if avg_year else "N/A")
    with col3:
        with_doi = len([a for a in abstracts if a.doi])
        st.metric("With DOI", with_doi)
    with col4:
        unique_journals = len(set(a.journal for a in abstracts if a.journal))
        st.metric("Unique Journals", unique_journals)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📊 Analyze with AI"):
            analyze_abstracts_with_ai(abstracts)
    
    with col2:
        if st.button("📝 Export Results"):
            export_results(abstracts)
    
    # Display individual abstracts
    for i, abstract in enumerate(abstracts, 1):
        with st.container():
            st.markdown(f"""
            <div class="abstract-card">
                <div class="abstract-title">{i}. {abstract.title or 'No title available'}</div>
                <div class="abstract-meta">
                    <strong>Authors:</strong> {', '.join(abstract.authors) if abstract.authors else 'N/A'} | 
                    <strong>Year:</strong> {abstract.year or 'N/A'} | 
                    <strong>Journal:</strong> {abstract.journal or 'N/A'}
                    {f' | <strong>DOI:</strong> {abstract.doi}' if abstract.doi else ''}
                </div>
                <div class="abstract-content">{abstract.abstract_content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons for each abstract
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if abstract.url:
                    st.link_button("🔗 View on PubMed", abstract.url)
            with col2:
                if st.button(f"💬 Discuss with AI", key=f"discuss_{i}"):
                    discuss_abstract_with_ai(abstract)

def analyze_abstracts_with_ai(abstracts):
    """Analyze search results with AI assistant"""
    if 'ai_model' not in st.session_state:
        from components.llm_offline import OfflineAIModel
        st.session_state.ai_model = OfflineAIModel()
    
    # Create summary of abstracts
    summary_text = f"Please analyze these {len(abstracts)} research abstracts:\n\n"
    for i, abstract in enumerate(abstracts[:5], 1):  # Limit to first 5 for analysis
        summary_text += f"{i}. {abstract.title}\nAbstract: {abstract.abstract_content[:300]}...\n\n"
    
    analysis_query = f"Based on these research papers, what are the main themes, findings, and research directions? {summary_text}"
    
    with st.spinner("🤔 AI is analyzing the research papers..."):
        response = st.session_state.ai_model.invoke(analysis_query)
        
        st.markdown("### 🧠 AI Analysis of Search Results")
        st.markdown(response)

def discuss_abstract_with_ai(abstract):
    """Start a discussion about a specific abstract with AI"""
    st.markdown(f"### 💬 Discussing: {abstract.title}")
    
    if 'ai_model' not in st.session_state:
        from components.llm_offline import OfflineAIModel
        st.session_state.ai_model = OfflineAIModel()
    
    discussion_prompt = f"""
    Please provide insights about this research paper:
    
    Title: {abstract.title}
    Authors: {', '.join(abstract.authors) if abstract.authors else 'N/A'}
    Year: {abstract.year}
    
    Abstract: {abstract.abstract_content}
    
    What are the key findings, implications, and potential follow-up research questions?
    """
    
    response = st.session_state.ai_model.invoke(discussion_prompt)
    st.markdown(response)

def ai_assistant_interface(system_status):
    """AI assistant chat interface"""
    st.subheader("💬 AI Research Assistant")
    
    if not system_status['ai_model_available']:
        st.error("❌ AI assistant is currently unavailable.")
        return
    
    # Initialize AI model in session state
    if "ai_model" not in st.session_state:
        st.session_state.ai_model = system_status['ai_model']
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """Hello! I'm your biomedical research assistant. I can help you with:

• **Literature Analysis** - Interpret research findings and methodologies
• **Research Questions** - Help formulate and refine research hypotheses  
• **Study Design** - Guidance on experimental approaches
• **Data Interpretation** - Explain statistical concepts and results
• **Background Knowledge** - Provide context on biomedical topics

What would you like to explore today?"""
            }
        ]
    
    # Quick action buttons
    st.markdown("**Quick Topics:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧬 Cancer Research"):
            st.session_state.quick_question = "What are the latest approaches in cancer immunotherapy?"
    with col2:
        if st.button("🧠 Neuroscience"):
            st.session_state.quick_question = "How do biomarkers help in early Alzheimer's detection?"
    with col3:
        if st.button("💊 Drug Development"):
            st.session_state.quick_question = "What are the main challenges in drug development today?"
    with col4:
        if st.button("📊 Research Methods"):
            st.session_state.quick_question = "How do I design a good clinical trial?"
    
    # Handle quick questions
    if hasattr(st.session_state, 'quick_question'):
        st.session_state.messages.append({
            "role": "user",
            "content": st.session_state.quick_question
        })
        
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.ai_model.invoke(st.session_state.quick_question)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })
        
        del st.session_state.quick_question
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about biomedical research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.ai_model.invoke(prompt)
                st.write(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})

def settings_interface(system_status):
    """Settings and system status interface"""
    st.subheader("⚙️ System Settings & Status")
    
    # System status
    st.markdown("### 📊 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌐 PubMed Access:**")
        if system_status['pubmed_available']:
            st.success("✅ Online - Ready for literature search")
        else:
            st.error("❌ Offline - Check internet connection")
        
        st.markdown("**🤖 AI Assistant:**")
        if system_status['ai_model_available']:
            st.success("✅ Available - Ready for queries")
        else:
            st.error("❌ Unavailable - Check model files")
    
    with col2:
        st.markdown("**📋 Error Messages:**")
        if system_status['error_messages']:
            for error in system_status['error_messages']:
                st.warning(f"⚠️ {error}")
        else:
            st.info("No errors detected")
    
    # Settings
    st.markdown("### ⚙️ Preferences")
    
    with st.expander("🔍 Search Settings", expanded=False):
        st.slider("Default max results", 5, 100, 20, key="default_max_results")
        st.checkbox("Enable query simplification by default", value=True, key="default_simplify")
        st.selectbox("Default year filter", ["All years", "Last 5 years", "Last 10 years"], key="default_year_filter")
    
    with st.expander("💬 AI Assistant Settings", expanded=False):
        st.slider("Response length preference", 1, 5, 3, key="response_length", 
                 help="1=Concise, 5=Detailed")
        st.checkbox("Include citations in responses", value=True, key="include_citations")
    
    # Actions
    st.markdown("### 🔧 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh System"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat History"):
            if "messages" in st.session_state:
                st.session_state.messages = st.session_state.messages[:1]  # Keep welcome message
            st.success("Chat history cleared!")
    
    with col3:
        if st.button("🌐 Test Connection"):
            test_system_connectivity()

def test_system_connectivity():
    """Test system connectivity"""
    with st.spinner("Testing connections..."):
        results = {}
        
        # Test internet
        try:
            import requests
            response = requests.get("https://httpbin.org/get", timeout=5)
            results['internet'] = response.status_code == 200
        except:
            results['internet'] = False
        
        # Test PubMed
        try:
            from metapub import PubMedFetcher
            fetcher = PubMedFetcher()
            test_pmids = fetcher.pmids_for_query("cancer", retmax=1)
            results['pubmed'] = len(test_pmids) > 0
        except:
            results['pubmed'] = False
        
        # Display results
        st.markdown("**Connection Test Results:**")
        for service, status in results.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {service.title()}: {'Connected' if status else 'Failed'}")

def export_results(abstracts):
    """Export search results"""
    if not abstracts:
        st.warning("No results to export")
        return
    
    # Create export data
    export_data = []
    for abstract in abstracts:
        export_data.append({
            'Title': abstract.title,
            'Authors': ', '.join(abstract.authors) if abstract.authors else '',
            'Year': abstract.year,
            'Journal': abstract.journal,
            'DOI': abstract.doi,
            'PMID': abstract.pmid,
            'Abstract': abstract.abstract_content,
            'URL': abstract.url
        })
    
    # Convert to CSV
    import pandas as pd
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"pubmed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()