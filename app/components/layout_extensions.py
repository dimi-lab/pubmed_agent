import streamlit as st

def render_app_info():
    """Render application information with enhanced styling"""
    st.title("🔬 PubMed Screener")
    
    st.markdown("""
    **AI-Powered Biomedical Research Assistant**
    
    This tool combines PubMed literature search with advanced AI analysis to help you:
    - Find relevant scientific literature
    - Get evidence-based answers to research questions
    - Explore complex biomedical topics through interactive chat
    """)
    
    # Enhanced example questions with better styling
    st.markdown("""
    <style>
    .example-questions {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .question-item {
        margin: 0.5rem 0;
        padding: 0.5rem;
        background-color: white;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .question-category {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.25rem;
    }
    </style>
    
    <div class="example-questions">
        <h4>💡 Example Research Questions</h4>
        
        <div class="question-item">
            <div class="question-category">🧠 Neuroscience</div>
            How can advanced imaging techniques and biomarkers be leveraged for early diagnosis and monitoring of disease progression in neurodegenerative disorders?
        </div>
        
        <div class="question-item">
            <div class="question-category">🧬 Regenerative Medicine</div>
            What are the potential applications of stem cell technology and regenerative medicine in the treatment of neurodegenerative diseases, and what are the associated challenges?
        </div>
        
        <div class="question-item">
            <div class="question-category">🦠 Microbiome Research</div>
            What are the roles of gut microbiota and the gut-brain axis in the pathogenesis of type 1 and type 2 diabetes, and how can these interactions be modulated for therapeutic benefit?
        </div>
        
        <div class="question-item">
            <div class="question-category">🎯 Cancer Therapeutics</div>
            What are the molecular mechanisms underlying the development of resistance to targeted cancer therapies, and how can these resistance mechanisms be overcome?
        </div>
    </div>
    """, unsafe_allow_html=True)