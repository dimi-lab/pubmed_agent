import streamlit as st
import os

# Ultra-simple app that only requires your offline AI model

def main():
    st.set_page_config(
        page_title="PubMed Abstract Screener",
        page_icon='🔬',
        layout='wide'
    )
    
    # Try to load AI model
    ai_model = None
    ai_error = None
    
    try:
        from components.llm_offline import OfflineAIModel
        ai_model = OfflineAIModel()
        ai_status = "✅ AI Assistant: Ready"
    except Exception as e:
        ai_error = str(e)
        ai_status = f"❌ AI Assistant: Failed - {ai_error}"
    
    # Header
    st.title("🔬 PubMed Abstract Screener")
    st.markdown("**AI-Powered Biomedical Research Assistant (Offline Mode)**")
    
    # Status
    with st.expander("🔍 System Status", expanded=bool(ai_error)):
        if ai_model:
            st.success(ai_status)
            st.info("💡 Running in offline mode with built-in biomedical knowledge")
        else:
            st.error(ai_status)
            st.markdown("**Troubleshooting:**")
            st.code("1. Check that components/llm_offline.py exists\n2. Ensure OfflineAIModel class is defined\n3. Try restarting the app")
    
    if not ai_model:
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Research Questions")
        
        # Quick example buttons
        st.markdown("**Try these examples:**")
        
        examples = [
            "How do cancer cells develop drug resistance?",
            "What role does gut microbiota play in diabetes?", 
            "How can AI help in drug discovery?",
            "What are the latest approaches in gene therapy?"
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
            placeholder="Type your question here...",
            height=100
        )
        
        ask_button = st.button("🧠 Ask AI Assistant", type="primary")
    
    with col2:
        st.subheader("🤖 AI Response")
        
        if ask_button and custom_question.strip():
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    response = ai_model.invoke(custom_question)
                    st.markdown("### Response:")
                    st.write(response)
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        elif ask_button:
            st.warning("Please enter a question first.")
        else:
            st.info("👈 Select an example or enter your own question to get started!")
    
    # Chat section
    st.markdown("---")
    st.subheader("💬 Interactive Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your biomedical research assistant. Ask me anything about medicine, biology, or health research!"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about biomedical research..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ai_model.invoke(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with additional features
    with st.sidebar:
        st.header("🛠️ Tools")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared! How can I help you?"}
            ]
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 About")
        st.markdown("""
        **Current Mode:** Offline Knowledge Base
        
        **Capabilities:**
        - Biomedical Q&A
        - Research guidance  
        - Scientific explanations
        - Medical terminology
        - Research methodology
        
        **Knowledge Areas:**
        - Cancer research
        - Neuroscience
        - Immunology
        - Genetics
        - Drug development
        - Clinical trials
        """)
        
        st.markdown("---")
        st.subheader("🎯 Tips")
        st.markdown("""
        **For best results:**
        - Be specific in your questions
        - Ask about mechanisms or processes
        - Request explanations of complex topics
        - Ask for research methodology guidance
        """)

if __name__ == "__main__":
    main()