from typing import List, Optional, Union
import streamlit as st
from langchain_core.documents.base import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import Output
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import VectorStore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatAgent:
    """Enhanced ChatAgent with better error handling and offline support"""
    
    def __init__(self, prompt: ChatPromptTemplate, llm: Runnable, use_offline_fallback: bool = True):
        """
        Initialize the Enhanced ChatAgent.

        Args:
            prompt (ChatPromptTemplate): The chat prompt template
            llm (Runnable): The language model runnable
            use_offline_fallback (bool): Whether to use offline AI as fallback
        """
        try:
            self.history = StreamlitChatMessageHistory(key="chat_history")
        except Exception as e:
            logger.warning(f"Failed to initialize StreamlitChatMessageHistory: {e}")
            # Fallback to session state
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            self.history = None
        
        self.llm = llm
        self.prompt = prompt
        self.use_offline_fallback = use_offline_fallback
        self.offline_model = None
        
        # Initialize offline model if needed
        if use_offline_fallback:
            try:
                from components.llm_offline import OfflineAIModel
                self.offline_model = OfflineAIModel()
                logger.info("Offline AI model loaded as fallback")
            except Exception as e:
                logger.warning(f"Failed to load offline model: {e}")
        
        self.chain = self.setup_chain()
    
    def reset_history(self) -> None:
        """Clean up chat history to start new chat session"""
        try:
            if self.history:
                self.history.clear()
            else:
                st.session_state.chat_messages = []
        except Exception as e:
            logger.error(f"Failed to reset history: {e}")

    def setup_chain(self) -> Optional[RunnableWithMessageHistory]:
        """Set up the chain for the ChatAgent with error handling"""
        try:
            if not self.history:
                return None
            
            chain = self.prompt | self.llm
            return RunnableWithMessageHistory(
                chain,
                lambda session_id: self.history,
                input_messages_key="question",
                history_messages_key="history",
            )
        except Exception as e:
            logger.error(f"Failed to setup chain: {e}")
            return None

    def display_messages(self, selected_query: str) -> None:
        """Display messages in the chat interface with fallback handling"""
        try:
            if self.history:
                if len(self.history.messages) == 0:
                    self.history.add_ai_message(f"Let's discuss your query: {selected_query}")
                
                for msg in self.history.messages:
                    st.chat_message(msg.type).write(msg.content)
            else:
                # Fallback to session state
                if not st.session_state.chat_messages:
                    welcome_msg = f"Let's discuss your query: {selected_query}"
                    st.session_state.chat_messages.append({"role": "assistant", "content": welcome_msg})
                
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
        except Exception as e:
            logger.error(f"Failed to display messages: {e}")
            st.error("Error displaying chat history")
    
    def format_retrieved_abstracts_for_prompt(self, documents: List[Document]) -> str:
        """Format retrieved documents for LLM prompt with better error handling"""
        if not documents:
            return "No relevant abstracts were retrieved."
        
        formatted_strings = []
        for i, doc in enumerate(documents):
            try:
                title = doc.metadata.get('title', f'Untitled Abstract {i+1}')
                content = doc.page_content if doc.page_content else 'No content available'
                doi = doc.metadata.get('source') or doc.metadata.get('doi', 'DOI not available')
                
                formatted_str = f"ABSTRACT TITLE: {title}; ABSTRACT CONTENT: {content}; ABSTRACT DOI: {doi}"
                formatted_strings.append(formatted_str)
            except Exception as e:
                logger.warning(f"Error formatting document {i}: {e}")
                continue
        
        return "\n\n".join(formatted_strings) if formatted_strings else "No abstracts could be formatted."
    
    def get_answer_from_llm(self, question: str, retrieved_documents: List[Document]) -> Union[Output, str]:
        """Get response from LLM with offline fallback"""
        try:
            if self.chain:
                config = {"configurable": {"session_id": "any"}}
                formatted_abstracts = self.format_retrieved_abstracts_for_prompt(retrieved_documents)
                
                response = self.chain.invoke({
                    "question": question, 
                    "retrieved_abstracts": formatted_abstracts,
                }, config)
                
                return response
            else:
                raise Exception("Chain not available")
                
        except Exception as e:
            logger.error(f"LLM response failed: {e}")
            
            # Fallback to offline model
            if self.offline_model:
                try:
                    formatted_abstracts = self.format_retrieved_abstracts_for_prompt(retrieved_documents)
                    fallback_prompt = f"""Based on these research abstracts, please answer: {question}

Research Literature:
{formatted_abstracts}

Please provide a comprehensive answer based on the available literature."""
                    
                    offline_response = self.offline_model.invoke(fallback_prompt)
                    return type('Response', (), {'content': f"[Offline Mode] {offline_response}"})()
                    
                except Exception as offline_error:
                    logger.error(f"Offline fallback failed: {offline_error}")
            
            # Final fallback
            return type('Response', (), {'content': f"I apologize, but I encountered an error processing your question. Error: {str(e)}"})()
    
    def retrieve_documents(self, retriever: VectorStore, question: str, cut_off: int = 5) -> List[Document]:
        """Retrieve documents using similarity search with error handling"""
        try:
            if not retriever:
                return []
            
            results = retriever.similarity_search(question, k=cut_off)
            logger.info(f"Retrieved {len(results)} documents for question: {question[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []

    def start_conversation(self, retriever: Optional[VectorStore], selected_query: str) -> None:
        """Start conversation with enhanced error handling"""
        try:
            self.display_messages(selected_query)
            
            user_question = st.chat_input(placeholder="Ask me anything about the research...")
            
            if user_question:
                # Display user message
                st.chat_message("human").write(user_question)
                
                # Add to history
                if self.history:
                    # History managed by StreamlitChatMessageHistory
                    pass
                else:
                    st.session_state.chat_messages.append({"role": "user", "content": user_question})
                
                # Get response
                with st.spinner("🤔 Analyzing the research..."):
                    if retriever:
                        documents = self.retrieve_documents(retriever, user_question)
                        response = self.get_answer_from_llm(user_question, documents)
                    else:
                        # No retriever available, use offline model directly
                        if self.offline_model:
                            response_text = self.offline_model.invoke(user_question)
                            response = type('Response', (), {'content': f"[Knowledge Base] {response_text}"})()
                        else:
                            response = type('Response', (), {'content': "I'm sorry, but I cannot access the research database at the moment."})()
                
                # Display response
                response_content = response.content if hasattr(response, 'content') else str(response)
                st.chat_message("ai").write(response_content)
                
                # Add to history
                if not self.history:
                    st.session_state.chat_messages.append({"role": "assistant", "content": response_content})
                
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            st.error(f"An error occurred during the conversation: {str(e)}")