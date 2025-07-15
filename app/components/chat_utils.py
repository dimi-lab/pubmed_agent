import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import Runnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage
import re

class ChatAgent:
    def __init__(self, prompt, llm: Runnable, is_chat_model=True):
        """Initialize the ChatAgent.
        
        Args:
            prompt: The prompt template (ChatPromptTemplate or PromptTemplate)
            llm (Runnable): The language model runnable
            is_chat_model (bool): Whether the model supports chat format or is a simple text generator
        """
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.llm = llm
        self.prompt = prompt
        self.is_chat_model = is_chat_model
        self.output_parser = StrOutputParser()
        self.chain = self.setup_chain()

    def setup_chain(self) -> RunnableWithMessageHistory:
        """Set up the chain for the ChatAgent.
        
        Returns:
            RunnableWithMessageHistory: The configured chain with message history.
        """
        if self.is_chat_model:
            # For chat models (works with ChatPromptTemplate)
            chain = self.prompt | self.llm | self.output_parser
            return RunnableWithMessageHistory(
                chain,
                lambda session_id: self.history,
                input_messages_key="question",
                history_messages_key="history",
            )
        else:
            # For simple text generation models (works with PromptTemplate)
            chain = self.prompt | self.llm | self.output_parser
            return RunnableWithMessageHistory(
                chain,
                lambda session_id: self.history,
                input_messages_key="question",
                history_messages_key="history",
            )

    def format_history_for_simple_model(self):
        """Format chat history for simple text generation models."""
        if not self.history.messages:
            return ""
        
        formatted_history = []
        for msg in self.history.messages:
            if msg.type == "human":
                formatted_history.append(f"Human: {msg.content}")
            elif msg.type == "ai":
                formatted_history.append(f"Assistant: {msg.content}")
        
        return "\n".join(formatted_history)

    def clean_model_response(self, response):
        """Clean up the model response for better display."""
        if isinstance(response, str):
            # Remove common artifacts from text generation models
            response = response.strip()
            
            # Remove repeated patterns
            response = re.sub(r'(.+?)\1{2,}', r'\1', response)
            
            # Stop at natural conversation boundaries
            stop_patterns = [
                r'\nHuman:',
                r'\nUser:',
                r'\n\nHuman:',
                r'\n\nUser:',
                '<|endoftext|>',
                '</s>',
            ]
            
            for pattern in stop_patterns:
                response = re.split(pattern, response, 1)[0]
            
            # Clean up any remaining formatting
            response = response.replace('Assistant:', '').strip()
            
            return response
        
        return str(response)

    def display_messages(self):
        """Display messages in the chat interface.
        
        If no messages are present, adds a default AI message.
        """
        if len(self.history.messages) == 0:
            self.history.add_ai_message("How can I help you?")
        
        for msg in self.history.messages:
            st.chat_message(msg.type).write(msg.content)

    def start_conversation(self):
        """Start a conversation in the chat interface.
        
        Displays messages, prompts user for input, and handles AI response.
        """
        self.display_messages()
        
        user_question = st.chat_input(placeholder="Ask me anything!")
        
        if user_question:
            st.chat_message("human").write(user_question)
            
            with st.spinner("Thinking..."):
                try:
                    config = {"configurable": {"session_id": "any"}}
                    
                    if self.is_chat_model:
                        # For chat models
                        response = self.chain.invoke({"question": user_question}, config)
                    else:
                        # For simple text models, format history manually
                        formatted_history = self.format_history_for_simple_model()
                        response = self.chain.invoke({
                            "question": user_question,
                            "history": formatted_history
                        }, config)
                    
                    # Clean up the response
                    cleaned_response = self.clean_model_response(response)
                    
                    if cleaned_response.strip():
                        st.chat_message("ai").write(cleaned_response)
                    else:
                        st.chat_message("ai").write("I'm sorry, I couldn't generate a proper response. Could you try rephrasing your question?")
                        
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.chat_message("ai").write("I encountered an error. Please try again.")

# Factory function to create ChatAgent with appropriate settings
def create_chat_agent(llm, model_type="chat"):
    """Create a ChatAgent with appropriate prompt and settings.
    
    Args:
        llm: The language model
        model_type: "chat" for chat models, "text" for simple text generation
    
    Returns:
        ChatAgent: Configured chat agent
    """
    if model_type == "chat":
        from components.prompts import create_chat_prompt_template
        prompt = create_chat_prompt_template()
        return ChatAgent(prompt, llm, is_chat_model=True)
    else:
        from components.prompts import create_simple_prompt_template
        prompt = create_simple_prompt_template()
        return ChatAgent(prompt, llm, is_chat_model=False)

# Usage examples:
# For DialoGPT or other conversational models:
# agent = create_chat_agent(llm, model_type="chat")

# For GPT-2, DistilGPT-2, or basic text generation models:
# agent = create_chat_agent(llm, model_type="text")