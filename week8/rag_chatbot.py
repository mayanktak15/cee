"""
Loan Approval RAG Chatbot - Week 8 Project
Celebal Technologies Data Science Internship

This is my learning project to understand:
- How chatbots work with real data
- Basic Natural Language Processing
- Building simple AI applications
- Working with financial datasets

Author: Data Science Intern
Goal: Learn by building a functional loan approval Q&A system
"""

from document_retriever import DocumentRetriever
from response_generator import ResponseGenerator, HuggingFaceLightModel
import pandas as pd
from typing import List, Dict, Any
import streamlit as st
import os

class LoanApprovalRAGChatbot:
    """
    Main chatbot class - this brings everything together!
    Think of this as the coordinator that takes your question,
    finds relevant information, and gives you a helpful answer.
    """
    
    def __init__(self, model_type='huggingface', model_name='distilgpt2'):
        """
        Set up the chatbot
        
        Args:
            model_type (str): What kind of AI to use ('openai', 'huggingface', 'simple')
            model_name (str): Which specific AI model
        """
        self.retriever = DocumentRetriever()
        
        if model_type == 'simple':
            self.generator = HuggingFaceLightModel()
        else:
            self.generator = ResponseGenerator(model_type=model_type, model_name=model_name)
        
        self.model_type = model_type
        self.conversation_history = []
    
    def setup(self, train_path: str, test_path: str, force_rebuild: bool = False):
        """
        Setup the chatbot with data
        
        Args:
            train_path (str): Path to training dataset
            test_path (str): Path to test dataset
            force_rebuild (bool): Force rebuild of index
        """
        index_path = "loan_rag_index"
        
        if not force_rebuild and self._index_exists(index_path):
            print("Loading existing index...")
            try:
                self.retriever.load_index(index_path)
                print("Index loaded successfully!")
                return
            except Exception as e:
                print(f"Error loading index: {e}")
                print("Rebuilding index...")
        
        print("Building new index...")
        self.retriever.load_and_process_data(train_path, test_path)
        self.retriever.save_index(index_path)
        print("Index built and saved successfully!")
    
    def _index_exists(self, path: str) -> bool:
        """Check if index files exist"""
        return (os.path.exists(f"{path}_state.pkl") and 
                os.path.exists(f"{path}_faiss.index") and 
                os.path.exists(f"{path}_embeddings.npy"))
    
    def ask_question(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Ask a question to the chatbot
        
        Args:
            query (str): User question
            top_k (int): Number of documents to retrieve
            
        Returns:
            Dict: Response with answer and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve_documents(query, top_k)
        
        # Generate response
        if self.model_type == 'simple':
            context = self.retriever._prepare_context(retrieved_docs)
            answer = self.generator.generate_simple_response(query, context)
        else:
            answer = self.generator.generate_response(query, retrieved_docs)
        
        # Store in conversation history
        conversation_entry = {
            'question': query,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'top_k': top_k
        }
        self.conversation_history.append(conversation_entry)
        
        return conversation_entry
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset"""
        train_df = self.retriever.train_df
        test_df = self.retriever.test_df
        
        summary = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'features': list(train_df.columns),
            'target_distribution': train_df['Loan_Status'].value_counts().to_dict(),
            'missing_values': train_df.isnull().sum().to_dict(),
            'approval_rate': (train_df['Loan_Status'] == 'Y').mean()
        }
        
        return summary
    
    def suggest_questions(self) -> List[str]:
        """Suggest sample questions users can ask"""
        questions = [
            "What is the overall loan approval rate?",
            "How does gender affect loan approval?",
            "What is the impact of education on loan approval?",
            "How does credit history affect loan approval chances?",
            "What are the income requirements for loan approval?",
            "How does property area (urban vs rural) affect approval?",
            "What is the average loan amount approved?",
            "How does marital status impact loan approval?",
            "What factors lead to loan rejection?",
            "How does self-employment status affect approval?",
            "What is the relationship between income and loan amount?",
            "How many dependents typically get approved?",
            "What is the most common loan term?",
            "Show me examples of approved vs rejected loans",
            "What are the key predictors of loan approval?"
        ]
        return questions

def create_streamlit_app():
    """Create Streamlit web interface"""
    
    st.set_page_config(
        page_title="Loan Approval RAG Chatbot",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 Loan Approval RAG Chatbot")
    st.markdown("Ask questions about loan approval patterns and get intelligent responses!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = LoanApprovalRAGChatbot(model_type='simple')
            
            # Setup with default paths
            train_path = "Training Dataset.csv"
            test_path = "Test Dataset.csv"
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                st.session_state.chatbot.setup(train_path, test_path)
                st.success("Chatbot initialized successfully!")
            else:
                st.error("Dataset files not found. Please ensure 'Training Dataset.csv' and 'Test Dataset.csv' are in the current directory.")
                st.stop()
    
    # Sidebar with dataset info
    with st.sidebar:
        st.header("📊 Dataset Information")
        
        if hasattr(st.session_state.chatbot, 'retriever') and hasattr(st.session_state.chatbot.retriever, 'train_df'):
            summary = st.session_state.chatbot.get_dataset_summary()
            
            st.metric("Training Samples", summary['train_size'])
            st.metric("Test Samples", summary['test_size'])
            st.metric("Approval Rate", f"{summary['approval_rate']:.1%}")
            
            # Target distribution
            st.subheader("Loan Status Distribution")
            approved = summary['target_distribution'].get('Y', 0)
            rejected = summary['target_distribution'].get('N', 0)
            st.write(f"✅ Approved: {approved}")
            st.write(f"❌ Rejected: {rejected}")
        
        st.subheader("💡 Sample Questions")
        questions = st.session_state.chatbot.suggest_questions()
        selected_question = st.selectbox("Choose a sample question:", [""] + questions)
        
        if st.button("Use Selected Question") and selected_question:
            st.session_state.current_question = selected_question
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with the Bot")
        
        # Question input
        if 'current_question' in st.session_state:
            question = st.text_area("Your Question:", value=st.session_state.current_question, height=100)
            del st.session_state.current_question
        else:
            question = st.text_area("Your Question:", height=100)
        
        col_ask, col_clear = st.columns([1, 1])
        
        with col_ask:
            if st.button("🔍 Ask Question", type="primary"):
                if question.strip():
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.ask_question(question)
                        
                        # Display response
                        st.subheader("🤖 Answer:")
                        st.write(response['answer'])
                        
                        # Display retrieved documents
                        with st.expander("📚 Retrieved Context (Top 3)"):
                            for i, doc in enumerate(response['retrieved_docs'][:3]):
                                st.write(f"**Document {i+1} (Score: {doc['score']:.3f})**")
                                st.text(doc['document'][:500] + "..." if len(doc['document']) > 500 else doc['document'])
                                st.divider()
                else:
                    st.warning("Please enter a question!")
        
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.chatbot.conversation_history = []
                st.success("Conversation history cleared!")
    
    with col2:
        st.subheader("📜 Conversation History")
        
        if st.session_state.chatbot.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.chatbot.conversation_history[-5:])):
                with st.expander(f"Q{len(st.session_state.chatbot.conversation_history)-i}: {entry['question'][:50]}..."):
                    st.write("**Question:**", entry['question'])
                    st.write("**Answer:**", entry['answer'])
        else:
            st.info("No conversation history yet. Ask a question to get started!")

def main():
    """Main function to run the chatbot"""
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we're here, we're running in Streamlit
        create_streamlit_app()
        return
    except:
        pass
    
    # Command line interface
    print("🏦 Loan Approval RAG Chatbot")
    print("Data Science Internship Project - Week 8")
    print("Celebal Technologies")
    print("=" * 60)
    
    # Initialize chatbot
    print("Initializing RAG chatbot system...")
    chatbot = LoanApprovalRAGChatbot(model_type='simple')
    
    # Setup with data
    train_path = "Training Dataset.csv"
    test_path = "Test Dataset.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("❌ Dataset files not found!")
        print("Please ensure 'Training Dataset.csv' and 'Test Dataset.csv' are in the current directory.")
        return
    
    chatbot.setup(train_path, test_path)
    print("✅ Chatbot ready!")
    
    # Display dataset summary
    summary = chatbot.get_dataset_summary()
    print(f"\\n📊 Dataset Summary:")
    print(f"Training samples: {summary['train_size']}")
    print(f"Test samples: {summary['test_size']}")
    print(f"Approval rate: {summary['approval_rate']:.1%}")
    
    # Suggest questions
    print(f"\\n💡 Sample questions:")
    for i, q in enumerate(chatbot.suggest_questions()[:5], 1):
        print(f"{i}. {q}")
    
    print(f"\\n💬 Start asking questions (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        question = input("\\n🔍 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("🤖 Thinking...")
        response = chatbot.ask_question(question)
        
        print(f"\\n📝 Answer:")
        print(response['answer'])
        
        print(f"\\n📚 Top retrieved documents:")
        for i, doc in enumerate(response['retrieved_docs'][:2], 1):
            print(f"\\n{i}. Score: {doc['score']:.3f}")
            print(doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document'])

if __name__ == "__main__":
    main()
