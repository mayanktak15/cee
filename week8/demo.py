"""
Demo script to test the RAG chatbot functionality
"""

from rag_chatbot import LoanApprovalRAGChatbot
import os

def demo_rag_chatbot():
    """Demo function to showcase the RAG chatbot"""
    
    print("🏦 Loan Approval RAG Chatbot Demo")
    print("=" * 50)
    
    # Check if dataset files exist
    train_path = "Training Dataset.csv"
    test_path = "Test Dataset.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("❌ Dataset files not found!")
        print("Please ensure 'Training Dataset.csv' and 'Test Dataset.csv' are in the current directory.")
        return
    
    # Initialize chatbot
    print("🔧 Initializing RAG chatbot...")
    chatbot = LoanApprovalRAGChatbot(model_type='simple')
    
    # Setup with data
    print("📊 Loading and processing dataset...")
    chatbot.setup(train_path, test_path)
    
    # Display dataset summary
    summary = chatbot.get_dataset_summary()
    print(f"\\n📈 Dataset Summary:")
    print(f"   Training samples: {summary['train_size']}")
    print(f"   Test samples: {summary['test_size']}")
    print(f"   Overall approval rate: {summary['approval_rate']:.1%}")
    print(f"   Approved loans: {summary['target_distribution'].get('Y', 0)}")
    print(f"   Rejected loans: {summary['target_distribution'].get('N', 0)}")
    
    # Demo questions and answers
    demo_questions = [
        "What is the overall loan approval rate?",
        "How does credit history affect loan approval?",
        "What is the impact of education on loan approval?",
        "How does gender affect loan approval rates?",
        "What are the key factors for loan rejection?"
    ]
    
    print(f"\\n🤖 Demo Q&A Session:")
    print("-" * 50)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\\n{i}. 🔍 Question: {question}")
        print("   🤔 Analyzing...")
        
        response = chatbot.ask_question(question)
        
        print(f"   📝 Answer: {response['answer']}")
        
        # Show top retrieved document
        if response['retrieved_docs']:
            top_doc = response['retrieved_docs'][0]
            print(f"   📚 Top Context (Score: {top_doc['score']:.3f}):")
            print(f"      {top_doc['document'][:150]}...")
        
        print("-" * 30)
    
    print(f"\\n✅ Demo completed!")
    print(f"💡 You can now run 'streamlit run streamlit_app.py' for the full web interface")
    
    # Interactive mode
    print(f"\\n💬 Interactive Mode (type 'quit' to exit):")
    while True:
        user_question = input("\\n🔍 Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            continue
        
        print("🤖 Thinking...")
        response = chatbot.ask_question(user_question)
        print(f"\\n📝 Answer: {response['answer']}")

if __name__ == "__main__":
    demo_rag_chatbot()
