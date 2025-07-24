import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rag_chatbot import LoanApprovalRAGChatbot
import os

def create_enhanced_streamlit_app():
    """Enhanced Streamlit app with visualizations"""
    
    st.set_page_config(
        page_title="Loan Approval RAG Chatbot - Celebal Technologies",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .intern-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🏦 Loan Approval RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data Science Internship Project - Week 8 | Celebal Technologies</div>', unsafe_allow_html=True)
    st.markdown('<div class="intern-badge">Intelligent Financial Analytics using RAG Technology</div>', unsafe_allow_html=True)
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🏦 Loan Approval RAG Chatbot</h1>
        <p style="color: lightblue; margin: 0;">Intelligent Q&A system for loan approval insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("🔧 Initializing chatbot..."):
            st.session_state.chatbot = LoanApprovalRAGChatbot(model_type='simple')
            
            train_path = "Training Dataset.csv"
            test_path = "Test Dataset.csv"
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                st.session_state.chatbot.setup(train_path, test_path)
                st.success("✅ Chatbot initialized successfully!")
            else:
                st.error("❌ Dataset files not found. Please ensure CSV files are in the directory.")
                st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Overview")
        
        if hasattr(st.session_state.chatbot, 'retriever'):
            summary = st.session_state.chatbot.get_dataset_summary()
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train Samples", summary['train_size'])
                st.metric("Approved", summary['target_distribution'].get('Y', 0))
            with col2:
                st.metric("Test Samples", summary['test_size'])
                st.metric("Rejected", summary['target_distribution'].get('N', 0))
            
            # Approval rate gauge
            approval_rate = summary['approval_rate']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = approval_rate * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Rate (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            fig_gauge.update_layout(height=200)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.divider()
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        questions = st.session_state.chatbot.suggest_questions()
        
        for i, q in enumerate(questions[:8]):
            if st.button(f"📝 {q[:40]}...", key=f"q_{i}", help=q):
                st.session_state.selected_question = q
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📈 Show Analytics", type="secondary"):
            st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "🔍 Explore Data"])
    
    with tab1:
        # Chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display chat messages
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 You:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message bot-message">
                            <strong>🤖 Assistant:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "retrieved_docs" in message:
                            with st.expander("📚 View Retrieved Context"):
                                for i, doc in enumerate(message["retrieved_docs"][:3]):
                                    st.text_area(
                                        f"Document {i+1} (Score: {doc['score']:.3f})",
                                        doc['document'][:500] + "..." if len(doc['document']) > 500 else doc['document'],
                                        height=100,
                                        key=f"doc_{len(st.session_state.messages)}_{i}"
                                    )
            
            # Question input
            question_input = st.text_input(
                "Ask a question about loan approvals:",
                value=st.session_state.get('selected_question', ''),
                key="question_input",
                placeholder="e.g., What factors affect loan approval the most?"
            )
            
            if 'selected_question' in st.session_state:
                del st.session_state.selected_question
            
            col_send, col_example = st.columns([1, 1])
            
            with col_send:
                if st.button("🚀 Send", type="primary") or question_input:
                    if question_input.strip():
                        # Add user message
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": question_input
                        })
                        
                        # Get bot response
                        with st.spinner("🤔 Analyzing..."):
                            response = st.session_state.chatbot.ask_question(question_input)
                            
                            # Add bot message
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response['answer'],
                                "retrieved_docs": response['retrieved_docs']
                            })
                        
                        st.rerun()
        
        with col2:
            st.subheader("🎯 Quick Stats")
            if hasattr(st.session_state.chatbot, 'retriever'):
                df = st.session_state.chatbot.retriever.train_df
                
                # Quick visualizations
                st.metric("Total Questions", len(st.session_state.messages) // 2)
                
                # Gender distribution
                gender_counts = df['Gender'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution"
                )
                fig_gender.update_layout(height=250)
                st.plotly_chart(fig_gender, use_container_width=True)
                
                # Education distribution
                edu_counts = df['Education'].value_counts()
                fig_edu = px.bar(
                    x=edu_counts.index,
                    y=edu_counts.values,
                    title="Education Levels"
                )
                fig_edu.update_layout(height=250)
                st.plotly_chart(fig_edu, use_container_width=True)
    
    with tab2:
        # Analytics dashboard
        st.subheader("📊 Loan Approval Analytics")
        
        if hasattr(st.session_state.chatbot, 'retriever'):
            df = st.session_state.chatbot.retriever.train_df
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Income",
                    f"${df['ApplicantIncome'].mean():,.0f}",
                    f"{df['ApplicantIncome'].median():,.0f} median"
                )
            
            with col2:
                st.metric(
                    "Avg Loan Amount",
                    f"${df['LoanAmount'].mean():,.0f}K",
                    f"{df['LoanAmount'].median():,.0f}K median"
                )
            
            with col3:
                credit_rate = df['Credit_History'].mean()
                st.metric(
                    "Credit History %",
                    f"{credit_rate:.1%}",
                    f"{(df['Credit_History'] == 1).sum()} applicants"
                )
            
            with col4:
                married_rate = (df['Married'] == 'Yes').mean()
                st.metric(
                    "Married %",
                    f"{married_rate:.1%}",
                    f"{(df['Married'] == 'Yes').sum()} applicants"
                )
            
            # Detailed visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Approval by property area
                area_approval = df.groupby('Property_Area')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
                fig_area = px.bar(
                    x=area_approval.index,
                    y=area_approval.values * 100,
                    title="Approval Rate by Property Area (%)",
                    labels={'y': 'Approval Rate (%)', 'x': 'Property Area'}
                )
                st.plotly_chart(fig_area, use_container_width=True)
                
                # Income distribution by loan status
                fig_income = px.box(
                    df,
                    x='Loan_Status',
                    y='ApplicantIncome',
                    title="Income Distribution by Loan Status"
                )
                st.plotly_chart(fig_income, use_container_width=True)
            
            with col2:
                # Approval by education
                edu_approval = df.groupby('Education')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
                fig_edu = px.bar(
                    x=edu_approval.index,
                    y=edu_approval.values * 100,
                    title="Approval Rate by Education (%)",
                    labels={'y': 'Approval Rate (%)', 'x': 'Education'}
                )
                st.plotly_chart(fig_edu, use_container_width=True)
                
                # Loan amount vs income
                fig_scatter = px.scatter(
                    df,
                    x='ApplicantIncome',
                    y='LoanAmount',
                    color='Loan_Status',
                    title="Loan Amount vs Applicant Income",
                    hover_data=['Credit_History', 'Education']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("🔥 Feature Correlation")
            numeric_cols = df.select_dtypes(include=['number']).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        # Data exploration
        st.subheader("🔍 Explore Dataset")
        
        if hasattr(st.session_state.chatbot, 'retriever'):
            df = st.session_state.chatbot.retriever.train_df
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender_filter = st.selectbox("Filter by Gender:", ['All'] + list(df['Gender'].unique()))
            
            with col2:
                education_filter = st.selectbox("Filter by Education:", ['All'] + list(df['Education'].unique()))
            
            with col3:
                status_filter = st.selectbox("Filter by Loan Status:", ['All', 'Y', 'N'])
            
            # Apply filters
            filtered_df = df.copy()
            if gender_filter != 'All':
                filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
            if education_filter != 'All':
                filtered_df = filtered_df[filtered_df['Education'] == education_filter]
            if status_filter != 'All':
                filtered_df = filtered_df[filtered_df['Loan_Status'] == status_filter]
            
            st.write(f"Showing {len(filtered_df)} records")
            
            # Display filtered data
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data",
                data=csv,
                file_name='filtered_loan_data.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    create_enhanced_streamlit_app()
