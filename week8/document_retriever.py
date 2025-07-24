import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from typing import List, Dict, Any
import logging

class DocumentRetriever:
    """
    Document retriever for RAG chatbot - helps find relevant information from loan data
    This class reads the loan dataset and creates searchable documents that the chatbot can use
    to answer questions. Think of it like creating a smart index for a library.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Set up the document retriever
        
        Args:
            model_name (str): The AI model used to understand text similarity
                             (we use a pre-trained model that's good for this task)
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def load_and_process_data(self, train_path: str, test_path: str):
        """
        Load and process the loan dataset to create document corpus
        
        Args:
            train_path (str): Path to training dataset
            test_path (str): Path to test dataset
        """
        # Load datasets
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # Create comprehensive document corpus
        self.documents = self._create_document_corpus()
        
        # Generate embeddings
        print("Generating embeddings for document corpus...")
        self.embeddings = self.model.encode(self.documents)
        
        # Build FAISS index
        self._build_faiss_index()
        
    def _create_document_corpus(self) -> List[str]:
        """
        Create a comprehensive document corpus from the loan dataset
        
        Returns:
            List[str]: List of documents for retrieval
        """
        documents = []
        
        # Data overview documents
        documents.extend(self._create_overview_documents())
        
        # Statistical documents
        documents.extend(self._create_statistical_documents())
        
        # Pattern documents
        documents.extend(self._create_pattern_documents())
        
        # Individual loan records as documents
        documents.extend(self._create_loan_record_documents())
        
        return documents
    
    def _create_overview_documents(self) -> List[str]:
        """Create overview documents about the dataset"""
        docs = []
        
        # Dataset overview
        docs.append(f"""
        Loan Approval Dataset Overview:
        This dataset contains {len(self.train_df)} training records and {len(self.test_df)} test records.
        
        Features include:
        - Loan_ID: Unique identifier for each loan application
        - Gender: Male/Female
        - Married: Yes/No - Marital status of applicant
        - Dependents: Number of dependents (0, 1, 2, 3+)
        - Education: Graduate/Not Graduate
        - Self_Employed: Yes/No - Employment type
        - ApplicantIncome: Income of primary applicant
        - CoapplicantIncome: Income of co-applicant
        - LoanAmount: Loan amount requested (in thousands)
        - Loan_Amount_Term: Term of loan in months
        - Credit_History: Credit history meets guidelines (1.0/0.0)
        - Property_Area: Urban/Semiurban/Rural
        - Loan_Status: Y (Approved) / N (Rejected) - Target variable
        """)
        
        # Approval rates
        approval_rate = (self.train_df['Loan_Status'] == 'Y').mean()
        docs.append(f"""
        Loan Approval Statistics:
        Overall approval rate: {approval_rate:.2%}
        Approved loans: {(self.train_df['Loan_Status'] == 'Y').sum()}
        Rejected loans: {(self.train_df['Loan_Status'] == 'N').sum()}
        """)
        
        return docs
    
    def _create_statistical_documents(self) -> List[str]:
        """Create statistical analysis documents"""
        docs = []
        
        # Income statistics
        docs.append(f"""
        Income Analysis:
        Average applicant income: ${self.train_df['ApplicantIncome'].mean():.2f}
        Median applicant income: ${self.train_df['ApplicantIncome'].median():.2f}
        Average co-applicant income: ${self.train_df['CoapplicantIncome'].mean():.2f}
        
        Income ranges:
        - Low income (< $3000): {(self.train_df['ApplicantIncome'] < 3000).sum()} applicants
        - Middle income ($3000-$6000): {((self.train_df['ApplicantIncome'] >= 3000) & (self.train_df['ApplicantIncome'] < 6000)).sum()} applicants
        - High income (>= $6000): {(self.train_df['ApplicantIncome'] >= 6000).sum()} applicants
        """)
        
        # Loan amount statistics
        docs.append(f"""
        Loan Amount Analysis:
        Average loan amount: ${self.train_df['LoanAmount'].mean():.2f}K
        Median loan amount: ${self.train_df['LoanAmount'].median():.2f}K
        Most common loan term: {self.train_df['Loan_Amount_Term'].mode().iloc[0]} months
        """)
        
        # Demographic statistics
        for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
            if col in self.train_df.columns:
                value_counts = self.train_df[col].value_counts()
                doc = f"{col} Distribution:\n"
                for value, count in value_counts.items():
                    percentage = (count / len(self.train_df)) * 100
                    doc += f"- {value}: {count} ({percentage:.1f}%)\n"
                docs.append(doc)
        
        return docs
    
    def _create_pattern_documents(self) -> List[str]:
        """Create documents about approval patterns"""
        docs = []
        
        # Approval by gender
        if 'Gender' in self.train_df.columns:
            gender_approval = self.train_df.groupby('Gender')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
            docs.append(f"""
            Loan Approval by Gender:
            {chr(10).join([f'- {gender}: {rate:.2%} approval rate' for gender, rate in gender_approval.items()])}
            """)
        
        # Approval by education
        if 'Education' in self.train_df.columns:
            education_approval = self.train_df.groupby('Education')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
            docs.append(f"""
            Loan Approval by Education:
            {chr(10).join([f'- {edu}: {rate:.2%} approval rate' for edu, rate in education_approval.items()])}
            """)
        
        # Approval by property area
        if 'Property_Area' in self.train_df.columns:
            area_approval = self.train_df.groupby('Property_Area')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
            docs.append(f"""
            Loan Approval by Property Area:
            {chr(10).join([f'- {area}: {rate:.2%} approval rate' for area, rate in area_approval.items()])}
            """)
        
        # Credit history impact
        if 'Credit_History' in self.train_df.columns:
            credit_approval = self.train_df.groupby('Credit_History')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
            docs.append(f"""
            Credit History Impact on Approval:
            - With credit history (1.0): {credit_approval.get(1.0, 0):.2%} approval rate
            - Without credit history (0.0): {credit_approval.get(0.0, 0):.2%} approval rate
            """)
        
        return docs
    
    def _create_loan_record_documents(self) -> List[str]:
        """Create documents from individual loan records"""
        docs = []
        
        # Sample approved and rejected loans
        approved_loans = self.train_df[self.train_df['Loan_Status'] == 'Y'].head(50)
        rejected_loans = self.train_df[self.train_df['Loan_Status'] == 'N'].head(50)
        
        for _, row in approved_loans.iterrows():
            doc = f"""
            Approved Loan Case (ID: {row['Loan_ID']}):
            - Gender: {row.get('Gender', 'N/A')}
            - Married: {row.get('Married', 'N/A')}
            - Dependents: {row.get('Dependents', 'N/A')}
            - Education: {row.get('Education', 'N/A')}
            - Self Employed: {row.get('Self_Employed', 'N/A')}
            - Applicant Income: ${row.get('ApplicantIncome', 0)}
            - Coapplicant Income: ${row.get('CoapplicantIncome', 0)}
            - Loan Amount: ${row.get('LoanAmount', 0)}K
            - Loan Term: {row.get('Loan_Amount_Term', 0)} months
            - Credit History: {row.get('Credit_History', 'N/A')}
            - Property Area: {row.get('Property_Area', 'N/A')}
            - Status: APPROVED
            """
            docs.append(doc)
        
        for _, row in rejected_loans.iterrows():
            doc = f"""
            Rejected Loan Case (ID: {row['Loan_ID']}):
            - Gender: {row.get('Gender', 'N/A')}
            - Married: {row.get('Married', 'N/A')}
            - Dependents: {row.get('Dependents', 'N/A')}
            - Education: {row.get('Education', 'N/A')}
            - Self Employed: {row.get('Self_Employed', 'N/A')}
            - Applicant Income: ${row.get('ApplicantIncome', 0)}
            - Coapplicant Income: ${row.get('CoapplicantIncome', 0)}
            - Loan Amount: ${row.get('LoanAmount', 0)}K
            - Loan Term: {row.get('Loan_Amount_Term', 0)} months
            - Credit History: {row.get('Credit_History', 'N/A')}
            - Property Area: {row.get('Property_Area', 'N/A')}
            - Status: REJECTED
            """
            docs.append(doc)
        
        return docs
    
    def _build_faiss_index(self):
        """Build FAISS index for similarity search"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} documents")
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: Retrieved documents with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call load_and_process_data first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                'document': self.documents[idx],
                'score': float(score),
                'rank': i + 1
            })
        
        return results
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents for response generation"""
        context_parts = []
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            context_parts.append(f"Document (Score: {doc['score']:.3f}):\\n{doc['document']}")
        
        return "\\n\\n".join(context_parts)
    
    def save_index(self, path: str):
        """Save the retriever state"""
        state = {
            'documents': self.documents,
            'train_df': self.train_df.to_dict(),
            'test_df': self.test_df.to_dict()
        }
        
        with open(f"{path}_state.pkl", 'wb') as f:
            pickle.dump(state, f)
        
        faiss.write_index(self.index, f"{path}_faiss.index")
        np.save(f"{path}_embeddings.npy", self.embeddings)
        
    def load_index(self, path: str):
        """Load the retriever state"""
        with open(f"{path}_state.pkl", 'rb') as f:
            state = pickle.load(f)
        
        self.documents = state['documents']
        self.train_df = pd.DataFrame(state['train_df'])
        self.test_df = pd.DataFrame(state['test_df'])
        
        self.index = faiss.read_index(f"{path}_faiss.index")
        self.embeddings = np.load(f"{path}_embeddings.npy")
