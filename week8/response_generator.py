import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class ResponseGenerator:
    """
    Response generator for RAG system - creates answers using different AI models
    This is like the "brain" that takes the found information and writes it in a friendly way
    """
    
    def __init__(self, model_type='huggingface', model_name='microsoft/DialoGPT-medium'):
        """
        Set up the response generator
        
        Args:
            model_type (str): What kind of AI to use ('openai', 'huggingface', or 'local')
            model_name (str): Which specific model to use
        """
        self.model_type = model_type
        self.model_name = model_name
        
        if model_type == 'openai':
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        elif model_type == 'huggingface':
            self._setup_huggingface_model()
        
        elif model_type == 'local':
            self._setup_local_model()
    
    def _setup_huggingface_model(self):
        """Setup Hugging Face model"""
        device = 0 if torch.cuda.is_available() else -1
        
        # Use a lightweight conversational model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Falling back to simpler text generation...")
            # Fallback to a smaller model
            self.generator = pipeline('text-generation', model='gpt2', max_length=512)
    
    def _setup_local_model(self):
        """Setup local model (placeholder for custom models)"""
        # This can be extended to support local models
        print("Local model setup - implement based on your specific model")
        self.generator = None
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate response using retrieved documents
        
        Args:
            query (str): User query
            retrieved_docs (List[Dict]): Retrieved documents from retriever
            
        Returns:
            str: Generated response
        """
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        if self.model_type == 'openai':
            return self._generate_openai_response(query, context)
        elif self.model_type == 'huggingface':
            return self._generate_huggingface_response(query, context)
        else:
            return self._generate_fallback_response(query, context)
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            context_parts.append(f"Document (Score: {doc['score']:.3f}):\\n{doc['document']}")
        
        return "\\n\\n".join(context_parts)
    
    def _generate_openai_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI API"""
        try:
            prompt = f"""
            You are a loan approval expert assistant. Use the following context to answer the user's question about loan approvals.
            
            Context:
            {context}
            
            Question: {query}
            
            Provide a comprehensive, accurate answer based on the context. Include specific statistics and examples when relevant.
            If the context doesn't contain enough information to fully answer the question, mention what additional information might be needed.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful loan approval expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback_response(query, context)
    
    def _generate_huggingface_response(self, query: str, context: str) -> str:
        """Generate response using Hugging Face model"""
        try:
            prompt = f"""Context: {context[:1000]}...

Question: {query}

Answer: Based on the loan approval data,"""
            
            # Generate response
            outputs = self.generator(
                prompt,
                max_length=len(prompt.split()) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            return answer if answer else self._generate_fallback_response(query, context)
            
        except Exception as e:
            print(f"Hugging Face generation error: {e}")
            return self._generate_fallback_response(query, context)
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate a rule-based fallback response"""
        query_lower = query.lower()
        
        # Extract key information from context
        lines = context.split('\\n')
        relevant_info = [line.strip() for line in lines if line.strip()]
        
        response = f"Based on the loan approval dataset analysis:\\n\\n"
        
        # Query-specific responses
        if any(word in query_lower for word in ['approval', 'approve', 'success']):
            response += "**Loan Approval Insights:**\\n"
            for info in relevant_info[:5]:
                if any(word in info.lower() for word in ['approval', 'rate', 'approved']):
                    response += f"• {info}\\n"
        
        elif any(word in query_lower for word in ['income', 'salary', 'earning']):
            response += "**Income Analysis:**\\n"
            for info in relevant_info[:5]:
                if any(word in info.lower() for word in ['income', '$', 'salary']):
                    response += f"• {info}\\n"
        
        elif any(word in query_lower for word in ['gender', 'male', 'female']):
            response += "**Gender-based Analysis:**\\n"
            for info in relevant_info[:5]:
                if any(word in info.lower() for word in ['gender', 'male', 'female']):
                    response += f"• {info}\\n"
        
        elif any(word in query_lower for word in ['education', 'graduate', 'degree']):
            response += "**Education Impact:**\\n"
            for info in relevant_info[:5]:
                if any(word in info.lower() for word in ['education', 'graduate']):
                    response += f"• {info}\\n"
        
        elif any(word in query_lower for word in ['credit', 'history', 'score']):
            response += "**Credit History Analysis:**\\n"
            for info in relevant_info[:5]:
                if any(word in info.lower() for word in ['credit', 'history']):
                    response += f"• {info}\\n"
        
        else:
            response += "**Key Insights:**\\n"
            for info in relevant_info[:5]:
                response += f"• {info}\\n"
        
        if len(response.strip()) <= len("Based on the loan approval dataset analysis:"):
            response += "The retrieved information shows various factors affecting loan approvals. Please ask a more specific question for detailed insights."
        
        return response

class HuggingFaceLightModel:
    """
    Lightweight alternative using smaller Hugging Face models
    """
    
    def __init__(self):
        self.model_name = "distilgpt2"  # Much smaller and faster
        try:
            self.generator = pipeline(
                'text-generation',
                model=self.model_name,
                device=-1,  # Use CPU
                max_length=256
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.generator = None
    
    def generate_simple_response(self, query: str, context: str) -> str:
        """Generate a simple response"""
        if not self.generator:
            return self._template_response(query, context)
        
        try:
            prompt = f"Loan Question: {query}\\nAnswer:"
            
            outputs = self.generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=50256  # GPT-2 EOS token
            )
            
            generated = outputs[0]['generated_text']
            answer = generated.split("Answer:")[1].strip() if "Answer:" in generated else ""
            
            return answer if len(answer) > 10 else self._template_response(query, context)
            
        except Exception:
            return self._template_response(query, context)
    
    def _template_response(self, query: str, context: str) -> str:
        """Template-based response as final fallback"""
        return f"""Based on the loan approval dataset:

The analysis shows various factors that influence loan approval decisions including income levels, credit history, education, and demographic factors.

Key factors typically include:
- Credit history and score
- Applicant income and co-applicant income
- Education level
- Employment status
- Property area (Urban/Rural/Semi-urban)

For specific statistics about "{query}", please refer to the detailed dataset analysis."""
