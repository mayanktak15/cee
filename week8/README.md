# Loan Approval RAG Chatbot
**Data Science Internship Project - Week 8**  
*Celebal Technologies*

A beginner-friendly Retrieval-Augmented Generation (RAG) chatbot for learning about loan approval patterns. This project helps understand the basics of NLP, data analysis, and building simple AI applications.

## 🎯 Learning Project Overview

As a Data Science fresher, this project helps me learn:
- **Basic Data Analysis** with pandas and visualization
- **Introduction to NLP** concepts and text processing
- **Simple Machine Learning** applications
- **Building Web Apps** with Streamlit
- **Working with Real Datasets** from financial domain

## 🚀 What This Project Does

- **Answers Questions**: Ask questions about loan data and get simple answers
- **Searches Information**: Finds relevant information from the loan dataset
- **Shows Data Insights**: Basic charts and statistics about loan approvals
- **Easy Interface**: Simple web interface to interact with the data
- **Learning Tool**: Helps understand how AI chatbots work

## 📊 About the Dataset

**Learning with Real Financial Data**: Loan Approval Information
- **Training File**: 614 loan applications with results (approved/rejected)
- **Test File**: 367 loan applications for testing
- **Why This Data**: Learn how banks decide on loan approvals

**What Information We Have**:
- Personal Details: Gender, Marriage status, Number of dependents, Education
- Money Details: How much they earn, How much loan they want
- Other Factors: Credit history, Where they live (city/village)
- **Result**: Whether loan was approved or not

## 🎓 What I'm Learning (Fresher Goals)

Through this project, I'm practicing:

1. **Basic Data Science**:
   - Reading and understanding CSV files
   - Making simple charts and graphs
   - Finding patterns in data

2. **Introduction to AI**:
   - How chatbots find relevant information
   - Basic text processing
   - Simple question-answering systems

3. **Programming Practice**:
   - Writing Python code step by step
   - Using libraries like pandas and streamlit
   - Building a complete project

4. **Real-World Application**:
   - Understanding business problems
   - Working with financial data
   - Creating user-friendly interfaces

## 🛠️ How to Run This Project

### What You Need First
- Python installed on your computer (3.8 or newer)
- Basic understanding of running Python scripts
- The dataset files (already provided)

### Easy Setup Steps

1. **Get All Files Ready**
   - Make sure all Python files and CSV files are in one folder
   - Check that you have `Training Dataset.csv` and `Test Dataset.csv`

2. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```
   (This installs all the tools we need)

3. **Test if Everything Works**
   ```bash
   python test_setup.py
   ```
   (This checks if everything is set up correctly)

4. **Optional: For Advanced Features**
   - You can add API keys later if you want to try advanced models
   - For now, the basic version works without any API keys

## 🖥️ How to Use the Chatbot

### Method 1: Pretty Web Interface (Easiest)
```bash
streamlit run streamlit_app.py
```
This opens a website in your browser where you can chat with the bot!

### Method 2: Simple Command Line
```bash
python rag_chatbot.py
```
This runs the chatbot in your terminal - just type questions and get answers.

### Method 3: Test Everything
```bash
python demo.py
```
This runs some tests to make sure everything is working properly.

## 💡 Questions You Can Ask

Here are some simple questions to get started:
- "How many loans were approved?"
- "Do men or women get more loan approvals?"
- "What's the average income of people who got loans?"
- "Which education level gets more approvals?"
- "How important is credit history?"
- "Do people in cities get more loans than villages?"

## 🏗️ How It Works (Simple Explanation)

Think of it like a smart library assistant:

1. **You ask a question** → "How does education affect loan approval?"
2. **The system searches** → Looks through all the loan data for relevant information
3. **Finds the best answers** → Picks the most useful facts and numbers
4. **Gives you a response** → Combines everything into an easy-to-understand answer

### The Main Parts

1. **Data Reader** (`document_retriever.py`)
   - Reads the loan data from CSV files
   - Creates searchable text from the numbers and categories
   - Finds relevant information when you ask questions

2. **Answer Creator** (`response_generator.py`)
   - Takes the found information
   - Writes human-friendly answers
   - Uses simple templates when advanced AI isn't available

3. **Chat Manager** (`rag_chatbot.py`)
   - Connects everything together
   - Manages the conversation
   - Remembers what you asked before

4. **Web Interface** (`streamlit_app.py`)
   - Creates the pretty website interface
   - Shows charts and graphs
   - Makes it easy to ask questions

## 🔧 Simple Settings

### Easy Mode (No Setup Needed)
The chatbot works right away with basic responses:
```python
chatbot = LoanApprovalRAGChatbot(model_type='simple')
```

### If You Want to Try Advanced Features Later
```python
# For better AI responses (requires internet)
chatbot = LoanApprovalRAGChatbot(model_type='huggingface')

# For professional AI responses (requires API key)
chatbot = LoanApprovalRAGChatbot(model_type='openai')
```

### Which One to Use?
- **Simple mode**: Works offline, good for learning, fast responses
- **Hugging Face**: Better answers, needs internet for first time
- **OpenAI**: Best answers, needs API key (costs money)

## 📈 What You'll See in the Web Interface

When you run the Streamlit app, you'll get:

- **Basic Statistics**: How many loans approved, average amounts, etc.
- **Simple Charts**: Bar graphs and pie charts showing the data
- **Chat Box**: Type questions and get answers
- **Example Questions**: Click to try pre-written questions
- **Answer Sources**: See where the answer came from in the data

## 🎯 Example Conversation

```
You: "How many people got their loans approved?"

Bot: "Based on the loan dataset analysis:

📊 Loan Approval Summary:
• Total applications: 614
• Approved loans: 422 (68.7%)
• Rejected loans: 192 (31.3%)

This means about 7 out of 10 people who applied got their loan approved."
```

## 🛡️ When Things Go Wrong

Don't worry if you see errors! Here's what to do:

- **"Module not found"**: Run `pip install -r requirements.txt` again
- **"File not found"**: Make sure the CSV files are in the right folder
- **Slow responses**: The first time takes longer as it sets everything up
- **Strange answers**: Try asking simpler, more direct questions

## 📚 What I Learned Building This

### Technical Skills
- How to read and analyze data with Python
- Basic natural language processing concepts
- Building simple web applications
- Working with external libraries and APIs

### Soft Skills
- Breaking down complex problems into smaller parts
- Reading documentation and learning new tools
- Testing and debugging code
- Presenting technical work clearly

## 🔄 Ideas for Improvement

As I learn more, I could add:
- More types of charts and visualizations
- Better question understanding
- More detailed analysis features
- Comparison tools for different loan types
- Export features for reports

## 🤝 Getting Help

If you're also learning and want to understand this project:
- Read the code comments - they explain what each part does
- Try the example questions first
- Start with the simple mode before trying advanced features
- Look at the CSV files to understand the data structure

---

**A learning project by a Data Science fresher at Celebal Technologies 📚**
