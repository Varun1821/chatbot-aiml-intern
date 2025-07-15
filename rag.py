from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
import os

file_path = "C:\Projects\Amazon QA RAG\Dataset\single_qna.csv"

loader = CSVLoader(file_path=file_path)
data = loader.load()
load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# Extract the content from the loaded documents
documents = [doc.page_content for doc in data]

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = text_splitter.create_documents(documents)

# Initialize HuggingFace embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Extract text for embeddings
texts = [doc.page_content for doc in split_docs]

# Build the FAISS vector store
vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)

# Save the FAISS index locally
vector_store.save_local("faiss_index")

print("FAISS index created and saved locally with HuggingFace embeddings.")


from huggingface_hub import InferenceClient

client = InferenceClient(api_key=os.getenv("API_KEY"))

# Define the Mistral-based answer generation function
def generate_answer(context, question, user_type):
    """
    Generate an answer using Mistral.
    Args:
        context (str): The relevant document context.
        question (str): The user query.
        user_type (str): User type ("Prime" or "Non-Prime").
    Returns:
        str: Generated answer from the Mistral model.
    """
    prompt = f"""
You are a knowledgeable assistant answering questions about products on Amazon.
- Always take the user's type into account, Prime users have added benefits like free delivery and exclusive deals.
- Always prioritize accuracy, clarity, and helpfulness in your responses.

Here is the information you have:
Context: {context}

The user has asked:
{question}

The user type is: {user_type} (Amazon Prime member or Non-Prime member).

Please tailor your answer accordingly.
"""
    print(context)
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=500,
    )
    return completion["choices"][0]["message"]["content"]


# Define the RAG pipeline
def rag_pipeline(query, user_type):
    """
    Retrieve relevant documents and generate a response.
    Args:
        query (str): User query.
        user_type (str): User type ("Prime" or "Non-Prime").
    Returns:
        str: Generated answer or fallback message.
    """
    # Retrieve relevant documents
    results = vector_store.similarity_search(query, k=1)
    if not results:
        return "No relevant documents found."

    # Combine results for context
    context = " ".join([doc.page_content for doc in results])

    # Generate an answer
    answer = generate_answer(context, query, user_type)
    return answer

from flask import Flask, render_template, request, jsonify 

app = Flask(__name__) 

@app.route("/", methods=['POST', 'GET']) 
def query_view(): 
	if request.method == 'POST': 
		prompt = request.form['prompt'] 
		userType = request.form['userType'] 
		print(prompt, userType)      
		response = rag_pipeline(prompt, userType) 
		print(userType)     

		return jsonify({'response': response}) 

	return render_template('index.html') 


if __name__ == "__main__": 
	app.run(debug=True) 
