from flask import Flask, render_template, request, jsonify
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, DocxLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

# Ollama setup
ollama_models = ["llama2", "gemma"] #add more models as needed
embeddings = OllamaEmbeddings(model="llama2") #default model
llm = Ollama(model="llama2") #default model

def load_and_process_document(file_path):
    """Loads and processes the document based on its type."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = DocxLoader(file_path)
    elif file_path.lower().endswith((".xls", ".xlsx")):
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    else:
        raise ValueError("Unsupported file type.")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vectorstore(texts):
    """Creates a vector store from the document texts."""
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    """Creates a question-answering chain."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_chain

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]
        model_name = request.form.get("model", "llama2") #get selected model, default to llama2

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            try:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    file.save(temp_file.name)
                    file_path = temp_file.name

                texts = load_and_process_document(file_path)
                vectorstore = create_vectorstore(texts)
                qa_chain = create_qa_chain(vectorstore)

                question = request.form["question"]
                response = qa_chain.run(question)

                os.unlink(file_path) #delete the temporary file.

                return jsonify({"response": response})

            except Exception as e:
                return jsonify({"error": str(e)})

    return render_template("index.html", models=ollama_models)

if __name__ == "__main__":
    app.run(debug=True)