from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Ollama setup
llm = Ollama(model="llama2")

def scrape_and_summarize(url):
    """Scrapes a webpage and summarizes its content using an LLM."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract relevant text (example: paragraphs)
        paragraphs = [p.text for p in soup.find_all("p")]
        text = "\n".join(paragraphs)

        # Text splitting for LLM processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Summarization prompt
        prompt_template = """
        Summarize the following text:
        {text}

        SUMMARY:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        summaries = []
        for chunk in chunks:
            summary = llm_chain.run(chunk)
            summaries.append(summary)

        final_summary = "\n".join(summaries)
        return final_summary

    except requests.exceptions.RequestException as e:
        return f"Error during scraping: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        summary = scrape_and_summarize(url)
        return jsonify({"summary": summary})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)