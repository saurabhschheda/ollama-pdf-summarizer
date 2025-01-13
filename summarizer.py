# Import the necessary libraries
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from IPython.display import display, Markdown, clear_output
import time
import streamlit as st

# Initialize the LLaMA model via Ollama
model_id = "llama3.1"
model = Ollama(model=model_id)

# Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in range(len(reader.pages)):
#             text += reader.pages[page].extract_text()
#     return text


# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def summarize_text(input_text: str):
    """
    Summarize a given text using the llama3.1 model.

    Args:
        input_text (str): The text to summarize.

    Returns:
        summary (str): Summarized text.
    """

    # Step 1: Prepare the summarization prompt
    prompt_template = """
    You are an advanced summarization model. Summarize the following text:
    
    {text}
    
    Provide a concise and informative summary.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Step 2: Inject the whole document into the prompt
    input_prompt = prompt.format(text=input_text)

    # Step 3: Generate the summary
    display(Markdown("**Generating Summary...**"))
    result = model(input_prompt)
    
    # Display the final result
    clear_output(wait=True)
    display(Markdown(f"### Final Summary:\n\n{result.strip()}"))
    
    return result.strip()

# Streamlit UI
st.title("PDF Summarizer with Llama3.1 Model")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
print(dir(uploaded_file))

if uploaded_file is not None:
    # bytes_data = uploaded_file.getvalue()
    # Extract text from uploaded PDF
    with st.spinner('Extracting text from PDF...'):
        document_text = extract_text_from_pdf(uploaded_file)

    # Summarize the extracted text
    if st.button("Summarize Text"):
        with st.spinner('Generating Summary...'):
            summary = summarize_text(document_text)
        
        st.subheader("Summary")
        st.markdown(summary)