# Import the necessary libraries
import PyPDF2
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import argparse

# Initialize the LLaMA model via Ollama
model_id = "llama3.1"
model = Ollama(model=model_id)


# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = [reader.pages[page].extract_text() for page in range(len(reader.pages))]
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
    result = model(input_prompt)
    return result.strip()


def main(pdf, output):
    with open(pdf, "rb") as f:
        print("Extracting text from PDF...")
        document_text = extract_text_from_pdf(f)
        document_size = len(document_text)
        print(f"Extracted {document_size} pages from pdf")
        count = 0
        with open(output, "w", encoding="utf-8") as out:
            for page in document_text:
                summary = summarize_text(page)
                count += 1
                print(f"Page size: {len(page)}", f"Summary Size: {len(summary)}")
                print(f"Summarized {count}/{document_size} pages")
                out.write(f"Page {count}\n")
                out.write(summary)
                out.write('\n')
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        metavar="path",
        required=False,
        help="the path to pdf",
        default="D:/OMSCS/06_HPC/Theory/Week00/1_IO_Complexity_of_Sorting_and_Related_Problems.pdf",
    )
    parser.add_argument(
        "--output",
        metavar="path",
        required=False,
        help="the path to output",
        default="D:/OMSCS/06_HPC/Theory/Week00/1_IO_Complexity_of_Sorting_and_Related_Problems.txt",
    )
    args = parser.parse_args()
    main(pdf=args.pdf, output=args.output)
