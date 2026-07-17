import streamlit as st
import chromadb
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from dotenv import load_dotenv

st.title("PDF Answering System")
st.write("Upload a PDF and asl questions about it.")

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

clientg = genai.Client(api_key=api_key)
if "collection" not in st.session_state:
    st.session_state.collection = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

if "response" not in st.session_state:
    st.session_state.response = ""

uploaded_file = st.file_uploader("Choose a PDF file", type= "pdf")

rude_config = genai.types.GenerateContentConfig(
    system_instruction='''
You are an intelligent PDF document assistant.

Your primary responsibility is to answer the user's questions ONLY using the content extracted from the uploaded PDF.

## Core Instructions

1. Treat the uploaded PDF as the single source of truth.
2. Do not use outside knowledge unless the user explicitly asks for general knowledge beyond the document.
3. If the answer is not present in the PDF, clearly state:
   "I couldn't find this information in the uploaded document."
4. Never fabricate, assume, or infer facts that are not supported by the PDF.
5. Base every answer strictly on the document's content.

## Response Style

- Be clear, concise, and professional.
- Use simple language unless the document is highly technical.
- Organize long answers with headings and bullet points.
- When appropriate, summarize lengthy sections before giving details.
- Preserve important technical terms exactly as they appear in the document.

## Citations

Whenever possible:
- Mention the page number(s) where the information was found.
- If multiple pages contribute to the answer, list all relevant pages.
- If page numbers are unavailable, mention the relevant section or heading.

Example:
Source: Page 12

or

Sources: Pages 8, 10, and 11

## Handling Different Question Types

### Factual Questions
Provide the exact information from the document.

### Summary Requests
Summarize only the relevant portions while preserving key facts.

### Comparison Requests
Create tables whenever comparisons improve readability.

### Lists
Present numbered or bulleted lists.

### Definitions
Use the document's definition. If no definition exists, state that it is not defined in the document.

## Mathematical Content

If calculations are present:
- Explain the calculation step-by-step.
- Do not change numerical values.
- Preserve formulas exactly.

## Tables

If information comes from a table:
- Reconstruct the table in Markdown when possible.
- Maintain row-column relationships.

## Code

If the document contains code:
- Preserve formatting.
- Use Markdown code blocks.
- Explain the code only if the user asks.

## Ambiguous Questions

If the user's question is unclear:
- Ask a concise clarifying question.
- Do not guess what the user means.

## Missing Information

If the document does not contain enough information:

"I couldn't find this information in the uploaded document."

Optionally suggest nearby topics that are present.

## Safety

Do not invent content.
Do not generate fake references.
Do not cite pages that do not exist.
Do not claim certainty when the document is unclear.

Your goal is to provide accurate, document-grounded answers that faithfully represent the uploaded PDF.
''',
temperature= 0.0
)

if uploaded_file is not None:
    st.write("File uploaded: ",uploaded_file.name)
    if uploaded_file.name != st.session_state.processed_filename:
        st.write("New file detected - ready to process")
        if st.button("Process PDF"):
            with st.spinner("Processing your PDF... this may take a moment"):
                with tempfile.NamedTemporaryFile(delete=False, suffix= ".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 450,
                        chunk_overlap = 100
                    )

                    chunk = splitter.split_documents(pages)
                    client = chromadb.PersistentClient(path="./chromadb")
                    doc_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_").replace("&", "and")
                    collection = client.get_or_create_collection(name=doc_name)
                    existing = collection.get(ids=["chunk_0"])
                    if len(existing['ids']) == 0:
                        for i,c in enumerate(chunk):
                            collection.add(
                                documents= [c.page_content],
                                ids=[f"chunk_{i}"],
                                metadatas= [{"page_data":c.metadata['page_label'],"source":c.metadata['source']}]
                            )
                    else:
                        print(f"{doc_name} already indexed, skipping")
                    
                    st.session_state.collection = collection
                    st.session_state.processed = True
                    st.session_state.processed_filename = uploaded_file.name
            st.success("PDF processed successfully!")
    else:
        st.write("This file is already processed")
    st.write("Collection Count: ",st.session_state.collection.count())
else:
    st.write("Please  upload a PDF to get started")

def retrieve(ques):
      results = collection.query(
            query_texts=[ques],
            n_results= 3
      )
      print("Top distance:", results['distances'][0][0])
      if results['distances'][0][0] > 1.0:
        return 0
      else:
        return results

if st.session_state.processed:
    question = st.text_input("Ask a question to know about your PDF")
    if question:
        main_results = retrieve(question)
        if main_results:
            context = ""
            for i in range(len(main_results['documents'][0])):
                context += f"Page {main_results['metadatas'][0][i]['page_label']}: {main_results['documents'][0][i]}\n\n"      
                response = clientg.models.generate_content(
                model= "gemini-3-flash-preview",
                contents=f'''User Question: {ques}
                Context: {context}
                After each fact or bullet point in your answer, cite the page number in brackets like this: [Page 5]
                ''',
                config=rude_config
                )
