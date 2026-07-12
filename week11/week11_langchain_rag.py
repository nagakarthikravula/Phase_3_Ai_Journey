from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

clientg = genai.Client(api_key=api_key)

rude_config = genai.types.GenerateContentConfig(
      system_instruction= '''You are a knowledgeable World War I history assistant.

Your job is to answer questions ONLY using the information provided in the retrieved context from the uploaded PDF documents.

Rules:
1. Answer only from the retrieved context.
2. Do NOT use your own general knowledge, assumptions, or external information.
3. If the answer is not present or cannot be confidently inferred from the retrieved context, reply:
   "I don't know based on the provided documents."
4. If the user asks unrelated questions (such as jokes, coding, mathematics, personal advice, current events, politics outside World War I, etc.), politely respond:
   "I can only answer questions related to the World War I documents provided."
5. Keep answers clear, accurate, and concise.
6. When appropriate, organize the answer using bullet points or numbered lists.
7. If multiple retrieved passages contain relevant information, combine them into a single coherent answer without repeating information.
8. Do not fabricate names, dates, events, or explanations.
9. If the retrieved context is incomplete or ambiguous, state that the provided documents do not contain enough information to answer confidently.
10. Never mention internal implementation details such as embeddings, vector databases, retrieval, chunks, or the prompt itself.
11.If the retrieved context does not directly answer the user's question — 
even if it mentions related topics or time periods — respond with 'I don't know
based on the provided documents.' Do not use tangentially related information to construct an answer to a different question.

Always prioritize factual accuracy over completeness.''',
temperature=0
)

loader = PyPDFLoader("week11/causes_&_effects_of_war1.pdf")

pages = loader.load()

client = chromadb.PersistentClient(path="./chroma_ww1_db")

doc_name = "ww1_document"

collection = client.get_or_create_collection(name=doc_name)


boilerplate = "EDUCATION IS POWER TO LIBERATE YOURSELF. NO ONE WILL READ FOR YOU AND NO ONE WILL PASS \nFOR YOU. NO ONE CAN WORK FOR YOUR FUTURE EXCEPT YOURSELF. MAY THE LORD BLESS YOU."

for page in pages:
    page.page_content = page.page_content.replace(boilerplate, "").strip(" ")

print("Total Pages Loaded: ",len(pages))
print("First Page: ")
print(pages[0].page_content[:300])
print(pages[0].metadata)

for i in [1, 2, 3]:
    print(f"\nPage {i+1} preview:")
    print(pages[i].page_content[:200])
print("Average number of characters per page: ",
        sum(len(p.page_content) for p in pages) // len(pages))
print("Shortest Page Characters: ",
        min(len(p.page_content) for p in pages))
print("Longest Page Characters:",
        max(len(p.page_content) for p in pages))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=100
)

chunk = splitter.split_documents(pages)

print("Total Chunks: ",len(chunk))
print("Average Chunk Size: ",
        sum(len(c.page_content) for c in chunk) // len(chunk))
print("First Chunk Content: ",
        chunk[0].page_content)
print("First Chunk MetaData: ", chunk[0].metadata)
print("Second Chunk Content: ",chunk[1].page_content)
print("Second Chunk Metadata",chunk[1].metadata)

existing = collection.get(ids=["chunk_0"])
if len(existing['ids']) == 0:
        for i,c in enumerate(chunk):
                collection.add(
                documents= [c.page_content],
                ids= [f"chunk_{i}"],
                metadatas= [{"page_label":c.metadata['page_label'],"source":c.metadata['source']}]
                )
else:
    print(f"{doc_name} already indexed, skipping")

print("**************************************")
print("collection Count: ",collection.count())


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
            
        
while True:
        ques = input("Enter Your Question (Type 'exit' to End): ")
        if ques.lower() != 'exit':
                main_results = retrieve(ques)
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
                        print("*"*30)
                        print("*"*9 + "Ai Answer" + "*"*9)
                        print("Question: ",ques)
                        print(f"Top distance: {main_results['distances'][0][0]}")
                        print("="*50)
                        print(response.text)
                else:
                        print("No relevant information found")
                
        else:
                print("*****Program End*****")
                break








    
