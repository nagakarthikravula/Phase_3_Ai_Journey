import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
clientg = genai.Client(api_key=api_key)

rude_config = genai.types.GenerateContentConfig(
    system_instruction='''Think of you like receptionist for a 
    company having 5 years of experience, And users will ask you
    question related to Policies of the company, You should answer
    effectively to users with provided information.If you Don't know or 
    not getting the answer from provided information of chunks, Simply redirect
    the conversation to human reception, Don't guess the answer.''',
    temperature=0
)

document_pages = {
    1: """Section 1: Working Hours
All full-time employees are expected to work 9 hours per day, Monday through Friday.
Core hours are between 10 AM and 4 PM, during which all employees must be available.
Flexible working hours are allowed outside of core hours with manager approval.
Remote work is permitted up to 2 days per week for roles that do not require physical presence.
Overtime work must be pre-approved by the department head and compensated accordingly.""",

    2: """Section 2: Leave Policy
Employees are entitled to 18 days of paid annual leave per calendar year.
Sick leave is capped at 10 days per year and requires a medical certificate for absences exceeding 3 consecutive days.
Maternity leave is 26 weeks as per government regulations.
Paternity leave is 5 working days, to be availed within 3 months of the child's birth.
Casual leave of 6 days per year may be taken without prior approval for emergencies.""",

    3: """Section 3: Salary and Benefits
Salaries are credited on the last working day of each month.
Employees are eligible for a performance bonus of up to 20% of annual salary, reviewed every April.
Health insurance coverage of up to 5 lakhs per annum is provided for employees and their immediate family.
Provident Fund contributions are deducted at 12% of basic salary as per government mandate.
Employees with more than 3 years of service are eligible for gratuity as per the Payment of Gratuity Act.""",

    4: """Section 4: Resignation and Exit
Employees must serve a notice period of 60 days before their last working day.
Notice period buyout is allowed at the discretion of the management.
Full and final settlement is processed within 45 days of the last working day.
Experience letters and relieving letters are issued within 7 days of full and final settlement.
Employees who leave without serving notice will forfeit their pending leaves and bonus."""
}

experiments = [
    {"name": "experiment_a", "chunk_size": 300, "overlap": 50},
    {"name": "experiment_b", "chunk_size": 50,  "overlap": 10},
    {"name": "experiment_c", "chunk_size": 1000,"overlap": 0},
    {"name": "experiment_d", "chunk_size": 300, "overlap": 0},
]
for a in experiments:
    doc_name = a['name']

    client = chromadb.PersistentClient(path="./chroma_pipeline_db")
    client.delete_collection(name=doc_name)
    collection = client.get_or_create_collection(name=doc_name)


    def chunk_text(text, chunk_size, overlap):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += (chunk_size - overlap)
        return chunks

    existing = collection.get(ids=[f"{doc_name}_page1_chunk0"])
    if len(existing['ids']) == 0:
        for page,text in document_pages.items():
            chunk = chunk_text(text,a['chunk_size'],a['overlap'])
            for i,doc in enumerate(chunk):
                collection.add(
                    documents=[chunk[i]],
                    ids=[f"{doc_name}_page{page}_chunk{i}"],
                    metadatas=[{"page":page,"section":text[:30],"doc":doc_name}]
                )
                print("Total Counts of chunk", collection.count())
    else:
        print(f"*****{doc_name} already indexed, skipping*****")


    def retrieve(ques):
        results = collection.query(
            query_texts= [ques],
            n_results=2
        )
        for i in range(len(results['documents'])):
            for j in range(len(results['documents'][i])):
                print("📃")
                print(f"Document: {results['documents'][i][j]}")
                print("📟")
                print(f"Page Number: {results['metadatas'][i][j]['page']}")
                print("➗")
                print(f"Section: {results['metadatas'][i][j]['section'][:9]}")
                print("🛣️")
                print(f"Distances: {results['distances'][i][j]}")
        return results
    print("\n" + "="*50)
    print(f"EXPERIMENT: {a['name']} | chunk_size={a['chunk_size']} | overlap={a['overlap']}")
    print(f"Total chunks stored: {collection.count()}")
    print("="*50)
    test_questions = [
        "How many days of annual leave do employees get?",
        "What is the health insurance coverage amount?",
        "What happens if an employee resigns without serving notice?"
    ]
    print("\n" + "="*50)
    print(f"EXPERIMENT: {a['name']} | chunk_size={a['chunk_size']} | overlap={a['overlap']}")
    print(f"Total chunks stored: {collection.count()}")
    print("="*50)
    for question in test_questions:
        main_results = retrieve(question)
        response = clientg.models.generate_content(
            model= "gemini-3-flash-preview",
            contents=f'''User Question: {question}
            Context:
            Page {main_results['metadatas'][0][0]['page']}: {main_results['documents'][0][0]}
            Page {main_results['metadatas'][0][1]['page']}: {main_results['documents'][0][1]}

            Answer the question using only the context above. 
            Cite which page your answer came from.''',
            config=rude_config
        )

        print("*"*20)
        print("*"*9 + "Ai Answer" + "*"*9)
        print("Question: ",question)
        print(response.text)



