import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection2 = client.get_or_create_collection(name="handbook_with_metadata")
print(collection2.count())
collection2.add(
    documents=[
        "Employees are entitled to 18 days of paid annual leave per calendar year.",
        "Salaries are credited on the last working day of each month.",
        "Employees must serve a notice period of 60 days before their last working day.",
        "Health insurance coverage of up to 5 lakhs per annum is provided.",
        "Maternity leave is 26 weeks as per government regulations."
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
    metadatas=[
        {"section": "Leave Policy", "page": 2},
        {"section": "Salary and Benefits", "page": 4},
        {"section": "Resignation and Exit", "page": 5},
        {"section": "Salary and Benefits", "page": 4},
        {"section": "Leave Policy", "page": 2}
    ]
)


collection2.delete(ids=["doc3"])


collection2.update(
    ids = ["doc1"],
    metadatas=[{"section": "Leave Policy",
    "page": 2,
    "last_updated": "2024"}]
)

results = collection2.query(
    query_texts=["How many annual leaves per year?"],
    n_results=2
)

for i in range(len(results['documents'][0])):
    print(f"Document: {results['documents'][0][i]}")
    print(f"MetaData: {results['metadatas'][0][i]}")
    print(f"Score: {results['distances'][0][i]}")


print(collection2.count())
print(collection2.get(ids=["doc3"]))