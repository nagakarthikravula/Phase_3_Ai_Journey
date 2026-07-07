import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_first_collection")

print("Collection created: ",collection.name)
print("Number of items: ",collection.count())

collection.add(
    documents=[
        "Employees are entitled to 18 days of paid annual leave per calendar year.",
        "Salaries are credited on the last working day of each month.",
        "Employees must serve a notice period of 60 days before their last working day.",
        "Health insurance coverage of up to 5 lakhs per annum is provided.",
        "Maternity leave is 26 weeks as per government regulations."
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print("Items in collection after adding:", collection.count())

results = collection.query(
    query_texts=["How many days of annual leave do I get?"],
    n_results=2
)

print("Top results:")
for i in range(len(results['documents'][0])):
    print(f"\nRank {i+1}:")
    print(f"  Document: {results['documents'][0][i]}")
    print(f"  ID: {results['ids'][0][i]}")
    print(f"  Distance: {results['distances'][0][i]}")