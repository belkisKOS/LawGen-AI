import chromadb


client = chromadb.PersistentClient(
    path="data/chroma_db"
)


collections = client.list_collections()

print(collections)


for c in collections:
    print("Collection:", c.name)
    print("Count:", c.count())
