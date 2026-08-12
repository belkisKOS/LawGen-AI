from retrieval.retriever import LawGenRetriever


retriever = LawGenRetriever()


results = retriever.vectorstore.similarity_search_with_score(
    "ما هو دين الدولة حسب الدستور الجزائري؟",
    k=5
)


print("="*80)
print("CHROMA DIRECT TEST")
print("="*80)


for doc, score in results:

    print("SCORE:", score)
    print(doc.page_content[:300])
    print("-"*50)
