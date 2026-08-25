from rag.pipeline import LawGenPipeline


bot = LawGenPipeline()


while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break


    answer = bot.ask(question)


    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)

    print(answer)
