from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain

def main():
    llm = get_hf_llm(temperature=0.7)
    rag_chain = build_rag_chain(llm, data_dir="./data_source/generative_ai", data_type="pdf")
    while True:
        question = input(">> Question: ")
        if question.lower() == "stop":
            print("Exiting...")
            break
        answer = rag_chain.invoke(question)
        print(">> Answer:", answer)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
