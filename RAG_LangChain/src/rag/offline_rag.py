import re

from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(
        self,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:
        print("---------------------------------------------------------------------")
        print("Text response:", text_response)

        match = re.search(pattern, text_response, re.DOTALL)
        print("---------------------------------------------------------------------")
        print("Match:", match)

        if match:
            answer_text = match.group(1).strip()
            print("---------------------------------------------------------------------")
            print("Answer text:", answer_text)

            return answer_text
        else:
            return text_response
        
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        print("------------------------------------------------------")
        print("Retriever:", retriever)
        
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        print("------------------------------------------------------")
        print("Input data:", input_data)

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        print("------------------------------------------------------")
        print("RAG chain:", rag_chain)

        return rag_chain

    def format_docs(self, docs):
        print("------------------------------------------------------")
        print("Docs:", docs)

        return "\n\n".join(doc.page_content for doc in docs)
