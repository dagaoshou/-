from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
"""
总结服务类：用户提问，搜索参考资料，将提问交给大模型，让模型总结回复
"""


def print_prompt(prompt:PromptTemplate)->PromptTemplate:
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService:
    def __init__(self):
        self.vector_service =VectorStoreService()
        self.retriever=self.vector_service.get_retriever()
        self.prompt_text=load_rag_prompts()
        self.prompt_template=PromptTemplate.from_template(self.prompt_text)
        self.model=chat_model
        self.chain=self._init_chain()
    def _init_chain(self):
        chain=self.prompt_template | print_prompt| self.model | StrOutputParser()
        return chain

    def rag_summary(self,query:str):
        context_docs=self.retriever.invoke(query)
        context_str=""
        counter=1
        for doc in context_docs:
            context_str+=f"文档{counter}:文档内容：{doc.page_content}|文档元数据：{doc.metadata}\n"
            counter+=1
        return self.chain.invoke({"input":query,"context":context_str})
if __name__ == '__main__':
    rag=RagSummarizeService()
    response=rag.rag_summary("小户型适合的扫地机器人有哪些？")
    print(response)
