from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[BaseChatModel|Embeddings]:
        pass
class ChatModel(BaseModelFactory):
    def generator(self)->Optional[BaseChatModel|Embeddings]:
        return init_chat_model(
            model=rag_conf["chat_model_name"],
            model_provider=rag_conf["model_provider"],
            api_key=rag_conf["api_key"],
            api_base=rag_conf["api_base"],
            extra_body={
                "thinking":{
                    "type":"disabled"
                }
            }
        )
class EmbeddingModel(BaseModelFactory):
    def generator(self)->Optional[BaseChatModel|Embeddings]:
        return OpenAIEmbeddings(
            model="text-embedding-v4",
            api_key=rag_conf["DASHSCOPE_API_KEY"],
            base_url=rag_conf["DASHSCOPE_API_BASE"],
            check_embedding_ctx_length=False,
            chunk_size=10
    )


chat_model=ChatModel().generator()
embedding_model=EmbeddingModel().generator()

