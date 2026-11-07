import os
import sys


from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embedding_model
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex


class VectorStoreService:
    def __init__(self):
        self.vector_store =Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_conf["persist_directory"]
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r") as f:
                for line in f.readlines():
                    line=line.strip()
                    if line == md5_for_check:
                        return True
                return False
        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a") as f:
                f.write(md5_for_save + "\n")
        def get_file_documents(read_path: str)->list[Document]:
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []
        allow_file_path = listdir_with_allowed_type(get_abs_path(chroma_conf["data_path"]), tuple(chroma_conf["allow_knowledge_file_type"]))
        for file_path in allow_file_path:
            if  check_md5_hex(get_file_md5_hex(file_path)):
                logger.info(f"[加载文档]文件{file_path}已存在，跳过加载")
                continue
            #存入向量数据库
            try:
                documents:list[Document]=get_file_documents(file_path)
                if not documents:
                    logger.warning(f"[加载文档]文件{file_path}无内容")
                    continue
                #对文档分片
                chunks: list[Document] = self.spliter.split_documents(documents)
                if not chunks:
                    logger.warning(f"[加载文档]文件{file_path}分片后无内容")
                    continue
                #存入向量数据库
                self.vector_store.add_documents(chunks)
                #保存md5_hex
                save_md5_hex(get_file_md5_hex(file_path))
            except Exception as e:
                logger.error(f"[加载文档]文件{file_path}存入向量数据库失败",exc_info=True)
                continue
if __name__ == '__main__':
    vs=VectorStoreService()
    vs.load_document()
    retriever=vs.get_retriever()
    res=retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("="*20)
