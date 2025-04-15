import os
import lazyllm

from common.config import Config
from lazyllm.tools.rag import DocField, DataType


current_dir = os.getcwd()
config = Config(current_dir + "/application.yml").parse()

milvu_store_conf = {
    "type": "milvus",  # milvus\chroma
    "kwargs": {
        "uri": config["rag"]["milvus"],
        "index_kwargs": {"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    },
}
print(milvu_store_conf)

chroma_store_conf = {
    "type": "chroma",  # milvus\chroma
    "kwargs": {"dir": "/Users/taoruicheng/python_work/RagLazyLLM/docs/db/chroma/"},
}


m = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

doc_fields = {
    "comment": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=" "),
    "signature": DocField(data_type=DataType.VARCHAR, max_size=32, default_value=" "),
}

documentsMilvus = lazyllm.Document(
    dataset_path=config["rag"]["docPath"],
    embed=m,
    manager=False,
    store_conf=milvu_store_conf,
    doc_fields=doc_fields,
)

documentsMilvus.create_node_group(
    name="block", transform=lambda s: s.split("\n") if s else ""
)

documentsChroma = lazyllm.Document(
    dataset_path=config["rag"]["docPath"],
    embed=m,
    manager=False,
    store_conf=chroma_store_conf,
)

retrieverCosine = lazyllm.Retriever(
    documentsMilvus,
    group_name="block",
    topk=3,
    # output_format="content",  # similarity="cosine",
)

retrieverBm25 = lazyllm.Retriever(
    doc=documentsChroma, group_name="CoarseChunk", similarity="bm25_chinese", topk=3
)  # 定义检索组件

resultCosine = retrieverCosine("图像压缩是什么？")
print("retrieverCosine:", resultCosine)
resultBm25 = retrieverBm25("图像压缩是什么？")
print("retrieverBm25:", resultBm25[0].text)

online_rerank = lazyllm.OnlineEmbeddingModule(type="rerank")
reranker = lazyllm.Reranker("ModuleReranker", model=online_rerank, topk=3)

rerankerResult = reranker(resultCosine + resultBm25, query="图像压缩是什么？")
print("reranker:", rerankerResult)
print("reranker 0 :", rerankerResult[0].text)

onlineChat = lazyllm.OnlineChatModule()
prompt = "你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n "
onlineChat.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=["context_str"]))
res = onlineChat(
    {
        "query": "图像压缩是什么？",
        "context_str": "".join([node.get_content() for node in rerankerResult]),
    }
)


print(f"Answer: {res}")
