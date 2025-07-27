import requests
import numpy as np
from typing import List, Union
from ollama import Client

OLLAMA_MODEL_DIM_MAP = {
    "bge-m3": 1024,
    "mxbai-embed-large": 768,
    "nomic-embed-text": 768,
}

class OllamaBGE:
    def __init__(self, model: str = "bge-m3", batch_size: int = 32, host: str = "http://localhost:11434"):
        """初始化Ollama BGE模型
        
        Args:
            model: 模型名称，默认为"bge-m3"
            batch_size: 批处理大小，默认为32
            host: Ollama服务器地址，默认为本地11434端口
        """
        self.model = model
        self.batch_size = batch_size
        self.dim = OLLAMA_MODEL_DIM_MAP.get(model, 1024)
        self.client = Client(host=host)
        
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """将文本编码为向量
        
        Args:
            texts: 单个文本字符串或文本列表
            normalize: 是否对输出向量进行归一化
            
        Returns:
            文本的嵌入向量
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 批量处理
        if self.batch_size > 0 and len(texts) > self.batch_size:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._get_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
        else:
            embeddings = self._get_embeddings(texts)
            
        embeddings = np.array(embeddings)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        return embeddings.squeeze()
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        # 一次性处理整个文本列表
        response = self.client.embed(model=self.model, input=texts)
        return response["embeddings"]
    
    @property
    def dimension(self) -> int:
        """获取当前模型的嵌入向量维度
        
        Returns:
            向量维度
        """
        return self.dim

if __name__ == "__main__":
    # 初始化模型
    encoder = OllamaBGE(model="bge-m3", batch_size=32)

    # 编码单个文本
    text = "这是一个测试句子"
    embedding = encoder.encode(text)
    print(f"单个文本向量维度: {embedding.shape}")

    # 编码多个文本
    texts = ["第一个句子", "第二个句子", "第三个句子", "第四个句子", "第五个句子", "第六个句子", "第七个句子", "第八个句子", "第九个句子", "第十个句子"]
    embeddings = encoder.encode(texts)
    print(f"多个文本向量维度: {embeddings.shape}")
