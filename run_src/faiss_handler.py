# app/faiss_handler.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaissHandler:
    def __init__(self, index_path, csv_path, model_name='all-MiniLM-L6-v2'):
        """
        初始化 FaissHandler。

        参数:
        - index_path (str): FAISS 索引文件的路径。
        - csv_path (str): evidence 数据的路径。
        - model_name (str): SentenceTransformer 模型名称。
        """
        self.index_path = index_path
        self.csv_path = csv_path
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.evidence_df = None

    def load_faiss_index(self):
        """
        从指定路径加载 FAISS 索引。
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS 索引文件不存在: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        logger.info(f"FAISS 索引已从 {self.index_path} 加载。")

    def load_evidence_dataframe(self):
        """
        从指定路径加载 evidence 数据。
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"evidence 数据文件不存在: {self.csv_path}")
        self.evidence_df = pd.read_csv(self.csv_path)
        logger.info(f"evidence 数据已从 {self.csv_path} 加载。")

    def prepare_handler(self):
        """
        加载 FAISS 索引和 evidence 数据。
        """
        self.load_faiss_index()
        self.load_evidence_dataframe()

    def retrieve(self, query, top_k=5):
        """
        根据查询语句检索最相似的 evidence 项。

        参数:
        - query (str): 查询语句。
        - top_k (int): 返回的最相似的 evidence 数量。

        返回:
        - List[Dict]: 包含 evidence 内容和相似度分数的字典列表。
        """
        if not self.model or not self.index or self.evidence_df is None:
            raise ValueError("FAISS 索引或 evidence 数据尚未加载。")

        # 生成查询嵌入
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')

        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.evidence_df):
                continue  # 跳过无效索引
            evidence = self.evidence_df.iloc[idx]['evidence']
            results.append({
                'evidence': evidence,
                'distance': float(distance)  # 转换为 float 类型以便序列化
            })
        return results

    def move_index_to_gpu(self, gpu_id=0):
        """
        将 FAISS 索引迁移到 GPU。

        参数:
        - gpu_id (int): 使用的 GPU 设备 ID。
        """
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
        logger.info(f"FAISS 索引已迁移到 GPU {gpu_id}。")
