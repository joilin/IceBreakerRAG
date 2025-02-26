import os
import pandas as pd
import numpy as np
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import hashlib

# 環境変数の読み込み
load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
Pinecone_API_KEY = os.environ['PINECONE_API_KEY']
Pinecone_ENV= os.environ['PINECONE_ENV']
client = OpenAI(api_key=OpenAI_API_Key)
pinecone = Pinecone(api_key=Pinecone_API_KEY)

class ExcelFileVectorizer:
    def __init__(self):
        self.date_format = "%Y-%m-%d"

    def ingest_excel(self, file_path: str):
        """
        Excelファイルの処理パイプライン
        """
        # データ読み込み
        df = self._read_excel(file_path)

        # データ変換
        processed_data = self._transform_data(df)

        # ベクトル生成
        vector_data = self._generate_vectors(processed_data)

        # Pineconeにアップロード
        self._upload_to_pinecone(vector_data)

    def _read_excel(self, file_path: str) -> pd.DataFrame:
        """Excelファイルの読み込み"""
        df = pd.read_excel(file_path, header=0)

        # 列名の検証
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            raise ValueError("最初の列が日付形式ではありません")

        return df

    def _transform_data(self, df: pd.DataFrame) -> List[Dict]:
        """データ構造の変換"""
        processed = []
        people = df.columns[1:].tolist()

        for _, row in df.iterrows():
            date = row[0].strftime(self.date_format)
            duty_flags = [1 if cell == '〇' else 0 for cell in row[1:]]
            duty_people = [people[i] for i, flag in enumerate(duty_flags) if flag == 1]

            processed.append({
                "date": date,
                "duty_flags": duty_flags,
                "duty_people": duty_people,
                "people_count": len(duty_people),
                "weekday": datetime.strptime(date, self.date_format).weekday()
            })

        return processed

    def _generate_vectors(self, data: List[Dict]) -> List[Dict]:
        # テキスト埋め込みの生成（テキストデータを保持するように修正）
        text_inputs = [
            f"{d['date']}の出張者: {', '.join(d['duty_people'])}"
            for d in data
        ]

        """ベクトル生成"""
        # テキスト埋め込みの生成
        text_embeddings = self._generate_text_embeddings(data)

        # 数値特徴量の生成
        numeric_features = np.array([
            [d['people_count'],d['weekday']] for d in data
        ])

        # 特徴量の結合と正規化
        combined = np.hstack([
            text_embeddings,
            numeric_features
        ])

        normalized = combined / np.linalg.norm(combined, axis=1, keepdims=True)

        # 最終データフォーマット
        return [{
            "id": self._generate_id(d['date']),
            "values": normalized[i].tolist(),
            "metadata": {
                "text": text_inputs[i],
                "date": d['date'],
                "people": d['duty_people'],
                "weekday": d['weekday'],
                "flags_hash": hashlib.md5(str(d['duty_flags']).encode()).hexdigest()
            }
        } for i, d in enumerate(data)]

    def _generate_text_embeddings(self, data: List[Dict]) -> np.ndarray:
        """テキスト埋め込みの生成"""
        text_inputs = [
            f"{d['date']}の出張者: {', '.join(d['duty_people'])}"
            for d in data
        ]

        response = client.embeddings.create(
            input=text_inputs,
            model="text-embedding-3-small",
            dimensions=1534
        )

        embeddings = [item.embedding for item in response.data]
        return np.array([item.embedding for item in response.data])

    def _generate_id(self, date: str) -> str:
        """一意のID生成"""
        return f"duty_{date.replace('-', '')}"

    def _upload_to_pinecone(self, data: List[Dict]):
        """Pineconeへのアップロード"""
        if INDEX_NAME not in [item["name"] for item in pinecone.list_indexes()]:
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=Pinecone_ENV
                )
            )

        index = pinecone.Index(INDEX_NAME)

        # データ形式変換
        vectors = [{
            "id": item["id"],
            "values": item["values"],
            "metadata": item["metadata"]
        } for item in data]

        # バッチアップロード
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i + 100])


if __name__ == "__main__":
    # excelファイルのパス
    excel_path = r"C:\Users\joili\PycharmProjects\IceBreaker_RAG\出張プラン.xlsx"

    vectorizer = ExcelFileVectorizer()
    vectorizer.ingest_excel(excel_path)