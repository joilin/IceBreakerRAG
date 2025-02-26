import os
import re

from pydantic import SecretStr
import hashlib
import openai
import pinecone
from dotenv import load_dotenv
from typing import List, Dict
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))


class TextFileVectorizer:
    def __init__(self):
        # テキスト分割の設定
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " "]
        )

    def ingest_txt(self, file_path: str) -> str:
        try:
            """単一ファイルの処理"""
            # テキスト読み込み
            text = self._read_text_file(file_path)

            # 前処理
            cleaned_text = self._preprocess_text(text)

            document = Document(
                page_content=cleaned_text,
                metadata={
                    'title': os.path.basename(file_path),
                    "char_length": len(cleaned_text)
                }
            )

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            documents = text_splitter.split_documents([document])

            print(f"Going to add {len(documents)} to Pinecone")
            PineconeVectorStore.from_documents(
                documents, embeddings, index_name=INDEX_NAME
            )
            print("****Loading to vectorstore done ***")

            return "Loaded"
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            raise

    def _read_text_file(self, file_path: str) -> str:
        """テキストファイルの読み込み（エンコーディング自動検出）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='shift_jis') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Encoding error: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """テキストの前処理"""
        # 不要な空白・改行の削除
        text = re.sub(r'\s+', ' ', text)

        # 制御文字の除去
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

        # URLの除去
        text = re.sub(r'https?://\S+', '', text)

        return text.strip()


if __name__ == "__main__":

    # txtファイルのパス
    txt_path = r"C:\Users\joili\PycharmProjects\IceBreaker_RAG\名古屋の天気.txt"

    # 使用例
    vectorizer = TextFileVectorizer()
    vectorizer.ingest_txt(txt_path)
