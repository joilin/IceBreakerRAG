import os
from dotenv import load_dotenv
from pydantic import SecretStr
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))

class CSVFileVectorizer:
    def ingest_csv(self,file_path) -> str:
        try:
            # CSVファイルを読み込む
            df = pd.read_csv(file_path, nrows=0)

            # 列名を動的に取得（1行目を列名として使用）
            columns = df.columns.tolist()  # 1行目を列名に設定

            csv_loader = CSVLoader(
                file_path=file_path,
                csv_args={
                    'delimiter': ',',
                    'fieldnames': columns
                })
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            documents = csv_loader.load_and_split(text_splitter)

            print(f"Going to add {len(documents)} to Pinecone")
            PineconeVectorStore.from_documents(
                documents, embeddings, index_name=INDEX_NAME
            )
            print("****Loading to vectorstore done ***")

            return "Loaded"
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            raise

if __name__ == '__main__':
    # 使用例
    # csvファイルのパス
    csv_path = r"C:\Users\joili\PycharmProjects\IceBreaker_RAG\customers-100.csv"

    loader = CSVFileVectorizer()
    loader.ingest_csv(csv_path)
