from backend.ingestion_docx import ingest_docx
from backend.ingestion_pdf import PDFLoader
from backend.ingestion_txt import TextFileVectorizer
from backend.ingestion_csv import CSVFileVectorizer
# from backend.ingestion_excel import ExcelFileVectorizer
from backend.ingestion_excel_process import ExcelToPineconeLoader

import os
from dotenv import load_dotenv
from pydantic import SecretStr
import pandas as pd

from docx import Document
import csv
from collections import defaultdict

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))

def load_file(file_path):
    """
    ファイルパスから拡張子を判定して適切な方法でファイルを読み込む
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.xlsx':
            excel_loader = ExcelToPineconeLoader()
            return excel_loader.ingest_excel(file_path)
        elif ext == '.csv':
            csv_loader = CSVFileVectorizer()
            return csv_loader.ingest_csv()
        elif ext == '.txt':
            txt_loader = TextFileVectorizer()
            return txt_loader.ingest_txt(file_path)
        elif ext == '.pdf':
            pdf_loader = PDFLoader()
            return pdf_loader.ingest_pdf(file_path)
        elif ext == '.docx':
            return ingest_docx(file_path)
        else:
            raise ValueError(f"未対応のファイル形式です: {ext}")
    except Exception as e:
        print(f"ファイル読み込みエラー ({file_path}): {str(e)}")
        return None



if __name__ == "__main__":
    # # Excelファイルのパス
    # excel_path = "../出張プラン.xlsx"
    #
    # # データの読み込み
    # df = load_excel_data(excel_path)
    files = []

    for file in files:
        print(f"\n=== Loading: {file} ===")
        content = load_file(file)

        if content is not None:
            if isinstance(content, pd.DataFrame):
                print("DataFrame形式で読み込み成功")
                print(content.head())
            else:
                print("テキスト内容:")
                print(content[:500])  # 長い場合は先頭500文字のみ表示
