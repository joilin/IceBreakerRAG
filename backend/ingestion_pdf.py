import os
from dotenv import load_dotenv
from pydantic import SecretStr
import re

from typing import List,Optional
import PyPDF2
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))

class PDFLoader:
    def __init__(self, extract_metadata: bool = True, clean_text: bool = True):
        self.extract_metadata = extract_metadata
        self.clean_text = clean_text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def ingest_pdf(self, file_path: str) -> str:
        try:
            # pdfテキスト抽出,前処理
            loaded_documents = self._load_pdf(file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            documents = text_splitter.split_documents(loaded_documents)

            print(f"Going to add {len(documents)} to Pinecone")
            PineconeVectorStore.from_documents(
                documents, embeddings, index_name=INDEX_NAME
            )
            print("****Loading to vectorstore done ***")

            return "Loaded"
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            raise

    # 1. PDFの読み込み,
    def _load_pdf(self, file_path: str) -> Optional[List[Document]]:
        """
        PDFファイルを読み込んでテキストとメタデータを返す
        """
        if not os.path.exists(file_path):
            logger.error(f"ファイルが存在しません: {file_path}")
            return None

        result = []
        try:
            pdf_text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    pdf_text += page.extract_text(x_tolerance=2, y_tolerance=2)

                if self.clean_text:
                    pdf_text = self._clean_text(pdf_text)

                document = Document(
                    page_content=pdf_text,
                    metadata=self._extract_metadata(pdf) if self.extract_metadata else {}
                )

                result.append(document)

                return result

        except Exception as e:
            logger.error(f"PDF読み込みエラー ({file_path}): {str(e)}")
            return None

    def _extract_text(self, pdf_reader: PyPDF2.PdfReader) -> str:
        """PDFからテキストを抽出"""
        text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    def _extract_metadata(self, pdf_reader: pdfplumber.PDF) -> dict:
        """メタデータを抽出"""
        metadata = pdf_reader.metadata
        return {
            'title': metadata.get('/Title', ''),
            'author': metadata.get('/Author', ''),
            'creator': metadata.get('/Creator', ''),
            'producer': metadata.get('/Producer', ''),
            'creation_date': metadata.get('/CreationDate', ''),
            'modification_date': metadata.get('/ModDate', ''),
            'num_pages': len(pdf_reader.pages)
        }

    def _clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        # 不要な改行と空白の削除
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # 特殊文字の除去
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

        # ヘッダー/フッターと思われるパターンの除去（例）
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        return text.strip()


if __name__ == '__main__':
    # 使用例
    # pdfファイルのパス
    pdf_path = r"C:\Users\joili\PycharmProjects\IceBreaker_RAG\the-little-mongodb-book-ja.pdf"

    loader = PDFLoader()
    loader.ingest_pdf(pdf_path)

