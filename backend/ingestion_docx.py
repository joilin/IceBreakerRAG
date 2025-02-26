import os
from docx import Document
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))

# 1. DOCXファイルの読み込み
def _load_docx(file_path: str) -> List[Dict]:
    """DOCXファイルからテキストとメタデータを抽出"""
    doc = Document(file_path)
    content = []

    # メタデータ取得
    metadata = {
        "source": file_path,
        "author": doc.core_properties.author,
        "created": doc.core_properties.created.isoformat()
    }

    # 段落ごとに処理
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            content.append({
                "text": para.text,
                "section": f"paragraph_{i}",
                **metadata
            })

    return content

# 2. テキストの前処理とチャンキング
def _chunk_text(data: List[Dict]) -> List[Dict]:
    """テキストを適切なサイズに分割"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = []
    for item in data:
        texts = text_splitter.split_text(item["text"])
        for i, text in enumerate(texts):
            chunks.append({
                "id": f"{item['section']}_chunk{i}",
                "text": text,
                "metadata": {
                    "source": item["source"],
                    "author": item["author"],
                    "created": item["created"],
                    "chunk_num": i
                }
            })
    return chunks

# 3. テキストのベクトル化,Pineconeへのデータ登録
def _generate_embeddings_to_pinecone(chunks: List[Dict]):
    """OpenAIでEmbedding生成"""
    texts = [item["text"] for item in chunks]

    print(f"Going to add {len(texts)} to Pinecone")
    PineconeVectorStore.from_texts(
        texts, embeddings, index_name=INDEX_NAME
    )
    print("****Loading to vectorstore done ***")

def ingest_docx(file_path: str) -> str:
    try:
        # 1. DOCX読み込み
        raw_data = _load_docx(file_path)

        # 2. チャンキング
        chunks = _chunk_text(raw_data)

        # 3. ベクトル生成（例: OpenAIを使用）,Pineconeへのデータ登録
        _generate_embeddings_to_pinecone(chunks)

        return "Loaded"
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":

    # docxファイルのパス
    docx_path = r"C:\Users\joili\PycharmProjects\IceBreaker_RAG\量子力学.docx"

    ingest_docx(docx_path)