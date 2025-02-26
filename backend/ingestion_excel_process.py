import os
import pandas as pd
from datetime import datetime
from langchain_pinecone import PineconeVectorStore
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
INDEX_NAME = os.environ['INDEX_NAME']
OpenAI_API_Key = os.environ['OPENAI_API_KEY']
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))


class ExcelToPineconeLoader:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=SecretStr(OpenAI_API_Key))
        self.index_name = INDEX_NAME
        self.weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]

        # チャンク分割の設定
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            separators=["\n", "。", "、", " ", ""],
            length_function=len
        )

    def ingest_excel(self, file_path: str) -> str:
        """Excelファイルを読み込みPineconeにアップロード"""
        try:
            # Excelデータの読み込みと前処理
            documents = self._process_excel(file_path)

            # チャンク分割
            chunked_documents = self._split_documents(documents)

            # Pineconeへのアップロード
            print(f"Going to add {len(documents)} to Pinecone")
            PineconeVectorStore.from_documents(
                documents=chunked_documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            print("****Loading to vectorstore done ***")

            return "Loaded"
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            raise

    def _process_excel(self, file_path: str) -> list[Document]:
        """ExcelデータをLangChain Document形式に変換"""
        # Excel読み込み
        df = pd.read_excel(file_path)

        # 列名の検証
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            raise ValueError("最初の列が日付形式ではありません")

        # ドキュメント生成
        documents = []
        people_columns = df.columns[1:]

        for _, row in df.iterrows():
            date = row[0]
            duty_people = self._get_duty_people(row, people_columns)

            # ドキュメント内容の生成
            doc = self._create_document(date, duty_people)
            documents.append(doc)

        return documents

    def _get_duty_people(self, row: pd.Series, columns: list) -> list[str]:
        """当番者リストを取得"""
        return [col for col in columns if row[col] == '〇']

    def _create_document(self, date: datetime, people: list[str]) -> Document:
        """LangChain Documentオブジェクトの生成"""
        # 日付情報のフォーマット
        date_str = date.strftime("%Y-%m-%d")
        weekday = self.weekdays_jp[date.weekday()]

        content = (
            f"日付: {date_str}（{weekday}）\n"
            f"出張予定者: {', '.join(people) if people else 'なし'}\n"
            f"関連情報: この日は{len(people)}人の当番が割り当てられています"
        )

        metadata = {
            "date": date_str,
            "weekday": weekday,
            "people": people,
            "timestamp": date.timestamp(),
            "source": "duty_schedule"
        }

        return Document(
            page_content=content,
            metadata=metadata
        )

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """ドキュメントをチャンク分割"""
        all_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])

            # チャンクにメタデータを追加
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata['date']}-{i}",
                    "total_chunks": len(chunks)
                })

            all_chunks.extend(chunks)

        return all_chunks

if __name__ == "__main__":
    # excelファイルのパス
    excel_path = r"C:\Users\joili\PycharmProjects\IceBreaker_RAG\出張プラン.xlsx"

    # 使用例
    loader = ExcelToPineconeLoader()

    # Excelファイルの処理
    loader.ingest_excel(excel_path)