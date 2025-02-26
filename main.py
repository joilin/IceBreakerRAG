import os
from backend.core import run_llm
from backend.ingestion import load_file
import streamlit as st
from streamlit_chat import message

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
# if "prompt" not in st.session_state:
#     st.session_state.prompt = ""

# サイドバー設定
with st.sidebar:
    st.title("⚙️ 設定")
    file_path = st.text_input("ドキュメントのファイルパスを入力:",key="filepath_input")

    if st.button("ドキュメントを埋め込み",key="file_load_button"):
        if os.path.exists(file_path):
            load_file(file_path)
            st.write("ファイルをロードしました")
        else:
            st.error("ファイルが見つかりません")

# メイン画面
st.title("💬 ドキュメントチャットボット")
st.caption("ドキュメントをベクトルストアに埋め込んで質問できます")

# # チャット入力フォーム
# def clear_text():
#     # 入力内容をクリア
#     st.session_state.prompt = ""

prompt = st.text_input("Prompt",key="prompt_input",placeholder="Enter your prompt here...")



if (
    "chat_answer_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history"  not in st.session_state
):
    st.session_state["chat_answer_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt,chat_history=st.session_state["chat_history"]
        )

        formatted_response = f"{generated_response['answer']}"

        st.session_state["chat_answer_history"].append(prompt)
        st.session_state["user_prompt_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human",prompt))
        st.session_state["chat_history"].append(("ai",generated_response['answer']))

        # # 入力内容をクリア
        # clear_text()

if st.session_state["chat_answer_history"]:
    # for generated_response,user_query in zip(st.session_state["chat_answer_history"],st.session_state["user_prompt_history"]):
    #     message(user_query,is_user=True,key="user_message")
    #     message(generated_response,key="llm_message")

    # 逆順でループ処理
    for i, (generated_response, user_query) in enumerate(reversed(list(zip(
            st.session_state["chat_answer_history"],
            st.session_state["user_prompt_history"]
    )))):
        # インデックスを逆順に合わせて調整
        original_index = len(st.session_state["chat_answer_history"]) - i - 1
        message(generated_response, key=f"bot_{original_index}")
        message(user_query, is_user=True, key=f"user_{original_index}")


