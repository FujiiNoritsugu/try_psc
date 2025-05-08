import streamlit as st
import MeCab
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import scipy.sparse as sp
import numpy as np

PROJECT_ID = os.environ.get("PROJECT_ID", "gen-lang-client-0471694923")
LOCATION = os.environ.get("LOCATION", "asia-northeast1")
INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID", "8864297927502200832")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "vs_hybridsearch_ja_deployed")

aiplatform.init(project=PROJECT_ID, location=LOCATION)

tagger = MeCab.Tagger("-r /dev/null -d /usr/local/lib/mecab/dic/unidic-lite")


def mecab_tokenizer(text):
    """日本語テキストをトークン化"""
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface != "":  # 空白行を除外
            tokens.append(node.surface)
        node = node.next
    return tokens


vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer, token_pattern=r"(?u)\b\w\w+\b")

dummy_corpus = [
    "東京は大阪の東にある",
    "大阪は東京の西にある",
    "京都は大阪の北にある",
    "札幌は東京の北にある",
    "那覇は大阪の南にある",
]
vectorizer.fit(dummy_corpus)


def get_sparse_embedding(text):
    """入力テキストを TF-IDF 疎ベクトルに変換"""
    tfidf_vector = vectorizer.transform([text])

    csr = sp.csr_matrix(tfidf_vector)

    values = [float(x) for x in csr.data]
    dims = [int(x) for x in csr.indices]

    return {"values": values, "dimensions": dims}


model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual")


def get_query_dense_embedding(text):
    """入力クエリを密ベクトルに変換"""
    input = TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")
    return model.get_embeddings([input])[0].values


def get_index_endpoint():
    endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    )
    return aiplatform.MatchingEngineIndexEndpoint(endpoint_name)


def hybrid_search(query, top_k=5):
    dense_embedding = get_query_dense_embedding(query)
    sparse_embedding = get_sparse_embedding(query)

    index_endpoint = get_index_endpoint()

    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[dense_embedding],
        num_neighbors=top_k,
        sparse_queries=[sparse_embedding],
    )

    return response


st.title("日本の観光地検索")
st.write("日本の観光地に関する情報を検索できます。キーワードを入力してください。")

query = st.text_input("検索キーワード", "京都の観光地")

if st.button("検索"):
    try:
        with st.spinner("検索中..."):
            results = hybrid_search(query)

            st.subheader("検索結果")

            if not results or not results[0]:
                st.write("検索結果が見つかりませんでした。")
            else:
                for i, match in enumerate(results[0]):
                    neighbor = match.neighbor
                    st.write(f"**{i+1}. {neighbor.id}**")
                    st.write(f"スコア: {match.distance:.4f}")
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.write("詳細なエラー情報:")
        st.exception(e)
