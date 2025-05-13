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

tagger = MeCab.Tagger()


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

import wikipedia
import pandas as pd

# Wikipedia の言語を日本語に設定
wikipedia.set_lang("ja")

# 都道府県名のリスト
prefectures = [
    "北海道",
    "青森県",
    "岩手県",
    "秋田県",
    "宮城県",
    "山形県",
    "福島県",
    "茨城県",
    "栃木県",
    "群馬県",
    "埼玉県",
    "千葉県",
    "東京都",
    "神奈川県",
    "新潟県",
    "富山県",
    "石川県",
    "福井県",
    "山梨県",
    "長野県",
    "岐阜県",
    "静岡県",
    "愛知県",
    "三重県",
    "滋賀県",
    "京都府",
    "大阪府",
    "兵庫県",
    "奈良県",
    "和歌山県",
    "鳥取県",
    "島根県",
    "岡山県",
    "広島県",
    "山口県",
    "徳島県",
    "香川県",
    "愛媛県",
    "高知県",
    "福岡県",
    "佐賀県",
    "長崎県",
    "熊本県",
    "大分県",
    "宮崎県",
    "鹿児島県",
    "沖縄県",
]

# 各都道府県ごとに「〇〇の観光地」というタイトルの Wikipedia ページを取得
pages = [
    wikipedia.page(prefecture + "の観光地", auto_suggest=False)
    for prefecture in prefectures
]

# 抽出したデータを Pandas DataFrame に格納
df = pd.DataFrame(
    {
        "title": [page.title for page in pages],  # 各 Wikipedia ページのタイトル
        "url": [page.url for page in pages],  # 各 Wikipedia ページの URL
        "content": [page.content for page in pages],  # 各 Wikipedia ページの内容
    }
)

# 各 Wikipedia ページの内容を corpus_ja に格納
corpus_ja = df.content.tolist()

vectorizer.fit(corpus_ja)


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


from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    HybridQuery,
)


def hybrid_search(query_text, top_k=5):

    query_dense_emb = get_query_dense_embedding(query_text)
    query_sparse_emb = get_sparse_embedding(query_text)
    query = HybridQuery(
        dense_embedding=query_dense_emb,
        sparse_embedding_dimensions=query_sparse_emb["dimensions"],
        sparse_embedding_values=query_sparse_emb["values"],
        rrf_ranking_alpha=0.5,
    )

    index_endpoint = get_index_endpoint()

    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query],
        num_neighbors=top_k,
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
