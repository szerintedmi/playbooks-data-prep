from embed_openAI import get_embeddings
# for different (free) model use this (but it's has 512  token limit over 8k with openai)):
# from embed_SentenceTransformer import get_embeddings
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

with st.spinner('Creating embeddings...'):
    df = pd.read_parquet('results/flattened_content.parquet')

    get_embeddings(df)

    df.to_parquet('results/embeddings_oai_ada.parquet')
    # df.to_parquet('results/embeddings_multi-qa-mpnet.parquet')
    print("Done.")

st.write(df)
