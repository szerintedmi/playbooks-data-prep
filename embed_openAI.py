import pandas as pd
import openai
import os
from openai.embeddings_utils import get_embedding
from transformers import GPT2TokenizerFast
import streamlit as st


# Based on openai cookbook example:
# https://github.com/openai/openai-cookbook/blob/838f000935d9df03e75e181cbcea2e306850794b/examples/Obtain_dataset.ipynb


# set your API key in os env var. Eg. conda env config vars set OPENAI_API_KEY=xxxx
openai.api_key = os.getenv("OPENAI_API_KEY")

tokenizer = GPT2TokenizerFast.from_pretrained(
    "gpt2")  # only used to get token length


def get_embeddings(df: pd.DataFrame) -> None:
    """
    Create embeddings using OpenAI API (model is set in the function)
        Works on the passed dataframe in place
    """

    df['tokensLength'] = df.apply(lambda row:
                                  len(tokenizer.encode(row["fullTitle"] + "\n" + row["content"])), axis=1)

    df['embeddings'] = df.apply(
        lambda row: get_embedding(row["fullTitle"] + "\n" + row["content"], engine='text-embedding-ada-002'), axis=1)


@st.experimental_memo
def get_single_embedding(text: str) -> object:
    """
    Get a single embedding from OpenAI API 
    returns the full response object (token usage data)
    """
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002',
    )
    return response
