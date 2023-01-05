from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torchUtils
import pandas as pd

torch_device = torchUtils.getDevice()


def get_embeddings(df: pd.DataFrame):
    """
    Create embeddings using sentence_transformers (model is set in the function)
    Fills in the embeddings, mergedTitles and tokensLength columns to the dataframe passed in
    """

    # We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
    # 'nq-distilbert-base-v1' was in the e.g. but not too good
    model_name = "multi-qa-mpnet-base-dot-v1"

    bi_encoder = SentenceTransformer(model_name, device=torch_device)
    print("Max Sequence Length:", bi_encoder.max_seq_length)

    # calculate the length of the tokens
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/" + model_name)

    # TODO: better way to get token length? eg. get it from bi_encoder.encode ?
    #   or avoid tokenizing twice by feeding these tokens to the encoder?
    df['tokensLength'] = df.apply(lambda row:
                                  len(tokenizer.encode([row.fullTitle, row.content])), axis=1)

    # FIXME: how to handle the max_seq_length?
    corpus_embeddings = bi_encoder.encode(
        df[["fullTitle", "content"]].values.tolist(),
        convert_to_numpy=True, show_progress_bar=True)

    df['embeddings'] = corpus_embeddings.tolist()

    print("**** Done encoding.\n")

    return
