from langchain_huggingface import HuggingFaceEmbeddings


def build_sentence_transformer_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )