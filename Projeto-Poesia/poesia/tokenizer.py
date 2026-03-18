"""Tokenizador e config para o gerador de poesia."""

from transformers import AutoTokenizer


def get_tokenizer(model_name: str = "distilgpt2"):
    """Carrega tokenizador do modelo."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_corpus(corpus_text: str, tokenizer) -> list[dict]:
    """
    Tokeniza corpus de poesia.
    
    Args:
        corpus_text: Texto contendo poesias separadas por '\\n---\\n'
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Lista de dicts com 'input_ids' e 'attention_mask'
    """
    poems = corpus_text.split("\n---\n")
    poems = [p.strip() for p in poems if p.strip()]
    
    tokenized = tokenizer(
        poems,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    
    return tokenized
