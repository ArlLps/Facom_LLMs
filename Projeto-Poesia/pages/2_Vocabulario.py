"""Pagina Streamlit para visualizacao do vocabulario do tokenizer."""

# pyright: reportMissingModuleSource=false

from __future__ import annotations

from pathlib import Path

import streamlit as st
from transformers import AutoTokenizer

try:
    import pandas as pd
except ImportError:  # pragma: no cover - fallback para ambientes sem pandas instalado
    pd = None


st.set_page_config(page_title="Vocabulario do Modelo", page_icon="🔤", layout="wide")
st.title("🔤 Vocabulario do Modelo")

workspace_root = Path(__file__).resolve().parents[1]

candidate_model_paths = [
    workspace_root / "models" / "distilgpt2-poesia-poems",
    workspace_root / "models" / "distilgpt2-poesia-clean",
    workspace_root / "models" / "distilgpt2-poesia",
]
model_path = next((path for path in candidate_model_paths if path.exists()), None)

if model_path is None:
    st.warning(
        "Nenhum modelo fine-tuned encontrado em models/. "
        "Treine ou copie um modelo antes de abrir esta pagina."
    )
    st.stop()


@st.cache_resource
def load_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(path)


def token_display(tokenizer, token: str) -> str:
    text = tokenizer.convert_tokens_to_string([token])
    text = text.replace("\n", "\\n")
    return text if text else "(vazio)"


def build_vocab_df(tokenizer):
    vocab = tokenizer.get_vocab()
    rows = []
    for token, token_id in vocab.items():
        rows.append(
            {
                "id": int(token_id),
                "token": token,
                "texto": token_display(tokenizer, token),
                "especial": token in tokenizer.all_special_tokens,
                "tamanho_token": len(token),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("id").reset_index(drop=True)


tokenizer = load_tokenizer(str(model_path))

if pd is None:
    st.error("Pandas nao encontrado no ambiente. Instale com: pip install pandas")
    st.stop()

df_vocab = build_vocab_df(tokenizer)

st.caption(f"Modelo em uso: {model_path.name}")

col1, col2, col3 = st.columns(3)
col1.metric("Tamanho do vocabulario", f"{len(df_vocab):,}".replace(",", "."))
col2.metric("Menor ID", int(df_vocab["id"].min()))
col3.metric("Maior ID", int(df_vocab["id"].max()))

st.subheader("Filtros")
with st.sidebar:
    st.header("🔎 Buscar no Vocabulario")
    search_text = st.text_input("Buscar token ou texto")
    min_id = st.number_input("ID minimo", min_value=0, value=0, step=1)
    max_id = st.number_input(
        "ID maximo",
        min_value=int(min_id),
        value=int(df_vocab["id"].max()),
        step=1,
    )
    only_special = st.checkbox("Apenas tokens especiais", value=False)
    max_points = st.selectbox("Pontos no grafico", [500, 1000, 2000, 5000, 10000], index=2)
    page_size = st.selectbox("Itens por pagina", [25, 50, 100, 250], index=1)

filtered = df_vocab[(df_vocab["id"] >= int(min_id)) & (df_vocab["id"] <= int(max_id))]

if search_text.strip():
    search_lower = search_text.lower()
    filtered = filtered[
        filtered["token"].str.lower().str.contains(search_lower, regex=False)
        | filtered["texto"].str.lower().str.contains(search_lower, regex=False)
    ]

if only_special:
    filtered = filtered[filtered["especial"]]

filtered = filtered.reset_index(drop=True)

total = len(filtered)
if total == 0:
    st.info("Nenhum token encontrado com os filtros atuais.")
else:
    st.subheader("Dispersao de Tokens")
    st.caption("Eixo X: ID do token • Eixo Y: tamanho do token")

    if total > max_points:
        plot_df = filtered.sample(n=max_points, random_state=42).sort_values("id")
        st.caption(f"Mostrando amostra de {max_points} pontos de um total de {total} tokens")
    else:
        plot_df = filtered

    st.scatter_chart(
        plot_df,
        x="id",
        y="tamanho_token",
        color="especial",
        use_container_width=True,
    )

    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Pagina", min_value=1, max_value=total_pages, value=1, step=1)
    start = (int(page) - 1) * page_size
    end = start + page_size

    st.write(f"Mostrando {start + 1}-{min(end, total)} de {total} tokens")
    st.dataframe(filtered.iloc[start:end], use_container_width=True, hide_index=True)

st.subheader("Tokenizacao Rapida")
input_text = st.text_input("Texto para tokenizar", value="Na caída da noite")
if input_text.strip():
    ids = tokenizer.encode(input_text, add_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)
    preview = pd.DataFrame({"id": ids, "token": toks})
    st.dataframe(preview, use_container_width=True, hide_index=True)
