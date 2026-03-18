"""Interface Streamlit para gerador de poesia."""

from pathlib import Path
import streamlit as st
from poesia import PoetryGenerator

st.set_page_config(page_title="PoesIA", page_icon="✨", layout="wide")
st.title("✨ PoesIA - Gerador de Poesia com GPT-2 Fine-tuned ✨")

workspace_root = Path(__file__).resolve().parent
candidate_model_paths = [
    workspace_root / "models" / "distilgpt2-poesia-poems",
    workspace_root / "models" / "distilgpt2-poesia-clean",
    workspace_root / "models" / "distilgpt2-poesia",
]
model_path = next((path for path in candidate_model_paths if path.exists()), None)

# Verificar se modelo fine-tuned existe
if model_path is None:
    st.warning(
        "⚠️ Nenhum modelo fine-tuned encontrado em models/\n\n"
        "Execute primeiro: `python poesia/trainer.py`"
    )
    st.stop()

st.caption(f"Modelo em uso: {model_path.name}")

# Carregar modelo
@st.cache_resource
def load_generator():
    return PoetryGenerator(
        model_name="distilgpt2",
        finetuned_path=str(model_path)
    )

generator = load_generator()

# Sidebar com controles
with st.sidebar:
    st.header("⚙️ Controles")
    
    prompt = st.text_input(
        "Verso inicial ou tema",
        value="Quando a lua brilha",
        placeholder="Ex: Amor, noite, esperança..."
    )
    
    max_length = st.number_input(
        "Comprimento máximo (tokens)",
        min_value=12,
        max_value=120,
        value=36,
        step=2,
        help="Para poemas completos, 30-50 costuma funcionar melhor"
    )
    
    temperature = st.number_input(
        "Temperatura (criatividade)",
        min_value=0.5,
        max_value=1.2,
        value=0.8,
        step=0.05,
        format="%.2f",
        help="Baixo=determinístico, Alto=criativo"
    )
    
    top_k = st.number_input(
        "Top-k (diversidade)",
        min_value=10,
        max_value=100,
        value=40,
        step=1,
        help="Considera apenas top-k tokens mais prováveis"
    )
    
    top_p = st.number_input(
        "Top-p (nucleus sampling)",
        min_value=0.75,
        max_value=0.98,
        value=0.9,
        step=0.01,
        format="%.2f",
        help="Evite valores muito baixos (ex: 0.25), que empobrecem a saída"
    )
    
    num_poems = st.number_input(
        "Quantidade de poesias",
        min_value=1,
        max_value=6,
        value=3,
        step=1
    )

# Botão de geração
if st.button("🎨 Gerar Poesia", type="primary", use_container_width=True):
    with st.spinner("Gerando poesia..."):
        poems = generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_poems,
        )
    
    for i, poem in enumerate(poems, 1):
        with st.container(border=True):
            st.markdown(f"**Poesia #{i}**")
            st.text(poem)
            st.caption(f"Tokens: {len(poem.split())}")

st.markdown("---")
st.caption(
    "Gerador de Poesia • Fine-tuning em distilgpt2 • GSI073"
)
