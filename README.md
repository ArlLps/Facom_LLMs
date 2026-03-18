# Facom_LLMs

Repositório da disciplina optativa sobre LLMs da Facom.

O objetivo deste espaço é concentrar materiais práticos da disciplina, incluindo
notebooks de aula e um projeto aplicado de geração de texto em português.

## Conteúdo

- Notebooks de aula com fundamentos de NLP e LLMs:
	- regressão logística e SVM;
	- seq2seq e atenção;
	- tokenização;
	- transformer decoder-only e inferência;
	- RAG (retrieval-augmented generation).
- Projeto aplicado em `Projeto-Poesia/`, com fine-tuning e interface Streamlit.

## Estrutura do repositório

- `Aula02_tokenizacao_pratica.ipynb`
- `Aula_3_Transformer_Decoder_only_and_Inference.ipynb`
- `aula6_retrieval_augmented_generation_rag.ipynb`
- `GSI073_aula0_luong_attention.ipynb`
- `GSI073_aula0_regressao_logistica.ipynb`
- `GSI073_aula0_seq2seq.ipynb`
- `GSI073_aula0_support_vector_machine.ipynb`
- `Projeto-Poesia/`

## Projeto-Poesia

Dentro de `Projeto-Poesia/` está o projeto de geração de poesia em português.

Passos rápidos:

```bash
cd Projeto-Poesia
pip install -r requirements.txt
streamlit run app.py
```

## Requisitos

- Python 3.10+
- Ambiente virtual recomendado (`venv`)

## Observações

- Alguns notebooks podem exigir GPU para execução completa.
- Os modelos treinados e artefatos podem ocupar bastante espaço em disco.

## Licença

Uso acadêmico.
