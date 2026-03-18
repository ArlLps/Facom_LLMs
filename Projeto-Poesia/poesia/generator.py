"""Gerador de poesia usando modelo fine-tuned."""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PoetryGenerator:
    """Gerador de poesia com controle fino de inferência."""
    
    def __init__(self, model_name: str = "distilgpt2", finetuned_path: str = None):
        """
        Inicializa gerador.
        
        Args:
            model_name: Nome do modelo base (HuggingFace)
            finetuned_path: Caminho para modelo fine-tuned. Se None, usa base.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = finetuned_path or model_name
        
        # Mantem tokenizer e modelo no mesmo snapshot para evitar incompatibilidades.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.bad_words_ids = self._build_bad_words_ids()

    def _build_bad_words_ids(self) -> list[list[int]]:
        """Constroi lista de tokens indesejados para bloquear na geracao."""
        blocked_terms = [
            "Advertisements",
            "Advertisement",
            "http",
            "www",
        ]
        bad_words_ids: list[list[int]] = []
        for term in blocked_terms:
            token_ids = self.tokenizer.encode(term, add_special_tokens=False)
            if token_ids:
                bad_words_ids.append(token_ids)
        return bad_words_ids

    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """Limpa artefatos comuns de decodificacao para melhorar legibilidade."""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("[", "").replace("]", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s*\.\s*", ". ", cleaned)
        cleaned = cleaned.strip()

        # Remove lixo textual conhecido que aparece em alguns corpora web.
        cleaned = re.sub(r"(?i)\badvertisements?\b", "", cleaned).strip()

        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt):].lstrip(" ,.;:-")

        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        filtered_lines: list[str] = []
        for line in lines:
            # Evita versos de um unico caractere, comuns em degradacao.
            if len(line) <= 1:
                continue
            if filtered_lines and line.lower() == filtered_lines[-1].lower():
                continue
            filtered_lines.append(line)

        if filtered_lines:
            return "\n".join(filtered_lines)
        return cleaned or prompt
    
    def generate(
        self,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """
        Gera poesia a partir de prompt.
        
        Args:
            prompt: Verso inicial ou tema
            max_length: Comprimento máximo em tokens
            temperature: Controla criatividade (0.1=determinístico, 2.0=aleatório)
            top_k: Mantém apenas top-k tokens mais prováveis
            top_p: Nucleus sampling (probabilidade cumulativa)
            num_return_sequences: Quantas poesias gerar
            
        Returns:
            Lista de poesias geradas
        """
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            bad_words_ids=self.bad_words_ids if self.bad_words_ids else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        poems = [
            self._clean_generated_text(
                self.tokenizer.decode(ids, skip_special_tokens=True),
                prompt,
            )
            for ids in output_ids
        ]
        
        return poems
