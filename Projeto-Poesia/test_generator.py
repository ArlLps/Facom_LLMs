#!/usr/bin/env python3
"""Teste do gerador de poesias fine-tuned."""

from poesia.generator import PoetryGenerator

# Carregar gerador com modelo fine-tuned
generator = PoetryGenerator(finetuned_path="models/distilgpt2-poesia")

# Testar gerações com diferentes prompts
prompts = ["amor", "noite", "estrela", "silêncio"]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*60}")
    
    poetry = generator.generate(
        prompt=prompt,
        max_length=150,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    print(poetry)
