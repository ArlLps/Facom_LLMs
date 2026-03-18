"""Fine-tuning de modelo para poesia."""

import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


def finetune_model(
    corpus_path: str,
    model_name: str = "distilgpt2",
    output_dir: str = "models/distilgpt2-poesia",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 128,
):
    """
    Fine-tune modelo em corpus de poesia.
    
    Args:
        corpus_path: Caminho para arquivo .txt com corpus
        model_name: Modelo base do HuggingFace
        output_dir: Onde salvar modelo fine-tuned
        epochs: Número de épocas
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Tamanho máximo de sequência para tokenização
    """
    print(f"Carregando tokenizador e modelo {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"Carregando corpus de {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    poems = [p.strip() for p in text.split("\n---\n") if p.strip()]
    dataset = Dataset.from_dict({"text": poems})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=100,
        save_total_limit=2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    print("Iniciando fine-tuning...")
    trainer.train()
    
    print(f"Salvando modelo em {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning concluído!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.txt", help="Corpus path")
    parser.add_argument("--output", default="models/distilgpt2-poesia", help="Output dir")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="LR")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent.parent
    corpus_path = script_dir / args.corpus
    output_dir = script_dir / args.output
    
    finetune_model(
        corpus_path=str(corpus_path),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
