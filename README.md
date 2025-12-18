# BhabhaLLM: A Homi J. Bhabha Persona Model

A hobby project designed to fine-tune the **Mistral 7B v0.3** model to emulate the personality, eloquence, and scientific visionary spirit of **Dr. Homi J. Bhabha**.

## Project Overview
This model uses **4-bit QLoRA** (via Unsloth) to align the base model with a custom dataset of scientific and philosophical dialogues inspired by Bhabha's legacy.

- **Base Model:** `mistralai/Mistral-7B-v0.3`
- **Training Method:** Parameter-Efficient Fine-Tuning (PEFT) / QLoRA
- **Format:** ChatML
- **Compute:** Trained on an NVIDIA Tesla T4 (Google Colab)

## Repository Structure
- `Train_Bhabha.ipynb`: The notebook used for training and inference.
- `bhabha_dataset.jsonl`: Curated conversation dataset for fine-tuning.
- `Bhabha-v1-Adapters/`: Model adapters (also hosted on [Hugging Face](https://huggingface.co/moonboyknm/BhabhaLLM)).

## Quick Start
To run inference using the adapters from Hugging Face:

```python
from unsloth import FastLanguageModel
from huggingface_hub import snapshot_download
import os

repo_path = snapshot_download(repo_id="moonboyknm/BhabhaLLM")
adapter_path = os.path.join(repo_path, "Bhabha-v1-Adapters")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapter_path,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
```

## Disclaimer
This is a small-scale research project for educational purposes and may not perfectly reflect the historical accuracy of Dr. Bhabha's views.
