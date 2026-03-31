# Customer Support LLM Fine-Tuning

This project fine-tunes Qwen2.5-0.5B-Instruct for customer-support style responses using a three-notebook pipeline:

1. Data preparation and dataset construction
2. LoRA SFT training
3. DPO preference optimization

The work is based on the Bitext customer support dataset and runs on Apple MPS (with CUDA/CPU fallback logic in the notebooks).

## Base Configuration

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- Training stack: transformers + peft + trl + datasets
- Training approach: LoRA SFT followed by DPO

## What Was Done

### 1) Data preparation

Notebook: [notebooks/data_preparation.ipynb](notebooks/data_preparation.ipynb)

- Loaded the Bitext customer support dataset.
- Applied stratified sampling by intent to create balanced subsets:
    - SFT subset: 150 samples per intent, target total 4050
    - DPO subset: 75 samples per intent, target total 2025
    - Test subset: 30 samples per intent, target total 810
- Created SFT training text using chat template roles:
    - system, user, assistant
- Created DPO preference pairs:
    - prompt: system + user
    - chosen: correct response
    - rejected: mismatched response from a different intent in the same category
- Split SFT and DPO into train/validation with a 90/10 split.
- Ran validation checks for column presence and basic sample consistency.
- Saved datasets to disk:
    - data/sft_dataset
    - data/dpo_dataset
    - data/test_dataset

### 2) Phase 1 training (LoRA SFT)

Notebook: [notebooks/LoRA_SFT_training.ipynb](notebooks/LoRA_SFT_training.ipynb)

- Loaded prepared SFT and test datasets from disk.
- Loaded tokenizer and base model (Qwen/Qwen2.5-0.5B-Instruct).
- Attached LoRA adapters with:
    - rank (r)=8
    - lora_alpha=16
    - lora_dropout=0.05
    - target modules: q_proj, k_proj, v_proj, o_proj
- Trained with SFTTrainer using key settings:
    - epochs=1
    - train batch size=2
    - gradient accumulation=8 (effective batch size 16)
    - learning rate=2e-4
    - cosine scheduler, warmup ratio=0.03
    - max sequence length=256
- Evaluated on the prepared test set using loss/perplexity.
- Performed qualitative inference checks with representative support queries.
- Saved SFT model artifacts to:
    - models/sft_model/sft_final

### 3) Phase 2 training (DPO)

Notebook: [notebooks/DPO_training.ipynb](notebooks/DPO_training.ipynb)

- Loaded DPO and test datasets from disk.
- Loaded SFT tokenizer and SFT model as DPO starting policy.
- Mounted SFT adapter as trainable policy model.
- Trained with DPOTrainer using key settings:
    - beta=0.1
    - loss_type=sigmoid
    - epochs=1
    - train batch size=1
    - gradient accumulation=8 (effective batch size 8)
    - learning rate=5e-5
    - cosine scheduler, warmup ratio=0.1
    - max length=256
- Evaluated on test data with response-only loss masking (prompt tokens ignored in labels).
- Performed deterministic inference checks after DPO.
- Saved DPO model artifacts to:
    - models/dpo_model/dpo_final

## Reproducible Run Order

1. [notebooks/data_preparation.ipynb](notebooks/data_preparation.ipynb)
2. [notebooks/LoRA_SFT_training.ipynb](notebooks/LoRA_SFT_training.ipynb)
3. [notebooks/DPO_training.ipynb](notebooks/DPO_training.ipynb)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional authentication (if pushing/pulling private assets on Hugging Face Hub):

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

## Current Artifact Layout

- Data artifacts:
    - data/sft_dataset
    - data/dpo_dataset
    - data/test_dataset
- Model artifacts:
    - models/sft_model/sft_final
    - models/dpo_model/dpo_final
    - models/merged_final_model
