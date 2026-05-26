# 🧠 AI-Healthcare-App Model Fine-Tuning & Training Guide

This guide details how to fine-tune a lightweight LLM like **Llama 3.2 (3B)** or **Llama 3 (8B)** on your custom healthcare insurance policies and documents, and load it into your local Ollama instance.

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Step 1: Dataset Preparation](#step-1-dataset-preparation)
3. [Step 2: Fine-Tuning with Hugging Face & QLoRA](#step-2-fine-tuning-with-hugging-face--qlora)
4. [Step 3: Merging & Exporting to GGUF](#step-3-merging--exporting-to-gguf)
5. [Step 4: Importing and Running in Ollama](#step-4-importing-and-running-in-ollama)

---

## 1. Prerequisites
To fine-tune models, you need:
- A machine with a CUDA-capable GPU (NVIDIA RTX 3090/4090, A100, or a free Google Colab T4/L4 instance).
- Python 3.10+ with PyTorch installed.
- Hugging Face account and CLI token (`huggingface-cli login`).

---

## Step 1: Dataset Preparation
Fine-tuning requires a dataset of input-output pairs reflecting how you want the model to analyze documents or answer questions. Create a file named `healthcare_dataset.json` containing conversational entries:

```json
[
  {
    "instruction": "Summarize the key coverages and exclusions for the following insurance policy text.",
    "input": "Policy Name: Care Premium. Coverage: In-patient hospitalization up to 10 Lakhs. ICU charges are covered up to the sum insured. Exclusion: Room rent capped at 1% of Sum Insured per day. Pre-existing diseases covered after a 48-month waiting period.",
    "output": "{\n  \"summary_text\": \"The Care Premium policy provides standard in-patient hospitalization up to 10 Lakhs with full ICU coverage. It features room rent restrictions and a 48-month pre-existing disease waiting period.\",\n  \"coverage_summary\": \"• In-patient hospitalization up to 10 Lakhs\\n• Full ICU coverage up to Sum Insured\",\n  \"exclusions_summary\": \"• Room rent limited to 1% of Sum Insured per day\\n• Pre-existing diseases excluded for first 48 months\"\n}"
  },
  {
    "instruction": "Extract the premium details and co-payment details for the following text.",
    "input": "The annual premium is ₹18,500 including 18% GST. A compulsory co-payment of 20% is applicable for claims filed by senior citizens over 60 years old.",
    "output": "Premium is ₹18,500/year (with 18% GST). Co-payment is 20% for senior citizens over 60."
  }
]
```

---

## Step 2: Fine-Tuning with Hugging Face & QLoRA
We recommend **Unsloth** or Hugging Face **PEFT (QLoRA)** as they require minimal GPU memory (fits on a single 16GB GPU).

Here is a Python script (`train.py`) to fine-tune Llama 3.2 using Hugging Face **TRL** and **PEFT**:

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Config
model_id = "meta-llama/Llama-3.2-3B-Instruct"
dataset_path = "healthcare_dataset.json"
output_dir = "./healthcare-llama-adapter"

# 2. BitsAndBytes 4-bit Quantization Config (saves GPU memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Configure LoRA (PEFT Adapter)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 5. Load Dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_prompts(batch):
    formatted = []
    for inst, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        # Standard chat template formatting
        text = f"<|im_start|>system\nYou are a healthcare insurance expert.<|im_end|>\n" \
               f"<|im_start|>user\n{inst}\nContext: {inp}<|im_end|>\n" \
               f"<|im_start|>assistant\n{out}<|im_end|>"
        formatted.append(text)
    return {"text": formatted}

dataset = dataset.map(format_prompts, batched=True)

# 6. Setup Trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,  # Adjust based on dataset size (epochs)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    save_strategy="steps",
    save_steps=50
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args
)

# 7. Start Fine-Tuning
trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Training completed! Adapter saved.")
```

---

## Step 3: Merging & Exporting to GGUF
Ollama requires models in **GGUF** format. First, merge the LoRA weights back into the original base model:

```python
# merge.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_dir = "./healthcare-llama-adapter"
save_merged_dir = "./healthcare-llama-merged"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Load adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = model.merge_and_unload()

# Save merged HF model
merged_model.save_pretrained(save_merged_dir)
tokenizer.save_pretrained(save_merged_dir)
print("Model merged successfully!")
```

### Convert to GGUF (via llama.cpp)
Clone `llama.cpp` and build/convert the merged Hugging Face model:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Run conversion script to output standard Q4_K_M (4-bit quantized) GGUF
python convert_hf_to_gguf.py ../healthcare-llama-merged --outfile ../healthcare-llama-q4.gguf --outtype q4_k_m
```

---

## Step 4: Importing and Running in Ollama
Once you have the `healthcare-llama-q4.gguf` file:

1. Create a `Modelfile` in the directory:
   ```dockerfile
   FROM ./healthcare-llama-q4.gguf

   # Define default parameters
   PARAMETER temperature 0.2
   PARAMETER num_predict 2048

   # Define system message
   SYSTEM """You are a helpful healthcare document assistant trained to analyze health insurance policies and medical reports. Always prioritize accuracy and facts found in the document."""
   ```

2. Register the model with Ollama:
   ```bash
   ollama create healthcare-llama -f ./Modelfile
   ```

3. Test your new model locally:
   ```bash
   ollama run healthcare-llama "What is the pre-existing disease waiting period for Care Premium?"
   ```

4. Update your backend environment variables (`backend/.env`):
   ```env
   OLLAMA_MODEL=healthcare-llama
   ```
   Restart the backend server, and the system will run on your custom fine-tuned model!
