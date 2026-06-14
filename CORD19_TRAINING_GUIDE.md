# 🧬 Fine-Tuning Gemma 3 (4B) on the CORD-19 Dataset

This guide explains how to fine-tune the **Gemma 3 (4B)** model using the **CORD-19** (COVID-19 Open Research Dataset) to create a specialized medical AI assistant. It covers dataset preprocessing, fine-tuning via Hugging Face TRL (QLoRA), merging weights, converting to GGUF, and loading the model into Ollama.

---

## 📋 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Step 1: Understanding & Preprocessing CORD-19](#step-1-understanding--preprocessing-cord-19)
3. [Step 2: Gemma 3 QLoRA Fine-Tuning Script](#step-2-gemma-3-qlora-fine-tuning-script)
4. [Step 3: Merging LoRA Adapters](#step-3-merging-lora-adapters)
5. [Step 4: Converting to GGUF using llama.cpp](#step-4-converting-to-gguf-using-llamacpp)
6. [Step 5: Importing and Running in Ollama](#step-5-importing-and-running-in-ollama)

---

## 1. Prerequisites & System Requirements
Fine-tuning **Gemma 3 (4B)** using 4-bit quantization (QLoRA) requires:
* **GPU**: NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3090, RTX 4090, A10G, or a free Google Colab T4/L4 instance).
* **RAM**: 16GB+ System RAM.
* **Storage**: 50GB+ free space (to download the model, dataset, and save intermediate checkpoints).
* **Python**: Version 3.10 or newer.

Install the required library stack:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets trl peft bitsandbytes accelerate huggingface_hub
```

Before training, authenticate with Hugging Face to access Gemma models:
```bash
huggingface-cli login
```

---

## Step 1: Understanding & Preprocessing CORD-19
The full CORD-19 dataset is massive (containing over 1,000,000 papers). For instruction fine-tuning, we want to convert paper titles and abstracts into instruction-following Q&A or summarization pairs.

Below is a Python script (`preprocess_cord19.py`) that uses the Hugging Face `datasets` library to load a subset of CORD-19, clean the abstracts, and generate structured instruction-response JSON files:

```python
# preprocess_cord19.py
import json
from datasets import load_dataset

print("Downloading/Loading CORD-19 subset...")
# We use the metadata subset or load a small slice to prevent out-of-memory errors
try:
    # Streaming allows us to process without downloading the full 100GB+ dataset
    dataset = load_dataset("allenai/cord19", "metadata", split="train", streaming=True)
except Exception as e:
    print(f"Error loading cord19: {e}")
    print("Falling back to downloading a smaller medical dataset or mock subset...")
    dataset = None

formatted_data = []
count = 0
limit = 5000  # Number of instruction pairs to generate

if dataset:
    for row in dataset:
        title = row.get("title", "").strip()
        abstract = row.get("abstract", "").strip()
        
        # We only want entries that have both a valid title and abstract
        if title and abstract and len(abstract) > 100:
            # We construct a task: Summarize or Extract medical details from the abstract
            entry = {
                "instruction": f"Analyze the following scientific medical literature abstract and summarize the key findings, including any methodologies mentioned.",
                "input": f"Title: {title}\nAbstract: {abstract}",
                "output": f"This scientific paper investigated '{title}'. Key findings and summary:\n{abstract[:500]}..." # In practice, write summary extraction or clean abstract
            }
            formatted_data.append(entry)
            count += 1
            if count >= limit:
                break
else:
    # Mock fallback for demonstration/local testing
    formatted_data = [
        {
            "instruction": "Analyze the following scientific medical literature abstract and summarize the key findings, including any methodologies mentioned.",
            "input": "Title: Clinical features of patients infected with 2019 novel coronavirus in Wuhan, China\nAbstract: A recent cluster of pneumonia cases in Wuhan, China, was caused by a novel coronavirus (2019-nCoV). We reports clinical features of 41 laboratory-confirmed patients. Most of the infected patients were men, and less than half had underlying diseases. Common symptoms at onset of illness were fever, cough, and myalgia or fatigue.",
            "output": "This study details the clinical features of 41 laboratory-confirmed 2019-nCoV patients in Wuhan. The cohort was predominantly male, with less than half showing underlying health conditions. Key clinical symptoms at onset included fever, cough, and myalgia/fatigue."
        }
    ]

# Save to disk
output_file = "cord19_instructions.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"Dataset preprocessed! Saved {len(formatted_data)} records to {output_file}")
```

---

## Step 2: Gemma 3 QLoRA Fine-Tuning Script
**Gemma 3 (4B)** uses specific architectural targets and system formatting. To train the model efficiently, we target the attention projection layers (`q_proj`, `v_proj`, etc.) and load it in 4-bit using `BitsAndBytesConfig`.

Save this script as `train_gemma.py`:

```python
# train_gemma.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Config
MODEL_ID = "google/gemma-3-4b-it"  # Use the instruction-tuned version of Gemma 3
DATASET_PATH = "cord19_instructions.json"
OUTPUT_DIR = "./gemma3-cord19-adapter"

print(f"Loading tokenizer and model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 2. 4-bit Quantization Config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for 8-bit/4-bit training (gradients checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA (PEFT)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. Load Preprocessed Dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 6. Apply Gemma Chat Template
def format_prompts(batch):
    formatted = []
    for inst, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        # Format matching the standard Gemma 3 chat template
        messages = [
            {"role": "system", "content": "You are a medical research AI assistant specializing in scientific literature."},
            {"role": "user", "content": f"{inst}\n\nContext:\n{inp}"},
            {"role": "assistant", "content": out}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append(text)
    return {"text": formatted}

dataset = dataset.map(format_prompts, batched=True)

# 7. Setup Training Arguments using SFTConfig
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=150,  # Adjust based on dataset size
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    save_strategy="steps",
    save_steps=50,
    gradient_checkpointing=True,
    report_to="none",  # Disable wandb reporting unless configured
    dataset_text_field="text",
    max_length=2048
)

# 8. Setup SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

print("Starting training...")
trainer.train()

# 9. Save Trained Adapter
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training completed successfully! Adapter saved to {OUTPUT_DIR}")
```

---

## Step 3: Merging LoRA Adapters
Ollama runs stand-alone models. We must merge our trained LoRA adapter weights back into the original 16-bit unquantized Gemma 3 weights.

Save this script as `merge_gemma.py`:

```python
# merge_gemma.py
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Clear GPU VRAM from training phase first to avoid OOM
try:
    del trainer
    del model
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()

BASE_MODEL = "google/gemma-3-4b-it"
ADAPTER_DIR = "./gemma3-cord19-adapter"
MERGED_DIR = "./gemma3-cord19-merged"

print("Loading base model in half precision on GPU...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda"  # Load on GPU VRAM to avoid exceeding Colab CPU RAM limit
)

print("Loading adapter and merging weights...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged_model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}...")
# Move to CPU before saving to avoid GPU VRAM overhead
merged_model = merged_model.to("cpu")
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print("Merge complete! Ready for GGUF conversion.")
```

Run the merge:
```bash
python merge_gemma.py
```

---

## Step 4: Converting to GGUF using llama.cpp
To convert the Hugging Face merged folder into a GGUF model that Ollama can parse, download and compile `llama.cpp`:

1. **Clone llama.cpp Repository**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Convert merged model to GGUF**:
   For **Gemma 3 (4B)**, run the conversion script (outputting a 4-bit quantized version, `q4_k_m`, which maintains high accuracy with minimal memory usage):
   ```bash
   python convert_hf_to_gguf.py ../gemma3-cord19-merged --outfile ../gemma3-cord19-q4.gguf --outtype q4_k_m
   ```

---

## Step 5: Importing and Running in Ollama
Now you can create a custom Ollama model using the GGUF file.

1. Create a `Modelfile` inside the root folder:
   ```dockerfile
   # Modelfile
   FROM ./gemma3-cord19-q4.gguf

   # Model Parameters
   PARAMETER temperature 0.2
   PARAMETER top_p 0.9
   PARAMETER num_predict 2048

   # System Prompt
   SYSTEM """You are a medical research AI assistant. You have been fine-tuned on the CORD-19 scientific literature database to provide accurate, context-aware information on virology, clinical symptoms, and medical publications. If you do not know the answer, state that you do not know."""
   ```

2. Import the model to Ollama:
   ```bash
   ollama create gemma3-cord19 -f ./Modelfile
   ```

3. Run the model in your terminal to test:
   ```bash
   ollama run gemma3-cord19 "What are the common symptoms of novel coronavirus based on Wuhan patient cohorts?"
   ```

4. Configure the model in your AI Healthcare app by updating `backend/.env`:
   ```env
   OLLAMA_MODEL=gemma3-cord19
   ```
   Then restart your backend.
