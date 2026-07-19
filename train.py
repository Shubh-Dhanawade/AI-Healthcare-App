import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os

def main():
    print("🚀 Starting Healthcare AI Fine-Tuning Pipeline...")
    
    # 1. Config
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_path = "healthcare_dataset.json"
    output_dir = "./healthcare-llama-adapter"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please make sure it is in the current directory.")
        
    print(f"📖 Loading base model: {model_id}")
    print(f"📊 Dataset path: {dataset_path}")
    
    # 2. BitsAndBytes 4-bit Quantization Config (saves GPU memory, fits in a single 16GB GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 3. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    if device_map == "cpu":
        print("⚠️ WARNING: CUDA GPU not detected. Training will run on CPU, which is extremely slow.")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map
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
    print("✅ LoRA adapters initialized.")
    
    # 5. Load Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def format_prompts(batch):
        formatted = []
        for inst, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
            text = f"<|im_start|>system\nYou are a healthcare insurance expert.<|im_end|>\n" \
                   f"<|im_start|>user\n{inst}\nContext: {inp}<|im_end|>\n" \
                   f"<|im_start|>assistant\n{out}<|im_end|>"
            formatted.append(text)
        return {"text": formatted}
        
    dataset = dataset.map(format_prompts, batched=True)
    print(f"✅ Loaded and preprocessed {len(dataset)} training examples.")
    
    # 6. Setup Trainer
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=50,  # 50 training steps for demonstration
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        save_strategy="steps",
        save_steps=25,
        dataset_text_field="text",
        max_length=2048
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args
    )
    
    # 7. Start Fine-Tuning
    print("🏋️ Starting SFT Training...")
    trainer.train()
    
    print(f"💾 Saving adapter weights to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("🎉 Fine-tuning finished successfully! Adapter weights saved.")

if __name__ == "__main__":
    main()
