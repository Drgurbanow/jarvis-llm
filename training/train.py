# Usage:
# 1. Set DATA_PATH and OUTPUT_DIR at the top of the file
# 2. python train.py

import torch, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer

MODEL_NAME = "microsoft/phi-4-mini-instruct"
DATA_PATH = "your_path_here/chat_temp_all_in_one.jsonl"
OUTPUT_DIR = "your_path_here/checkpoints"

if torch.cuda.is_available():
    print(True, torch.cuda.get_device_name(0))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # {"": "cuda:0"}
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    #trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #,"gate_proj", "up_proj", "down_proj"
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id


def load(path):
    result = []
    with open(path, mode="r", encoding="utf-8") as file:
        for n, dialog in enumerate(file):
            jsonl = json.loads(dialog)
            text = jsonl[f"dialog_{n}"]
            result.append({"text": f"{text}"})
    return result


dialog_container = load(
    path=DATA_PATH)
data = Dataset.from_list(dialog_container)
data = data.shuffle(seed=69)
data = data.train_test_split(test_size=0.05)
train_data = data['train']
eval_data = data["test"]

def tokenizing_function(data_batch):
    text = data_batch["text"]
    return tokenizer(text,
                     padding=False,  # "longest", #dataCollator will do it
                     truncation=True,
                     max_length=1024,
                     )

tokenized_train_data = train_data.map(tokenizing_function, batched=True, remove_columns=train_data.column_names)
tokenized_eval_data = eval_data.map(tokenizing_function, batched=True, remove_columns=eval_data.column_names)

response_template = "<|assistant|>"
response_template_ids = tokenizer(response_template, add_special_tokens=False)["input_ids"]
print(tokenizer.convert_ids_to_tokens(response_template_ids))
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False,
    ignore_index=-100,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=6,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    warmup_steps=100,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="steps",
    save_steps=75,
    save_total_limit=1,
    report_to='none',
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    label_names=["labels"],
    eval_strategy="steps",
    eval_steps=75,
    load_best_model_at_end=True,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_eval_data,
    data_collator=data_collator,
    # tokenizer=tokenizer  # processing_class=tokenizer
)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    trainer.train()
