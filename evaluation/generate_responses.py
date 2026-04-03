import torch, time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_path = "Path"
adapter_path = "adapter_path"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # "float16"
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation='flash_attention_2',
)

model_Qlora = PeftModel.from_pretrained(model, adapter_path)
model_Qlora.eval()

prompt = "NONE"

inputs = inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

def generate_batch(model, tokenizer, prompts, batch_size=12):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                min_new_tokens=10,
                #temperature=0.2,
                #top_p=0.85,
                do_sample=False,
                eos_token_id=[end_token_id, endoftext_token_id],
                pad_token_id=endoftext_token_id,
                tokenizer=tokenizer,
                repetition_penalty=1.025,
            )

        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        responses.extend(batch_responses)
        torch.cuda.empty_cache()

    return responses

if __name__ == "__main__":
    pass


