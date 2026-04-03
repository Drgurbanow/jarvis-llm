import torch, time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prepare_to_input import PrepareToInput, PrepareToOutput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "microsoft/phi-3.5-mini-instruct"
ADAPTER_PATH  = os.path.join(BASE_DIR, "adapter")
TEST_CASES_PATH = os.path.join(BASE_DIR, "test_cases.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs.jsonl")
BATCH_SIZE = 12  # optimized for 8GB VRAM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH , trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # "float16"
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH ,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation='flash_attention_2',
)

model_Qlora = PeftModel.from_pretrained(model, ADAPTER_PATH)
model_Qlora.eval()


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
                repetition_penalty=1.02,
            )

        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        responses.extend(batch_responses)

    return responses

def main():
    prepare_to_input = PrepareToInput()
    prepare_to_output = PrepareToOutput()
    #---------------------------------------------------------
    iterator_read = prepare_to_input.read_iter(TEST_CASES_PATH)
    prompts = [prepare_to_input.prep_input_prompt(rag, question) for rag, question in iterator_read]
    responses = generate_batch(model, tokenizer, prompts, batch_size=BATCH_SIZE)
    # ---------------------------------------------------------
    prepare_to_output.write_iter(OUTPUT_PATH, responses)

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time:", time.time() - start)


