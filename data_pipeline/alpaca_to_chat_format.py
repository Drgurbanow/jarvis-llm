import json, torch, re
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

paths = [
    os.path.join(BASE_DIR, "ready_cat_1.jsonl"),
    os.path.join(BASE_DIR, "ready_cat_2.jsonl"),
    os.path.join(BASE_DIR, "ready_cat_3.jsonl"),
    os.path.join(BASE_DIR, "ready_cat_4.jsonl"),
    os.path.join(BASE_DIR, "ready_cat_5.jsonl"),
    os.path.join(BASE_DIR, "ready_cat_6.jsonl"),
    os.path.join(BASE_DIR, "ready_cat_7.jsonl"),
]

output_path = os.path.join(BASE_DIR, "all_dialogues.jsonl")


def write_to_jsonl(output_path, dialog, indx):
    with open(output_path, mode="a", encoding="utf-8") as file:
        packed_dialog = {f"dialog_{indx}": dialog}
        x = json.dumps(packed_dialog) + "\n"
        file.write(x)
        file.flush()


def read_all(paths: List[str], key: str = "deepseek_answer"):
    # dialogs_list = list()
    for i in paths:
        with open(i, mode="r", encoding="utf-8") as file:
            for obj in file:
                try:
                    json_l = json.loads(obj)
                    deepseek_answer = json_l[key]
                    yield deepseek_answer
                except Exception:
                    continue


# -----------------------------------------------------------------------------------------------
def split_and_filtr(dialog):
    filterd = list()
    splited = re.findall(r"(### Human:.*?### Jarvis:.*?(?=\s*### Human:|$))", dialog, flags=re.DOTALL)
    for part in splited:
        filterd.append(part)
    return filterd


def extract_all_after_marker(part):
    patern3 = r"### (Human|Jarvis):([\s\S]*?)(?=\n### |$)"
    match = re.finditer(patern3, part, flags=re.DOTALL)
    if match:
        return match
    return None


def prepare_extracted_replica(dialog):
    for part in split_and_filtr(dialog):
        for replica in extract_all_after_marker(part):
            role = replica.group(1).strip()
            content = replica.group(2).strip()
            yield (role, content)


def prepare_message(dialog):
    splited_to_replicas = iter(prepare_extracted_replica(dialog))
    message = list()
    for role, content in splited_to_replicas:
        if "Human" in role:
            message.append({"role": "user", "content": content})
        elif "Jarvis" in role:
            message.append({"role": "assistant", "content": content})
    return message


def apply_chat_temp(dialog):
    message_list = prepare_message(dialog)
    x = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=False)
    return x


if __name__ == "__main__":
    read_list = read_all(paths)
    for n, dialog in enumerate(read_list):
        result = apply_chat_temp(dialog)
        write_to_jsonl(output_path, result, n)
