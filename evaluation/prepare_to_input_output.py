import json


class PrepareToOutput:
    def clean_tokens(self, text):
        return text.replace('<|endoftext|>', "") + '\n<|endoftext|>'

    def write_iter(self, path, responses):
        with open(path, mode="a", encoding="utf-8") as file:
            for n, item in enumerate(responses):
                cleaned_item = self.clean_tokens(item)
                x = {f"dialog_{n}": cleaned_item}
                jsonl = json.dumps(x, ensure_ascii=False) + "\n"
                file.write(jsonl)


class PrepareToInput:
    def read_iter(self, path):
        with open(path, mode="r", encoding="utf-8") as file:
            for item in file:
                x = json.loads(item)
                rag, question = x["rag_cont"], x["question"]
                yield (rag, question)

    def prep_input_prompt(rag=None, question=None, sys_promt=True):
        s_p = ""
        if sys_promt:
            s_p = """<|system|>\nYou are JARVIS. ALWAYS address the user as "Sir". Use ONLY information from the context. Do NOT invent dates, numbers, or facts not in the context.<|end|>\n"""
        if rag:
            return f"""{s_p}<|user|>\nContext:\n{rag}\n\nQuestion:\n{question}<|end|>\n<|assistant|>\n"""
        return f"""{s_p}<|user|>\n{question}<|end|>\n<|assistant|>\n"""

if __name__ == "__main__":
    pass
