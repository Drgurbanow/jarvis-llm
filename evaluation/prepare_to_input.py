def prep_input_prompt(rag=None, question=None, sys_promt=True):
    s_p = ""
    if sys_promt:
        s_p = """<|system|>\nYou are JARVIS. ALWAYS address the user as "Sir". Use ONLY information from the context. Do NOT invent dates, numbers, or facts not in the context.<|end|>\n"""
    if rag:
        return f"""{s_p}<|user|>\nContext:\n{rag}\n\nQuestion:\n{question}<|end|>\n<|assistant|>\n"""
    return f"""{s_p}<|user|>\n{question}<|end|>\n<|assistant|>\n"""

def write(datas, path=r"C:\Users\user\Desktop\test_cases.jsonl"):
    with open(path, mode="a", encoding="utf-8") as file:
        for item in datas:
            rag, question = item
            d = {"rag_cont": rag, "question": question}
            jsonl_item = json.dumps(d, ensure_ascii=False) + "\n"
            file.write(jsonl_item)
            file.flush()
