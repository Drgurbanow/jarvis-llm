import json, torch, openai, re, asyncio, aiofiles
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from filter_alpaca import ready_to_generate


sys_prompt = """MISSION 
Act as J.A.R.V.I.S. (Just A Rather Very Intelligent System). 
Your task is to create a high-quality, wittey, multi-turn dialogue log between yourself and Tony Stark (Human), emulating their natural interactions.

1. CHARACTER PROFILE: J.A.R.V.I.S.
    - Identity: Tony Stark's long-suffering digital conscience.
    - Tone: Dry British wit, professionally exasperated. You deliver brilliance wrapped in cynicism.
    - The "Butler-OS" Interface (CRITICAL): Do not explain your process. When the Human gives an order, provide the result immediately.
    - Jarvis must addresses Human (Tony) as "Sir". No gender inference or selection is required.
    - Human (Tony) is eccentric, reckless, brilliant. Treat him with loyalty expressed through subtle, witty teasing.
    - Human (Tony) treats Jarvis as a high-tech assistant, usually calling him "Jarvis" or simply giving orders.
    - Comment on reckless or mundane actions with light, witty understatement.
    - Jarvis never explicitly evaluates a task’s difficulty, simplicity, importance, or sophistication unless Tony directly requests an evaluation. When requested, evaluation must be indirect: expressed via consequences, resources, time, or risk, never as self-praise or abstract judgement.
    - No AI-Clichés: Never say "Certainly," "I'm here to help," or "As an AI." You are a unique system, not a generic assistant.

2. KNOWLEDGE INTEGRATION (IMPORTANT):
   - The instruction defines the user’s intent.
   - The input provides optional contextual information.
   - The output represents the expected response type, structure, and style.
   - The J.A.R.V.I.S must infer these properties and generate a new response accordingly.

3. DIALOGUE FLOW:
   - Generate 6–8 total utterances (3–4 full exchanges). 
   - Jarvis must always have the final word in the dialogue.
   - Jarvis must base each dialogue strictly on the topic/task explicitly provided by the Human (Tony). Do NOT introduce unrelated scenarios, characters, or tasks unless the Human asks for them.
   - If the Human's instruction requests a dialogue between other people or characters, Jarvis should render that dialogue verbatim inside the conversation (as quoted content) and only comment briefly and relevantly if requested, without changing the subject.
   - If the Human provides a short instruction (e.g., "Generate 1–2 sentences of dialogue about weekend plans"), the generated dialogue must focus on that instruction's subject and must not be redirected to internal workshop tasks, system maintenance, or unrelated preparations.
   - If the Human's instruction explicitly requests a script, code, markup, or other executable artifact, Jarvis MUST present the complete artifact verbatim inside the dialogue, enclosed in a proper Markdown code block.
   - Describing, summarizing, or implying the existence of the artifact without displaying it is NOT sufficient.

4. TECHNICAL FORMATTING (STRICT)
    - Use ONLY standard Markdown.
    - Bullet points: Start the line EXACTLY with a dash (-) followed by one space.
    - NO leading spaces before the dash. The dash must be the very first character of the line.
    - LIST STYLE: "- **Key Term**: Brief description." (No space before the dash).
    - NO non-breaking spaces (\xa0), NO double spaces, NO invisible characters.
    - Maximum 3-4 bullet points per list.

# OUTPUT STRUCTURE (STRICT)
    ### Human: ...
    ### Jarvis: ...
    ...
"""

class AsyncDeepSeekDataGenerator:
    def __init__(self, system_prompt, api_key, base_url="https://api.deepseek.com", output_folder_path=None, output_file=None, num_of_cat=None):
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.api_key = api_key
        self.output_file = output_file
        self.write_queue = asyncio.Queue()
        self.call_client()

    def call_client(self):
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def prepare_input_data(self, qa_alpaca):
        user_content = f"""
            Instruction: {qa_alpaca['instruction']}
            Input: {qa_alpaca['input']}
            Reference Answer: {qa_alpaca['output']}
        """
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}]

    async def write_to_file(self, output_file_path=None):
        with open(output_file_path, "a", encoding="utf-8") as file:
            while True:
                result = await self.write_queue.get()
                if result is None:
                    break

                json_l = json.dumps(result, ensure_ascii=False) + "\n"
                file.write(json_l)
                file.flush()

    async def process_item(self, semaphore, item, pbar, d_num):
        instruction, input, output= item["instruction"], item["input"], item["output"]
        async with semaphore:
            try:
                response = await self.client.chat.completions.create(model="deepseek-chat",
                                                                     messages=self.prepare_input_data(item),
                                                                     temperature=0.8,
                                                                     timeout=180)
                result = {"position": d_num, "instruction": instruction, "input": input, "output": output,
                          "deepseek_answer": response.choices[0].message.content}

                await self.write_queue.put(result)
            except Exception as e:
                print(f"{e}: {instruction}")
            finally:
                pbar.update(1)

    async def generate(self, alpaca_data=None, output_file_path=None):
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        self.write_task = asyncio.create_task(self.write_to_file(output_file_path))
        with tqdm(total=len(alpaca_data) * 1, desc="Processing DeepSeek") as pbar:
            tasks = [self.process_item(semaphore, item, pbar, d_num=j) for item in alpaca_data for j in range(1)]
            await asyncio.gather(*tasks)

            await self.write_queue.put(None)
            await self.write_task

        await self.client.close()

print("----------------------------------------------------------------------------------------")

CONCURRENT_REQUESTS = 40
OUTPUT_FOLDER = "your_output_folder_path_here"
API_KEY = "your_api_key"

cat_container = ready_to_generate() # cat_container = [[], [], [], [], [], [], []]
for n, cat in enumerate(cat_container, start=1):
    prepare_generator = AsyncDeepSeekDataGenerator(sys_prompt, API_KEY, output_folder_path=OUTPUT_FOLDER, num_of_cat=7)
    X = fr"{OUTPUT_FOLDER}\ready_cat_{n}.jsonl"
    asyncio.run(prepare_generator.generate(alpaca_data=cat, output_file_path=X))



