Evaluation was performed using conversational LLM-as-a-Judge approach.  
Each metric was evaluated **independently using separate prompts** to reduce bias.

Each response was evaluated by external LLM judges (DeepSeek, Kimi 2.5) using the following criteria:

- Style: does the response sound like JARVIS? (0-10)
- Context Utilization: does the model use provided context? (0-10)
- Reasoning: are calculations and logic correct? (0-10)
- Hallucination: does the model fabricate facts? (binary + rate)

## Example Prompt (Reasoning)

```text
Evaluate the reasoning (calculations and logic) correctness from 0 to 10.

- 0 = completely incorrect
- 5 = partially correct
- 10 = fully correct

Answer:
{answer}
