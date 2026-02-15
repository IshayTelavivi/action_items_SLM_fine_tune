The purpose of this project was to extract action items from a call transcription automatically, by training a SLM to do it.

The project had 2 parts, the first one was to generate the labels (the action items) with a LLM (Claude-3.5) using LangChain package, and then validating with GPT-4o.
The second part was the SLM fine-tune.

The action items format was: "-Action: action. When: due time. By who: action owner".
Examples:
```
- Action: Follow-up call to discuss after work hours. When: 5:30 PM (today). By Who: Agent
- Action: Follow-up call to Ms Anoki. When: Today at 13:30. By Who: Agent
- Action: Send an email to find a Russian-speaking colleague to assist the client. When: N/A. By Who: Agent
- Action: Client to open a trade on oil. When: Monday. By Who: Client
- Action: Client to make withdrawal of 300 to Apple Pay. When: N/A. By Who: Client
- Action: Client to monitor gold and silver prices. When: tomorrow at 13:30 British time. By Who: Client
```

If no action items were argeed, then the model needs to return: `No follow-up action items agreed upon.`

The SLM that I chose was `mistralai/Mistral-7B-Instruct-v0.1` as this was small enough to run locally and large enough for good performance. 


General notes:
- I used LoRA as the framework for training (peft package), using all modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- The number of total samples I used was 1500 (for train/val/test), and it was enough to get good results
- I ran the training on Google Colab, with a NVIDIA A100 GPU


Potential future improvements:
- Enriching the user prompt
- Modifying the chat template (specifically the system prompt in it)
- Add more samples, especially samples with action items
- Tuning more hyperparameters





