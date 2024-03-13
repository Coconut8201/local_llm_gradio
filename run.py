import time

import gradio as gr
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "YC-Chen/Breeze-7B-Instruct-v1_0-GGUF",
    model_file="breeze-7b-instruct-v1_0-q6_k.gguf",
    model_type="mistral",
    context_length=8000,
    gpu_layers=50)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")

gen_kwargs = dict(
    max_new_tokens=1024,
    repetition_penalty=1.1,
    stop=["[INST]"],
    temperature=0.0,
    top_p=0.0,
    top_k=1,
)




def askLLm(message):
    print(f"message input: {message}")
    chat = [
            {"role": "system", "content": "You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan."},
            {"role": "user", "content": f"{message}"}
          ]
    for text in llm(tokenizer.apply_chat_template(chat, tokenize=False), stream=True, **gen_kwargs):
        print("我的回答是: ",text, end="", flush=True)
        # yield text


demo = gr.ChatInterface(askLLm).queue()
demo.launch(server_name="163.13.201.153", server_port=2761)