import torch
from model.gpt import MyGPT, GPT_CONFIG_124M
from utils.tokenizer import tokenizer
import tiktoken
from utils.generate import generate_text_simple, text_to_token_ids, token_ids_to_text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = GPT_CONFIG_124M
model = MyGPT(cfg).to(device)

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer).to(device),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["max_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

##python3 -m test.test_data 