import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Crop the context to fit the model’s maximum context size
        idx_cond = idx if idx.size(1) <= context_size else idx[:,-context_size:
        ]
        # Compute predictions
        logits = model(idx_cond)
        # Select the next token based on the highest probability prediction
        probs = torch.softmax(logits[:,-1, :], dim=-1)
        next_idx = torch.argmax(probs, dim=-1).unsqueeze(1)
        # Append the next token to the input sequence
        idx = torch.cat((idx, next_idx), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor # 添加批次维度
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 移除批次维度
    return tokenizer.decode(flat.tolist())
