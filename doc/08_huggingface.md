# attention mask
Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).
一般是因为有padding token，所以padding对应的就是0，其余都是1

# loss mask
通常用于掩盖住prompt
| 对象                  | 作用点             | 控制内容         | 举例                                   |
| ------------------- | --------------- | ------------ | ------------------------------------ |
| **attention\_mask** | 前向传播（attention） | token 能不能被看到 | `[1, 1, 1, 1, 0, 0]` 表示最后2个是 padding |
| **loss mask**       | 反向传播（loss）      | token 是否参与训练 | `labels[i] = -100` 表示跳过这个位置          |



直接看文档 transformers库的使用： https://huggingface.co/docs/transformers/index

Tokenizer库： https://huggingface.co/docs/tokenizers/index

accelerate使用： https://huggingface.co/docs/accelerate/basic_tutorials/migration



