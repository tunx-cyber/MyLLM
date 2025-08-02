# SFT
| 模型阶段      | 训练目标               | `labels` 构造方式                             | 是否使用 shift |
| --------- | ------------------ | ----------------------------------------- | ---------- |
| 预训练 GPT   | 预测下一个 token        | `labels = input_ids[1:]`                  | ✅ 是        |
| 指令微调（SFT） | 模仿助手回复（completion） | `labels = input_ids.copy()` 并屏蔽 prompt 部分 | ❌ 否        |

# distribute and parralle
pytorch官方tutorial有

# RL
## PPO
## GRPO