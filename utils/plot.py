import matplotlib.pyplot as plt

def parse_log_file(file):
    # 存储数据
    losses = []

    # 解析日志文件
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            loss_idx = line.find("INFO - ")
            loss_len = len("INFO - ")
            if loss_idx >= 0:
                loss = float(line[loss_idx+loss_len:])
                losses.append(loss)
    return range(1,len(losses)+1), losses

epochs, train_loss = parse_log_file("./log.txt")
# 创建画布
plt.figure(figsize=(10, 6))

# 绘制训练损失曲线
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
# 添加标题和标签
plt.title('Loss Curve', fontsize=16)
plt.xlabel('step', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=12)

# 显示图形
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)