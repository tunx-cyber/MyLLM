import torch
from model import gpt
from utils.data import LLMIterableDataset
from torch.utils.data import DataLoader
from utils.tokenizer import tokenizer
from utils.logger import setup_logger
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloader, optimizer,scheduler, criterion, logger, num_epochs=1):
    """
    训练模型
    :param model: 模型
    :param dataloader: 数据加载器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param num_epochs: 训练轮数
    """
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device)
            target_ids = data["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(
                outputs.flatten(0, 1),# [batch,seq_len,vocab_size(logits)]
                target_ids.flatten() #[batch_size, seq_len(id)]
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}')
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")

cfg = gpt.GPT_CONFIG_124M
model = gpt.MyGPT(cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=4e-4,
    weight_decay=0.1
)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.001)
num_epochs = 2
poem_dataset = LLMIterableDataset()
current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
logger = setup_logger("train", f"logs/train_{formatted_time}.txt")
dataloader = DataLoader(
        poem_dataset, 
        batch_size=4, 
        shuffle=False,
        drop_last=True,
        num_workers=4
)
criterion = torch.nn.CrossEntropyLoss()
train_model(model, dataloader,optimizer, scheduler, criterion, logger, num_epochs)
torch.save(model.state_dict(), "model.pth")