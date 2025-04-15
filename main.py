import torch
from model import gpt
from utils.data import get_data, PoemDataset
from torch.utils.data import DataLoader
from utils.tokenizer import tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
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
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(
                outputs.flatten(0, 1),# [batch,seq_len,vocab_size(logits)]
                target_ids.flatten() #[batch_size, seq_len(id)]
            )
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}')

cfg = gpt.GPT_CONFIG_124M
model = gpt.MyGPT(cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=4e-4,
    weight_decay=0.1
)
num_epochs = 10
data = get_data()
poem_dataset = PoemDataset(data[:10],cfg["max_length"], stride=1,tokenizer=tokenizer)
print(f"数据长度: {len(poem_dataset)}")

poen_dataloader = dataloader = DataLoader(
        poem_dataset, 
        batch_size=4, 
        shuffle=False,
        drop_last=True, 
        num_workers=4
)
criterion = torch.nn.CrossEntropyLoss()
train_model(model, dataloader, optimizer, criterion, num_epochs)
torch.save(model.state_dict(), "model.pth")