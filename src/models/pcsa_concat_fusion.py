import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from multiprocessing import Manager

def create_item(post_content='', cmt_lv1_content='', cmt_lv2_content='', cmt_lv3_content='', label=None):
    return {
        'post_content': post_content,
        'cmt_content_lv1': cmt_lv1_content,
        'cmt_content_lv2': cmt_lv2_content,
        'cmt_content_lv3': cmt_lv3_content,
        'label': label
    }

def create_dataset(data):
    dataset = []
    for post in data:
        for cmt1 in post['comments']:
            item = create_item(post['post_content'], cmt_lv1_content=cmt1['cmt_content'], label=cmt1['label'])
            dataset.append(item)
            for cmt2 in cmt1['comments']:
                item = create_item(post['post_content'], cmt_lv1_content=cmt1['cmt_content'],
                                   cmt_lv2_content=cmt2['cmt_content'], label=cmt2['label'])
                dataset.append(item)
                for cmt3 in cmt2['comments']:
                    item = create_item(post['post_content'], cmt_lv1_content=cmt1['cmt_content'],
                                       cmt_lv2_content=cmt2['cmt_content'], cmt_lv3_content=cmt3['cmt_content'],
                                       label=cmt3['label'])
                    dataset.append(item)
    return dataset

class PCSA_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        p_token = self.tokenizer(self.dataset[idx]['post_content'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        c1_token = self.tokenizer(self.dataset[idx]['cmt_content_lv1'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        c2_token = self.tokenizer(self.dataset[idx]['cmt_content_lv2'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        c3_token = self.tokenizer(self.dataset[idx]['cmt_content_lv3'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        p_token = {k: v.squeeze(0) for k, v in p_token.items()}
        c1_token = {k: v.squeeze(0) for k, v in c1_token.items()}
        c2_token = {k: v.squeeze(0) for k, v in c2_token.items()}
        c3_token = {k: v.squeeze(0) for k, v in c3_token.items()}
        label = self.dataset[idx]['label']
        return p_token, c1_token, c2_token, c3_token, label

def batch_to_device(batch, device):
    p_token, c1_token, c2_token, c3_token, label = batch
    p_token = {k: v.to(device, non_blocking=True) for k, v in p_token.items()}
    c1_token = {k: v.to(device, non_blocking=True) for k, v in c1_token.items()}
    c2_token = {k: v.to(device, non_blocking=True) for k, v in c2_token.items()}
    c3_token = {k: v.to(device, non_blocking=True) for k, v in c3_token.items()}
    return p_token, c1_token, c2_token, c3_token, label.to(device, non_blocking=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class PCSA_Concat_Fusion_Model(nn.Module):
    def __init__(self, Phobert_path, num_label, dropout_prob=0.0):
        super(PCSA_Concat_Fusion_Model, self).__init__()
        config = AutoConfig.from_pretrained(Phobert_path)
        config.hidden_dropout_prob = dropout_prob
        self.Phobert = AutoModel.from_pretrained(Phobert_path, config=config)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(768 * 4, num_label)

    def forward(self, p_token, c1_token, c2_token, c3_token):
        p_embs = self.Phobert(**p_token).pooler_output
        c1_embs = self.Phobert(**c1_token).pooler_output
        c2_embs = self.Phobert(**c2_token).pooler_output
        c3_embs = self.Phobert(**c3_token).pooler_output
        fusion_embs = torch.cat((p_embs, c1_embs, c2_embs, c3_embs), dim=1)
        fusion_embs = self.dropout(fusion_embs)
        return self.dense(fusion_embs)

def get_train_loader(dataset, batch_size, world_size, rank):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)

def get_test_loader(dataset, batch_size, world_size, rank):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)

def evaluate(model, device, data_loader):
    predictions = []
    labels = []
    model.eval()
    for batch in data_loader:
        p_token, c1_token, c2_token, c3_token, label = batch_to_device(batch, device)
        labels.extend(label.cpu().numpy())
        with torch.no_grad():
            logits = model(p_token, c1_token, c2_token, c3_token)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    f1_classes = f1_score(labels, predictions, average=None)
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return f1_classes, weighted_f1, accuracy

def train_ddp(rank, world_size, train_dataset, test_dataset, Phobert_path, num_label, dropout_prob, lr, epochs, batch_size, return_dict):
    print(f"[Rank {rank}] Bắt đầu huấn luyện trên GPU {rank}")
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    train_loader = get_train_loader(train_dataset, batch_size, world_size, rank)
    test_loader = get_test_loader(test_dataset, batch_size, world_size, rank)

    model = PCSA_Concat_Fusion_Model(Phobert_path, num_label, dropout_prob=dropout_prob).to(device)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    loss_values = []
    f1_train_values = []
    f1_test_values = []

    for epoch in range(epochs):
        ddp_model.train()
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            p_token, c1_token, c2_token, c3_token, label = batch_to_device(batch, device)
            output = ddp_model(p_token, c1_token, c2_token, c3_token)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)

        f1_classes_train, weighted_f1_train, _ = evaluate(ddp_model.module, device, train_loader)
        f1_classes_test, weighted_f1_test, _ = evaluate(ddp_model.module, device, test_loader)
        f1_train_values.append(weighted_f1_train)
        f1_test_values.append(weighted_f1_test)

        print(f"[Rank {rank}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train F1: {weighted_f1_train:.4f} | Test F1: {weighted_f1_test:.4f} | Time: {time.time()-start_time:.2f} sec")

    if rank == 0:
        hyperparams = {
            'model_arch': 'PCSA_Concat_Fusion_Model',
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout_prob': dropout_prob,
            'epochs': epochs
        }
        checkpoint = {
            'hyperparams': hyperparams,
            'model_state_dict': ddp_model.module.state_dict(),
            'train_loss': loss_values,
            'train_f1': f1_train_values,
            'test_f1': f1_test_values,
        }
        save_path = "/kaggle/working/PCSA_concatenate(5).pth"
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
        return_dict[rank] = (loss_values, f1_train_values, f1_test_values)

    cleanup()


def main():
    PhoBert_path = 'vinai/phobert-base-v2'
    data_path = '/kaggle/input/post-comment-dataset/preprocessed_data.json'
    batch_size = 32
    lr = 3e-5
    dropout_prob = 0.3
    num_label = 3
    epochs = 30

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['posts']
    dataset = create_dataset(data)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(PhoBert_path)
    pcsa_train_dataset = PCSA_Dataset(train_set, tokenizer, max_length=128)
    pcsa_test_dataset = PCSA_Dataset(test_set, tokenizer, max_length=128)

    world_size = torch.cuda.device_count()
    print("Số lượng GPU dùng cho DDP:", world_size)

    manager = Manager()
    return_dict = manager.dict()

    mp.spawn(train_ddp, args=(world_size, pcsa_train_dataset, pcsa_test_dataset, PhoBert_path, num_label, dropout_prob, lr, epochs, batch_size, return_dict),
             nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
