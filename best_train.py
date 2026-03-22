# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # 引入混合精度
from dataset import SpikingVideoDataset  
from clip.simple_tokenizer import SimpleTokenizer
from clip.best_model import CLIP
import gc
import time

# --- 配置设备和路径 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./weight_sa/best_hdbm51_b8_accum.pth" # 建议换个名字以区分
TRAIN_JSON_PATH = "./train.json"
VAL_JSON_PATH = "./val.json"

# --- 核心超参数 ---
SPIKE_H, SPIKE_W = 240, 320
PHYSICAL_BATCH_SIZE = 8       # 你的显卡能承受的最大 Batch Size
TARGET_BATCH_SIZE = 8        # 你想要达到的理想 Batch Size
ACCUMULATION_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE # 8 / 8 = 1

TOTAL_EPOCHS = 100
LEARNING_RATE = 1e-4          # 增大LR以匹配较大的有效Batch Size
WEIGHT_DECAY = 0.001           # 适中的正则化
NUM_WORKERS = 8               # 开启多进程加载，加速 CPU 读取
VALIDATE_EVERY_N_EPOCHS = 1   # 验证频率

print(f"Running on {DEVICE}")
print(f"Physical Batch Size: {PHYSICAL_BATCH_SIZE}")
print(f"Gradient Accumulation Steps: {ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS}")

def initialize_model():
    model = CLIP(
        embed_dim=256,
        image_resolution=(SPIKE_H, SPIKE_W),
        vision_layers=(2, 2, 2, 2), # 保持和你现有权重结构一致
        vision_width=256,
        context_length=77,
        vocab_size=49408,
        transformer_width=128,
        transformer_heads=4,
        transformer_layers=4,
        input_channels=64
    ).to(DEVICE)
    
    # 如果有之前的权重，可以选择加载；或者从头开始
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print(f"Loaded model weights from {MODEL_PATH}")
        except:
            print("Weight mismatch or error, starting from scratch.")
    else:
        print("Starting from scratch.")
    return model

def create_data_loaders():
    train_dataset = SpikingVideoDataset(
        json_file=TRAIN_JSON_PATH,
        spike_h=SPIKE_H,
        spike_w=SPIKE_W,
        device=DEVICE, # 注意：Dataset里尽量不要直接to(device)，留给DataLoader取出来后再to(device)
        target_frames=25,
        channels_per_sample=10,
        is_training=True  # 开启数据增强
    )
    val_dataset = SpikingVideoDataset(
        json_file=VAL_JSON_PATH,
        spike_h=SPIKE_H,
        spike_w=SPIKE_W,
        device=DEVICE,
        target_frames=25,
        channels_per_sample=10,
        is_training=False # 关闭数据增强
    )
    
    # 关键优化：多进程加载 + 锁页内存
    train_loader = DataLoader(
        train_dataset, 
        batch_size=PHYSICAL_BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=NUM_WORKERS, 
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=PHYSICAL_BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=NUM_WORKERS, 
        persistent_workers=True,
        drop_last=False
    )
    return train_loader, val_loader

def encode_texts(tokenizer, captions):
    tokenized_texts = []
    for caption in captions:
        tokenized = tokenizer.encode(caption)[:77]
        tokenized += [0] * (77 - len(tokenized))
        tokenized_texts.append(tokenized)
    return torch.tensor(tokenized_texts).to(DEVICE)

def train_and_validate(model, train_loader, val_loader):
    tokenizer = SimpleTokenizer()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 初始化 AMP Scaler
    scaler = GradScaler()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = 15

    for epoch in range(TOTAL_EPOCHS):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        correct_i = correct_t = total = 0
        
        optimizer.zero_grad() # 也就是 set_to_none=True

        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} started...")

        for idx, (spikes, texts) in enumerate(train_loader):
            # 数据移动到 GPU
            spikes = spikes.to(DEVICE, non_blocking=True)
            text_tokens = encode_texts(tokenizer, texts) # 文本编码一般很快

            # === 混合精度前向传播 ===
            with autocast():
                logits_per_image, logits_per_text = model(spikes, text_tokens)
                targets = torch.arange(len(spikes)).to(DEVICE)

                # 添加 Label Smoothing (0.1)
                loss_i = nn.functional.cross_entropy(logits_per_image, targets, label_smoothing=0.1)
                loss_t = nn.functional.cross_entropy(logits_per_text, targets, label_smoothing=0.1)
                loss = (loss_i + loss_t) / 2

                # 关键：Loss 除以累积步数，保证梯度大小正确
                loss = loss / ACCUMULATION_STEPS

            # === 混合精度反向传播 ===
            scaler.scale(loss).backward()

            # === 梯度累积更新 ===
            if (idx + 1) % ACCUMULATION_STEPS == 0:
                # Unscale 梯度以进行裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新权重
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 还原 Loss 数值用于统计
            current_loss = loss.item() * ACCUMULATION_STEPS
            epoch_train_loss += current_loss
            
            # 计算准确率 (不需要梯度)
            with torch.no_grad():
                correct_i += (logits_per_image.argmax(1) == targets).sum().item()
                correct_t += (logits_per_text.argmax(1) == targets).sum().item()
                total += len(targets)

            if idx % 10 == 0:
                print(f"\rBatch [{idx+1}/{len(train_loader)}] "
                      f"Loss: {current_loss:.4f} | Acc: {100 * correct_i / total:.1f}%", end='')

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        print(f"\nTrain Loss: {avg_train_loss:.4f} | Acc I->T: {100*correct_i/total:.2f}%")

        # === 验证阶段 ===
        if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0:
            model.eval()
            epoch_val_loss = 0.0
            val_correct_i = val_correct_t = val_total = 0

            with torch.no_grad():
                for spikes, texts in val_loader:
                    spikes = spikes.to(DEVICE, non_blocking=True)
                    text_tokens = encode_texts(tokenizer, texts)

                    # 验证时也可以开启 autocast 加速推理
                    with autocast():
                        logits_per_image, logits_per_text = model(spikes, text_tokens)
                        targets = torch.arange(len(spikes)).to(DEVICE)
                        
                        loss = (nn.functional.cross_entropy(logits_per_image, targets, label_smoothing=0.1) +
                                nn.functional.cross_entropy(logits_per_text, targets, label_smoothing=0.1)) / 2

                    epoch_val_loss += loss.item()
                    val_correct_i += (logits_per_image.argmax(1) == targets).sum().item()
                    val_correct_t += (logits_per_text.argmax(1) == targets).sum().item()
                    val_total += len(targets)

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            val_acc = (val_correct_i + val_correct_t) / (2 * val_total) * 100
            print(f"Val Loss: {avg_val_loss:.4f} | Val Acc I->T: {100*val_correct_i/val_total:.2f}% | Time: {(time.time()-start_time)/60:.1f} min")

            # 保存最优模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Saved Best Model ({best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement: {epochs_without_improvement}/{early_stop_patience}")

            scheduler.step(avg_val_loss)

            if epochs_without_improvement >= early_stop_patience:
                print("Early stopping triggered.")
                break
        
        gc.collect()

    print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    model = initialize_model()
    train_loader, val_loader = create_data_loaders()
    train_and_validate(model, train_loader, val_loader)