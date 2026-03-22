# -*- coding: UTF-8 -*-
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SpikingVideoDataset  
from clip.simple_tokenizer import SimpleTokenizer
from clip.best_model_Neural_Mask import CLIP
import gc
import time
import torch.nn as nn

# Configure device and paths
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./weight_sa/best_hdbm51_b8_Neural_Mask.pth"
TRAIN_JSON_PATH = "./train.json"
VAL_JSON_PATH = "./val.json"
SPIKE_H, SPIKE_W = 240, 320
BATCH_SIZE = 8
TOTAL_EPOCHS = 100
LEARNING_RATE = 1e-5
torch.cuda.empty_cache()

def initialize_model():
    model = CLIP(
        embed_dim=256,
        image_resolution=(SPIKE_H, SPIKE_W),
        vision_layers=(2, 2, 2, 2),
        vision_width=256,
        context_length=77,
        vocab_size=49408,
        transformer_width=128,
        transformer_heads=4,
        transformer_layers=4,
        input_channels=64
    ).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print("Starting from scratch.")
    return model

def create_data_loaders(batch_size=BATCH_SIZE):
    train_dataset = SpikingVideoDataset(
        json_file=TRAIN_JSON_PATH,
        spike_h=SPIKE_H,
        spike_w=SPIKE_W,
        device=DEVICE,
        target_frames=25,
        channels_per_sample=10
    )
    val_dataset = SpikingVideoDataset(
        json_file=VAL_JSON_PATH,
        spike_h=SPIKE_H,
        spike_w=SPIKE_W,
        device=DEVICE,
        target_frames=25,
        channels_per_sample=10
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)
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
    
    # Optimizer Configuration
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE, # Suggest 1e-4 or 5e-5
        weight_decay=1e-4, 
        betas=(0.9, 0.999)
    )

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses = [], []
    # best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0
    early_stop_patience = 10

    for epoch in range(TOTAL_EPOCHS):
        # ==========================
        #       Training Phase
        # ==========================
        model.train()
        epoch_train_loss = 0.0
        correct_i = correct_t = total = 0
        start_time = time.time()

        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} started...")

        for idx, (spikes, texts) in enumerate(train_loader):
            spikes = spikes.to(DEVICE, non_blocking=True)
            text_tokens = encode_texts(tokenizer, texts)
            
            # 1. Generate Padding Mask (Non-zero positions are valid)
            # text_tokens: [B, 77] -> mask: [B, 77] (Valid word is 1, Padding is 0)
            text_pad_mask = (text_tokens != 0).float() 

            # Clear gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # [Critical Update] Receive 5 return values now: 
                # Global Logits, Local Features, and MSM Loss
                logits_per_image, logits_per_text, v_local, t_local, msm_loss = model(spikes, text_tokens)
                targets = torch.arange(len(spikes)).to(DEVICE)

                # --- Loss Part A: Global Contrastive Loss ---
                loss_i = nn.functional.cross_entropy(logits_per_image, targets, label_smoothing=0.1)
                loss_t = nn.functional.cross_entropy(logits_per_text, targets, label_smoothing=0.1)
                loss_global = (loss_i + loss_t) / 2

                # --- Loss Part B: Fine-Grained Loss (FILIP) ---
                loss_fine = 0.0
                if v_local is not None and t_local is not None:
                    logit_scale = model.logit_scale.exp()
                    
                    # 1. Compute Similarity Matrix: [B, B, T, L]
                    # Note the dimension order: b(video), c(text), t(frame), l(word)
                    sim_matrix = torch.einsum('btd, cld -> bctl', v_local, t_local) * logit_scale
                    
                    # 2. Max over Vision (T): [B, B, L]
                    sim_max_vision = sim_matrix.max(dim=2)[0]
                    
                    # 3. Mean over Text (L) with Masking
                    mask = text_pad_mask.unsqueeze(0) # [1, B, L]
                    
                    # Mask padding positions
                    masked_sim = sim_max_vision * mask
                    
                    # Compute weighted mean
                    logits_fine_i2t = masked_sim.sum(dim=2) / (mask.sum(dim=2) + 1e-6) # [B, B]
                    logits_fine_t2i = logits_fine_i2t.t()

                    # Compute Loss
                    loss_fine_i = nn.functional.cross_entropy(logits_fine_i2t, targets, label_smoothing=0.1)
                    loss_fine_t = nn.functional.cross_entropy(logits_fine_t2i, targets, label_smoothing=0.1)
                    loss_fine = (loss_fine_i + loss_fine_t) / 2

                # --- Loss Part C: Masked Spike Modeling (MSM) ---
                # Add the MSM loss component
                # msm_loss is a scalar tensor returned by the model
                
                # --- Total Loss ---
                # Weights: Global 1.0, Fine 0.5, MSM 1.0
                loss = loss_global + 0.5 * loss_fine + 1.0 * msm_loss

            # Backward pass (Use Scaler)
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            epoch_train_loss += loss.item()
            correct_i += (logits_per_image.argmax(1) == targets).sum().item()
            correct_t += (logits_per_text.argmax(1) == targets).sum().item()
            total += len(targets)

            if idx % 10 == 0:
                # Handle msm_loss printing (it might be a tensor)
                msm_val = msm_loss.item() if isinstance(msm_loss, torch.Tensor) else msm_loss
                print(f"\rBatch [{idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (G:{loss_global:.3f} F:{loss_fine:.3f} M:{msm_val:.3f}) | Acc: {100 * correct_i / total:.1f}%", end='')

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"\nTrain Loss: {avg_train_loss:.4f} | Acc I->T: {100 * correct_i / total:.2f}%")

        # ==========================
        #      Validation Phase
        # ==========================
        model.eval()
        epoch_val_loss = 0.0
        val_correct_i = val_correct_t = val_total = 0

        with torch.no_grad():
            for idx, (spikes, texts) in enumerate(val_loader):
                spikes = spikes.to(DEVICE, non_blocking=True)
                text_tokens = encode_texts(tokenizer, texts)
                text_pad_mask = (text_tokens != 0).float()

                # Validation uses autocast for consistency and speed
                with torch.cuda.amp.autocast():
                    # [Critical] Receive 5 return values even in validation
                    # Note: msm_loss will be 0.0 here because model.training is False
                    logits_per_image, logits_per_text, v_local, t_local, msm_loss = model(spikes, text_tokens)
                    targets = torch.arange(len(spikes)).to(DEVICE)

                    # Compute Loss (Consistent with training logic, but usually exclude MSM for val metric)
                    loss_i = nn.functional.cross_entropy(logits_per_image, targets)
                    loss_t = nn.functional.cross_entropy(logits_per_text, targets)
                    loss_global = (loss_i + loss_t) / 2
                    
                    loss_fine = 0.0
                    if v_local is not None and t_local is not None:
                        logit_scale = model.logit_scale.exp()
                        sim_matrix = torch.einsum('btd, cld -> bctl', v_local, t_local) * logit_scale
                        sim_max_vision = sim_matrix.max(dim=2)[0]
                        mask = text_pad_mask.unsqueeze(0)
                        logits_fine_i2t = (sim_max_vision * mask).sum(dim=2) / (mask.sum(dim=2) + 1e-6)
                        logits_fine_t2i = logits_fine_i2t.t()
                        loss_fine = (nn.functional.cross_entropy(logits_fine_i2t, targets) + 
                                     nn.functional.cross_entropy(logits_fine_t2i, targets)) / 2

                    # Total Val Loss (Usually we don't add MSM loss to validation metric as it's a self-supervised task)
                    loss = loss_global + 0.5 * loss_fine

                epoch_val_loss += loss.item()
                val_correct_i += (logits_per_image.argmax(1) == targets).sum().item()
                val_correct_t += (logits_per_text.argmax(1) == targets).sum().item()
                val_total += len(targets)
                
                if idx % 10 == 0:
                     print(f"\rVal Batch [{idx+1}/{len(val_loader)}]", end='')

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_acc_i = 100 * val_correct_i / val_total
        print(f"\nVal Loss: {avg_val_loss:.4f} | Acc: I->T {val_acc_i:.2f}% | Time: {(time.time()-start_time)/60:.1f}m")

        # Save Best Model
        # [修改] Save Best Model (Based on Accuracy)
        # 逻辑：只要当前的验证集准确率 (val_acc_i) 超过了历史最佳 (best_val_acc)，就保存
        if val_acc_i > best_val_acc:
            best_val_acc = val_acc_i
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved Best Model (Acc: {best_val_acc:.2f}%) at {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in Acc: {epochs_without_improvement}/{early_stop_patience}")

        # [保持不变] Scheduler 依然建议基于 Loss 调整
        # 原因：Loss 比 Acc 更平滑，更适合指导学习率下降。
        # 如果你想改成基于 Acc 调整 LR，需要将 scheduler 的 mode 改为 'max'
        scheduler.step(avg_val_loss)

        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered. Best Acc: {best_val_acc:.2f}%")
            break
        
        gc.collect()

    print(f"Training finished. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    model = initialize_model()
    train_loader, val_loader = create_data_loaders()
    train_and_validate(model, train_loader, val_loader)    