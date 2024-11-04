import numpy as np
import torch
from tqdm import tqdm


def train_ssl_epoch(cfg, model, dataloader, loss_fn, optimizer):
    
    model.train()
    loss_sum = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for n, (x_weak, y_weak, mask_weak, x_strong, y_strong, mask_strong, _) in pbar:
        optimizer.zero_grad()
        
        cur_batch_size = x_weak.shape[0]
        
        if cfg.dataset_window_size > cfg.model_input_length:
            start = np.random.choice(cfg.dataset_window_size - cfg.model_input_length + 1)
            end = start + cfg.model_input_length
        else:
            start = 0
            end = cfg.model_input_length
                
        x_weak = x_weak[:, start:end]
        y_weak = y_weak[:, start:end]
        mask_weak = mask_weak[:, start:end]
        x_strong = x_strong[:, :cfg.model_input_length]
        y_strong = y_strong[:, :cfg.model_input_length]
        mask_strong = mask_strong[:, :cfg.model_input_length]
        
        x_weak, y_weak, mask_weak = x_weak.to(cfg.device), y_weak.to(cfg.device), mask_weak.to(cfg.device)
        x_strong, y_strong, mask_strong = x_strong.to(cfg.device), y_strong.to(cfg.device), mask_strong.to(cfg.device)

        x = torch.cat([x_weak, x_strong], dim=0)
        y = torch.cat([y_weak, y_strong], dim=0)
        mask = torch.cat([mask_weak, mask_strong], dim=0)
        pad_mask = torch.ones(*x.shape[:-1], dtype=torch.bool, device=cfg.device)

        # output_pred, output_emb, output_emb_pool, output_emb_pool_reconstructed, output_emb_proj, output_emb_fin_proj
        prediction, reconstruction_gt, _, reconstruction_pred, _, embedings = model(x, pad_mask, return_all=True)
        
        embedings_weak, embedings_strong = embedings[:cur_batch_size], embedings[cur_batch_size:]
        
        loss = loss_fn(prediction, y, mask, embedings_weak, embedings_strong, reconstruction_pred, reconstruction_gt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        
        loss_sum += loss.item()

        pbar.update()
        pbar.set_description(f"Loss: {round(loss_sum / (n + 1), 4)}")

    return loss_sum / (n + 1)

def train_scan_epoch(cfg, epoch, encoder, clustering_model, loader, loss_fn, encoder_optimizer, clustering_optimizer):
    clustering_model = clustering_model.train()
    
    flag_update_encoder = epoch >= cfg.clustering_finetuning_epochs
    if flag_update_encoder:
        encoder = encoder.train()
    else :
        encoder = encoder.eval()

    loss_sum = torch.zeros(1, device = cfg.device)

    for x, neighbour, _ in tqdm(loader, total= len(loader)):
        clustering_optimizer.zero_grad()
        if epoch >= cfg.clustering_finetuning_epochs:
            encoder_optimizer.zero_grad()
        
        x, neighbour = x.to(cfg.device), neighbour.to(cfg.device)
        
        with torch.set_grad_enabled(flag_update_encoder):
            pad_mask_x = torch.ones(*x.shape[:-1], dtype = torch.bool, device = cfg.device)
            _, _, _, features_x, _ = encoder(x, pad_mask_x, return_all = True)
            pad_mask_neighbour = torch.ones(*neighbour.shape[:-1], dtype = torch.bool, device = cfg.device)
            _, _, _, features_neighbour, _ = encoder(neighbour, pad_mask_neighbour, return_all = True)
        
        x_outputs = clustering_model(features_x)
        neighbour_outputs = clustering_model(features_neighbour)
        
        total_loss = loss_fn(x_outputs, neighbour_outputs)
        total_loss.backward()
        
        clustering_optimizer.step()
        if flag_update_encoder:
            encoder_optimizer.step()
        
        loss_sum += total_loss.detach()

    return loss_sum.item() / len(loader)


def train_triplet_epoch(cfg, model, dataloader, loss_fn, optimizer):
    model.train()

    loss = 0
    triplet_loss = 0

    for (anchors, positives, negatives, a_masks, p_masks, n_masks) in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()

        anchors = anchors.to(cfg.device)
        positives = positives.to(cfg.device)
        negatives = negatives.to(cfg.device)

        a_masks = a_masks.to(cfg.device)
        p_masks = p_masks.to(cfg.device)
        n_masks = n_masks.to(cfg.device)

        anchor_embs = model(anchors, a_masks)[1]
        positive_embs = model(positives, p_masks)[1]
        negative_embs = model(negatives, n_masks)[1]

        triplet_loss_, intra_loss_ = loss_fn(anchor_embs, positive_embs, negative_embs)
        triplet_loss += triplet_loss_.item()
        loss = (1 - cfg.alpha) * triplet_loss_ + cfg.alpha * intra_loss_

        # loss_ = loss_fn(anchor_embs, positive_embs, negative_embs)
        loss.backward()
        optimizer.step()
        # loss += loss_.item()

    return triplet_loss / len(dataloader)
