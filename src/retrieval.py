import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  

def select_subset_indices(dataset, subset_file, subset_size=1000):
    """
    Attempt to read subset indices from `subset_file`.
    If it doesn't exist, sample a random subset of size `subset_size`,
    write them, and return them.
    """
    import os
    import random

    if os.path.exists(subset_file):
        with open(subset_file, 'r') as f:
            indices = json.load(f)
        print(f"Loaded {len(indices)} subset indices from {subset_file}")
        return indices
    else:
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        subset = all_indices[:subset_size]
        with open(subset_file, 'w') as f:
            json.dump(subset, f)
        print(f"Created new subset of size {subset_size} and wrote to {subset_file}")
        return subset

def embed_av_subset(model, dataset, subset_indices, device='cuda', batch_size=8):
    """
    Extract audio and video embeddings for the given subset of indices in `dataset`.

    Returns:
        audio_feats_list: list of length N, each is a (Na_i, D) FloatTensor (CPU)
        video_feats_list: list of length N, each is a (Nv_i, D) FloatTensor (CPU)
        video_paths_list: list of length N, the string paths for debugging
    """
    model.eval()
    audio_feats_list = [None]*len(subset_indices)
    video_feats_list = [None]*len(subset_indices)
    video_paths_list = [None]*len(subset_indices)

    def collate_eval_fn(batch):

        frames = []
        audios = []
        paths = []
        for item in batch:
            frames.append(item['video_frames'])
            audios.append(item['audio'])
            paths.append(item['video_path'])

        max_len = max(a.shape[0] for a in audios)
        audio_padded = torch.zeros(len(audios), max_len)
        for i,aud in enumerate(audios):
            audio_padded[i,:aud.shape[0]] = aud
        return {
            'frames': torch.stack(frames),
            'audio': audio_padded,
            'paths': paths
        }

    class AVSubset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices):
            self.base = base_dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]

            sample = self.base.__getitem__(real_idx, apply_augmentation=False)
            return sample

    subset_ds = AVSubset(dataset, subset_indices)
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_eval_fn, drop_last=False)

    print(f"Embedding A/V subset of size {len(subset_indices)} ...")
    idx_offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding AV subset"):
            frames = batch['frames'].to(device)
            audio  = batch['audio'].to(device)
            paths  = batch['paths']

            vfeats = model.visual_embedder(frames)
            afeats = model.audio_embedder(audio)

            vfeats = F.normalize(vfeats, dim=2)
            afeats = F.normalize(afeats, dim=2)

            B = vfeats.shape[0]
            for b in range(B):
                audio_feats_list[idx_offset + b] = afeats[b].cpu()
                video_feats_list[idx_offset + b] = vfeats[b].cpu()
                video_paths_list[idx_offset + b] = paths[b]

            idx_offset += B

    return audio_feats_list, video_feats_list, video_paths_list

def aggregator_av_a2v(a_feats, v_feats, temperature):

    token_sims = torch.matmul(a_feats, v_feats.t()) / temperature
    max_sims = token_sims.max(dim=1).values
    return max_sims.mean().item()

def aggregator_av_v2a(a_feats, v_feats, temperature):
    token_sims = torch.matmul(a_feats, v_feats.t()) / temperature
    max_sims = token_sims.max(dim=0).values
    return max_sims.mean().item()

def compute_recall_at_k(sim_matrix):
    """
    Given NxN sim_matrix, where sim_matrix[i,j] is the similarity
    of query i to item j, the correct match is j=i.
    We'll compute R@1, R@5, R@10, R@20.

    Returns: dict with {'r1':..., 'r5':..., 'r10':..., 'r20':...}
    """
    N = sim_matrix.shape[0]
    ranks = []
    for i in range(N):
        row = sim_matrix[i]

        sorted_indices = np.argsort(-row)
        rank_of_correct = np.where(sorted_indices == i)[0][0]
        ranks.append(rank_of_correct)
    ranks = np.array(ranks)

    r1  = np.mean(ranks < 1)
    r5  = np.mean(ranks < 5)
    r10 = np.mean(ranks < 10)
    r20 = np.mean(ranks < 20)
    return {
        'r1':  r1,
        'r5':  r5,
        'r10': r10,
        'r20': r20
    }

def compute_av_retrieval_metrics(model, dataset, subset_file, device='cuda'):
    """
    1) Select subset indices
    2) Embed the subset
    3) Build NxN sim_matrices for A->V aggregator and V->A aggregator
    4) Compute retrieval metrics
    5) Return a dictionary
    """
    indices = select_subset_indices(dataset, subset_file, subset_size=1000)
    audio_feats_list, video_feats_list, _ = embed_av_subset(model, dataset, indices, device=device, batch_size=8)
    N = len(indices)
    temperature = model.temperature.item()

    print(f"Computing A->V retrieval on {N} items ...")
    sim_mat_a2v = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Aggregator A->V"):
        afeats_i = audio_feats_list[i].to(device)
        for j in range(N):
            vfeats_j = video_feats_list[j].to(device)
            sim_mat_a2v[i, j] = aggregator_av_a2v(afeats_i, vfeats_j, temperature)
    av_metrics = compute_recall_at_k(sim_mat_a2v)

    print(f"Computing V->A retrieval on {N} items ...")
    sim_mat_v2a = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Aggregator V->A"):
        vfeats_i = video_feats_list[i].to(device)
        for j in range(N):
            afeats_j = audio_feats_list[j].to(device)
            sim_mat_v2a[i, j] = aggregator_av_v2a(afeats_j, vfeats_i, temperature)
    va_metrics = compute_recall_at_k(sim_mat_v2a)

    results = {
        'A->V_r1':  av_metrics['r1'],
        'A->V_r5':  av_metrics['r5'],
        'A->V_r10': av_metrics['r10'],
        'A->V_r20': av_metrics['r20'],

        'V->A_r1':  va_metrics['r1'],
        'V->A_r5':  va_metrics['r5'],
        'V->A_r10': va_metrics['r10'],
        'V->A_r20': va_metrics['r20'],
    }
    return results

def aggregator_tv_t2v(t_feats, v_feats, temperature):
    token_sims = torch.matmul(t_feats, v_feats.t()) / temperature
    max_sims = token_sims.max(dim=1).values
    return max_sims.mean().item()

def aggregator_tv_v2t(t_feats, v_feats, temperature):
    token_sims = torch.matmul(t_feats, v_feats.t()) / temperature
    max_sims = token_sims.max(dim=0).values
    return max_sims.mean().item()

def embed_tv_subset(model, dataset, subset_indices, device='cuda', batch_size=8):
    """
    Extract text and image embeddings for the 1000 chosen items in dataset.
    We'll do a small subset DataLoader, no augmentation, etc.
    Returns:
        text_feats_list[i]: (Nt_i, D)
        image_feats_list[i]: (Ni_i, D)
    """
    model.eval()
    text_feats_list = [None]*len(subset_indices)
    image_feats_list = [None]*len(subset_indices)

    def collate_tv_eval(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        return images, list(captions)

    class TVSubset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices):
            self.base = base_ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]

            return self.base.__getitem__(real_idx)

    subset_ds = TVSubset(dataset, subset_indices)
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_tv_eval)

    print(f"Embedding T/V subset of size {len(subset_indices)} ...")
    idx_offset = 0
    with torch.no_grad():
        for batch_images, batch_captions in tqdm(loader, desc="Embedding TV subset"):
            batch_images = batch_images.to(device)
            vfeats = model.visual_embedder(batch_images)
            tfeats, attn_mask = model.text_embedder(batch_captions)

            B = vfeats.shape[0]
            for b in range(B):

                n_tokens = attn_mask[b].sum().item()
                text_feats_list[idx_offset + b] = tfeats[b, :n_tokens].cpu()
                image_feats_list[idx_offset + b] = vfeats[b].cpu()
            idx_offset += B

    return text_feats_list, image_feats_list

def compute_tv_retrieval_metrics(model, dataset, subset_file, device='cuda'):
    """
    1) select subset
    2) embed
    3) build NxN sim matrices for T->V and V->T
    4) compute recall
    5) return dictionary
    """
    indices = select_subset_indices(dataset, subset_file, subset_size=1000)
    text_feats_list, image_feats_list = embed_tv_subset(model, dataset, indices, device=device, batch_size=8)
    N = len(indices)
    temperature = model.temperature.item()

    print(f"Computing T->V retrieval on {N} items ...")
    sim_mat_t2v = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Aggregator T->V"):
        tfeats_i = text_feats_list[i].to(device)
        for j in range(N):
            vfeats_j = image_feats_list[j].to(device)
            sim_mat_t2v[i, j] = aggregator_tv_t2v(tfeats_i, vfeats_j, temperature)
    tv_metrics = compute_recall_at_k(sim_mat_t2v)

    print(f"Computing V->T retrieval on {N} items ...")
    sim_mat_v2t = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Aggregator V->T"):
        vfeats_i = image_feats_list[i].to(device)
        for j in range(N):
            tfeats_j = text_feats_list[j].to(device)
            sim_mat_v2t[i, j] = aggregator_tv_v2t(tfeats_j, vfeats_i, temperature)
    vt_metrics = compute_recall_at_k(sim_mat_v2t)

    results = {
        'T->V_r1':  tv_metrics['r1'],
        'T->V_r5':  tv_metrics['r5'],
        'T->V_r10': tv_metrics['r10'],
        'T->V_r20': tv_metrics['r20'],

        'V->T_r1':  vt_metrics['r1'],
        'V->T_r5':  vt_metrics['r5'],
        'V->T_r10': vt_metrics['r10'],
        'V->T_r20': vt_metrics['r20'],
    }
    return results

