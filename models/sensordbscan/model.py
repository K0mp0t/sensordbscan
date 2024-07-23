import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import (BatchNorm1d, Dropout, Linear, MultiheadAttention,
                              TransformerEncoderLayer)
# from sklearn.cluster import DBSCAN
from cuml.cluster import DBSCAN
from cuml.neighbors import KNeighborsClassifier


from .reference_attention import MultiHeadAttention as ReferenceMultiHeadAttention


def build_encoder(cfg):

    model = TSTransformerEncoder(feat_dim = cfg.num_features, max_len = cfg.model_input_length, d_model = cfg.model_dim, n_heads = cfg.num_heads,
                               num_layers = cfg.num_layers, dim_feedforward = cfg.ff_dim, dropout= cfg.dropout_rate, norm = cfg.norm)
    
    # model = model.apply(init_weights)
    model = model.to(cfg.device)

    return model

def build_clustering(cfg):
    model = nn.Sequential(
        nn.Linear(cfg.encoder_dim, cfg.clustering_dim),
        nn.BatchNorm1d(cfg.clustering_dim),
        nn.ReLU(),
        nn.Linear(cfg.clustering_dim, cfg.num_clusters)
    )

    model = model.apply(init_weights)

    model = model.to(cfg.device)

    return model

def init_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.)

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super(MultiHeadAttentionWithRoPE, self).__init__()

        assert d_model % n_heads == 0, "nheads should divide d_model"

        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.head_dim = d_model // n_heads

        self.compute_query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.compute_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.compute_value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.compute_output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.batch_first = False


    def _apply_rotary_emb(self, query, key, theta=10000.0):
        seqlen, _, _, _ = query.shape
        device = query.device

        query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
        key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

        thetas = torch.pow(theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim).to(device)

        sines = torch.sin(thetas.unsqueeze(0) * torch.arange(seqlen, device=device).unsqueeze(1))
        cosines = torch.cos(thetas.unsqueeze(0) * torch.arange(seqlen, device=device).unsqueeze(1))

        query_real_rotated = torch.zeros_like(query, device=device)
        query_imag_rotated = torch.zeros_like(query, device=device)
        key_real_rotated = torch.zeros_like(key, device=device)
        key_imag_rotated = torch.zeros_like(key, device=device)

        query_real_rotated[..., 0::2] = query_real * cosines.unsqueeze(1)
        query_real_rotated[..., 1::2] = query_real * sines.unsqueeze(1)

        query_imag_rotated[..., 0::2] = -query_imag * sines.unsqueeze(1)
        query_imag_rotated[..., 1::2] = query_imag * cosines.unsqueeze(1)

        key_real_rotated[..., 0::2] = key_real * cosines.unsqueeze(1)
        key_real_rotated[..., 1::2] = key_real * sines.unsqueeze(1)

        key_imag_rotated[..., 0::2] = -key_imag * sines.unsqueeze(1)
        key_imag_rotated[..., 1::2] = key_imag * cosines.unsqueeze(1)

        query_out = query_real_rotated + query_imag_rotated
        key_out = key_real_rotated + key_imag_rotated

        return query_out, key_out

    def compute_query_key_value_scores(self, query, key, value, src_key_padding_mask=None):
        # attn_scores = query.permute(1, 2, 0, 3) @ key.permute((1, 2, 3, 0)) / math.sqrt(self.head_dim)
        attn_scores = query @ key.permute((0, 2, 1)) / math.sqrt(self.head_dim)

        # if src_key_padding_mask is not None:
        #     src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, attn_scores.shape[1], attn_scores.shape[2], -1)
        #     attn_scores[src_key_padding_mask.bool()] = -torch.inf

        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        # output = attn_scores @ value.permute(1, 2, 0, 3)
        # output = output.permute(2, 0, 1, 3)
        output = attn_scores @ value

        return output, attn_scores

    def forward(self, query, key, value, attn_mask=None, src_key_padding_mask=None):
        seqlen, batch_size, _ = query.shape

        query = self.compute_query(query)
        key = self.compute_key(key)
        value = self.compute_value(value)
        query = query.view(seqlen, batch_size, self.n_heads, self.head_dim).permute(2, 1, 0, 3).reshape(-1, seqlen, self.head_dim)
        key = key.view(seqlen, batch_size, self.n_heads, self.head_dim).permute(2, 1, 0, 3).reshape(-1, seqlen, self.head_dim)
        value = value.view(seqlen, batch_size, self.n_heads, self.head_dim).permute(2, 1, 0, 3).reshape(-1, seqlen, self.head_dim)

        # query, key = self._apply_rotary_emb(query, key)

        output, attn_score = self.compute_query_key_value_scores(query, key, value, src_key_padding_mask=src_key_padding_mask)
        output = output.reshape(self.n_heads, batch_size, seqlen, self.head_dim).permute(1, 2, 0, 3)
        attn_score = attn_score.reshape(self.n_heads, batch_size, seqlen, self.head_dim).permute(1, 2, 0, 3)

        output = output.transpose(1, 2).contiguous().view(seqlen, batch_size, -1)
        output = self.resid_dropout(self.compute_output(output))
        return output, attn_score


class TransformerBatchNormEncoderLayer(nn.modules.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  
        src = src.permute(1, 2, 0)  
       
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  
        src = src.permute(1, 2, 0)  
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  
        return src


class TransformerEncoderLayerWithCustomAttention(nn.modules.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayerWithCustomAttention, self).__init__()
        # self.self_attn = MultiHeadAttentionWithRoPE(d_model, nhead, dropout=dropout)
        self.self_attn = ReferenceMultiHeadAttention(d_model, nhead)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(TransformerEncoderLayerWithCustomAttention, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)

        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class SeqPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)
          
    def forward(self, x):
        w = self.attention_pool(x)
        w = F.softmax(w, dim=1)
        w = w.transpose(1, 2)
        
        y = torch.matmul(w, x)
        
        y = y.squeeze(1)
        
        return y

class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0, activation='gelu',
                 norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        # self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayerWithCustomAttention(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = F.gelu

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        
        self.seq_pooling = SeqPooling(self.d_model)
        
        self.projection_linear = nn.Linear(self.d_model, self.d_model // 2)
        self.projection_bn = nn.BatchNorm1d(self.d_model // 2)
        self.projection_relu = nn.ReLU(inplace=True)
        self.projection_fin_linear = nn.Linear(self.d_model // 2, self.d_model // 4)

    def forward(self, X, padding_masks, return_all = False):
        
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        # inp = self.pos_enc(inp)
      
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  
        output = self.act(output)  
        output_emb = output.permute(1, 0, 2)  
        output_pred = self.dropout1(output_emb)
        
        output_pred = self.output_layer(output_pred)
        
        output_emb_pool = self.seq_pooling(output_emb)
        output_emb_proj = self.projection_linear(output_emb_pool)
        output_emb_fin_proj = self.projection_bn(output_emb_proj)
        output_emb_fin_proj = self.projection_relu(output_emb_fin_proj)
        output_emb_fin_proj = self.projection_fin_linear(output_emb_fin_proj)
        # output_emb_fin_proj = F.normalize(output_emb_fin_proj, p=2, dim=1)  # change distance function?

        if return_all:
            return output_pred, output_emb, output_emb_pool, output_emb_proj, output_emb_fin_proj
            
        return output_pred, output_emb_fin_proj
    
class SensorSCAN(nn.Module):
    def __init__(self, encoder, clustering_model, device):
        super().__init__()
        self.encoder = encoder
        self.clustering_model = clustering_model
        self.device = device
    def forward(self, X):
        pad_mask = torch.ones(*X.shape[:-1], dtype=torch.bool, device=self.device)
        _, _, _, encoder_embedings, _ = self.encoder(X, pad_mask, return_all=True)
        clutsering_output = self.clustering_model(encoder_embedings)
        return clutsering_output


class SensorDBSCAN(object):
    def __init__(self, encoder, cfg, avg_loss):
        self.encoder = encoder
        self.cfg = cfg
        self.avg_loss = avg_loss

        self.clustering_algorithm = DBSCAN

    def get_embs(self, X):
        pad_masks = torch.ones(*X.shape[:-1], dtype=torch.bool, device=self.cfg.device)

        embs = self.encoder(X, pad_masks)[1]

        return embs

    def cluster_embs(self, embs):
        cluster_labels = self.clustering_algorithm(eps=(self.cfg.epsilon-self.avg_loss)*self.cfg.dbscan_epsilon_multiplier,
                                                   min_samples=self.cfg.min_samples, metric=self.cfg.metric,
                                                   max_mbytes_per_batch=self.cfg.max_mbytes_per_batch).fit_predict(embs)

        if self.cfg.handle_outliers:
            knn = KNeighborsClassifier(n_neighbors=self.cfg.knn_neighbors, metric=self.cfg.metric)
            knn.fit(embs[cluster_labels != -1], cluster_labels[cluster_labels != -1])

            cluster_labels[cluster_labels == -1] = knn.predict(embs[cluster_labels == -1])

        return cluster_labels

