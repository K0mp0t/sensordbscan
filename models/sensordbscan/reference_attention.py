import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

        self.batch_first = False
        self.head_dim = in_features // head_num

    def _apply_rotary_emb(self, query, key, theta=10000.0):
        _, seqlen, _ = query.shape
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

        query_real_rotated[..., 0::2] = query_real * cosines
        query_real_rotated[..., 1::2] = query_real * sines

        query_imag_rotated[..., 0::2] = -query_imag * sines
        query_imag_rotated[..., 1::2] = query_imag * cosines

        key_real_rotated[..., 0::2] = key_real * cosines
        key_real_rotated[..., 1::2] = key_real * sines

        key_imag_rotated[..., 0::2] = -key_imag * sines
        key_imag_rotated[..., 1::2] = key_imag * cosines

        query_out = query_real_rotated + query_imag_rotated
        key_out = key_real_rotated + key_imag_rotated

        return query_out, key_out

    def forward(self, q, k, v, attn_mask=None, src_key_padding_mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        q, k = self._apply_rotary_emb(q, k)

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.unsqueeze(1).repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, src_key_padding_mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(seq_len, batch_size, self.head_num, sub_dim)\
                .permute(1, 2, 0, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(2, 0, 1, 3)\
                .reshape(seq_len, batch_size, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )