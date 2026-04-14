import math
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import  Linear,LayerNorm


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.p_to_r_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.r_to_p_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.p_norm = nn.LayerNorm(embed_dim)
        self.r_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, protein_feat, rna_feat, protein_mask=None, rna_mask=None):

        p2r_output, p2r_weights = self.p_to_r_attn(
            query=protein_feat, key=rna_feat, value=rna_feat,
            key_padding_mask=~rna_mask if rna_mask is not None else None
        )

        r2p_output, r2p_weights = self.r_to_p_attn(
            query=rna_feat, key=protein_feat, value=protein_feat,
            key_padding_mask=~protein_mask if protein_mask is not None else None
        )

        protein_out = self.dropout(p2r_output)
        rna_out = self.dropout(r2p_output)

        return protein_out, rna_out


class Mlp(nn.Module):
    def __init__(self, params):
        super(Mlp, self).__init__()
        self.fc1 = Linear(params.hidden_size, params.mlp_dim)
        self.fc2 = Linear(params.mlp_dim, params.hidden_size)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class Atten_Block(nn.Module):
    def __init__(self, params, vis):
        super(Atten_Block, self).__init__()
        self.hidden_size = params.hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)

        self.p_ffn = Mlp(params)
        self.r_ffn = Mlp(params)
        self.cross_attn = CrossAttention(embed_dim=params.hidden_size, num_heads=params.num_heads,
                                         dropout=params.dropout_rate)

    def forward(self, x_p, x_r):
        hp = x_p
        hr = x_r
        x_p = self.attention_norm(x_p)
        x_r = self.attention_norm(x_r)
        x_p, x_r = self.cross_attn(x_p, x_r)
        xp = x_p + hp
        xr = x_r + hr

        hp = xp
        hr = xr
        xp = self.ffn_norm(xp)
        xr = self.ffn_norm(xr)
        xp = self.p_ffn(xp)
        xr = self.r_ffn(xr)
        xp = xp + hp
        xr = xr + hr
        return xp, xr


class CA_Encoder(nn.Module):
    def __init__(self, params, vis):
        super(CA_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(params.hidden_size, eps=1e-6)
        for _ in range(params.num_layers):
            layer = Atten_Block(params, vis)
            self.layer.append(layer)

    def forward(self, x_p, x_r):
        for layer_block in self.layer:
            x_p, x_r = layer_block(x_p, x_r)
        p_encoded = self.encoder_norm(x_p)
        r_encoded = self.encoder_norm(x_r)
        return p_encoded, r_encoded

class CA_former(nn.Module):
    def __init__(self, params, vis):
        super(CA_former, self).__init__()
        self.ca_encoder = CA_Encoder(params, vis)

    def forward(self, protein_features,rna_features):
        protein_encoder,rna_encoder= self.ca_encoder(protein_features,rna_features)
        return protein_encoder,rna_encoder

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.1):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def mmd_loss_rbf(self, seq_embeddings, str_embeddings, bandwidth=1.0):
        diffs = seq_embeddings - str_embeddings
        l2_squared = torch.sum(diffs ** 2, dim=1)

        sim = torch.exp(-l2_squared / (2 * bandwidth ** 2))
        loss = 1.0 - sim
        return loss.mean()

    def forward(self, seq_embeddings, str_embeddings):

        seq_embeddings = F.normalize(seq_embeddings, p=2, dim=1)
        str_embeddings = F.normalize(str_embeddings, p=2, dim=1)
        N = seq_embeddings.shape[0]

        sim_seq_str = torch.matmul(seq_embeddings, str_embeddings.t()) / self.tau
        sim_seq_seq = torch.matmul(seq_embeddings, seq_embeddings.t()) / self.tau
        sim_str_str = torch.matmul(str_embeddings, str_embeddings.t()) / self.tau

        sim_pos = torch.diag(sim_seq_str)

        mask = ~torch.eye(N, dtype=torch.bool, device=seq_embeddings.device)

        numerator = torch.exp(sim_pos)

        denom_seq_str = torch.exp(sim_seq_str)[mask].view(N, N - 1).sum(dim=1)
        denom_seq_seq = torch.exp(sim_seq_seq)[mask].view(N, N - 1).sum(dim=1)
        denom_str_str = torch.exp(sim_str_str)[mask].view(N, N - 1).sum(dim=1)

        denominator = denom_seq_str + denom_seq_seq + denom_str_str
        denominator = denominator.clamp(min=1e-8)

        loss = -torch.log(numerator / denominator).mean()

        return loss

class Fusion(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim = hidden_dim
        attn_dim = hidden_dim // 4

        self.W1_attention = nn.Linear(hidden_dim, attn_dim)
        self.W2_attention = nn.Linear(hidden_dim, attn_dim)

        self.proj_dropout = nn.Dropout(dropout_rate)

        self.w = nn.Parameter(torch.ones(attn_dim))

        self.feature_dropout = nn.Dropout(dropout_rate)

        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, h1, h2):

        h1 = h1.squeeze(0)
        h2 = h2.squeeze(0)

        x1 = self.W1_attention(h1)
        x2 = self.W2_attention(h2)

        m1 = x1.size(0)
        m2 = x2.size(0)

        x1_expanded = x1.unsqueeze(1).expand(m1, m2, -1)
        x2_expanded = x2.unsqueeze(0).expand(m1, m2, -1)

        d = torch.tanh(x1_expanded + x2_expanded)
        alpha = torch.matmul(d, self.w).view(m1, m2)

        b1 = torch.mean(alpha, 1)
        p1 = torch.softmax(b1, 0)

        b2 = torch.mean(alpha, 0)
        p2 = torch.softmax(b2, 0)

        s1 = torch.matmul(x1.t(), p1).view(-1, 1)
        s2 = torch.matmul(x2.t(), p2).view(-1, 1)

        fusion_vec = torch.cat((s1, s2), 0).view(1, -1)

        return fusion_vec


class FiLMLayer(nn.Module):

    def __init__(self, prot_dim, token_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prot_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, token_dim * 2)
        )

    def forward(self, x, prot_vec):
        gb = self.mlp(prot_vec)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * x + beta


class SimpleSelfAttention(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
        super(SimpleSelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # 简单的线性变换，不使用多头
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.embed_dim)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        output = residual + self.dropout(attention_output)

        return output, attention_weights


class SelfAttentionModule(nn.Module):

    def __init__(self, p_dim, r_dim, params, vis):
        super(SelfAttentionModule, self).__init__()
        self.protein_linear = nn.Linear(p_dim, params.hidden_size)
        self.RNA_linear = nn.Linear(r_dim, params.hidden_size)
        self.vis = vis

        self.p_self_attention = SimpleSelfAttention(
            embed_dim=params.hidden_size,
            dropout=params.dropout_rate
        )
        self.r_self_attention = SimpleSelfAttention(
            embed_dim=params.hidden_size,
            dropout=params.dropout_rate
        )

    def forward(self, protein_features, rna_features):

        protein_embed = self.protein_linear(protein_features)
        rna_embed = self.RNA_linear(rna_features)
        protein_encoder, protein_attn = self.p_self_attention(protein_embed)
        rna_encoder, rna_attn = self.r_self_attention(rna_embed)

        return protein_encoder, rna_encoder, protein_attn, rna_attn


class PANIPmodel(nn.Module):
    def __init__(self, params,device):
        super(PANIPmodel, self).__init__()
        self.device = device
        self.dim = params.hidden_size
        self.film = FiLMLayer(prot_dim=params.hidden_size, token_dim=params.hidden_size)
        self.ca_former = CA_former(params, vis=True)
        self.seq_selfattention = SelfAttentionModule(960, 1024, params, vis=True)
        # self.str_selfattention = SelfAttentionModule(128, 128, params, use_pe=False, vis=True)
        self.head = nn.Linear(self.dim // 2, 1)
        self.pool = nn.AdaptiveAvgPool1d(128)
        # self.cl_loss = ContrastiveLoss(tau=0.1)
        self.proj_p = nn.Linear(params.hidden_size, params.hidden_size // 4)
        self.proj_r = nn.Linear(params.hidden_size, params.hidden_size // 4)
        self.site_predictor = nn.Sequential(
            nn.Conv1d(in_channels=params.hidden_size // 4, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
        )
        self.fusion = Fusion(hidden_dim=params.hidden_size,dropout_rate=params.dropout_rate)

    def augment_with_sites(self, x, site_scores, alpha=0.5, detach_hint=True):

        B, Lp, D = x.shape
        if site_scores.dim() == 1:
            site_scores = site_scores.unsqueeze(0).expand(B, Lp)
        prob = torch.sigmoid(site_scores)
        if detach_hint:
            prob = prob.detach()
        gate = (1 - alpha) + alpha * prob
        x_aug = x * gate.unsqueeze(-1)

        return x_aug

    def forward(self, batch):
        site_predicted, site_classes = [], []
        batch_y,ids= [],[]
        for item in batch:
            name = item["pdb_id"]
            protein_features = torch.as_tensor(item["protein_features"]).to(self.device)
            rna_features = torch.as_tensor(item["rna_features"]).to(self.device)
            ids.append(name)
            # g_protein_features = item["p_G_features"]
            # g_rna_features = item["r_G_features"]

            p_x, r_x, p_attn, r_attn = self.seq_selfattention(protein_features, rna_features)
            prot_vec = p_x.mean(dim=1)
            r_x = self.film(r_x, prot_vec)
            # g_p_x, g_r_x, g_p_attn, g_r_attn = self.g_transformer(g_protein_features.unsqueeze(0),g_rna_features.unsqueeze(0))

            p_x2, r_x2 = self.ca_former(p_x, r_x)
            q_p = self.proj_p(p_x2)
            q_r = self.proj_r(r_x2)
            p1 = self.site_predictor(q_p.transpose(1, 2)).squeeze()

            y = self.fusion(p_x2, r_x2)
            y = self.head(y)

            batch_y.append(y)

            p1 = torch.sigmoid(p1)
            site_classe = (p1 > 0.5).long()
            site_classes.append(site_classe.squeeze().cpu().numpy())
            site_predicted.append(p1.squeeze().detach().cpu().numpy())

            # p_pool = torch.mean(self.pool(p_x.squeeze(0)), dim=0, keepdim=True).unsqueeze(0)
            # r_pool = torch.mean(self.pool(r_x.squeeze(0)), dim=0, keepdim=True).unsqueeze(0)
            # pg_pool = torch.mean(self.pool(g_p_x.squeeze(0)), dim=0, keepdim=True).unsqueeze(0)
            # pr_pool = torch.mean(self.pool(g_r_x.squeeze(0)), dim=0, keepdim=True).unsqueeze(0)
            # p_seq.append(p_pool.squeeze(1))
            # r_seq.append(r_pool.squeeze(1))
            # p_str.append(pg_pool.squeeze(1))
            # r_str.append(pr_pool.squeeze(1))

        # p_con_loss = self.cl_loss(torch.cat(p_seq, dim=0), torch.cat(p_str, dim=0))
        # r_con_loss = self.cl_loss(torch.cat(r_seq, dim=0), torch.cat(r_str, dim=0))
        batch_ys = torch.cat(batch_y, dim=0).squeeze()
        predicted = torch.sigmoid(batch_ys)
        predicted_classes = (predicted > 0.5).long().detach().cpu().numpy()

        return ids,predicted_classes,predicted.detach().cpu().numpy(), site_classes, site_predicted,batch_ys.detach().cpu().numpy()



