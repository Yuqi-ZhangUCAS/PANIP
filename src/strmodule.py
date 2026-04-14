import copy
import csv
import math
from dgl.nn.pytorch import HeteroGraphConv
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Softmax, LayerNorm, CrossEntropyLoss
import torch as th
from torch import nn

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GATDotConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__
    """

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, residual=False, activation=None,
                 allow_zero_in_degree=False, bias=True):
        super(GATDotConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # Linear transformations for source and destination nodes
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # Dropout layers for feature and attention
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Residual connection
        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        # Bias term (optional)
        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        """
        Forward pass of the GATConv layer.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            graph.srcdata.update({"ft": feat_src})
            graph.dstdata.update({"ft": feat_dst})
            graph.apply_edges(fn.u_dot_v("ft", "ft", "e"))

            e = graph.edata.pop("e")  #
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))  # softmax  dropout

            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst

class ResiSC_Encoder(nn.Module):
    """
    The following code is improved based on JmcPPI. Paper link:
    https://doi.org/10.48550/arXiv.2503.04650
    """
    def __init__(self, param, input_dim, data_loader):
        super(ResiSC_Encoder, self).__init__()

        self.data_loader = data_loader
        self.num_layers = param.resid_num_layers
        self.dropout = nn.Dropout(param.dropout_ratio)
        self.gnnlayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.param = param
        self.input_dim = input_dim

        self.fc_dim = nn.Linear(self.input_dim, param.resid_hidden_dim)

        for i in range(self.num_layers):
            self.gnnlayers.append(HeteroGraphConv({'SEQ': GATDotConv(param.resid_hidden_dim,
                                                                     param.resid_hidden_dim,
                                                                     param.num_heads1, param.dropout_ratio,
                                                                     allow_zero_in_degree=True),  #
                                                   'STR_KNN': GATDotConv(param.resid_hidden_dim,
                                                                         param.resid_hidden_dim,
                                                                         param.num_heads1,
                                                                         param.dropout_ratio,
                                                                         allow_zero_in_degree=True),
                                                   'STR_DIS': GATDotConv(param.resid_hidden_dim,
                                                                         param.resid_hidden_dim,
                                                                         param.num_heads1,
                                                                         param.dropout_ratio,
                                                                         allow_zero_in_degree=True)},
                                                  aggregate='sum'))
            self.fcs.append(nn.Linear(param.resid_hidden_dim, param.resid_hidden_dim))
            self.norms.append(nn.BatchNorm1d(param.resid_hidden_dim))

    def forward(self):
        prot_embed_list = []

        self.eval()
        with torch.no_grad():
            for iter, batch_graph in enumerate(self.data_loader):
                batch_graph = batch_graph.to(device)

                x = batch_graph.ndata['x'].to(device)
                batch_graph.ndata['h'] = self.encoding(batch_graph, x)
                num_nodes_per_graph = batch_graph.batch_num_nodes()

                all_node_features = batch_graph.ndata['h']
                start_idx = 0
                for n_nodes in num_nodes_per_graph:
                    graph_node_features = all_node_features[start_idx:start_idx + n_nodes]
                    prot_embed_list.append(graph_node_features.detach().cpu())
                    start_idx += n_nodes

                del batch_graph, x, all_node_features
                torch.cuda.empty_cache()

        return prot_embed_list

    def encoding(self, batch_graph, x):

        x = self.fc_dim(x)

        for l, layer in enumerate(self.gnnlayers):
            x = torch.mean(layer(batch_graph, {'amino_acid': x})['amino_acid'], dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x


class ResiSC_Decoder(nn.Module):
    """
        The following code is improved based on JmcPPI. Paper link:
        https://doi.org/10.48550/arXiv.2503.04650
    """
    def __init__(self, param, input_dim):
        super(ResiSC_Decoder, self).__init__()
        self.num_layers = param.resid_num_layers
        self.dropout = nn.Dropout(param.dropout_ratio)
        self.gnnlayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for i in range(self.num_layers):
            self.gnnlayers.append(
                HeteroGraphConv({
                    'SEQ': GATDotConv(param.resid_hidden_dim, param.resid_hidden_dim, param.num_heads1,
                                      param.dropout_ratio, allow_zero_in_degree=True),
                    'STR_KNN': GATDotConv(param.resid_hidden_dim, param.resid_hidden_dim, param.num_heads1,
                                          param.dropout_ratio, allow_zero_in_degree=True),
                    'STR_DIS': GATDotConv(param.resid_hidden_dim, param.resid_hidden_dim, param.num_heads1,
                                          param.dropout_ratio, allow_zero_in_degree=True),
                }, aggregate='sum')
            )
            self.fcs.append(nn.Linear(param.resid_hidden_dim, param.resid_hidden_dim))
            self.norms.append(nn.BatchNorm1d(param.resid_hidden_dim))

        self.fc_dim = nn.Linear(param.resid_hidden_dim, input_dim)

        self.edge_decoders = nn.ModuleDict({
            etype: nn.Bilinear(param.resid_hidden_dim, param.resid_hidden_dim, 1)
            for etype in ['SEQ', 'STR_KNN', 'STR_DIS']
        })

    def decoding(self, batch_graph, x):
        for l, gnn in enumerate(self.gnnlayers):
            x = torch.mean(gnn(batch_graph, {'amino_acid': x})['amino_acid'], dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers - 1:
                x = self.dropout(x)
        node_recon = self.fc_dim(x)

        edge_preds = {}
        for etype, decoder in self.edge_decoders.items():
            src, dst = batch_graph.edges(etype=etype)
            h_src = x[src]
            h_dst = x[dst]
            logits = decoder(h_src, h_dst).squeeze(-1)
            edge_preds[etype] = torch.sigmoid(logits)

        return node_recon, edge_preds


class RecNet(nn.Module):
    """
    The following code is improved based on JmcPPI. Paper link:
    https://doi.org/10.48550/arXiv.2503.04650
    """
    def __init__(self, param, data_loader):
        super(RecNet, self).__init__()

        self.param = param
        first_batch_graph = next(iter(data_loader))
        first_batch_graph.to(device)
        x = first_batch_graph.ndata['x']
        input_dim = x.shape[1]

        self.Encoder = ResiSC_Encoder(param, input_dim, data_loader).to(device)
        self.Decoder = ResiSC_Decoder(param, input_dim).to(device)

        self.edge_embedding = nn.Embedding(3, param.resid_hidden_dim)

    def forward(self, batch_graph):
        batch_graph = batch_graph.to(device)
        x = batch_graph.ndata['x'].to(device)
        z = self.Encoder.encoding(batch_graph, x)
        recon_x, edge_preds = self.Decoder.decoding(batch_graph, z)

        recon_loss = F.mse_loss(recon_x, batch_graph.ndata['x'])
        edge_recon_loss = 0.0
        for etype, pred in edge_preds.items():
            pos_label = torch.ones_like(pred)
            num_pos = pred.shape[0]

            all_src = torch.randint(0, z.size(0), (num_pos,), device=z.device)
            all_dst = torch.randint(0, z.size(0), (num_pos,), device=z.device)
            neg_idx = torch.randperm(all_src.size(0))[:num_pos]
            neg_src, neg_dst = all_src[neg_idx], all_dst[neg_idx]
            neg_logits = self.Decoder.edge_decoders[etype](z[neg_src], z[neg_dst]).squeeze(-1)
            neg_pred = torch.sigmoid(neg_logits)
            neg_label = torch.zeros_like(neg_pred)

            preds = torch.cat([pred, neg_pred], dim=0)
            labels = torch.cat([pos_label, neg_label], dim=0)
            edge_recon_loss += F.binary_cross_entropy(preds, labels)

        edge_recon_loss = edge_recon_loss / len(edge_preds)

        mask_x = batch_graph.ndata['x'].clone()
        num_masked_rows = int(self.param.rec_mask_ratio * mask_x.shape[0])
        mask_index = torch.randperm(mask_x.shape[0])[:num_masked_rows]
        mask_x[mask_index] = 0.0
        mask_z = self.Encoder.encoding(batch_graph, mask_x)

        mask_recon_x, mask_edge_preds = self.Decoder.decoding(batch_graph, mask_z)
        mask_edge_recon_loss = 0.0
        for etype, pred in mask_edge_preds.items():
            pos_label = torch.ones_like(pred)
            num_pos = pred.shape[0]
            all_src, all_dst = torch.repeat_interleave(torch.arange(mask_z.size(0)), mask_z.size(0)), torch.arange(
                mask_z.size(0)).repeat(mask_z.size(0))
            neg_idx = torch.randperm(all_src.size(0))[:num_pos]
            neg_src, neg_dst = all_src[neg_idx], all_dst[neg_idx]
            neg_logits = self.Decoder.edge_decoders[etype](mask_z[neg_src], mask_z[neg_dst]).squeeze(-1)
            neg_pred = torch.sigmoid(neg_logits)
            neg_label = torch.zeros_like(neg_pred)

            preds = torch.cat([pred, neg_pred], dim=0)
            labels = torch.cat([pos_label, neg_label], dim=0)
            mask_edge_recon_loss += F.binary_cross_entropy(preds, labels)

        mask_edge_recon_loss = mask_edge_recon_loss / len(mask_edge_preds)

        x = F.normalize(mask_recon_x[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index],
                        p=2,
                        dim=-1,
                        eps=1e-12)
        mask_recon_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param.sce_scale))

        return z, recon_loss + edge_recon_loss, mask_edge_recon_loss + (mask_recon_loss.sum() / (
                mask_recon_loss.shape[0] + 1e-12))





