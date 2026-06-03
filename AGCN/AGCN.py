import torch
import torch.nn.functional as F
import torch.nn as nn

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x, node_embeddings, node_mask=None):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        logits = F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)))
        if node_mask is not None:
            m = node_mask.to(device=logits.device, dtype=logits.dtype).view(-1)
            if m.numel() != node_num:
                raise ValueError(f"node_mask length {m.numel()} != node_num {node_num}")
            active = m > 0.0
            if not torch.all(active):
                # Mask inactive columns so softmax only attends to active nodes.
                logits = logits.masked_fill(~active.view(1, -1), -1e9)
        supports = F.softmax(logits, dim=1)
        if node_mask is not None:
            # Zero out inactive rows and keep them as identity to avoid NaNs downstream.
            active = (node_mask.to(device=supports.device) > 0.0).view(-1)
            if not torch.all(active):
                supports = supports * active.view(-1, 1).to(supports.dtype)
                supports = supports + torch.eye(node_num, device=supports.device, dtype=supports.dtype) * (~active).to(supports.dtype).view(-1, 1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        if node_mask is not None:
            active = (node_mask.to(device=x_gconv.device) > 0.0).view(1, -1, 1)
            x_gconv = x_gconv * active.to(x_gconv.dtype)
        return x_gconv