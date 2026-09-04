# model.py
"""
Large-Scale Representation Learning on Graphs via Bootstrapping, ICLR 2022
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for node embeddings.
    Works on single graph or batched (disjoint union) graph.
    """
    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_out_norm: bool = True,   # ✅ 可开关
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hid_dim, "mean"))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_dim, hid_dim, "mean"))
        self.layers.append(SAGEConv(hid_dim, out_dim, "mean"))
        self.dropout = dropout

        # ✅ LayerNorm for hidden layers
        self.norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers - 1)])

        # ✅ LayerNorm for output layer (helps reduce clean-eval anisotropy)
        self.use_out_norm = use_out_norm
        self.out_norm = nn.LayerNorm(out_dim) if use_out_norm else nn.Identity()

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != last:
                h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = self.out_norm(h)  # ✅ normalize final embeddings

        return h  # (N, out_dim)



class Predictor(nn.Module):
    """
    BYOL/BGRL predictor head (student-only).
    """
    def __init__(self, dim: int, hid_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def byol_cosine_loss(p: torch.Tensor, z_target: torch.Tensor, center: bool = True) -> torch.Tensor:
    """
    BGRL/BYOL loss: 2 - 2*cosine similarity (averaged).
    """
    if center:
        # ✅ remove batch-wise common component (helps fight anisotropy)
        p = p - p.mean(dim=0, keepdim=True)
        z_target = z_target - z_target.mean(dim=0, keepdim=True)

    p = F.normalize(p, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    return 2 - 2 * (p * z_target).sum(dim=-1).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float = 0.99):
    """
    EMA update: teacher <- tau*teacher + (1-tau)*student
    """
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=(1 - tau))
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


@torch.no_grad()
def drop_edge(g, p):
    if p <= 0:
        return g
    num_edges = g.num_edges()
    device = g.device
    mask = torch.rand(num_edges, device=device) > p
    drop_eids = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    return dgl.remove_edges(g, drop_eids)


@torch.no_grad()
def drop_feature(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Feature dropout augmentation.
    """
    if p <= 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask

@torch.no_grad()
def drop_feature_dim(x: torch.Tensor, p: float) -> torch.Tensor:
    """Mask feature dimensions (columns) instead of individual elements."""
    if p <= 0:
        return x
    d = x.shape[1]
    keep = (torch.rand(d, device=x.device) > p).float()  # [D]
    return x * keep  # broadcast to [N, D]


class BGRL(nn.Module):
    """
    BGRL wrapper:
    - student encoder + predictor (trainable)
    - teacher encoder (EMA, no grad)
    """
    def __init__(self,
                 in_dim: int,
                 hid_dim: int = 256,
                 out_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 pred_hid_dim: int = 256,
                 tau: float = 0.99):
        super().__init__()
        self.tau = tau

        self.student = GraphSAGEEncoder(in_dim, hid_dim, out_dim, num_layers, dropout)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.predictor = Predictor(out_dim, pred_hid_dim)

    def forward(self, g1: dgl.DGLGraph, x1: torch.Tensor,
                      g2: dgl.DGLGraph, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric BGRL loss on two augmented views.
        """
        # student
        z1_s = self.student(g1, x1)
        z2_s = self.student(g2, x2)
        p1 = self.predictor(z1_s)
        p2 = self.predictor(z2_s)

        # teacher (stop-grad)
        with torch.no_grad():
            z1_t = self.teacher(g1, x1)
            z2_t = self.teacher(g2, x2)

        loss = byol_cosine_loss(p1, z2_t.detach()) + byol_cosine_loss(p2, z1_t.detach())
        return loss

    @torch.no_grad()
    def update_teacher(self):
        ema_update(self.teacher, self.student, tau=self.tau)

    @torch.no_grad()
    def encode(self, g: dgl.DGLGraph, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Get node embeddings from the student encoder (for retrieval / export).
        """
        self.student.eval()
        z = self.student(g, x)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z