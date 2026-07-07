"""FRM student — the ONLY trained part (Section 3).

Cross-attention with the gaze token as the QUERY and the global tokens as keys:

    q_vec = W_q @ gaze_tok                       # [d_h]
    s[j]  = (W_k @ G[j]) . q_vec / sqrt(d_h)     # relevance logit for candidate j

Multi-head: split d_h into H heads, average per-head logits. Fovea cells are
masked out (Section 3.1) BEFORE the softmax so the student ranks context only.

Two read-outs:
  r_dist = softmax(s over candidates)   # for listwise KL distillation (training)
  r      = sigmoid(s)                    # independent per-token, for KEEP/MERGE split

DEPTH>1 stacks additional cross-attn blocks (Exp 7 capacity sweep): the query is
refined by the attention-weighted context read-out, then re-queries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

NEG_INF = -1e9


class CrossAttnBlock(nn.Module):
    def __init__(self, d, d_h, n_heads):
        super().__init__()
        assert d_h % n_heads == 0, "d_h must be divisible by n_heads"
        self.n_heads, self.dh_head = n_heads, d_h // n_heads
        self.W_q = nn.Linear(d, d_h, bias=False)   # gaze  -> query
        self.W_k = nn.Linear(d, d_h, bias=False)   # global tokens -> keys
        self.W_v = nn.Linear(d, d_h, bias=False)   # values (for refining the query when DEPTH>1)
        self.scale = self.dh_head ** -0.5

    def logits(self, q_vec, G, cand_mask):
        """q_vec [B,d_h], G [B,K,d], cand_mask [B,K] bool -> s [B,K] (masked)."""
        B, K, _ = G.shape
        q = self.W_q(q_vec) if q_vec.dim() == 2 else q_vec        # [B,d_h]
        k = self.W_k(G)                                            # [B,K,d_h]
        q = q.view(B, self.n_heads, self.dh_head)                 # [B,H,dh]
        k = k.view(B, K, self.n_heads, self.dh_head)              # [B,K,H,dh]
        s = torch.einsum("bhd,bkhd->bhk", q, k) * self.scale      # [B,H,K]
        s = s.mean(dim=1)                                         # average heads -> [B,K]
        s = s.masked_fill(~cand_mask, NEG_INF)
        return s

    def readout(self, s, G, cand_mask):
        """Attention-weighted context vector (used to refine the query for DEPTH>1)."""
        a = torch.softmax(s, dim=-1).unsqueeze(-1)                # [B,K,1]
        v = self.W_v(G)                                           # [B,K,d_h]
        return (a * v).sum(dim=1)                                 # [B,d_h]


class FRM(nn.Module):
    def __init__(self, d, d_h=C.D_H, n_heads=C.N_HEADS, depth=C.DEPTH):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([CrossAttnBlock(d, d_h, n_heads) for _ in range(depth)])
        # for stacked depth, project the refined query (d_h) back to a query the next
        # block consumes (it expects a d_h vector already, so W_q there is identity-like);
        # we keep it simple: subsequent blocks take the readout directly as q_vec (d_h).
        self.d, self.d_h = d, d_h

    def forward(self, gaze_tok, G, cand_mask):
        """gaze_tok [B,d], G [B,K,d], cand_mask [B,K] bool -> s [B,K] logits."""
        # first block: query is the raw gaze token (projected inside via W_q)
        s = self.blocks[0].logits(gaze_tok, G, cand_mask)
        if self.depth == 1:
            return s
        # stacked depth (Exp 7): refine the query with the attention-weighted context
        # read-out, then re-query. Later blocks receive an already-d_h query.
        q = self.blocks[0].readout(s, G, cand_mask)               # [B,d_h]
        for blk in self.blocks[1:]:
            s = _logits_from_projected(blk, q, G, cand_mask)
            q = blk.readout(s, G, cand_mask)
        return s

    @torch.no_grad()
    def relevance(self, gaze_tok, G, cand_mask):
        """Deploy read-out: sigmoid relevance r [B,K] (fovea cells forced to 0)."""
        s = self.forward(gaze_tok, G, cand_mask)
        r = torch.sigmoid(s)
        return r * cand_mask.float()


def _logits_from_projected(blk, q_vec_dh, G, cand_mask):
    """Like CrossAttnBlock.logits but the query is ALREADY a d_h vector (stacked depth)."""
    B, K, _ = G.shape
    k = blk.W_k(G).view(B, K, blk.n_heads, blk.dh_head)
    q = q_vec_dh.view(B, blk.n_heads, blk.dh_head)
    s = torch.einsum("bhd,bkhd->bhk", q, k) * blk.scale
    s = s.mean(dim=1).masked_fill(~cand_mask, NEG_INF)
    return s


def masked_kl(teacher_p, student_logits, cand_mask):
    """Listwise KL over candidates: KL(p || softmax(student_logits)).

    teacher_p     : [B,K] non-negative teacher importance (0 on fovea cells)
    student_logits: [B,K] (fovea already -inf)
    cand_mask     : [B,K] bool
    """
    p = teacher_p * cand_mask.float()
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)          # renormalize over cand
    log_q = torch.log_softmax(student_logits, dim=-1)            # [B,K]
    kl = (p * (torch.log(p.clamp_min(1e-12)) - log_q)).sum(dim=1)
    return kl.mean()


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
