import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_row_orthogonal_matrix(out_dim, in_dim, seed=0):
    if out_dim > in_dim:
        raise ValueError("out_dim must be <= in_dim for row-orthogonal Q")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(in_dim, out_dim, generator=generator)
    q_col, _ = torch.linalg.qr(matrix, mode="reduced")
    return q_col.t().contiguous()


class FixedQLowRankHead(nn.Module):
    def __init__(self, in_dim, code_dim, rank=0, anchor_seed=0,
                 residual_scale=0.0, use_norm=True, train_scale=True,
                 train_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.code_dim = code_dim
        self.rank = int(rank)
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False) if use_norm else nn.Identity()
        self.register_buffer(
            "q",
            fixed_row_orthogonal_matrix(code_dim, in_dim, seed=anchor_seed))

        if train_scale:
            self.scale = nn.Parameter(torch.ones(code_dim))
        else:
            self.register_buffer("scale", torch.ones(code_dim))
        if train_bias:
            self.bias = nn.Parameter(torch.zeros(code_dim))
        else:
            self.register_buffer("bias", torch.zeros(code_dim))

        if self.rank > 0:
            self.lowrank_down = nn.Linear(in_dim, self.rank, bias=False)
            self.lowrank_up = nn.Linear(self.rank, code_dim, bias=False)
            nn.init.xavier_uniform_(self.lowrank_down.weight)
            nn.init.zeros_(self.lowrank_up.weight)
            self.initial_residual_scale = float(residual_scale)
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        else:
            self.lowrank_down = None
            self.lowrank_up = None
            self.register_buffer("residual_scale", torch.tensor(0.0))

    def reset_lowrank_residual(self):
        if self.lowrank_up is not None:
            nn.init.zeros_(self.lowrank_up.weight)
            self.residual_scale.data.fill_(self.initial_residual_scale)

    def forward(self, features):
        features = self.norm(features)
        code = F.linear(features, self.q)
        code = code * self.scale + self.bias
        if self.lowrank_down is not None:
            residual = self.lowrank_up(self.lowrank_down(features))
            code = code + self.residual_scale * residual
        return code


class FixedCodebookHead(nn.Module):
    def __init__(self, in_dim, code_dim, codebook_size=1024, anchor_seed=0,
                 temperature=1.0, use_norm=True):
        super().__init__()
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        self.in_dim = in_dim
        self.code_dim = code_dim
        self.codebook_size = int(codebook_size)
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False) if use_norm else nn.Identity()
        self.assignment = nn.Linear(in_dim, self.codebook_size)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(anchor_seed))
        codebook = torch.randn(self.codebook_size, code_dim, generator=generator)
        codebook = F.normalize(codebook, dim=1) * math.sqrt(code_dim)
        self.register_buffer("codebook", codebook)

    def forward(self, features):
        features = self.norm(features)
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        alpha = torch.softmax(self.assignment(features) / temperature, dim=1)
        return alpha.matmul(self.codebook.to(alpha.device, alpha.dtype))


def build_canonical_head(kind, in_dim, code_dim, anchor_seed=0,
                         lowrank_rank=0, lowrank_scale=0.0,
                         codebook_size=1024, codebook_temperature=1.0):
    if kind in (None, "", "none"):
        return nn.Linear(in_dim, code_dim)
    if kind == "fixed_q_lowrank":
        return FixedQLowRankHead(
            in_dim,
            code_dim,
            rank=lowrank_rank,
            anchor_seed=anchor_seed,
            residual_scale=lowrank_scale)
    if kind == "codebook":
        return FixedCodebookHead(
            in_dim,
            code_dim,
            codebook_size=codebook_size,
            anchor_seed=anchor_seed,
            temperature=codebook_temperature)
    raise ValueError(f"Unknown canonical head: {kind}")


class AnchorTargetBuilder:
    def __init__(self, target_type="none", code_dim=512, channel=2, nt=32,
                 nc=32, train_path=None, device="cpu"):
        self.target_type = target_type
        self.code_dim = code_dim
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.input_dim = channel * nt * nc
        self.device = device
        self.mean = None
        self.basis = None

        if target_type == "pca":
            if train_path is None:
                raise ValueError("train_path is required for PCA anchor target")
            self._fit_pca(train_path)
        elif target_type == "dct":
            self._build_dct_basis()
        elif target_type not in (None, "", "none"):
            raise ValueError(f"Unknown anchor target type: {target_type}")

    def _load_flat_data(self, train_path):
        data = torch.load(train_path, weights_only=True, map_location="cpu").float()
        if data.ndim == 2:
            data = data.view(-1, self.channel, self.nt, self.nc)
        return data.view(data.size(0), -1)

    def _fit_pca(self, train_path):
        flat = self._load_flat_data(train_path)
        self.mean = flat.mean(dim=0)
        centered = flat - self.mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:self.code_dim].contiguous()
        if basis.size(0) < self.code_dim:
            pad = torch.zeros(self.code_dim - basis.size(0), self.input_dim)
            basis = torch.cat([basis, pad], dim=0)
        self.basis = basis

    def _build_dct_basis(self):
        basis_1d_nt = self._dct_matrix(self.nt)
        basis_1d_nc = self._dct_matrix(self.nc)
        basis_2d = torch.kron(basis_1d_nt, basis_1d_nc)
        freq_order = []
        for i in range(self.nt):
            for j in range(self.nc):
                freq_order.append((i + j, i, j, i * self.nc + j))
        freq_order.sort()
        selected = [item[-1] for item in freq_order]

        rows = []
        per_channel = math.ceil(self.code_dim / self.channel)
        for ch in range(self.channel):
            offset = ch * self.nt * self.nc
            for idx in selected[:per_channel]:
                row = torch.zeros(self.input_dim)
                row[offset:offset + self.nt * self.nc] = basis_2d[idx]
                rows.append(row)
                if len(rows) == self.code_dim:
                    break
            if len(rows) == self.code_dim:
                break
        self.mean = torch.zeros(self.input_dim)
        self.basis = torch.stack(rows, dim=0).contiguous()

    @staticmethod
    def _dct_matrix(size):
        n = torch.arange(size, dtype=torch.float32)
        k = torch.arange(size, dtype=torch.float32).unsqueeze(1)
        mat = torch.cos(math.pi / size * (n + 0.5) * k)
        mat[0] *= math.sqrt(1.0 / size)
        mat[1:] *= math.sqrt(2.0 / size)
        return mat

    def __call__(self, x):
        if self.target_type in (None, "", "none"):
            return None
        flat = x.view(x.size(0), -1)
        mean = self.mean.to(flat.device, flat.dtype)
        basis = self.basis.to(flat.device, flat.dtype)
        return (flat - mean).matmul(basis.t())
