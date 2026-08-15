import pywt
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Custom LayerNorm that normalizes over the [C, N] dimensions."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps  # A small constant used to prevent division by zero
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine  # Whether to use learnable scale and bias parameters

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        # input: [B, C, N, T]
        # Compute the mean and variance over the channel dimension C and node dimension N,
        # while keeping dimensions for broadcasting.
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)

        # Optionally apply a learnable affine transformation.
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class Conv(nn.Module):
    """1x1 convolution followed by dropout for channel-wise feature transformation."""

    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    """Generates temporal embeddings from time-of-day and day-of-week information."""

    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time  # Number of time slots per day, e.g. 288 means one slot every 5 minutes

        # Time-of-day embedding.
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        # Day-of-week embedding, indexed from 0 to 6.
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # x: [B, T, N, input_dim], where x[..., 1] stores the time-of-day feature.
        day_emb = x[..., 1]
        # Use the time-of-day index from the final time step.
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1]

        # x[..., 2] stores the day-of-week information.
        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1]

        # Combine the time-of-day and day-of-week embeddings.
        tem_emb = time_day + time_week
        return tem_emb


class TemporalEmbedding2(nn.Module):
    """Alternative temporal embedding module for a different input dimension ordering."""

    def __init__(self, time, features):
        super(TemporalEmbedding2, self).__init__()
        self.time = time

        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # This implementation assumes that x uses a different dimension ordering
        # from the one expected by TemporalEmbedding.
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, :, -1] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, :, -1]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb.permute(0, 1, 3, 2)


class GatedUpdate(nn.Module):
    """Gated update unit that updates hidden state h using statistics from the current chunk."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        # The input concatenates h, mean, maximum, and minimum,
        # so its channel size is channels * 4.
        self.z = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.h_hat = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, h, c):
        # h: Historical hidden state [B, C, N, 1]
        # c: Statistical features of the current chunk [B, 3C, N, 1]
        inp = torch.cat([h, c], dim=1)
        z = torch.sigmoid(self.z(inp))  # Update gate controlling the balance between old and new information
        h_new = self.h_hat(inp)  # Candidate hidden state
        return (1 - z) * h + z * h_new


class DCTAconv(nn.Module):
    """Dual-branch temporal compression module for high- and low-frequency signals."""

    def __init__(
            self,
            channels=64,
            chunk_num_high=3,
            chunk_num_low=4,
            dropout=0.1
    ):
        super().__init__()
        self.chunk_num_high = chunk_num_high  # Number of chunks in the high-frequency branch
        self.chunk_num_low = chunk_num_low  # Number of chunks in the low-frequency branch

        # Use separate gated update units for the high- and low-frequency branches.
        self.update_s = GatedUpdate(channels, dropout)
        self.update_t = GatedUpdate(channels, dropout)

        # Branch gates determine how much high- or low-frequency information is used.
        self.branch_gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1))
        self.branch_gate2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1))

        # Output normalization and projection.
        self.out_norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _compress_one_branch(self, x, updater, chunk_num):
        # x: [B, C, N, T]
        # Split the sequence into multiple chunks along the temporal dimension T.
        chunks = torch.chunk(x, chunk_num, dim=-1)

        # Initialize the hidden state with the temporal mean of the first chunk.
        h = chunks[0].mean(dim=-1, keepdim=True)

        for ck in chunks:
            # Extract the mean, maximum, and minimum from each temporal chunk
            # to summarize its dynamic behavior.
            x_avg = ck.mean(dim=-1, keepdim=True)
            x_max = ck.max(dim=-1, keepdim=True)[0]
            x_min = ck.min(dim=-1, keepdim=True)[0]

            # Concatenate the statistical features and update the hidden state through the gate.
            other = torch.cat([x_avg, x_max, x_min], dim=1)
            h = updater(h, other)

        return h

    def forward(self, xs, xt, time_emb, last_time_emb):
        # xs: High-frequency features [B, C, N, T]
        # xt: Low-frequency features [B, C, N, T]
        # last_time_emb: Temporal embedding of the final time step [B, 2C, N, 1]

        hs = self._compress_one_branch(xs, self.update_s, self.chunk_num_high)  # High-frequency/variation information
        ht = self._compress_one_branch(xt, self.update_t, self.chunk_num_low)  # Low-frequency/trend information

        # Adaptively fuse the high- and low-frequency branches.
        g = torch.sigmoid(self.branch_gate(hs) + self.branch_gate2(ht))
        h = (1 - g) * hs + g * ht

        # Apply normalization and channel projection.
        h = self.out_proj(self.out_norm(h))

        # Concatenate the fused feature, final observation, and temporal embedding
        # to form the input to the spatiotemporal module.
        last_obs = xs[..., -1:] + xt[..., -1:]
        out = torch.cat([h, last_obs, last_time_emb], dim=1)
        return out


class SharedMemorySpatialattention(nn.Module):
    """Shared-memory spatial attention module for modeling relationships among nodes."""

    def __init__(
            self,
            device,
            d_model,
            head,
            num_nodes,
            seq_length=1,
            dropout=0.1,
            local_dim=64,
            mem_slots=16,
            mem_dim=64,
    ):
        super(SharedMemorySpatialattention, self).__init__()
        assert d_model % head == 0
        assert seq_length == 1, "This version is specialized for T=1."

        self.device = device
        self.d_model = d_model
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.local_dim = local_dim
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim

        self.dropout = nn.Dropout(p=dropout)
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )

        # Globally shared memory slots used to capture common patterns among nodes.
        self.SharedMemory = nn.Parameter(torch.randn(mem_slots, mem_dim))
        nn.init.xavier_uniform_(self.SharedMemory)

        # Use separate query projections for writing to and reading from memory.
        self.mem_write_q = nn.Linear(d_model, mem_dim, bias=False)
        self.mem_read_q = nn.Linear(d_model, mem_dim, bias=False)

        # Gated transformations used to control the strength of the memory output.
        self.g = Conv(d_model)
        self.t = Conv(d_model)
        self.conv = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

        # Node-adaptive bias that enhances node-specific representations.
        self.adaptive_bias = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(d_model, num_nodes, seq_length))
        )

    def forward(self, input):
        # input: [B, C, N, 1]
        B, C, N, T = input.shape
        assert T == 1

        # Convert to a node-major representation: [B, N, C].
        H = input.squeeze(-1).permute(0, 2, 1).contiguous()

        # Write stage: each node computes attention weights over the shared memory.
        q_write = self.mem_write_q(H)  # [B, N, mem_dim]
        k_mem_w = self.SharedMemory  # [M, mem_dim]
        score_write = torch.matmul(q_write, k_mem_w.t()) / math.sqrt(self.mem_dim)
        attn_write = F.softmax(score_write, dim=-1)  # [B, N, M]

        # Aggregate node features into memory_state: [B, M, C].
        memory_state = torch.einsum('bnm,bnc->bmc', attn_write, H)

        # Read stage: each node retrieves information relevant to itself from memory_state.
        q_read = self.mem_read_q(H)
        k_mem_r = self.SharedMemory
        score_read = torch.matmul(q_read, k_mem_r.t()) / math.sqrt(self.mem_dim)
        attn_read = F.softmax(score_read, dim=-1)
        HS = torch.einsum('bnm,bmc->bnc', attn_read, memory_state)  # [B, N, C]

        # Convert back to convolutional format: [B, C, N, 1].
        HS = HS.permute(0, 2, 1).unsqueeze(-1)

        # Apply gated output transformation.
        g = self.g(HS)
        t = torch.sigmoid(self.t(HS))
        HS = g * t

        HS = self.dropout(HS)
        HO = self.conv(HS) + HS * self.adaptive_bias
        HO = self.LayerNorm(HO)
        HO = self.dropout(HO)

        return HO


class WMSTA(nn.Module):
    """Main model: wavelet decomposition, dual-branch temporal compression, shared-memory spatial attention, and regression prediction."""

    def __init__(
            self,
            device,
            input_dim=3,
            channels=64,
            num_nodes=170,
            input_len=12,
            output_len=12,
            dropout=0.1,
    ):
        super().__init__()

        # Basic hyperparameters.
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 8

        # Determine the number of time slots per day based on the dataset's node count.
        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358 or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes > 200:
            time = 96

        # Numbers of temporal chunks used by the two DCTAconv branches.
        high = input_len // 3
        low = input_len // 4
        chunk_num_high = high
        chunk_num_low = low

        # The temporal embedding outputs channels * 2 features for later concatenation.
        self.Temb = TemporalEmbedding(time, channels * 2)

        self.DCTAconv = DCTAconv(
            channels=channels,
            chunk_num_high=chunk_num_high,
            chunk_num_low=chunk_num_low,
            dropout=dropout,
        )

        # Project the high- and low-frequency signals from wavelet decomposition
        # into the channels-dimensional feature space.
        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(1, channels, kernel_size=(1, 1))

        # The DCTA output concatenates h, last_obs, and time_emb: C + C + 2C = 4C.
        self.network_channel = channels * 4

        # Spatial attention module for modeling relationships among nodes.
        self.SpatialBlock = SharedMemorySpatialattention(
            device=device,
            d_model=self.network_channel,
            head=8,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=0.1,
            local_dim=64,
            mem_slots=32,
            mem_dim=64,
        ).to(device)

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )
        self.fc_st2 = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )
        # Output layer that maps the channel dimension to output_len future time steps.
        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        """Returns the total number of model parameters."""
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        # history_data: [B, input_dim, N, T]
        # Feature 0 is typically the primary target variable, such as traffic flow or speed.
        input_data = history_data[:, :1, :, :]  # [B, 1, N, T]

        # PyWavelets operates on CPU/NumPy arrays, so convert the tensor to NumPy first.
        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()

        # Two-level wavelet decomposition: coef[0] contains the low-frequency approximation
        # coefficients, while coef[1:] contains the high-frequency detail coefficients.
        coef = pywt.wavedec(residual_numpy, 'db1', level=2)
        coefl = [coef[0]] + [None] * (len(coef) - 1)  # Keep only the low-frequency coefficients
        coefh = [None] + coef[1:]  # Keep only the high-frequency coefficients

        # Reconstruct the low-frequency trend signal and high-frequency variation signal.
        low_freq_signal = pywt.waverec(coefl, 'db1')
        high_freq_signal = pywt.waverec(coefh, 'db1')

        # Convert back to tensors and move them to the specified device.
        low_freq_feature = torch.from_numpy(low_freq_signal).to(self.device)
        high_freq_signal = torch.from_numpy(high_freq_signal).to(self.device)

        # Use 1x1 convolutions to map single-channel signals into the channels-dimensional feature space.
        high_freq_feature = self.start_conv(high_freq_signal)  # [B, 64, N, T]
        low_freq_feature = self.start_conv2(low_freq_feature)  # [B, 64, N, T]

        # Rearrange dimensions for the temporal embedding module: [B, T, N, input_dim].
        history_data = history_data.permute(0, 3, 2, 1)
        temporal_last_embedding = self.Temb(history_data)  # [B, 2C, N, 1]

        # Compress the temporal dimension and fuse high- and low-frequency features,
        # producing an output of shape [B, 4C, N, 1].
        data_st = self.DCTAconv(
            high_freq_feature,
            low_freq_feature,
            1,
            temporal_last_embedding
        )

        # Apply spatial memory attention and gated residual enhancement.
        data_st = self.SpatialBlock(data_st) + \
                  self.fc_st(data_st) * torch.sigmoid(self.fc_st2(data_st))

        # Regression prediction: [B, output_len, N, 1].
        prediction = self.regression_layer(data_st)
        return prediction
