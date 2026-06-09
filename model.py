import pywt
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """自定义 LayerNorm：对 [C, N] 两个维度做归一化。"""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps  # 防止除零的小常数
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine  # 是否使用可学习的缩放和平移参数

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        # input: [B, C, N, T]
        # 对通道维 C 和节点维 N 求均值/方差，保留维度便于广播
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)

        # 可选：加入可学习的仿射变换
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class Conv(nn.Module):
    """1x1 卷积 + Dropout，用于通道特征变换。"""
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    """根据一天内时间片和星期信息生成时间嵌入。"""
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time  # 一天被划分的时间片数量，例如 288 表示 5 分钟一个片段

        # 一天内时间片 embedding
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        # 星期 embedding，范围为 0~6
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # x: [B, T, N, input_dim]，其中 x[..., 1] 为 day/time-of-day 特征
        day_emb = x[..., 1]
        # 取最后一个时间步的 day embedding 索引
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1]

        # x[..., 2] 为星期信息
        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1]

        # 合并日内时间和星期信息
        tem_emb = time_day + time_week
        return tem_emb


class TemporalEmbedding2(nn.Module):
    """另一种索引方式的时间嵌入模块，适配不同输入维度排列。"""
    def __init__(self, time, features):
        super(TemporalEmbedding2, self).__init__()
        self.time = time

        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # 这里默认 x 的维度排列与 TemporalEmbedding 不同
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
    """门控更新单元：用当前分块统计信息更新隐藏状态 h。"""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        # 输入由 h、均值、最大值、最小值拼接而成，所以通道数为 channels * 4
        self.z = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.h_hat = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, h, c):
        # h: 历史隐藏状态 [B, C, N, 1]
        # c: 当前分块统计特征 [B, 3C, N, 1]
        inp = torch.cat([h, c], dim=1)
        z = torch.sigmoid(self.z(inp))  # 更新门，控制新旧信息比例
        h_new = self.h_hat(inp)         # 候选隐藏状态
        return (1 - z) * h + z * h_new


class DCTAconv(nn.Module):
    """双分支时间压缩模块：分别压缩高频/低频信号，再通过门控融合。"""
    def __init__(
        self,
        channels=64,
        chunk_num_high=3,
        chunk_num_low=4,
        dropout=0.1
    ):
        super().__init__()
        self.chunk_num_high = chunk_num_high  # 高频分支切分块数
        self.chunk_num_low = chunk_num_low    # 低频分支切分块数

        # 高频/低频分支各自使用独立的门控更新器
        self.update_s = GatedUpdate(channels, dropout)
        self.update_t = GatedUpdate(channels, dropout)

        # 分支融合门，决定更多采用高频还是低频信息
        self.branch_gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1))
        self.branch_gate2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1))

        # 输出归一化和投影
        self.out_norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _compress_one_branch(self, x, updater, chunk_num):
        # x: [B, C, N, T]
        # 沿时间维 T 将序列切成多个块
        chunks = torch.chunk(x, chunk_num, dim=-1)

        # 用第一个块的时间均值初始化隐藏状态
        h = chunks[0].mean(dim=-1, keepdim=True)

        for ck in chunks:
            # 对每个时间块提取均值、最大值、最小值，概括该块的动态变化
            x_avg = ck.mean(dim=-1, keepdim=True)
            x_max = ck.max(dim=-1, keepdim=True)[0]
            x_min = ck.min(dim=-1, keepdim=True)[0]

            # 拼接统计特征，并通过门控更新隐藏状态
            other = torch.cat([x_avg, x_max, x_min], dim=1)
            h = updater(h, other)

        return h

    def forward(self, xs, xt, time_emb, last_time_emb):
        # xs: 高频特征 [B, C, N, T]
        # xt: 低频特征 [B, C, N, T]
        # last_time_emb: 最后一个时间步的时间嵌入 [B, 2C, N, 1]

        hs = self._compress_one_branch(xs, self.update_s, self.chunk_num_high)  # 高频/波动信息
        ht = self._compress_one_branch(xt, self.update_t, self.chunk_num_low)   # 低频/趋势信息

        # 自适应融合高频和低频分支
        g = torch.sigmoid(self.branch_gate(hs) + self.branch_gate2(ht))
        h = (1 - g) * hs + g * ht

        # 归一化与通道投影
        h = self.out_proj(self.out_norm(h))

        # 拼接融合特征、最后观测值和时间嵌入，形成时空模块输入
        last_obs = xs[..., -1:] + xt[..., -1:]
        out = torch.cat([h, last_obs, last_time_emb], dim=1)
        return out


class SharedMemorySpatialattention(nn.Module):
    """共享记忆空间注意力模块：通过可学习 memory 建模节点间关系。"""
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

        # 全局共享记忆槽，用于捕获节点间共性模式
        self.SharedMemory = nn.Parameter(torch.randn(mem_slots, mem_dim))
        nn.init.xavier_uniform_(self.SharedMemory)

        # 写入 memory 和读取 memory 使用不同的 query 投影
        self.mem_write_q = nn.Linear(d_model, mem_dim, bias=False)
        self.mem_read_q = nn.Linear(d_model, mem_dim, bias=False)

        # 门控变换，用于控制 memory 输出强度
        self.g = Conv(d_model)
        self.t = Conv(d_model)
        self.conv = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

        # 节点自适应偏置，增强不同节点的个性化表达
        self.adaptive_bias = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(d_model, num_nodes, seq_length))
        )

    def forward(self, input):
        # input: [B, C, N, 1]
        B, C, N, T = input.shape
        assert T == 1

        # 转为节点优先表示：[B, N, C]
        H = input.squeeze(-1).permute(0, 2, 1).contiguous()

        # 写入阶段：每个节点根据共享 memory 计算注意力权重
        q_write = self.mem_write_q(H)  # [B, N, mem_dim]
        k_mem_w = self.SharedMemory    # [M, mem_dim]
        score_write = torch.matmul(q_write, k_mem_w.t()) / math.sqrt(self.mem_dim)
        attn_write = F.softmax(score_write, dim=-1)  # [B, N, M]

        # 将节点特征聚合写入 memory_state：[B, M, C]
        memory_state = torch.einsum('bnm,bnc->bmc', attn_write, H)

        # 读取阶段：节点从 memory_state 中读取与自身相关的信息
        q_read = self.mem_read_q(H)
        k_mem_r = self.SharedMemory
        score_read = torch.matmul(q_read, k_mem_r.t()) / math.sqrt(self.mem_dim)
        attn_read = F.softmax(score_read, dim=-1)
        HS = torch.einsum('bnm,bmc->bnc', attn_read, memory_state)  # [B, N, C]

        # 转回卷积格式：[B, C, N, 1]
        HS = HS.permute(0, 2, 1).unsqueeze(-1)

        # 门控输出
        g = self.g(HS)
        t = torch.sigmoid(self.t(HS))
        HO = g * t


        HO = self.dropout(HO)
        HO = self.conv(HO) + HO * self.adaptive_bias
        HO = self.LayerNorm(HO)
        HO = self.dropout(HO)

        return HO


class WMSTA(nn.Module):
    """主模型：小波分解 + 双分支时间压缩 + 共享记忆空间注意力 + 回归预测。"""
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

        # 基本超参数
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 8

        # 根据数据集节点数确定一天内时间片数量
        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358 or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes > 200:
            time = 96

        # 高频/低频时间块数量，分别用于 DCTAconv 两个分支
        high = input_len // 2
        low = input_len // 4
        chunk_num_high = high
        chunk_num_low = low

        # 时间嵌入输出 channels * 2，后续与特征拼接
        self.Temb = TemporalEmbedding(time, channels * 2)

        self.DCTAconv = DCTAconv(
            channels=channels,
            chunk_num_high=chunk_num_high,
            chunk_num_low=chunk_num_low,
            dropout=dropout,
        )

        # 小波分解后的高频/低频信号分别升维到 channels
        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(1, channels, kernel_size=(1, 1))

        # DCTA 输出由 h、last_obs、time_emb 拼接得到：C + C + 2C = 4C
        self.network_channel = channels * 4

        # 空间注意力模块，建模节点间关系
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

        # 输出层：将通道维映射为未来 output_len 个时间步预测
        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        """统计模型参数总量。"""
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        # history_data: [B, input_dim, N, T]
        # 第 0 个特征通常是目标交通流/速度等主变量
        input_data = history_data[:, :1, :, :]  # [B, 1, N, T]

        # PyWavelets 运行在 CPU/numpy 上，因此先从 Tensor 转为 numpy
        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()

        # 二层小波分解：coef[0] 为低频近似系数，coef[1:] 为高频细节系数
        coef = pywt.wavedec(residual_numpy, 'db1', level=2)
        coefl = [coef[0]] + [None] * (len(coef) - 1)  # 仅保留低频系数
        coefh = [None] + coef[1:]                     # 仅保留高频系数

        # 小波重构得到低频趋势信号和高频波动信号
        low_freq_signal = pywt.waverec(coefl, 'db1')
        high_freq_signal = pywt.waverec(coefh, 'db1')

        # 转回 Tensor 并移动到指定设备
        low_freq_feature = torch.from_numpy(low_freq_signal).to(self.device)
        high_freq_signal = torch.from_numpy(high_freq_signal).to(self.device)

        # 通过 1x1 卷积将单通道信号映射到 channels 维特征空间
        high_freq_feature = self.start_conv(high_freq_signal)  # [B, 64, N, T]
        low_freq_feature = self.start_conv2(low_freq_feature)  # [B, 64, N, T]

        # 调整维度给时间嵌入模块使用：[B, T, N, input_dim]
        history_data = history_data.permute(0, 3, 2, 1)
        temporal_last_embedding = self.Temb(history_data)  # [B, 2C, N, 1]

        # 时间压缩与高低频融合，输出 [B, 4C, N, 1]
        data_st = self.DCTAconv(
            high_freq_feature,
            low_freq_feature,
            1,
            temporal_last_embedding
        )

        # 空间记忆注意力 + 门控残差增强
        data_st = self.SpatialBlock(data_st) + \
                  self.fc_st(data_st)

        # 回归预测：[B, output_len, N, 1]
        prediction = self.regression_layer(data_st)
        return prediction
