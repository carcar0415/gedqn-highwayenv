# model/clmha_model.py
# CLMHA 模型：CNN + LSTM + 多头注意力，用于预测车辆冲突风险等级

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):  # x: [B, T, H]
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class CLMHA(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CLMHA, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.attn = MultiHeadAttention(hidden_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):  # 输入 x: [B, T, F]
        if x.shape[-1] < 8:
            pad_dim = 8 - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_dim))  # 在最后一维填充0
            # print(f"⚠️ 自动补齐状态维度：新x.shape = {x.shape}")

        B, T, F = x.size()
        x = x.reshape(B * T, F).unsqueeze(-1)       # ✅ [B*T, F, 1] → 通道数 F=input_dim
        cnn_out = self.cnn(x).squeeze(-1)           # ✅ [B*T, 64]
        cnn_out = cnn_out.reshape(B, T, -1)         # ✅ [B, T, 64]
        lstm_out, _ = self.lstm(cnn_out)            # ✅ [B, T, H]
        attn_out = self.attn(lstm_out)              # ✅ [B, T, H]
        output = self.fc(attn_out[:, -1, :])        # ✅ [B, 1]
        return output

