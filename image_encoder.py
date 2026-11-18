"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        # 加载 QKV 联合投影的权重和偏置 [3D, D] -> [D, D]
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']  # [3D]
        # 加载输出投影的权重和偏置 [D,D] -> [D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        # 计算每个头的特征维度 (D/H)
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """将特征维度拆分为多个头，并调整维度顺序"""
        # 输入 x 形状: [B, N, D] (Batch, Token数, 特征维度)
        B, N, D = x.shape
        # 重塑为 [B, N, H, D/H]，然后转置为 [B, H, N, D/H] 以便后续头间并行计算
        x = x.reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        return x

    def __call__(self, x, causal=False):
        # 线性投影生成 Q, K, V (权重形状为 [out, in]，故转置后相乘)
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        # 按特征维度切分 Q, K, V 并拆分多头
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2 * self.width])
        v = self.split_heads(qkv[:, :, 2 * self.width:])

        # 计算注意力分数: Q * K^T / sqrt(d_k)，缩放防止梯度消失
        # q,k,v 形状: [B, H, N, 64]，scores 形状: [B, H, N, N]
        scores = np.float32(q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim))

        # 如果是因果掩码（用于文本生成），则掩盖未来信息
        if causal:
            scores = causal_mask(scores)
        # 计算注意力权重（Softmax）
        coefficients = attention_weights(scores)

        # 加权求和 Value，得到上下文信息
        context = coefficients @ v  # [B, 12, N, 64]
        B_, H_, N_, _ = context.shape
        # 合并多头：转置回 [B, N, H, D/H] 并重塑为 [B, N, D]
        merged = context.transpose(0, 2, 1, 3).reshape(B_, N_, self.width)  # [B, N, 768]

        # 输出投影层
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        # 初始化前后层归一化
        self.norm1 = LayerNorm(weights[prefix + '.ln_1.weight'], weights[prefix + '.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix + '.ln_2.weight'], weights[prefix + '.ln_2.bias'])
        # 初始化多头注意力机制
        self.attention = MultiHeadAttention(weights, prefix + '.attn', heads)
        # 初始化 MLP (多层感知机) 的两层线性映射权重和偏置 [4D,D] 和 [D,4D]
        self.fc1_weight = weights[prefix + '.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix + '.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix + '.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix + '.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # 第一部分：层归一化 -> 多头注意力 -> 残差连接
        update = self.attention(self.norm1(x), causal=causal)
        residual = x + update  # 第一个残差连接

        # 第二部分：层归一化 -> MLP -> 残差连接
        u = self.norm2(residual)
        # MLP 计算: GELU激活后接线性投影
        mlp = self.gelu(u @ self.fc1_weight.T + self.fc1_bias) @ self.fc2_weight.T + self.fc2_bias
        return mlp + residual  # 第二个残差连接


class ImageEncoder:
    def __init__(self, weights):
        # 卷积核重塑为线性投影矩阵 [768, 3072]
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        # CLS 令牌嵌入 [1, 1, 768]
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        # 位置编码 [50, 768]
        self.position = weights['visual.positional_embedding']  # [50,768]
        # 预处理和后处理层归一化
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        # 12 层 Transformer Block 堆叠
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        # 最终特征投影矩阵 [768, 512]
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """将输入图像切分为 Patch（图像块）"""
        # 输入形状: [B, 3, 224, 224]
        B, C, H, W = images.shape
        patches = 32  # Patch 大小
        h, w = H // patches, W // patches  # 计算高宽方向上的 Patch 数量 (7x7=49)

        # 1. 分离 Patch 网格轴: [B, C, h, 32, w, 32]
        images = images.reshape(B, C, h, patches, w, patches)
        # 2. 调整维度顺序 (先高后宽，再通道和Patch内部): [B, h, w, C, 32, 32]
        images = images.transpose(0, 2, 4, 1, 3, 5)
        # 3. 展平每个 Patch 的特征: [B, 49, 3072]
        images = images.reshape(B, h * w, C * patches * patches)
        return images

    def embed(self, images):
        # 将图像切分并投影到嵌入空间 [B, 49, 3072] -> [B, 49, 768]
        patches = self.patchify(images)
        patches = patches @ self.patch_weight.T  # 无偏置的线性投影

        # 为每张图像复制一个 CLS 令牌，并拼接到 Patch 序列最前面
        cls = np.tile(self.cls, (images.shape[0], 1, 1))
        x = np.concatenate([cls, patches], axis=1)  # [B, 50, 768]

        # 加上可学习的位置编码
        return x + self.position

    def encode_tokens(self, images):
        """组装 Token 输入并依次通过 12 个 Transformer 块"""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """提取图像特征：取 CLS Token，归一化并投影"""
        # 获取 Transformer 输出 [B, 50, 768]
        tokens = self.encode_tokens(images)
        # 提取 CLS Token (代表整张图像的特征) [B, 768]
        cls_token = tokens[:, 0, :]
        # 后层归一化 + 线性投影到 512 维 [B, 512]
        cls_token = self.post_norm(cls_token) @ self.projection

        # 返回 L2 归一化后的单位向量
        return unit_vector(cls_token)
