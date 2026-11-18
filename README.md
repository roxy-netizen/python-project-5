# Project 5: Transformer/ViT Forward Propagation

## 项目简介
本项目为 Python 程序设计 Project 5 实验作业。
主要利用 NumPy 实现 Transformer/ViT 的前向传播流程，包括 Patch 分块、多头注意力机制、MLP 及残差连接等模块，并与参考实现进行数值一致性验证。

## 环境依赖
- Python 3.8+
- NumPy 1.x
- Matplotlib (用于可视化输出)

## 安装与配置
1. 克隆本仓库到本地：git clone https://github.com/roxy-netizen.git
cd 你的仓库名
2. 创建并激活虚拟环境（可选但推荐）： python -m venv venv
source venv/bin/activate  # Windows 使用: venv\Scripts\activate
3. 安装依赖：pip install numpy matplotlib
## 运行与复现结果
本项目的实验结果可通过以下命令一键复现验证：python check.py --stage all
运行后将会依次检查预处理、Patch 嵌入、Token 拼接、Transformer Block 及最终输出等 8 个阶段，并显示 PASS 及误差绝对值（所有误差均在容差范围内）。

## 核心实现说明
我的核心实现部分已在代码及实验报告插图中用**红色**高亮标出：
1. **Patch 嵌入与展平**：严格按 `(C, H, W)` 顺序展平，确保与 `conv1.weight` 布局对齐。
2. **多头拆分与合并**：拆分时 `reshape` 后接 `transpose(0, 2, 1, 3)`，合并时严格逆操作，防止 Token 交错。
3. **数值稳定性**：在多头注意力中使用 $QK^T / \sqrt{d_k}$ 缩放因子防止 Softmax 饱和。

## 文件结构
├── check.py          # 验收与数值一致性检查脚本
├── model.py          # 核心 ViT/Transformer 前向传播实现
├── figs/             # 实验报告及 README 所需图片
│   ├── vit_overview.png
│   └── output_heatmap.png
└── README.md
