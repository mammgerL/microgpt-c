# microgpt-mac-c

这是一个C版本的minigpt，并使用 macOS `Accelerate` 做计算加速。

## 已实现功能

- 从 `input.txt`（每行一条样本）构建字符级 tokenizer/vocab（含 `BOS`）。
- 当 `input.txt` 不存在时，自动尝试下载 Karpathy 的 `names.txt` 作为输入（网络可用时）。
- GPT 解码器前向结构：
  - token + position embedding
  - RMSNorm
  - 因果自注意力（causal self-attention）
  - MLP（`ReLU^2`）
  - 残差连接
  - LM head logits
- 完整训练流程：
  - next-token cross-entropy loss
  - 手写反向传播（attention + MLP + RMSNorm + embedding）
  - Adam 优化器（`beta1=0.9`, `beta2=0.95`, `eps=1e-8`）
  - cosine 学习率衰减
- `train/val` 划分 + 周期评估（`eval_interval`, `eval_iters`）
- 最优 checkpoint 保存/加载（`ckpt_best.bin`）
- 训练后自回归采样生成
- 训练损失日志：`losses.csv`（`step,loss`）
- 评估日志：`eval.csv`（`step,train_loss,val_loss,lr`）

## 构建与运行

```bash
make
./microgpt_mac                             # steps=500, temp=0.5, samples=20, eval_interval=100, eval_iters=100
./microgpt_mac 1000                        # steps=1000
./microgpt_mac 1000 0.7 30                 # 额外指定 temperature 和 sample 数
./microgpt_mac 2000 0.7 30 200 200         # 额外指定 eval_interval 和 eval_iters
```

请将语料放在当前目录的 `input.txt` 中（每行一条样本）。

## 说明

- 默认超参数对齐 Python gist 的小模型配置：`n_embd=16`, `n_head=4`, `n_layer=1`, `block_size=8`。
- 核心数学计算使用 `Accelerate`（`cblas_sgemv/sdot/saxpy` 与 `vDSP`）。

## 性能

本节基准环境：
- 日期：2026-02-12
- 系统：macOS 26.2（`Darwin 25.2.0`, `arm64`）
- 数据集：Karpathy `names.txt`（`input.txt`，约 3.2 万条名字）
- 测试命令模式：`./microgpt_mac <steps> 0.7 1 <steps> 100`
- 说明：`eval_interval=<steps>` 表示训练结束时仅评估一次

训练吞吐（程序内部 `CLOCK_MONOTONIC` 计时）：
- `500` steps：`train_time_sec=0.008959`, `train_steps_per_sec=55809.80`
- `1000` steps：`train_time_sec=0.016074`, `train_steps_per_sec=62212.27`
- `2000` steps：`train_time_sec=0.027773`, `train_steps_per_sec=72012.39`

可重复性检查（`2000` steps，连续 3 次）：
- Run 1：`75038.46 steps/s`
- Run 2：`85397.10 steps/s`
- Run 3：`90432.27 steps/s`
- 平均：`83622.61 steps/s`
