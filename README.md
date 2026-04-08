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
./microgpt_mac                             # steps=500, temp=0.5, samples=20, eval_interval=100, eval_iters=100, model=32x4x2, block=16, lr=3e-3
./microgpt_mac 1000                        # steps=1000
./microgpt_mac 1000 0.7 30                 # 额外指定 temperature 和 sample 数
./microgpt_mac 2000 0.7 30 200 200         # 额外指定 eval_interval 和 eval_iters
./microgpt_mac 2000 0.7 20 200 100 64 4 2 24 0.001
                                          # 额外指定 n_embd n_head n_layer block_size learning_rate
```

请将语料放在当前目录的 `input.txt` 中（每行一条样本）。

## Guppy Mode

当前版本加入了一个受 `guppylm` 启发的最小 C 版功能集：

- 生成单轮鱼人格聊天语料
- 用现有字符级训练器训练聊天格式样本
- 训练后基于用户 prompt 做单轮回复

生成语料：

```bash
./microgpt_mac guppy-data                 # 生成 guppy_input.txt（默认 60000 条）
./microgpt_mac guppy-data my_guppy.txt 5000
```

生成的每一行样本格式为：

```text
<|user|> hi guppy <|assistant|> hello. the water feels nice today. <|end|>
```

训练 Guppy 语料：

```bash
./microgpt_mac 500 0.7 3 250 50 64 4 2 128 0.001 guppy_input.txt
```

训练后直接给一个 prompt：

```bash
./microgpt_mac 500 0.7 3 250 50 64 4 2 128 0.001 guppy_input.txt "hi guppy"
```

只使用已有 checkpoint 聊天：

```bash
./microgpt_mac chat "tell me a joke" guppy_input.txt ckpt_best.bin 0.7
```

说明：

- 这是在当前字符级 C 训练器上的第一版 Guppy 功能迁移，不是 `guppylm` 的完整复刻。
- 当前没有移植 BPE tokenizer、batch 训练、ONNX 导出和浏览器推理。
- 要得到更像样的聊天输出，通常需要更长训练步数和更大的 `block_size`。

## 说明

- 当前默认实验配置：`n_embd=32`, `n_head=4`, `n_layer=2`, `block_size=16`, `learning_rate=3e-3`。
- 如需回到最小教学配置，可使用：`./microgpt_mac 500 0.5 20 100 100 16 4 1 8 0.01`
- 核心数学计算使用 `Accelerate`（`cblas_sgemv/sdot/saxpy` 与 `vDSP`）。
- 2026-04-06 修复了长样本触发的 `prepare_example` 越界写问题，并修正了多层模型缓存索引；现在可以稳定跑更长训练和更大的 `n_layer`。

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
