# microgpt-mac-c

这是一个C版本的minigpt，并使用 macOS `Accelerate` 做计算加速。

当前和 `guppylm` 对齐的推进说明，见 [GUPPY_NEXT_STEPS.md](/Users/jun/ai/train/microgpt-c/GUPPY_NEXT_STEPS.md)。

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
- 对聊天语料自动启用更接近 `guppylm` 的模型配置：`LayerNorm + tied embeddings`
- 聊天和 token 模式默认改用更接近原项目的 `ReLU + dropout`

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

如果数据看起来像聊天样本（包含 `<|user|>` 和 `<|assistant|>`），程序会自动：

- 将 `norm` 切到 `LayerNorm`
- 启用 `tied embeddings`
- 至少使用 `block_size=128`
- 将过高的学习率压到 `1e-3`

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
- 当前已经迁入的关键点：聊天数据格式、单轮 prompt 推理、`LayerNorm`、`tied embeddings`。
- 当前还没有移植：纯 C BPE tokenizer、ONNX 导出和浏览器推理。
- 要得到更像样的聊天输出，通常需要更长训练步数和更大的 `block_size`。

## Guppy BPE Mode

当前版本还加入了一个更接近 `guppylm` 训练方式的离线路径：

- 使用 Python 脚本生成 Guppy 聊天语料
- 训练 ByteLevel BPE tokenizer
- 导出连续 token 流：
  - `train.bin`
  - `eval.bin`
  - `meta.json`
  - `tokenizer.json`
- C 训练器直接读取 `train.bin` 做 token-stream 训练
- BPE special tokens 和 prompt 格式已经改成和原项目同一套：
  - `<pad>`
  - `<|im_start|>`
  - `<|im_end|>`

准备依赖：

```bash
python3 -m pip install --user tokenizers
```

准备 BPE 数据：

```bash
python3 scripts/prepare_guppy_bpe.py --out-dir data/guppy_bpe --samples 60000 --vocab-size 1024
```

训练 token 模式：

```bash
./microgpt_mac 1000 0.7 1 200 50 64 4 2 128 0.001 data/guppy_bpe/train.bin
```

用训练好的 checkpoint 聊天：

```bash
./microgpt_mac chat "tell me a joke" data/guppy_bpe/train.bin ckpt_best.bin 0.7
```

当 `dataset_path` 是 `train.bin` 时，程序会自动：

- 从同目录推断 `eval.bin` 和 `meta.json`
- 读取 `u16` token 流
- 启用 token-stream 随机窗口采样
- 使用 `LayerNorm + tied embeddings + ReLU + dropout`

当前限制：

- `tokenizer.json` 的编码/解码仍在 Python 侧
- `chat` 命令已经能直接对 `.bin` 数据集聊天，但底层仍会调用 Python 脚本做 BPE 编码/解码
- 如果你之前已经生成过 `data/guppy_bpe` 或 `data/guppy_hf`，在这次格式对齐后需要重新生成一次

## 原版 Guppy 数据

如果你想尽量贴近原项目，可以直接下载原作者公开的 60K 数据集，再转成当前 C 训练器能吃的格式。

准备依赖：

```bash
python3 -m pip install --user datasets tokenizers
```

准备原版数据：

```bash
python3 scripts/prepare_guppy_hf.py --out-dir data/guppy_hf --vocab-size 4096
```

这会下载 `arman-bd/guppylm-60k-generic`，再导出：

- `train.txt`
- `eval.txt`
- `train.bin`
- `eval.bin`
- `tokenizer.json`
- `meta.json`

如果你要尽量对齐原项目的模型大小，可以用这一组参数：

```bash
MICROGPT_BATCH_SIZE=32 MICROGPT_ACT=relu MICROGPT_DROPOUT=0.1 MICROGPT_FFN_DIM=768 \
./microgpt_mac 10000 0.7 1 200 50 384 6 6 128 0.0003 data/guppy_hf/train.bin
```

这一组配置对应：

- `n_embd=384`
- `n_head=6`
- `n_layer=6`
- `ffn_dim=768`
- `block_size=128`
- `BPE vocab=4096`

也就是和原项目 README 里写的参数基本对齐。

## 自动测例

项目里现在有一个固定测例脚本，用来盯住“像不像 Guppy”。

运行方式：

```bash
python3 scripts/eval_guppy.py --dataset data/guppy_hf/train.bin --ckpt ckpt_best.bin
```

它会：

- 跑一组固定 prompt
- 抓取 `guppy>` 输出
- 用关键词命中和基本风格规则给出一个粗分

这个分数不是什么标准 benchmark，但很适合拿来比较不同训练配置有没有往原项目的风格靠近。

## 训练开关

当前版本加了三个环境变量，便于做更像真实训练的实验，不用再改一堆位置参数：

```bash
MICROGPT_BATCH_SIZE=8
MICROGPT_ACT=relu
MICROGPT_DROPOUT=0.1
MICROGPT_FFN_DIM=256
```

例子：

```bash
MICROGPT_BATCH_SIZE=8 MICROGPT_ACT=gelu MICROGPT_DROPOUT=0.1 \
./microgpt_mac 1000 0.7 1 200 50 64 4 2 128 0.001 data/guppy_bpe/train.bin
```

说明：

- `MICROGPT_BATCH_SIZE` 用梯度累积做 mini-batch，默认 `1`
- `MICROGPT_ACT` 可选 `relu2`、`relu`、`gelu`
- `MICROGPT_DROPOUT` 范围是 `[0, 1)`，默认名字任务是 `0`，聊天和 token 模式默认会切到 `0.1`
- `MICROGPT_FFN_DIM` 控制 MLP 隐层宽度；聊天和 token 模式默认会切到 `2 * n_embd`，更接近原项目

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
