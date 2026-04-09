# Guppy 下一步

这份文档只说一件事：当前项目离 `guppylm` 还差什么，接下来该先做什么。

## 当前状态

已经有的东西：

- 字符级训练和采样
- Guppy 风格数据生成
- BPE 离线预处理
- token-stream 训练
- `chat` 命令可以直接对 `.bin` 数据集聊天
- `LayerNorm`
- `tied embeddings`
- `mini-batch` 梯度累积
- `ReLU` / `ReLU^2` / `GELU`
- `dropout`
- `ffn_dim` 可调
- 原版 `arman-bd/guppylm-60k-generic` 数据准备脚本
- 固定测例脚本

已经验证过的事情：

- 普通名字训练还能正常跑
- BPE 数据准备和聊天链路能跑通
- 原版 60K 数据能下载并转成 `train.bin` / `eval.bin`
- 当前 C 版已经能起到接近原项目尺寸的配置：
  - `n_embd=384`
  - `n_head=6`
  - `n_layer=6`
  - `ffn_dim=768`
  - `block_size=128`
  - 参数量大约 `8.0M`

## 现在真正卡住的点

### 1. 还没有长时间训练结果

这是当前最大的缺口。

代码和数据链路已经能跑，但还没有一次像样的长训练结果，所以还不能说“效果接近原项目”。

建议先跑这一条：

```bash
python3 scripts/prepare_guppy_hf.py --out-dir data/guppy_hf --vocab-size 4096

MICROGPT_BATCH_SIZE=32 MICROGPT_ACT=relu MICROGPT_DROPOUT=0.1 MICROGPT_FFN_DIM=768 \
./microgpt_mac 10000 0.7 1 200 50 384 6 6 128 0.0003 data/guppy_hf/train.bin
```

训练过程中定期跑：

```bash
python3 scripts/eval_guppy.py --dataset data/guppy_hf/train.bin --ckpt ckpt_best.bin
```

### 2. tokenizer 已经基本对齐，但还要验证效果

现在已经对齐的部分：

- special tokens 改成了 `<pad>`、`<|im_start|>`、`<|im_end|>`
- token 训练文本改成了：
  - `<|im_start|>user\n...<|im_end|>`
  - `<|im_start|>assistant\n...<|im_end|>`
- `chat` 走 token 模式时，也会用同一套 prompt 格式

现在剩下的问题不是“格式不一样”，而是“这套 tokenizer 训出来以后，效果是不是已经足够接近原项目”。

目前还要继续看的点：

- 用 `--vocab-size 4096` 训练出来的实际词表还没有到 4096
- 这会影响参数量和最终效果

下一步该做的事：

1. 跑长训练，看固定测例分数是不是明显提升
2. 对比原项目输出风格，确认 tokenizer 不再是主要短板
3. 如果效果还是差，再继续细抠 tokenizer 训练细节

优先文件：

- `scripts/prepare_guppy_bpe.py`
- `scripts/prepare_guppy_hf.py`

### 3. 评测还比较粗

现在的 [scripts/eval_guppy.py](/Users/jun/ai/train/microgpt-c/scripts/eval_guppy.py) 只是关键词命中和简单风格检查，够看趋势，但不够细。

下一步可以补：

1. 保存每次评测结果到 `eval_guppy.json`
2. 记录每个 case 的历史分数
3. 增加“回复过长”“出现乱码”“重复过多”的扣分
4. 增加和原项目样例输出的人工对照

### 4. 训练配置还没系统扫过

现在只是把可能有用的开关接上了，还没做系统对比。

优先尝试这几组：

1. `batch=8, 16, 32`
2. `dropout=0.0, 0.1`
3. `ffn_dim=2*n_embd` 和 `ffn_dim=768`
4. `relu` 对比 `gelu`
5. `lr=3e-4` 对比 `1e-3`

建议每组都记录：

- `eval.csv`
- `losses.csv`
- `eval_guppy.py` 分数
- 5 条固定 prompt 输出

### 5. 还没有更省内存的训练路径

如果后面要认真跑 `384x6x6` 很多步，内存和速度都可能成为问题。

如果遇到卡顿，优先做这两个方向：

1. 减少训练缓存的常驻大小
2. 把 batch 从“完全串行累积”改成更省开销的路径

这个优先级在长训练真正遇到瓶颈前，可以先放后面。

## 推荐顺序

建议按这个顺序往下做：

1. 用原版 60K 数据跑一轮长训练
2. 用 `eval_guppy.py` 记下基线分数
3. 修 tokenizer，让词表更接近原项目
4. 扫一轮训练配置
5. 只在速度或内存真的卡住时，再做训练器优化

## 目标

短期目标不是“把所有功能都补齐”，而是先做到这三件事：

1. 原版数据能稳定长训
2. 固定测例分数明显高于现在的小模型
3. 输出风格看起来像原项目里的 Guppy

做到这三点，再去补纯 C tokenizer、前端导出、浏览器推理，性价比才对。
