# TinyStories Plan

## Goal

在当前 `microgpt-c` 项目基础上，做一个面向学习原理的 TinyStories 版本。

目标不是完整复刻 `gpt-tinystories` 或做可用生产训练器，而是把当前项目从：

- `input.txt` 字符级名字生成

推进到：

- TinyStories token-stream 训练
- 能生成短小、带儿童故事风格的文本
- 依然保持代码可读、可推导、便于理解前向/反向传播

## Principles

- 优先保留教学价值，不先追求工程完备性。
- 优先改数据形态，再改模型细节。
- 尽量把外部依赖留在预处理脚本中，不把 Hugging Face tokenizer 直接塞进 C 训练器。
- 每一阶段都保持可运行、可验证。

## Current Baseline

当前代码的关键入口：

- 数据读取与样本构造：`microgpt_mac.c`
- 单样本前向/反向训练：`microgpt_mac.c`
- 自回归采样：`microgpt_mac.c`

当前限制：

- 训练数据格式是“每行一个样本”
- tokenizer 是字符级
- 训练目标是单样本窗口，不是 token-stream
- 默认结构更接近教学版 GPT，而不是 GPT-2/TinyStories 训练配置

## Target Shape

第一版 TinyStories 改造后的形态：

- 使用离线脚本下载和预处理 TinyStories
- 生成 `train.bin`、`val.bin`、`meta.json`
- C 程序读取 token id 流并随机切片训练
- 继续保留当前手写前向、反向和 Adam
- 先不引入完整 PyTorch/Hugging Face 训练栈

## Phase 1: Data Pipeline

目标：

- 从 TinyStories 构建可供当前项目直接读取的 token 数据

工作项：

- 新增 `scripts/prepare_tinystories.py`
- 使用 Hugging Face 下载 `roneneldan/TinyStories`
- 使用参考实现同类 tokenizer 做离线编码
- 将训练集和验证集写成连续 token 流
- 文档边界插入 `EOS`
- 输出：
  - `data/tinystories/train.bin`
  - `data/tinystories/val.bin`
  - `data/tinystories/meta.json`

设计要求：

- C 端不依赖 Python tokenizer 运行
- `meta.json` 至少包含 `vocab_size`、`eos_token_id`、`tokenizer_name`
- 预处理脚本支持限制样本数，便于小规模调试

完成标准：

- 能稳定生成二进制 token 数据文件
- 能打印 token 数量、文档数量、词表大小

## Phase 2: Token-Stream Training

目标：

- 让当前训练器支持从连续 token 流中随机采样训练窗口

工作项：

- 在 `microgpt_mac.c` 中增加二进制数据加载逻辑
- 保留旧的 `input.txt` 字符级模式，新增 `tinystories` 模式
- 新增随机窗口采样函数：
  - 输入连续 token 流
  - 随机选择起点
  - 取长度 `block_size + 1` 的片段
  - 构造 `in_tokens` 和 `targets`
- 训练入口根据模式切换数据来源

设计要求：

- 第一版先支持 `uint16` 或 `uint32` token 存储，按词表大小决定
- 不引入 batch 维度，先保留单样本 SGD
- 日志中输出当前数据模式和 token 统计信息

完成标准：

- 用 TinyStories token 数据跑通训练
- 不再依赖名字数据也能完成前向、反向、评估和采样

## Phase 3: Sampling And Decoding

目标：

- 让训练后的采样结果能还原成文本

工作项：

- 增加词表和 tokenizer 元信息读取
- 第一版可采用两种方案之一：
  - 方案 A：C 端只输出 token ids，再由 Python 脚本解码
  - 方案 B：C 端直接读取 tokenizer 导出的 vocab/merges 并解码

建议：

- 先做方案 A

原因：

- 学习重点在训练器，不在 BPE 解码器实现
- 可以避免把 tokenizer 复杂度带进当前单文件 C 程序

完成标准：

- 能从 checkpoint 采样 token ids
- 能用脚本把样本解码成可读故事文本

## Phase 4: Minimal Model Upgrades

目标：

- 只补最值得学的 GPT-2 风格组件，不立刻全面工程化

优先顺序：

- `LayerNorm` 开关
- `GELU` 开关
- `wte/lm_head` 绑权
- 更大的 `block_size`

暂缓项：

- dropout
- mixed precision
- data loader workers
- distributed training
- 完整 checkpoint 兼容 Hugging Face

完成标准：

- 同一套 TinyStories 数据上，比较不同组件对 loss 和样本的影响

## Phase 5: Validation

每一阶段都要做这些验证：

- 编译通过
- 小数据调试集训练通过
- `ASan` 或等价方式验证没有新增越界
- 训练日志和评估日志完整输出
- 至少保留一个可复现命令

阶段性验证命令示例：

- 小规模预处理
- 小规模 TinyStories 训练 200 到 1000 steps
- 训练后采样 3 到 5 个样本

## Non-Goals

本计划明确不做：

- 在本机从零训练出可用通用模型
- 完整复刻 `gpt-tinystories`
- 一开始就支持大 batch、LoRA、分布式训练
- 追求和 MLX/PyTorch 栈同级的训练效率

## Expected Outcome

完成 Phase 1 到 Phase 3 后，项目会从“字符级名字生成器”升级为：

- 一个可读的 TinyStories 教学版 GPT 训练器
- 能在当前机器上训练并生成简短故事片段
- 仍然适合逐函数理解 Transformer 和语言模型训练过程

## Recommended Execution Order

建议严格按以下顺序实施：

1. `scripts/prepare_tinystories.py`
2. `data/tinystories/*.bin` 与 `meta.json` 格式确定
3. C 端 token-stream 数据加载
4. C 端训练入口切换
5. 采样 token ids 输出
6. Python 端解码脚本
7. 再做 `LayerNorm/GELU/tied embeddings`

## First Deliverable

第一批实现建议只交付以下内容：

- TinyStories 预处理脚本
- C 端 token-stream 训练
- 训练日志
- token id 采样

这样可以先把“数据和训练闭环”打通，再决定模型结构要不要继续向 GPT-2 靠拢。
