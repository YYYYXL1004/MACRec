# MACRec 复现指南

> **Multi-Aspect Cross-modal Quantization for Generative Recommendation (AAAI 2026 Oral)**
>
> 基于 T5 的生成式推荐模型，核心创新：跨模态 RQ-VAE 量化 + 跨模态对比学习微调 + 双路 Ensemble 推理。

---

## 1. Environment Setup（环境配置）

### 1.1 硬件要求

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 1× A100 40GB | 2× A100 80GB |
| 内存 | 32GB | 64GB |
| 磁盘 | 50GB（不含原始图片） | 200GB（含图片和模型缓存） |

### 1.2 核心依赖

| 包名 | 用途 | 备注 |
|------|------|------|
| PyTorch >= 2.0 | 框架 | 需要 CUDA 支持 |
| transformers | T5/LLaMA 模型 | |
| accelerate | 分布式训练 | |
| deepspeed | 混合精度/ZeRO | |
| sentencepiece | T5 Tokenizer | |
| scikit-learn | KMeans 聚类 | 原 requirements.txt 未列出 |
| numpy | 数值计算 | 原 requirements.txt 未列出 |
| Pillow | 图像加载 | CLIP 特征提取需要 |
| requests | 下载图片 | 数据预处理需要 |
| tqdm | 进度条 | |

### 1.3 一键安装

```bash
conda create -n macrec python=3.10 -y
conda activate macrec

# PyTorch（根据 CUDA 版本调整）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 项目依赖
pip install transformers accelerate deepspeed sentencepiece evaluate peft bitsandbytes tqdm

# 隐含依赖（requirements.txt 未列出但必须有）
pip install scikit-learn numpy Pillow requests
```

---

## 2. Data Flow & Script Architecture（数据流与脚本架构）

### 2.1 全局数据流图

```
┌─────────────────────────────── 数据预处理阶段 ───────────────────────────────┐
│                                                                              │
│  Amazon 原始数据                                                              │
│  (Ratings CSV + Metadata gz + 图片 URL)                                      │
│       │                                                                      │
│       ▼                                                                      │
│  ① load_all_figures.py ──► 下载商品图片到本地                                  │
│       │                       Images/{Dataset}/xxx.jpg                       │
│       ▼                                                                      │
│  ② amazon18_data_process.py ──► K-core 过滤 + Leave-one-out 划分             │
│       │   输出:  {Dataset}.inter.json     用户交互序列                         │
│       │          {Dataset}.item.json      商品元信息                           │
│       │          {Dataset}.train.inter    训练集                              │
│       │          {Dataset}.valid.inter    验证集                              │
│       │          {Dataset}.test.inter     测试集                              │
│       │          {Dataset}.user2id        用户ID映射                          │
│       │          {Dataset}.item2id        商品ID映射                          │
│       ▼                                                                      │
│  ③ amazon_text_emb.py ──► LLaMA-7B 提取文本嵌入                              │
│       │   输出:  {Dataset}.emb-llama-td.npy   [num_items, 4096]              │
│       ▼                                                                      │
│  ④ clip_feature.py ──► CLIP ViT-L/14 提取图像嵌入                            │
│       │   输出:  {Dataset}.emb-ViT-L-14.npy   [num_items, 768]              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────── 量化索引生成阶段 ─────────────────────────────────┐
│                                                                              │
│  ⑤ data/kmeans.py ──► KMeans 聚类生成伪标签                                   │
│       │   输入:  .emb-llama-td.npy / .emb-ViT-L-14.npy                      │
│       │   输出:  {Dataset}.index_lemb_kmeans512.json    文本伪标签            │
│       │          {Dataset}.index_vitemb_kmeans512.json   图像伪标签           │
│       ▼                                                                      │
│  ⑥ cross_index/main.py ──► 训练跨模态 CrossRQVAE                            │
│       │   输入:  .emb-llama-td.npy + .emb-ViT-L-14.npy + kmeans 伪标签      │
│       │   输出:  best_text_collision_model.pth                               │
│       │          best_image_collision_model.pth                              │
│       ▼                                                                      │
│  ⑦ cross_index/generate_indices_distance.py ──► 生成离散索引码                │
│       │   输入:  训练好的 CrossRQVAE 权重 + embedding .npy                    │
│       │   输出:  {Dataset}.index_lemb_{name}.json    文本索引                 │
│       │          {Dataset}.index_vitemb_{name}.json   图像索引                │
│       │                                                                      │
│       │   索引格式示例:                                                       │
│       │   文本: {"0": ["<a_12>","<b_45>","<c_89>","<d_3>"], ...}             │
│       │   图像: {"0": ["<A_12>","<B_45>","<C_89>","<D_3>"], ...}             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────── 模型训练阶段 ────────────────────────────────────┐
│                                                                              │
│  ⑧ finetune_contrastive.py ──► 跨模态对比学习微调 T5                          │
│       │   输入:  config/ckpt/ (T5-small 配置)                                │
│       │          .inter.json + .index_lemb_*.json + .index_vitemb_*.json     │
│       │          .train.inter / .valid.inter                                 │
│       │   输出:  log/{Dataset}-{name}-contrastive-*/                         │
│       │          ├── pytorch_model.bin                                       │
│       │          ├── tokenizer.json                                          │
│       │          ├── config.json                                             │
│       │          └── train.log                                               │
│       │                                                                      │
│       │   训练 6 个任务:                                                      │
│       │   seqrec / seqimage / item2image / image2item                        │
│       │   seqimage2item / seqitem2image                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────── 推理与评估阶段 ──────────────────────────────────┐
│                                                                              │
│  ⑨ test_ddp_save.py (seqrec) ──► 文本路 Beam Search                         │
│       │   输出:  save_seqrec_20.json + results_seqrec_20.json               │
│       ▼                                                                      │
│  ⑩ test_ddp_save.py (seqimage) ──► 图像路 Beam Search                       │
│       │   输出:  save_seqimage_20.json + results_seqimage_20.json           │
│       ▼                                                                      │
│  ⑪ ensemble.py ──► 双路分数融合                                               │
│       │   输入:  save_seqrec_20.json + save_seqimage_20.json + 索引文件      │
│       │   输出:  results_ensemble_20.json                                    │
│       │          (包含 hit@1/5/10, ndcg@5/10)                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心脚本职责表

| 脚本 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `data_process/load_all_figures.py` | 从 Amazon 下载商品高清图 | Metadata gz + Ratings CSV | `Images/{Dataset}/xxx.jpg` + `{Dataset}_images_info.json` |
| `data_process/amazon18_data_process.py` | K-core 过滤、Leave-one-out 划分 | Ratings CSV + Metadata gz | `.inter.json`, `.item.json`, `.train/.valid/.test.inter`, `.user2id`, `.item2id` |
| `data_process/amazon_text_emb.py` | LLaMA-7B 提取 title+desc 文本嵌入 | `.item.json` | `.emb-llama-td.npy` |
| `data_process/clip_feature.py` | CLIP ViT-L/14 提取图像嵌入 | 商品图片 jpg | `.emb-ViT-L-14.npy` |
| `data/kmeans.py` | KMeans 聚类生成 RQ-VAE 伪标签 | `.emb-*.npy` | `.index_lemb_kmeans512.json`, `.index_vitemb_kmeans512.json` |
| `cross_index/main.py` | 训练 CrossRQVAE | `.emb-*.npy` + kmeans 伪标签 | `best_*_collision_model.pth` |
| `cross_index/generate_indices_distance.py` | 生成离散索引码 + 冲突消解 | RQ-VAE 权重 + `.emb-*.npy` | `.index_lemb_{name}.json`, `.index_vitemb_{name}.json` |
| `finetune_contrastive.py` | CrossModalContrastive T5 多任务微调 | 索引文件 + 交互数据 + T5 config | 模型 checkpoint 目录 |
| `test_ddp_save.py` | DDP 分布式 Beam Search 测试 | checkpoint + 测试数据 | `save_*.json` + `results_*.json` |
| `ensemble.py` | 文本路+图像路分数融合 | 两路 `save_*.json` + 索引文件 | `results_ensemble_*.json` |
| `modeling.py` | 定义 CrossModalContrastive 模型 | — | — |
| `data.py` | 定义 SeqRecDataset / FusionSeqRecDataset / ItemImageDataset | — | — |
| `collator.py` | 训练/测试 batch 整理 | — | — |
| `evaluate.py` | 计算 hit@k / ndcg@k 指标 | — | — |
| `generation_trie.py` | Trie 前缀树约束解码 | — | — |

### 2.3 关键数据格式说明

**`.inter.json`** — 用户交互序列
```json
{"0": [12, 45, 89, 3, 77], "1": [5, 22, 108], ...}
```

**`.index_lemb_*.json`** — 文本量化索引（小写字母）
```json
{"0": ["<a_12>", "<b_45>", "<c_89>", "<d_3>"], "1": ["<a_7>", "<b_102>", "<c_55>", "<d_200>"], ...}
```

**`.index_vitemb_*.json`** — 图像量化索引（大写字母）
```json
{"0": ["<A_12>", "<B_45>", "<C_89>", "<D_3>"], ...}
```

**`.train.inter`** — TSV 格式训练集
```
user_id:token	item_id_list:token_seq	item_id:token
0	12 45 89	77
0	12 45	89
```

**T5 模型输入样本格式**（由 Dataset 类生成）
```json
{
  "input_ids": "<extra_id_0><extra_id_1><extra_id_2><extra_id_3><a_12><b_45><c_89><d_3><a_7><b_102><c_55><d_200>",
  "labels": "<a_88><b_33><c_11><d_5>",
  "task_flag": 0
}
```

---

## 3. Step-by-Step Reproduction（分步复现指南）

### Step 0: 获取 Amazon 2018 原始数据

从 [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) 下载以下文件：

```
/your/amazon18/
├── Ratings/          # 评分 CSV: Musical_Instruments.csv 等
├── Metadata/         # 元数据 gz: meta_Musical_Instruments.json.gz 等
└── Review/           # 评论 gz: Musical_Instruments_5.json.gz 等
```

### Step 1: 数据预处理

所有脚本在 `data_process/` 目录下执行。

**1a. 下载商品图片**

```bash
cd MACRec/data_process

python load_all_figures.py \
    --dataset Instruments \
    --meta_data_path /your/amazon18/Metadata \
    --rating_data_path /your/amazon18/Ratings \
    --review_data_path /your/amazon18/Review \
    --save_path /your/amazon18/Images
```

输出：`/your/amazon18/Images/Instruments/` 下的 jpg 图片 + `Instruments_images_info.json`。

**1b. 处理交互数据**

```bash
python amazon18_data_process.py \
    --dataset Instruments \
    --input_path /your/amazon18 \
    --output_path ../data \
    --user_k 5 \
    --item_k 5
```

| 参数 | 说明 |
|------|------|
| `--user_k` | 用户 K-core 过滤阈值（至少有 K 条交互的用户才保留） |
| `--item_k` | 商品 K-core 过滤阈值 |
| `--input_path` | Amazon 原始数据根目录（需包含 Ratings/, Metadata/, Images/ 子目录） |
| `--output_path` | 输出目录，生成的文件将保存到 `{output_path}/{dataset}/` |

输出到 `data/Instruments/`：

```
Instruments.inter.json      # {user_id: [item_id_list]}
Instruments.item.json       # {item_id: {title, description, brand, categories}}
Instruments.train.inter     # TSV: user_id \t item_seq \t target_item
Instruments.valid.inter
Instruments.test.inter
Instruments.user2id         # TSV: original_id \t remapped_id
Instruments.item2id
```

**1c. 提取文本嵌入**

```bash
export CUDA_VISIBLE_DEVICES=0

python amazon_text_emb.py \
    --dataset Instruments \
    --root ../data \
    --model_name_or_path huggyllama/llama-7b \
    --model_cache_dir /your/cache_models \
    --gpu_id 0
```

输出：`data/Instruments/Instruments.emb-llama-td.npy`，shape `[num_items, 4096]`。

**1d. 提取图像嵌入**

```bash
export CUDA_VISIBLE_DEVICES=0

python clip_feature.py \
    --dataset Instruments \
    --image_root /your/amazon18/Images \
    --save_root ../data \
    --model_cache_dir /your/cache_models/clip
```

输出：`data/Instruments/Instruments.emb-ViT-L-14.npy`，shape `[num_items, 768]`。

### Step 2: 生成量化索引

**2a. KMeans 聚类伪标签**

```bash
cd MACRec/data

# 修改 kmeans.py 中 dataset 变量为目标数据集
# dataset = "Instruments"
python kmeans.py
```

输出：
- `Instruments/Instruments.index_lemb_kmeans512.json`（文本伪标签）
- `Instruments/Instruments.index_vitemb_kmeans512.json`（图像伪标签）

**2b. 训练跨模态 CrossRQVAE**

```bash
cd MACRec/cross_index

# 参数说明：begin_cross_layer=0, save_name, dataset, text_contrast, image_contrast, recon_contrast
bash scripts/run_cross_rqvae.sh 0 Savename Instruments 0.1 0.1 0.001
```

等价的完整命令：

```bash
python -u main.py \
    --num_emb_list 256 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0 0.0 \
    --device cuda:0 \
    --text_data_path ../data/Instruments/Instruments.emb-llama-td.npy \
    --image_data_path ../data/Instruments/Instruments.emb-ViT-L-14.npy \
    --ckpt_dir log/Instruments/Savename \
    --eval_step 2 \
    --batch_size 2048 \
    --begin_cross_layer 0 \
    --use_cross_rq True \
    --text_class_info ../data/Instruments/Instruments.index_lemb_kmeans512.json \
    --image_class_info ../data/Instruments/Instruments.index_vitemb_kmeans512.json \
    --text_contrast_weight 0.1 \
    --image_contrast_weight 0.1 \
    --recon_contrast_weight 0.001 \
    --epochs 1000
```

| 参数 | 说明 |
|------|------|
| `--num_emb_list` | 每层 RQ 的码本大小，4 层各 256 个码字 |
| `--begin_cross_layer` | 从第几层开始跨模态交互（0 = 所有层都跨模态） |
| `--text_contrast_weight` | 文本侧类内对比学习损失权重 |
| `--image_contrast_weight` | 图像侧类内对比学习损失权重 |
| `--recon_contrast_weight` | 跨模态重建对齐损失权重 |

输出到 `cross_index/log/Instruments/Savename/`：
- `best_text_collision_model.pth`（文本侧最优模型）
- `best_image_collision_model.pth`（图像侧最优模型）

**2c. 生成离散索引码**

```bash
cd MACRec/cross_index

bash scripts/gen_code.sh Savename Instruments
```

该脚本会分别用文本最优和图像最优模型生成两份索引：

```bash
# 文本索引（用 best_text_collision_model.pth）
python -u generate_indices_distance.py \
    --dataset Instruments \
    --text_data_path ../data/Instruments/Instruments.emb-llama-td.npy \
    --image_data_path ../data/Instruments/Instruments.emb-ViT-L-14.npy \
    --ckpt_path log/Instruments/Savename/best_text_collision_model.pth \
    --output_dir ../data/Instruments \
    --output_file Instruments.index_lemb_Savename.json \
    --content text

# 图像索引（用 best_image_collision_model.pth）
python -u generate_indices_distance.py \
    --dataset Instruments \
    --text_data_path ../data/Instruments/Instruments.emb-llama-td.npy \
    --image_data_path ../data/Instruments/Instruments.emb-ViT-L-14.npy \
    --ckpt_path log/Instruments/Savename/best_image_collision_model.pth \
    --output_dir ../data/Instruments \
    --output_file Instruments.index_vitemb_Savename.json \
    --content image
```

输出到 `data/Instruments/`：
- `Instruments.index_lemb_Savename.json`
- `Instruments.index_vitemb_Savename.json`

### Step 3: 模型训练

```bash
cd MACRec

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=29500 finetune_contrastive.py \
    --data_path ./data/ \
    --dataset Instruments \
    --output_dir ./log/Instruments-Savename-contrastive \
    --base_model ./config/ckpt \
    --per_device_batch_size 1024 \
    --learning_rate 1e-3 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file .index_lemb_Savename.json \
    --image_index_file .index_vitemb_Savename.json \
    --tasks seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image \
    --valid_task seqrec
```

| 参数 | 说明 |
|------|------|
| `--base_model` | T5 模型配置目录（`config/ckpt/`，含 config.json + tokenizer） |
| `--per_device_batch_size` | 每张卡的 batch size |
| `--max_his_len` | 用户历史序列最大长度（截取最近 N 条交互） |
| `--prompt_num` | Soft prompt token 数量 |
| `--patient` | Early stopping 耐心值（连续 N epoch 验证集无提升则停止） |
| `--tasks` | 多任务训练，逗号分隔 |
| `--valid_task` | 验证集评估使用的任务 |
| `--index_file` | 文本量化索引文件（相对于 `{data_path}/{dataset}/` 的后缀） |
| `--image_index_file` | 图像量化索引文件 |

Checkpoint 保存位置：`./log/Instruments-Savename-contrastive/`，包含 `pytorch_model.bin`、`tokenizer.json`、`config.json` 等。

### Step 4: 模型推理与评估

**4a. 文本路推理**

```bash
torchrun --nproc_per_node=2 --master_port=29500 test_ddp_save.py \
    --ckpt_path ./log/Instruments-Savename-contrastive \
    --data_path ./data/ \
    --dataset Instruments \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file .index_lemb_Savename.json \
    --image_index_file .index_vitemb_Savename.json \
    --test_task seqrec \
    --results_file ./log/Instruments-Savename-contrastive/results_seqrec_20.json \
    --save_file ./log/Instruments-Savename-contrastive/save_seqrec_20.json \
    --filter_items
```

**4b. 图像路推理**

```bash
torchrun --nproc_per_node=2 --master_port=29500 test_ddp_save.py \
    --ckpt_path ./log/Instruments-Savename-contrastive \
    --data_path ./data/ \
    --dataset Instruments \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file .index_lemb_Savename.json \
    --image_index_file .index_vitemb_Savename.json \
    --test_task seqimage \
    --results_file ./log/Instruments-Savename-contrastive/results_seqimage_20.json \
    --save_file ./log/Instruments-Savename-contrastive/save_seqimage_20.json \
    --filter_items
```

**4c. 双路 Ensemble**

```bash
python ensemble.py \
    --output_dir ./log/Instruments-Savename-contrastive \
    --dataset Instruments \
    --data_path ./data/ \
    --index_file .index_lemb_Savename.json \
    --image_index_file .index_vitemb_Savename.json \
    --num_beams 20
```

评估结果输出位置：

| 文件 | 内容 |
|------|------|
| `results_seqrec_20.json` | 文本路指标（hit@1/5/10, ndcg@5/10） |
| `results_seqimage_20.json` | 图像路指标 |
| `results_ensemble_20.json` | **融合后最终指标** |
| `train.log` | 完整训练日志 |

### 一键脚本（推荐）

上述 Step 3 + Step 4 可通过一个脚本串行执行：

```bash
cd MACRec

# 编辑 scripts/run_finetune.sh 中的 names 和 datasets 数组
# names=(Savename)
# datasets=(Instruments)

export CUDA_VISIBLE_DEVICES=0,1
bash scripts/run_finetune.sh
```

`scripts/finetune_full.sh` 会依次执行：训练 → seqrec 测试 → seqimage 测试 → ensemble。

---

## 4. Troubleshooting（常见踩坑点预警）

### 4.1 硬编码路径

| 文件 | 位置 | 硬编码内容 | 修改方案 |
|------|------|-----------|---------|
| `data_process/amazon18_data_process.py` | L270-271 | `--input_path` 默认 `/datasets/datasets/amazon18`，`--output_path` 默认 `/datasets/datasets/LC-Rec_image` | 命令行传参覆盖 |
| `data_process/amazon_text_emb.py` | L111-115 | `--root` 默认 `/userhome/dataset/MQL4GRec`，`--model_cache_dir` 默认 `/userhome/cache_models` | 命令行传参覆盖 |
| `data_process/clip_feature.py` | L84-86 | `--image_root` 默认 `/userhome/dataset/amazon18/Images`，`--model_cache_dir` 默认 `/userhome/cache_models/clip` | 命令行传参覆盖 |
| `data_process/load_all_figures.py` | L181-184 | `--meta_data_path` 等多个路径默认 `/datasets/datasets/amazon18/` | 命令行传参覆盖 |
| `data/kmeans.py` | L4 | `dataset = "Games"` 硬编码在代码中 | **需要手动修改源码**中的 dataset 变量 |
| `ensemble.py` | L202-205 | `--data_path` 默认 `/userhome/dataset/LC-Rec_images`，`--output_dir` 默认一个长路径 | 命令行传参覆盖 |

### 4.2 缺失目录结构

- `cross_index/log/` — 训练 RQ-VAE 前需确保该目录存在（脚本中有 `mkdir -p`）
- `log/` — T5 训练输出目录（脚本中有 `mkdir -p`）
- `train_logs/` — `run_finetune.sh` 需要该目录（脚本中有 `mkdir -p`）
- `results/` — `test.py` 默认输出到 `./results/test-ddp.json`，需手动创建

### 4.3 OOM 风险

| 阶段 | 风险点 | 建议 |
|------|--------|------|
| 文本嵌入提取 | LLaMA-7B 至少需要 ~14GB 显存（FP16） | 确保单卡显存 >= 16GB，或使用 `bitsandbytes` 8bit 量化 |
| RQ-VAE 训练 | `batch_size=2048` 较大 | 单卡 24GB 通常够用；若 OOM 可降低 batch_size |
| T5 微调 | `per_device_batch_size=1024` | T5 模型极小（d_model=128, 4 层），1024 batch 在 A100 上可行；较小显存 GPU 可降低到 256~512 |
| Beam Search 推理 | `num_beams=20` 乘以序列长度 | 若 OOM 可降低 `num_beams`（如 10）或减小 `test_batch_size` |

### 4.4 其他注意事项

1. **`requirements.txt` 不完整**：缺少 `torch`、`numpy`、`scikit-learn`、`Pillow`、`requests`，需手动安装。

2. **NCCL 环境变量**：`finetune_full.sh` 中设置了 `NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1`，在某些 InfiniBand 集群上可能导致通信效率低下。如果你的集群支持 P2P/IB，可以去掉这两行。

3. **`kmeans.py` 缺少命令行参数**：dataset 名称硬编码在代码中（第 4 行），切换数据集需要手动改源码。

4. **T5 模型不是标准预训练权重**：`config/ckpt/` 中的 T5 是一个自定义小模型（d_model=128, 4 层, 6 头），不是 HuggingFace 上的 `t5-small`（d_model=512, 6 层, 8 头）。不要误替换为标准 T5 权重。

5. **图片下载可能不完整**：`load_all_figures.py` 从 Amazon CDN 下载图片，部分 URL 可能失效。脚本会跳过下载失败的图片，但后续 CLIP 特征提取时需要确保 `images_info.json` 中记录的图片确实存在。

6. **对比学习超参数**：`modeling.py` 中 `temperature=0.1`，`contrastive_weight=0.01` 是硬编码的，如需调整要改源码。

7. **冲突消解迭代次数**：`generate_indices_distance.py` 中冲突消解最多迭代 2 次（`tt == 2` 时 break），极端情况下可能还存在少量冲突。

8. **Ensemble 中的 magic number**：`ensemble.py` 第 66/79 行中，同一物品在两路都出现时分数公式为 `(score + old_score) / 2 + 1`，+1 是经验调参值，不可配置。
