"""
训练 SASRec 提取 256 维协同嵌入

输入: data/{Dataset}/{Dataset}.inter.json  (用户交互序列)
输出: data/{Dataset}/{Dataset}.emb-collab-256.npy  shape=(N_items, 256)

说明:
    - 从 inter.json 构造 RecBole 所需的 .inter 文件
    - 训练 SASRec，提取 item_embedding.weight
    - 输出按 MACRec 的 item_id (0~N-1) 排列，与 item.json 的 key 对齐

用法:
    python gen_collab_emb.py --dataset Instruments
    python gen_collab_emb.py --dataset Arts --gpu_id 1
"""

import argparse
import json
import os
import numpy as np
import torch

# PyTorch 2.6+ 兼容
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from recbole.utils import init_seed
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_root = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments',
                        help='Instruments / Arts / Games')
    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--stopping_step', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--gpu_id', type=str, default='0')
    return parser.parse_args()


def convert_inter_json_to_recbole(inter_json_path, output_dir, dataset_name):
    """
    将 MACRec 的 inter.json 转换为 RecBole 格式的 .inter 文件
    
    inter.json 格式: {"user_id": [item_id_0, item_id_1, ...], ...}
    序列按时间顺序排列，用位置索引作为 timestamp
    """
    with open(inter_json_path, 'r') as f:
        inters = json.load(f)

    recbole_name = f'{dataset_name}_collab_recbole'
    recbole_dir = os.path.join(output_dir, recbole_name)
    os.makedirs(recbole_dir, exist_ok=True)

    inter_file = os.path.join(recbole_dir, f'{recbole_name}.inter')
    total_inters = 0

    with open(inter_file, 'w') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        for uid_str, item_list in inters.items():
            # item_list 里是 int 类型的 item_id，按时间排序
            for t, iid in enumerate(item_list):
                # item_id 加前缀 'i' 避免 RecBole 把纯数字当成连续特征
                f.write(f'u{uid_str}\ti{iid}\t{t}\n')
                total_inters += 1

    print(f"转换完成: {len(inters)} 用户, {total_inters} 条交互")
    print(f"RecBole 文件: {inter_file}")
    return recbole_name, os.path.dirname(recbole_dir)


def train_sasrec(recbole_name, data_path, args):
    """训练 SASRec，返回 (model, dataset)"""
    config_dict = {
        'model': 'SASRec',
        'dataset': recbole_name,
        'data_path': data_path,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'seed': args.seed,
        'reproducibility': True,
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'full',
        },
        # 模型参数
        'hidden_size': args.hidden_size,
        'inner_size': args.hidden_size,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'loss_type': 'CE',
        'max_seq_length': args.max_seq_length,
        # 训练参数
        'train_neg_sample_args': None,
        'epochs': args.epochs,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.train_batch_size,
        'learner': 'adam',
        'learning_rate': args.learning_rate,
        'eval_step': 1,
        'stopping_step': args.stopping_step,
        # 评估
        'metrics': ['Recall', 'NDCG', 'Hit'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        # 设备
        'gpu_id': args.gpu_id,
        'use_gpu': True,
        'checkpoint_dir': os.path.join(args.data_root, args.dataset, 'saved_sasrec'),
        'show_progress': True,
    }

    config = Config(model='SASRec', dataset=recbole_name, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRec(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)

    print(f"\n训练 SASRec: epochs={args.epochs}, hidden={args.hidden_size}, "
          f"layers={args.n_layers}, early_stop={args.stopping_step}")

    best_valid_score, _ = trainer.fit(
        train_data, valid_data=valid_data, saved=True, show_progress=True)
    print(f"最佳验证 NDCG@10: {best_valid_score:.4f}")

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    print("\n测试集结果:")
    for metric, value in test_result.items():
        print(f"  {metric}: {value:.4f}")

    return model, dataset


def extract_and_remap(model, dataset, n_macrec_items, output_path):
    """
    提取 item_embedding 并映射回 MACRec 的 item_id 空间
    
    RecBole 内部 ID (1~N) 对应 token 'i0'~'i{N-1}'
    MACRec 的 item_id 就是 token 去掉前缀 'i' 后的数字
    """
    raw_emb = model.item_embedding.weight.data.cpu().numpy()  # (recbole_N+1, 256), [0]=PAD
    emb_dim = raw_emb.shape[1]

    # 构建 RecBole内部ID → MACRec item_id 的映射
    token2id = dataset.field2token_id['item_id']

    # 初始化输出 embedding：(n_macrec_items, 256)
    output_emb = np.zeros((n_macrec_items, emb_dim), dtype=np.float32)

    mapped_count = 0
    for token, recbole_idx in token2id.items():
        if recbole_idx == 0:
            continue  # 跳过 PAD
        # token 格式为 'i{macrec_item_id}'
        macrec_id = int(str(token).lstrip('i'))
        if 0 <= macrec_id < n_macrec_items:
            output_emb[macrec_id] = raw_emb[recbole_idx]
            mapped_count += 1

    print(f"映射: {mapped_count}/{n_macrec_items} 个 item 成功匹配")
    print(f"输出 shape: {output_emb.shape}")
    print(f"L2 norm 均值: {np.linalg.norm(output_emb, axis=1).mean():.4f}")

    np.save(output_path, output_emb)
    print(f"已保存: {output_path}")


def main():
    args = parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    inter_json_path = os.path.join(dataset_dir, f'{args.dataset}.inter.json')
    item_json_path = os.path.join(dataset_dir, f'{args.dataset}.item.json')
    output_path = os.path.join(dataset_dir, f'{args.dataset}.emb-collab-256.npy')

    # 获取 item 总数
    with open(item_json_path, 'r') as f:
        n_items = len(json.load(f))
    print(f"数据集: {args.dataset}, item 数: {n_items}")

    # 1. 转换格式
    recbole_name, recbole_data_path = convert_inter_json_to_recbole(
        inter_json_path, dataset_dir, args.dataset)

    # 2. 训练 SASRec
    model, dataset = train_sasrec(recbole_name, recbole_data_path, args)

    # 3. 提取并映射 embedding
    extract_and_remap(model, dataset, n_items, output_path)

    print("\n完成!")


if __name__ == '__main__':
    main()
