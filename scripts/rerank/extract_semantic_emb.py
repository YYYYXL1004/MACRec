"""
从训练好的 RQVAE 提取 text/image 的 768d 语义嵌入 (align space 的量化重建)

输出:
  {dataset}.emb-text-aligned-768.npy   (N_items, 768)  text_decoder(text_x_q) 输出
  {dataset}.emb-image-aligned-768.npy  (N_items, 768)  image_decoder(image_x_q) 输出

用法:
  cd MACRec
  python extract_semantic_emb.py \
      --rqvae_ckpt cross_index/log/Instruments/R1_E1_st-concat-caq/best_text_collision_model.pth \
      --text_data_path data/Instruments/Instruments.emb-st-768.npy \
      --image_data_path data/Instruments/Instruments.emb-ViT-L-14.npy \
      --collab_data_path data/Instruments/Instruments.emb-collab-256.npy \
      --output_dir data/Instruments \
      --dataset Instruments \
      --device cuda:0
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cross_index'))
from datasets import DualEmbDataset, TripleEmbDataset
from models.rqvae import CrossRQVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rqvae_ckpt', type=str, required=True,
                        help='RQVAE checkpoint 路径 (best_text_collision_model.pth)')
    parser.add_argument('--text_data_path', type=str, required=True)
    parser.add_argument('--image_data_path', type=str, required=True)
    parser.add_argument('--collab_data_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 加载 checkpoint，恢复训练时的配置
    ckpt = torch.load(args.rqvae_ckpt, map_location='cpu', weights_only=False)
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # 根据训练时配置决定数据集类型
    collab_data_path = getattr(ckpt_args, 'collab_data_path', '')
    collab_fusion = getattr(ckpt_args, 'collab_fusion', 'concat')
    collab_dim = 0

    if collab_data_path and collab_fusion == "concat" and args.collab_data_path:
        data = TripleEmbDataset(args.text_data_path, args.image_data_path, args.collab_data_path)
        print(f"[拼接模式] text_dim={data.text_dim}, img_dim={data.img_dim}")
    elif collab_data_path and collab_fusion == "proj_add" and args.collab_data_path:
        data = DualEmbDataset(args.text_data_path, args.image_data_path)
        collab_emb = np.load(args.collab_data_path)
        collab_dim = collab_emb.shape[-1]
    else:
        data = DualEmbDataset(args.text_data_path, args.image_data_path)

    # 构建模型
    model = CrossRQVAE(
        text_in_dim=data.text_dim,
        image_in_dim=data.img_dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
        use_cross_rq=ckpt_args.use_cross_rq,
        begin_cross_layer=ckpt_args.begin_cross_layer,
        collab_dim=collab_dim,
    )
    model.load_state_dict(state_dict)
    if collab_dim > 0:
        model.register_collab_embeddings(torch.FloatTensor(collab_emb))
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    text_aligned_list = []
    image_aligned_list = []

    print(f"提取语义嵌入: {len(data)} items...")
    with torch.no_grad():
        for batch_idx, (text_d, img_d, indices) in enumerate(data_loader):
            text_d = text_d.to(device)
            img_d = img_d.to(device)

            # 复现 RQVAE forward 中的关键路径，提取 align space 的量化重建
            text_align_in = model.text_align_encoder(text_d)
            image_align_in = model.image_align_encoder(img_d)

            # 投影加法: 在 align 空间加入 collab
            if model.collab_dim > 0 and model._collab_embeddings is not None:
                collab_x = model._collab_embeddings[indices].to(device)
                text_align_in = text_align_in + model.text_collab_proj(collab_x)
                image_align_in = image_align_in + model.image_collab_proj(collab_x)

            text_x = model.text_encoder(text_align_in)
            image_x = model.image_encoder(image_align_in)

            # RQ 量化
            if model.use_cross_rq:
                text_x_q = torch.zeros_like(text_x)
                image_x_q = torch.zeros_like(image_x)
                residual_text = text_x.clone()
                residual_image = image_x.clone()
                for i in range(model.num_rq_layers):
                    text_vq = model.text_rq.vq_layers[i]
                    image_vq = model.image_rq.vq_layers[i]
                    text_res, _, _, _ = text_vq(residual_text, use_sk=False)
                    image_res, _, _, _ = image_vq(residual_image, use_sk=False)
                    residual_text = residual_text - text_res
                    residual_image = residual_image - image_res
                    text_x_q = text_x_q + text_res
                    image_x_q = image_x_q + image_res
            else:
                text_x_q, _, _, _ = model.text_rq(text_x, use_sk=False)
                image_x_q, _, _, _ = model.image_rq(image_x, use_sk=False)

            # 解码: text_decoder(32d→768d) → text_align_decoder(768d→原始输入维度)
            # 最终输出是 RQVAE 的完整重建，维度与输入一致 (如 collab+text=1024d)
            text_align_out = model.text_decoder(text_x_q)
            image_align_out = model.image_decoder(image_x_q)
            text_recon = model.text_align_decoder(text_align_out)
            image_recon = model.image_align_decoder(image_align_out)

            text_aligned_list.append(text_recon.cpu().numpy())
            image_aligned_list.append(image_recon.cpu().numpy())

    text_aligned = np.concatenate(text_aligned_list, axis=0)
    image_aligned = np.concatenate(image_aligned_list, axis=0)

    # 输出维度取决于 RQVAE 输入维度 (如 concat 模式下 collab+text=1024d)
    out_dim = text_aligned.shape[1]
    text_path = os.path.join(args.output_dir, f'{args.dataset}.emb-text-aligned-{out_dim}.npy')
    image_path = os.path.join(args.output_dir, f'{args.dataset}.emb-image-aligned-{out_dim}.npy')
    np.save(text_path, text_aligned)
    np.save(image_path, image_aligned)

    print(f"text  语义嵌入: {text_aligned.shape} → {text_path}")
    print(f"image 语义嵌入: {image_aligned.shape} → {image_path}")
    print(f"text  L2 norm 均值: {np.linalg.norm(text_aligned, axis=1).mean():.4f}")
    print(f"image L2 norm 均值: {np.linalg.norm(image_aligned, axis=1).mean():.4f}")


if __name__ == '__main__':
    main()
