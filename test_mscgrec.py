"""MSCGRec evaluation — DDP constrained beam search over the collaborative-code trie.
Mirrors test_ddp.py; only the dataset (multimodal history -> collab target) differs."""
import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import (parse_global_args, parse_dataset_args, parse_test_args,
                   set_seed, prefix_allowed_tokens_fn)
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie
from data_mscgrec import MSCGRecDataset


def test_ddp(args):
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path, low_cpu_mem_usage=True, device_map={"": local_rank})
    model = DistributedDataParallel(model, device_ids=[local_rank])

    args.soft_prompts = {"seqrec": ""}
    test_data = MSCGRecDataset(args, task="seqrec", mode="test", sample_num=args.sample_num)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    candidate_trie = Trie([[0] + tokenizer.encode(c) for c in all_items])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, num_workers=2, pin_memory=True)
    if local_rank == 0:
        print("data num:", len(test_data))

    model.eval()
    metrics = args.metrics.split(",")
    metrics_results = {}
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            bs = len(targets)
            num_beams = args.num_beams
            while True:
                try:
                    output = model.module.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=10,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    num_beams -= 1
                    print("OOM, beam:", num_beams)

            output_ids = output["sequences"]
            scores = output["sequences_scores"]
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            topk_res = get_topk_results(decoded, scores, targets, num_beams,
                                        all_items=all_items if args.filter_items else None)

            bs_gather = [None for _ in range(world_size)]
            dist.all_gather_object(obj=bs, object_list=bs_gather)
            total += sum(bs_gather)
            res_gather = [None for _ in range(world_size)]
            dist.all_gather_object(obj=topk_res, object_list=res_gather)

            if local_rank == 0:
                all_res = []
                for g in res_gather:
                    all_res += g
                batch_res = get_metrics_results(all_res, metrics)
                for m, r in batch_res.items():
                    metrics_results[m] = metrics_results.get(m, 0) + r
                if (step + 1) % 50 == 0:
                    print({m: metrics_results[m] / total for m in metrics_results})
            dist.barrier()

    dist.barrier()
    if local_rank == 0:
        mean_results = {m: metrics_results[m] / total for m in metrics}
        print("======================================================")
        print("Mean results:", mean_results)
        print("======================================================")
        save_data = {"mean_results": mean_results, "all_prompt_results": [metrics_results]}
        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print("Save file:", args.results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSCGRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument("--collab_index_file", type=str, default=".index_collab_mscgrec.json")
    args = parser.parse_args()
    test_ddp(args)
