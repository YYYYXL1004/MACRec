"""MSCGRec training entry — reuses the CAGRec T5 pipeline with a multimodal-history /
collaborative-target dataset. Non-invasive: does not modify data.py / utils.py."""
import argparse
import os

import torch
import transformers
from torch.utils.data import ConcatDataset
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers.trainer_callback import EarlyStoppingCallback

from utils import (parse_global_args, parse_train_args, parse_dataset_args,
                   set_seed, ensure_dir, print_trainable_parameters)
from collator import Collator
from modeling import create_contrastive_model
from data_mscgrec import MSCGRecDataset


def load_mscgrec_datasets(args):
    train = MSCGRecDataset(args, task="seqrec", mode="train")
    valid = MSCGRecDataset(args, task="seqrec", mode="valid")
    return ConcatDataset([train]), ConcatDataset([valid])


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
    args.deepspeed = None

    # MSCGRec uses a single multimodal->collab generation task; soft prompts unused.
    args.soft_prompts = {"seqrec": ""}

    train_data, valid_data = load_mscgrec_datasets(args)

    add_num = 0
    for dataset in train_data.datasets:
        add_num += tokenizer.add_tokens(dataset.get_new_tokens())

    collator = Collator(args, tokenizer)
    model = create_contrastive_model(config)  # CAGRecModel; cpa_weight=0 + task_flag=0 -> plain T5
    model.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data.datasets[0][100])
        print_trainable_parameters(model)

    early_stop = EarlyStoppingCallback(early_stopping_patience=args.patient)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=args.gradient_checkpointing,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        ),
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[early_stop],
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSCGRec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--collab_index_file", type=str, default=".index_collab_mscgrec.json",
                        help="collaborative-modality code index (decoding target space)")
    args = parser.parse_args()
    train(args)
