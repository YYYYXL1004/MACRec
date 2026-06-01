"""
MSCGRec (2026) dataset — Multimodal Semantic & Collaborative Generative Rec.

Faithful core (pragmatic-baseline, see reproduce_unger_mscgrec.md):
  * Each item carries THREE independent modality codes (not fused):
      text  : residual-RQ of Llama text emb   -> prefixes a/b/c/d  (index_file)
      image : residual-RQ of CLIP image emb    -> prefixes A/B/C/D  (image_index_file)
      collab: residual-RQ of SASRec item emb   -> prefixes P/Q/R/S  (collab_index_file)
  * History items are encoded with ALL modalities concatenated (multimodal encoding).
  * The next item is DECODED by a SINGLE modality — the collaborative codes (Eq.2),
    with constrained beam search over the collaborative-code trie at inference.

Approximated/omitted (documented): DINO self-supervised image quantization is replaced
by standard RQ on CLIP features; dual relative-position embeddings, empty-leaf
redistribution and training-time random modality masking are omitted. Constrained
sequence learning is applied at inference (trie) rather than in the training softmax.
"""
import json
import os

from data import SeqRecDataset


class MSCGRecDataset(SeqRecDataset):

    def _load_data(self):
        # index_file=text, image_index_file=image, collab_index_file=collab
        with open(os.path.join(self.data_path, self.dataset + ".inter.json")) as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.args.index_file)) as f:
            self.text_indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.args.image_index_file)) as f:
            self.image_indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.args.collab_index_file)) as f:
            self.collab_indices = json.load(f)
        # `self.indices` is the RETRIEVAL/target space = collaborative codes
        self.indices = self.collab_indices

    def _remap_items(self):
        # history representation: multimodal (text + image + collab) codes concatenated
        self.remapped_mm = {}
        # target representation: collaborative codes only (single-modality decoding)
        self.remapped_cb = {}
        for uid, items in self.inters.items():
            mm, cb = [], []
            for i in items:
                si = str(i)
                mm.append("".join(self.text_indices[si] + self.image_indices[si] + self.collab_indices[si]))
                cb.append("".join(self.collab_indices[si]))
            self.remapped_mm[uid] = mm
            self.remapped_cb[uid] = cb

    def _process_train_data(self):
        inter_data = []
        for uid in self.remapped_mm:
            mm = self.remapped_mm[uid][:-2]
            cb = self.remapped_cb[uid][:-2]
            orig_ids = self.inters[uid][:-2]
            for i in range(1, len(mm)):
                history = mm[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                inter_data.append({
                    "item": cb[i],                       # target = collaborative codes
                    "target_item_id": int(orig_ids[i]),
                    "inters": history,                   # history = multimodal codes
                })
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_mm:
            mm, cb = self.remapped_mm[uid], self.remapped_cb[uid]
            history = mm[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inter_data.append({
                "item": cb[-2],
                "target_item_id": int(self.inters[uid][-2]),
                "inters": history,
            })
        return inter_data

    def _process_test_data(self):
        import numpy as np
        inter_data = []
        for uid in self.remapped_mm:
            mm, cb = self.remapped_mm[uid], self.remapped_cb[uid]
            history = mm[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inter_data.append({
                "item": cb[-1],
                "target_item_id": int(self.inters[uid][-1]),
                "inters": history,
            })
        if self.sample_num > 0:
            idx = np.random.choice(range(len(inter_data)), self.sample_num, replace=False)
            inter_data = np.array(inter_data)[idx].tolist()
        return inter_data

    def get_all_items(self):
        # retrieval candidates = collaborative code strings (unique per item)
        if self.all_items is not None:
            return self.all_items
        self.all_items = set("".join(v) for v in self.collab_indices.values())
        return self.all_items

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        toks = set()
        for src in (self.text_indices, self.image_indices, self.collab_indices):
            for code in src.values():
                toks.update(code)
        self.new_tokens = sorted(toks)
        return self.new_tokens
