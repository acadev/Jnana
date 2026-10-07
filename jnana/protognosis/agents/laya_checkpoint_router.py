"""Load archived Laya LoRA checkpoints as routers for native Jnana ensembles.

Heavy ML dependencies are imported only when a router is instantiated, so Jnana
can still be imported and tested without PyTorch/Laya installed.
"""
from __future__ import annotations

import json
from pathlib import Path


class CheckpointLayaRouter:
    """Laya ``predict``/``predict_batch`` adapter backed by one LoRA checkpoint."""

    def __init__(self, base_model, checkpoint, *, device="cuda", batch_size=32):
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from laya.agent import _fix_tokenizer_config
        from laya.common import build_model

        self.torch = torch
        self.base_model = Path(base_model).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.batch_size = batch_size
        _fix_tokenizer_config(str(self.base_model))
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model / "tokenizer")
        self.model_config = json.loads((self.base_model / "rl_agent_config.json").read_text())
        checkpoint_config = json.loads((self.checkpoint / "config.json").read_text())
        model = build_model(self.model_config, encoder_dir=str(self.base_model / "encoder"))
        model.load_state_dict(load_file(self.base_model / "model.safetensors"), strict=True)
        model.encoder = get_peft_model(model.encoder, LoraConfig(
            r=checkpoint_config["lora_rank"],
            lora_alpha=checkpoint_config["lora_alpha"],
            lora_dropout=checkpoint_config["lora_dropout"],
            target_modules=["Wqkv", "Wo"], bias="none",
        ))
        model.load_state_dict(load_file(self.checkpoint / "model.safetensors"), strict=True)
        self.model = model.to(self.device).eval()
        self.provenance = {
            "kind": "laya_lora_checkpoint",
            "base_model": str(self.base_model),
            "checkpoint": str(self.checkpoint),
        }

    def _collate(self, items):
        torch = self.torch
        max_tokens = max(len(item["ids"]) for item in items)
        max_markers = max(len(item["markers"]) for item in items)
        ids = torch.full(
            (len(items), max_tokens), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(ids)
        marker_positions = torch.zeros((len(items), max_markers), dtype=torch.long)
        marker_mask = torch.zeros((len(items), max_markers), dtype=torch.bool)
        qtypes = torch.zeros((len(items),), dtype=torch.long)
        for index, item in enumerate(items):
            ids[index, :len(item["ids"])] = torch.tensor(item["ids"])
            attention[index, :len(item["ids"])] = 1
            marker_positions[index, :len(item["markers"])] = torch.tensor(item["markers"])
            marker_mask[index, :len(item["markers"])] = True
            qtypes[index] = item["qtype"]
        return tuple(x.to(self.device) for x in (
            ids, attention, marker_positions, marker_mask, qtypes
        ))

    def predict_batch(self, requests, *, batch_size=None, sort_by_length=True):
        del sort_by_length
        from laya.common import build_sequence, QTYPES

        flat, metadata = [], []
        for request_index, request in enumerate(requests):
            for key, question in request["questions"].items():
                specification = {
                    "t": question["type"], "ins": question["instructions"],
                    "crit": question["criteria"],
                }
                sequence, markers = build_sequence(
                    self.tokenizer, request["state"], specification,
                    self.model_config["max_len"], self.model_config["head_max_len"],
                )
                flat.append({
                    "ids": sequence, "markers": markers,
                    "qtype": QTYPES[question["type"]],
                })
                metadata.append((request_index, key, list(question["criteria"])))
        outputs = [{"answers": {}} for _ in requests]
        size = batch_size or self.batch_size
        with self.torch.inference_mode():
            for offset in range(0, len(flat), size):
                chunk = flat[offset:offset + size]
                ids, attention, marker_positions, marker_mask, qtypes = self._collate(chunk)
                with self.torch.autocast("cuda", dtype=self.torch.float16,
                                         enabled=self.device.type == "cuda"):
                    logits, _ = self.model(
                        ids, attention, marker_positions, marker_mask, qtypes
                    )
                probabilities = self.torch.softmax(
                    logits.float().masked_fill(~marker_mask, -1e4), -1
                ).cpu()
                for local_index, row in enumerate(probabilities):
                    request_index, key, labels = metadata[offset + local_index]
                    values = {label: float(row[index]) for index, label in enumerate(labels)}
                    choice = max(values, key=values.get)
                    outputs[request_index]["answers"][key] = {
                        "choice": choice,
                        "answer_confidence": values[choice],
                        "probabilities": values,
                    }
        return outputs

    def predict(self, state, questions, **kwargs):
        return self.predict_batch(
            [{"state": state, "questions": questions}],
            batch_size=kwargs.get("batch_size"),
        )[0]


def checkpoint_router_specs(base_model, checkpoints, *, device="cuda", batch_size=32):
    """Build named router specifications accepted by ``LayaRankingAgent``.

    ``checkpoints`` entries are mappings with ``name``, ``path``, and optional
    ``weight``/``role`` fields.
    """
    specs = []
    for item in checkpoints:
        router = CheckpointLayaRouter(
            base_model, item["path"], device=device, batch_size=batch_size
        )
        specs.append({
            "name": item["name"], "router": router,
            "weight": float(item.get("weight", 1.0)),
            "provenance": {**router.provenance, "role": item.get("role")},
        })
    return specs
