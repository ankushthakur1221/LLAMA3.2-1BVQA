"""Frozen language model + trainable bridge: loss for training, greedy answers for evaluation."""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bridges import build_bridge

# How the image and the question are laid out for each chat format. The visual tokens sit between
# `before` and `after`; the answer follows `after` and is closed by `end` (None: the eos token).
TEMPLATES = {
    "llama3": {
        "bos": True, "before": "<|start_header_id|>user<|end_header_id|>\n\n",
        "after": "Question: {question} Short answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "sep": "", "end": "<|eot_id|>"},
    "chatml": {
        "bos": False, "before": "<|im_start|>user\n",
        "after": "Question: {question} Short answer:<|im_end|>\n<|im_start|>assistant\n",
        "sep": "", "end": "<|im_end|>"},
    "plain": {"bos": True, "before": "", "after": " Question: {question} Short answer:", "sep": " ", "end": None},
}


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def clean_answer(text):
    text = text.split("\n")[0].strip().lower()
    return text[:-1].strip() if text.endswith(".") else text


class BridgedVQA(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg, self.device = cfg, device
        llm = cfg["llm"]
        self.tok = AutoTokenizer.from_pretrained(llm["name"])
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(llm["name"], torch_dtype=dtype)
        self.lm.requires_grad_(False)
        self.lm.eval()
        if llm["lora"]["enabled"]:
            from peft import LoraConfig, get_peft_model
            lora = llm["lora"]
            # LoRA layers are injected into self.lm in place, so self.lm keeps working as the model.
            get_peft_model(self.lm, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                               target_modules=list(lora["targets"]), task_type="CAUSAL_LM"))
        self.embed = self.lm.get_input_embeddings()
        self.bridge = build_bridge(cfg, self.embed.weight.shape[1])
        self._match_embedding_scale()
        self.blind = cfg["bridge"]["blind"]
        self.to(device)

        t = TEMPLATES[llm["template"]]
        self.after, self.sep = t["after"], t["sep"]
        bos = [self.tok.bos_token_id] if t["bos"] and self.tok.bos_token_id is not None else []
        self.prefix = bos + self.tok(t["before"], add_special_tokens=False).input_ids
        self.end = self.tok(t["end"] or self.tok.eos_token, add_special_tokens=False).input_ids
        self.stop_ids = sorted(set(self.end[:1] + [self.tok.eos_token_id]))

    def _match_embedding_scale(self):
        """Start the visual tokens at the same scale as the language model's own word embeddings."""
        v = self.cfg["vision"]
        with torch.no_grad():
            self.bridge.eval()
            out = self.bridge(torch.randn(4, v["tokens"], v["dim"]))
            scale = self.embed.weight.float().std() / out.std()
            self.bridge.last.weight.mul_(scale)
            self.bridge.last.bias.mul_(scale)

    def autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def trainable(self):
        """Parameter groups: the bridge, and the LoRA weights when enabled."""
        lora = [p for p in self.lm.parameters() if p.requires_grad]
        groups = [{"params": list(self.bridge.parameters()), "lr": self.cfg["train"]["lr"]}]
        if lora:
            groups.append({"params": lora, "lr": self.cfg["llm"]["lora"]["lr"]})
        return groups

    def trainable_state(self):
        return {"bridge": self.bridge.state_dict(),
                "lora": {n: p.detach().cpu() for n, p in self.lm.named_parameters() if p.requires_grad}}

    def load_trainable_state(self, state):
        self.bridge.load_state_dict(state["bridge"])
        if state["lora"]:
            self.lm.load_state_dict(state["lora"], strict=False)

    def encode_image(self, feats, need_attn=False):
        feats = feats.to(self.device, dtype=torch.float32)
        if self.blind:
            feats = torch.zeros_like(feats)
        return self.bridge(feats, need_attn)

    def _assemble(self, vis, prompts, answers=None):
        """Embeddings for [prefix, visual tokens, prompt, (answer, end)].
        Training rows are padded on the right; generation rows on the left."""
        b, k = vis.shape[:2]
        rows = []
        for i in range(b):
            ctx = self.prefix + [self.tok.pad_token_id] * k + prompts[i]
            rows.append((ctx + (answers[i] + self.end if answers else []), len(ctx)))
        width = max(len(r) for r, _ in rows)
        ids = torch.full((b, width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((b, width), dtype=torch.long)
        labels = torch.full((b, width), -100, dtype=torch.long)
        start = torch.zeros(b, dtype=torch.long)
        for i, (row, n_ctx) in enumerate(rows):
            off = 0 if answers else width - len(row)
            ids[i, off:off + len(row)] = torch.tensor(row)
            mask[i, off:off + len(row)] = 1
            labels[i, off + n_ctx:off + len(row)] = torch.tensor(row[n_ctx:], dtype=torch.long)
            start[i] = off + len(self.prefix)
        embeds = self.embed(ids.to(self.device))
        cols = start[:, None] + torch.arange(k)[None, :]
        embeds[torch.arange(b)[:, None], cols.to(self.device)] = vis.to(embeds.dtype)
        return embeds, mask.to(self.device), labels.to(self.device)

    def _tokenize(self, texts):
        return self.tok(texts, add_special_tokens=False).input_ids

    def loss(self, feats, questions, answers):
        """Cross-entropy on the answer tokens only; the output layer runs on those positions alone."""
        vis = self.encode_image(feats)
        prompts = self._tokenize([self.after.format(question=q) for q in questions])
        embeds, mask, labels = self._assemble(vis, prompts, self._tokenize([self.sep + a for a in answers]))
        hidden = self.lm.base_model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        target = labels[:, 1:]
        keep = target != -100
        logits = self.lm.get_output_embeddings()(hidden[:, :-1][keep])
        return F.cross_entropy(logits.float(), target[keep])

    @torch.no_grad()
    def answer(self, feats, questions, need_attn=False):
        out = self.encode_image(feats, need_attn)
        vis, attns = out if need_attn else (out, None)
        prompts = self._tokenize([self.after.format(question=q) for q in questions])
        embeds, mask, _ = self._assemble(vis, prompts)
        tokens = self.lm.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False, temperature=None,
                                  top_p=None, top_k=None, max_new_tokens=self.cfg["eval"]["max_new_tokens"],
                                  eos_token_id=self.stop_ids, pad_token_id=self.tok.pad_token_id)
        answers = [clean_answer(t) for t in self.tok.batch_decode(tokens, skip_special_tokens=True)]
        return (answers, attns) if need_attn else answers

    def count_trainable(self):
        return sum(p.numel() for g in self.trainable() for p in g["params"])
