"""Train the CRNN-CTC recogniser on synthetic cheque text lines.

Handwriting font families and printed fonts are split train/val/test, so the
reported test CER is on writing styles never seen in training.

    python -m training.train_text_recognizer --steps 3500
"""
import argparse

import numpy as np

from training.common import Timer, parallel_map, save_metrics, seed_everything
from cheque_forensics.config import TEXT_RECOGNIZER_WEIGHTS


def _make(args):
    seed, split, n = args
    from cheque_forensics.synth.text_lines import render_line
    from cheque_forensics.text_recognizer import prepare_line
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g, text = render_line(rng, split)
        out.append(((prepare_line(g) * 255).astype(np.uint8), text))
    return out


def build(split, n, seed):
    chunks = parallel_map(_make, [(seed * 100_000 + i, split, 250) for i in range((n + 249) // 250)], chunksize=1)
    items = [x for c in chunks for x in c][:n]
    return np.stack([a for a, _ in items]), [t for _, t in items]


def cer(pred, gold):
    import difflib
    sm = difflib.SequenceMatcher(None, pred, gold)
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return edits / max(1, len(gold))


def evaluate(model, x, texts, batch=128):
    import torch
    from cheque_forensics.device import get_device
    from cheque_forensics.text_recognizer import ctc_decode
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(x), batch):
            xb = torch.from_numpy(x[k:k + batch, None].astype(np.float32) / 255.0).to(get_device())
            lp = model(xb).log_softmax(2).cpu().numpy()
            preds += [ctc_decode(p)[0] for p in lp]
    cers = [cer(p, t.strip()) for p, t in zip(preds, texts)]
    exact = [p == t.strip() for p, t in zip(preds, texts)]
    return {"cer": float(np.mean(cers)), "line_accuracy": float(np.mean(exact)), "n": len(texts)}, preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F
    from cheque_forensics.device import get_device, save_checkpoint
    from cheque_forensics.nets import CRNN
    from cheque_forensics.text_recognizer import CHARSET, encode

    seed_everything(0)
    t = Timer()
    xt, tt = build("train", args.n_train, 1)
    xv, tv = build("val", args.n_eval, 2)
    xs, ts = build("test", args.n_eval, 3)
    print(f"data built: train {xt.shape} val {xv.shape} test {xs.shape} ({t()})", flush=True)
    device = get_device()
    model = CRNN(len(CHARSET) + 1).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=args.steps)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    rng = np.random.default_rng(0)
    best, best_state, losses = 1e9, None, []
    for step in range(1, args.steps + 1):
        model.train()
        idx = rng.integers(0, len(xt), args.batch)
        xb = xt[idx].astype(np.float32) / 255.0
        xb = xb * rng.uniform(0.7, 1.0, (len(xb), 1, 1)) + rng.normal(0, 0.03, xb.shape)   # contrast/noise jitter
        xb = torch.from_numpy(np.clip(xb, 0, 1)[:, None].astype(np.float32)).to(device)
        xb = xb.contiguous(memory_format=torch.channels_last)
        targets = [encode(tt[i]) for i in idx]
        flat = torch.tensor([c for tgt in targets for c in tgt], dtype=torch.long)
        lens = torch.tensor([len(tgt) for tgt in targets], dtype=torch.long)
        logp = model(xb).log_softmax(2).permute(1, 0, 2)          # (T, B, C)
        in_lens = torch.full((len(idx),), logp.shape[0], dtype=torch.long)
        loss = ctc(logp.cpu(), flat, in_lens, lens)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 500 == 0 or step == args.steps:
            m, _ = evaluate(model, xv[:1000], tv[:1000])
            print(f"step {step} loss {np.mean(losses[-500:]):.3f} val CER {m['cer']:.4f} line acc {m['line_accuracy']:.3f} ({t()})", flush=True)
            if m["cer"] < best:
                best = m["cer"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val, _ = evaluate(model, xv, tv)
    test, preds = evaluate(model, xs, ts)
    examples = [{"gold": g, "pred": p} for g, p in list(zip(ts, preds))[:25]]
    save_checkpoint(model, TEXT_RECOGNIZER_WEIGHTS, charset=CHARSET, val=val, test=test)
    save_metrics("text_recognizer", {
        "val": val, "test": test, "steps": args.steps, "n_train": len(tt),
        "data": "synthetic cheque text lines; handwriting font families and printed fonts disjoint across splits",
        "caveat": "measured on synthetic renderings only; real handwriting accuracy is unmeasured",
        "test_examples": examples,
    })
    print("val", val, "test", test)


if __name__ == "__main__":
    main()
