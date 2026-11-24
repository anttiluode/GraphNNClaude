"""Capacity gate: K images in one fixed memory, edge transport vs addressed slots.

Question: the 2-image GraphNN task is at ceiling for every arm. Does a memory
stored in the transport (edge valves read by a ping) degrade more gracefully
than the same bytes split into K fixed slots, as K grows at a fixed budget?

Arms reuse the transports of GraphNNSol Gate 1 (Sol, MIT) so results line up:
  positive  [0,1] edge gain, fixed degree normalisation        896 bytes
  signed    [-1,1] edge gain, fixed degree normalisation       896 bytes
  binary    exactly -1/+1 edges (STE in training, packed bits) 112 bytes
  phase     unit-modulus conjugate edge rotations              896 bytes
  slots     K disjoint channel groups, one per angle, direct   896 bytes
            read (no transport); each image owns 896/K bytes

Same writer, ping, decoder and loss for every arm. Item k is written at angle
90*k/(K-1). Evaluation always round-trips memory through uint8 / packed bits.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from graphnn.data import procedural_pairs
from graphnn.half_memory import HalfEdgeWriter
from graphnn.model import JanusModel, ModelConfig, _angles
from graphnn.rewrite import PALETTE

MODES = ("positive", "signed", "binary", "phase", "slots")
# Gate 2 (reader bottleneck and cuff read). "direct", "signed_sweep" and
# "elastic_sweep" share signed bytes; only how the memory is READ differs.
GATE2 = ("direct", "elastic", "elastic_sweep", "signed_sweep")
SWEEP = (0.5, 1.0, 2.0)


def pack_bits(x):
    shifts = torch.arange(8, device=x.device, dtype=torch.int64)
    return (x.to(torch.int64).reshape(x.shape[0], -1, 8) << shifts).sum(-1).to(torch.uint8)


def unpack_bits(x):
    shifts = torch.arange(8, device=x.device, dtype=torch.int64)
    return ((x.to(torch.int64)[..., None] >> shifts) & 1).flatten(1).float()


class Memory:
    def __init__(self, values, bits, packed=False):
        self.values, self.bits, self.packed = values, bits, packed

    def flat(self):
        if not self.packed:
            return self.values
        return unpack_bits(self.values) if self.bits == 1 else self.values.float() / 255

    def quantized(self):
        if self.packed:
            return self
        x = self.values.detach().clamp(0, 1)
        if self.bits == 1:
            return Memory(pack_bits((x >= .5).to(torch.uint8)), 1, True)
        return Memory((255 * x).round().to(torch.uint8), 8, True)

    def roll(self):
        return Memory(self.values.roll(1, 0), self.bits, self.packed)


class Valve(nn.Module):
    def __init__(self, mode, k, cfg=ModelConfig()):
        super().__init__()
        if cfg.channels % k:
            raise ValueError("K must divide the channel count for equal slots")
        self.mode, self.k, self.cfg = mode, k, cfg
        self.bits = 1 if mode == "binary" else 8
        self.writer = HalfEdgeWriter(cfg)
        self.backbone = JanusModel(cfg)
        self.backbone.writer = nn.Identity()
        self.n = 2 * cfg.channels * cfg.grid_size * (cfg.grid_size - 1)
        self.payload_bytes = self.n * self.bits // 8

    def angles(self):
        return [90. * j / (self.k - 1) for j in range(self.k)]

    def planes(self, x):
        c, g = self.cfg.channels, self.cfg.grid_size
        h, v = x.chunk(2, 1)
        return h.reshape(-1, c, g, g - 1), v.reshape(-1, c, g - 1, g)

    def slot_mask(self, theta, dtype):
        c, g = self.cfg.channels, self.cfg.grid_size
        item = (theta / (math.pi / 2) * (self.k - 1)).round().long()
        group = torch.arange(c, device=theta.device)[None, :] // (c // self.k)
        mask = (group == item[:, None]).to(dtype)
        return mask.repeat_interleave(g * (g - 1), dim=1).repeat(1, 2)

    def write(self, image, angle, previous=None):
        theta = _angles(angle, len(image), image.device)
        if previous is None:
            old = image.new_full((len(image), self.n), .5)
        else:
            old = previous.flat()
        x = self.writer(image, theta, self.planes(old)).flat()
        if self.mode == "binary":
            x = x + ((x >= .5).to(x.dtype) - x).detach()
        if self.mode == "slots":
            if previous is None:
                old = torch.full_like(old, 128 / 255)
            mask = self.slot_mask(theta, x.dtype)
            x = x * mask + old * (1 - mask)
        return Memory(x, self.bits)

    def transport(self, r, i, h, v):
        def nbr(a, b, c, d):
            return (F.pad(a, (0, 1, 0, 0)) + F.pad(b, (1, 0, 0, 0))
                    + F.pad(c, (0, 0, 0, 1)) + F.pad(d, (0, 0, 1, 0)))
        deg = nbr(torch.ones_like(h), torch.ones_like(h), torch.ones_like(v), torch.ones_like(v))
        if self.mode == "phase":
            hc = torch.polar(torch.ones_like(h), (2 * h - 1) * math.pi)
            vc = torch.polar(torch.ones_like(v), (2 * v - 1) * math.pi)
            z = torch.complex(r, i)
            y = .5 * z + .5 * nbr(hc * z[..., 1:], hc.conj() * z[..., :-1],
                                  vc * z[..., 1:, :], vc.conj() * z[..., :-1, :]) / deg
            return y.real, y.imag
        wh, wv = (h, v) if self.mode == "positive" else (2 * h - 1, 2 * v - 1)
        if self.mode.startswith("elastic"):
            # Elastic valve: a byte sets a signed opening, and how much pressure the
            # edge needs before it conducts. Strong edges open at low pressure, weak
            # edges only at high pressure, so the transport is NONLINEAR in the ping.
            m = torch.sqrt(r.square() + i.square() + 1e-8)
            ph = .5 * (m[..., :-1] + m[..., 1:])
            pv = .5 * (m[..., :-1, :] + m[..., 1:, :])
            wh = torch.tanh(3 * wh) * torch.sigmoid(6 * (ph - (1 - wh.abs())))
            wv = torch.tanh(3 * wv) * torch.sigmoid(6 * (pv - (1 - wv.abs())))
        def scalar(z):
            return nbr(wh * z[..., 1:], wh * z[..., :-1], wv * z[..., 1:, :], wv * z[..., :-1, :])
        return .5 * r + .5 * scalar(r) / deg, .5 * i + .5 * scalar(i) / deg

    def read(self, memory, angle):
        x = memory.flat()
        theta = _angles(angle, len(x), x.device)
        z = self.backbone._fresh_ping(theta)
        r, i = z.real, z.imag
        if self.mode == "slots":
            h, v = self.planes((2 * x - 1) * self.slot_mask(theta, x.dtype))
            h, v = F.pad(h, (0, 1, 0, 0)), F.pad(v, (0, 0, 0, 1))
            features = [r, i, h, v, r * h, i * v]
        elif self.mode.endswith("sweep"):
            # Cuff read: the same ping at three pressures, each answer rescaled.
            # On a linear medium the three rescaled answers are identical.
            h, v = self.planes(x)
            features = []
            for amp in SWEEP:
                rr, ii = amp * r, amp * i
                for _ in range(self.cfg.steps):
                    rr, ii = self.transport(rr, ii, h, v)
                features.extend((rr / amp, ii / amp))
        elif self.mode == "direct":
            # Signed transport, plus the raw bytes handed straight to the decoder.
            h, v = self.planes(x)
            hd, vd = F.pad(2 * h - 1, (0, 1, 0, 0)), F.pad(2 * v - 1, (0, 0, 0, 1))
            for _ in range(self.cfg.steps):
                r, i = self.transport(r, i, h, v)
            features = [z.real, z.imag, r, i, hd, vd]
        else:
            h, v = self.planes(x)
            features = [r, i]
            for step in range(1, self.cfg.steps + 1):
                r, i = self.transport(r, i, h, v)
                if step in (self.cfg.steps // 2, self.cfg.steps):
                    features.extend((r, i))
        size = (self.cfg.image_size,) * 2
        return self.backbone.decoder(F.interpolate(torch.cat(features, 1), size=size,
                                                   mode="bilinear", align_corners=False))


def make_items(count, k, seed):
    """[count, K, 3, 32, 32] procedural images, each with one 6x6 two-colour mark."""
    pairs = [procedural_pairs(count, seed=seed * 31 + j) for j in range((k + 1) // 2)]
    images = torch.cat(pairs, dim=1)[:, :k].clone()
    gen = torch.Generator().manual_seed(seed + 5)
    side, size = 6, images.shape[-1]
    pos = torch.randint(0, size - side + 1, (count, k, 2), generator=gen)
    labels = torch.randint(0, 2, (count, k), generator=gen)
    target = torch.randint(0, k, (count,), generator=gen)
    pal = torch.tensor(PALETTE)
    masks = torch.zeros(count, k, 1, size, size)
    for n in range(count):
        for j in range(k):
            y, x = pos[n, j].tolist()
            masks[n, j, :, y:y + side, x:x + side] = 1
            images[n, j, :, y:y + side, x:x + side] = pal[labels[n, j]][:, None, None]
    flipped = images.clone()
    for n in range(count):
        for j in range(k):
            y, x = pos[n, j].tolist()
            flipped[n, j, :, y:y + side, x:x + side] = pal[1 - labels[n, j]][:, None, None]
    return {"images": images, "flipped": flipped, "masks": masks,
            "labels": labels, "target": target}


def take(batch, idx):
    return {key: value[idx] for key, value in batch.items()}


def to(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def mark_mse(pred, target, mask):
    return ((pred - target).square() * mask).sum() / (mask.sum() * 3).clamp_min(1)


def write_all(model, batch, quantize):
    memory = None
    for j, angle in enumerate(model.angles()):
        memory = model.write(batch["images"][:, j], angle, memory)
        if quantize:
            memory = memory.quantized()
    return memory


def rewrite_target(model, batch, memory, quantize):
    """Rewrite one item per example (its mark flipped), at that item's own angle."""
    rows = torch.arange(len(batch["target"]), device=batch["target"].device)
    image = batch["flipped"][rows, batch["target"]]
    angle = torch.tensor(model.angles(), device=image.device)[batch["target"]]
    memory = model.write(image, angle, memory)
    return memory.quantized() if quantize else memory


def train_step(model, opt, batch):
    model.train()
    opt.zero_grad(set_to_none=True)
    full = write_all(model, batch, quantize=False)
    after = rewrite_target(model, batch, full, quantize=False)
    terms = []
    for j, angle in enumerate(model.angles()):
        changed = (batch["target"] == j)[:, None, None, None].float()
        new_target = changed * batch["flipped"][:, j] + (1 - changed) * batch["images"][:, j]
        mask = batch["masks"][:, j]
        for memory, target in ((full, batch["images"][:, j]), (after, new_target)):
            pred = model.read(memory, angle)
            terms.append((pred - target).square().mean() + .5 * mark_mse(pred, target, mask))
    loss = torch.stack(terms).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()
    return float(loss.detach())


def hits(pred, mask, label):
    colour = (pred * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    pal = pred.new_tensor(PALETTE)
    return (colour[:, None, :] - pal[None, :, :]).square().sum(-1).argmin(-1) == label


@torch.no_grad()
def score(model, memory, batch, flipped_items):
    """flipped_items: [N,K] bool, which items should now show the flipped mark."""
    per_hit, per_mse = [], []
    for j, angle in enumerate(model.angles()):
        f = flipped_items[:, j]
        target = torch.where(f[:, None, None, None], batch["flipped"][:, j], batch["images"][:, j])
        label = torch.where(f, 1 - batch["labels"][:, j], batch["labels"][:, j])
        pred = model.read(memory, angle)
        per_hit.append(hits(pred, batch["masks"][:, j], label))
        per_mse.append((pred - target).square().mean((1, 2, 3)))
    h, m = torch.stack(per_hit, 1).float(), torch.stack(per_mse, 1)
    return {"item_mark": float(h.mean()), "all_marks": float(h.all(1).float().mean()),
            "rgb_mse": float(m.mean())}


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    k, n = model.k, len(batch["target"])
    none = torch.zeros(n, k, dtype=torch.bool, device=batch["target"].device)
    one = none.clone()
    one[torch.arange(n), batch["target"]] = True
    full = write_all(model, batch, quantize=True)
    after = rewrite_target(model, batch, full, quantize=True)
    # chain: rewrite every item once, in order, then read all
    chain = full
    for j, angle in enumerate(model.angles()):
        chain = model.write(batch["flipped"][:, j], angle, chain).quantized()
    everyone = torch.ones_like(none)
    # untouched-item drift caused by the single rewrite
    drift = []
    for j, angle in enumerate(model.angles()):
        keep = ~one[:, j]
        if keep.any():
            d = (model.read(after, angle) - model.read(full, angle)).square().mean((1, 2, 3))
            drift.append(d[keep])
    out = {"full": score(model, full, batch, none),
           "rewrite": score(model, after, batch, one),
           "chain": score(model, chain, batch, everyone),
           "shuffled_floor": score(model, full.roll(), batch, none),
           "untouched_drift": float(torch.cat(drift).mean()),
           "payload_bytes": model.payload_bytes,
           "bytes_per_item": model.payload_bytes / k,
           "shared_params": sum(p.numel() for p in model.parameters())}
    return out


def self_test():
    torch.manual_seed(0)
    for k in (2, 4, 8):
        batch = make_items(4, k, seed=3)
        assert batch["images"].shape == (4, k, 3, 32, 32)
        assert not torch.equal(batch["images"], batch["flipped"])
        params = set()
        # linear transport: the rescaled cuff answers must be identical
        lin = Valve("signed_sweep", k)
        x = torch.rand(4, lin.n)
        h, v = lin.planes(x)
        z = torch.randn(4, 8, 8, 8)
        a = lin.transport(2 * z, 2 * z, h, v)[0] / 2
        b = lin.transport(z, z, h, v)[0]
        assert torch.allclose(a, b, atol=1e-5), "signed transport should be linear"
        el = Valve("elastic", k)
        a = el.transport(2 * z, 2 * z, h, v)[0] / 2
        b = el.transport(z, z, h, v)[0]
        assert not torch.allclose(a, b, atol=1e-3), "elastic transport should not be linear"
        for mode in MODES + GATE2:
            model = Valve(mode, k)
            params.add(sum(p.numel() for p in model.parameters()))
            m = write_all(model, batch, quantize=True)
            assert m.values.dtype == torch.uint8
            assert m.values.shape[1] == model.payload_bytes
            if mode == "slots":
                # writing item 0 must not touch any other item's slot
                m0 = model.write(batch["images"][:, 0], 0., None).quantized()
                base = torch.full((4, model.n), 128 / 255)
                other = model.slot_mask(_angles(0., 4, "cpu"), torch.float32) == 0
                assert torch.allclose(m0.flat()[other], (base[other] * 255).round() / 255)
            evaluate(model, batch)
        assert len(params) == 1, params
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 97, 127])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-count", type=int, default=256)
    ap.add_argument("--test-count", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path("runs/capacity"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"config": {key: (str(v) if isinstance(v, Path) else v)
                          for key, v in vars(args).items()}, "runs": {}}
    for k in args.ks:
        for seed in args.seeds:
            train = make_items(args.train_count, k, seed)
            test = to(make_items(args.test_count, k, seed + 10000), args.device)
            for mode in args.modes:
                torch.manual_seed(seed)
                model = Valve(mode, k).to(args.device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                gen = torch.Generator().manual_seed(seed)
                start = time.perf_counter()
                for step in range(1, args.steps + 1):
                    idx = torch.randint(0, args.train_count, (args.batch_size,), generator=gen)
                    loss = train_step(model, opt, to(take(train, idx), args.device))
                    if step == 1 or step % 250 == 0 or step == args.steps:
                        print(f"K={k} seed={seed} {mode:8s} step {step}: {loss:.4f}", flush=True)
                report = evaluate(model, test)
                report["train_seconds"] = round(time.perf_counter() - start, 1)
                results["runs"][f"K{k}/seed{seed}/{mode}"] = report
                print(f"K={k} seed={seed} {mode:8s} " + json.dumps(
                    {s: report[s] for s in ("full", "rewrite", "chain", "shuffled_floor")}), flush=True)
                (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    means = {}
    for k in args.ks:
        for mode in args.modes:
            runs = [results["runs"][f"K{k}/seed{s}/{mode}"] for s in args.seeds]
            means[f"K{k}/{mode}"] = {
                block: {m: round(sum(r[block][m] for r in runs) / len(runs), 5)
                        for m in runs[0][block]}
                for block in ("full", "rewrite", "chain", "shuffled_floor")}
            means[f"K{k}/{mode}"]["untouched_drift"] = round(
                sum(r["untouched_drift"] for r in runs) / len(runs), 6)
            means[f"K{k}/{mode}"]["bytes_per_item"] = runs[0]["bytes_per_item"]
    results["means"] = means
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(means, indent=1))


if __name__ == "__main__":
    main()
