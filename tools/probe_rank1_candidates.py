#!/usr/bin/env python3
"""Is a published "abliterated" checkpoint actually a rank-1 directional edit?

    python3 tools/probe_rank1_candidates.py                      # the three from §2
    python3 tools/probe_rank1_candidates.py --abl OTHER/REPO     # any candidate
    python3 tools/probe_rank1_candidates.py --abl A --abl B --json out.json

Answers it for ~430 MB per candidate instead of a 54 GB download, by reading
individual tensors out of the remote safetensors over HTTP range requests: 8 bytes
of header length, then the JSON header, then exactly the byte range of the tensor
you asked for. Nothing else crosses the wire.

WHY THIS EXISTS. "Abliterated" in a repo name tells you nothing about *how* the
model was edited. Of the three published Qwen3.8-27B ablations measured for §2,
only one was a clean rank-1 directional edit; the other two are projection-shaped
or plain fine-tunes, and the runtime dial cannot reproduce them at any λ. Measuring
that ahead of time costs minutes. Finding out afterwards costs a full extraction
run and a wrong conclusion.

WHAT IT REPORTS, per module:

  s0/s1        top singular value of ΔW over the second. Near 1 means NO dominant
               direction exists — whatever that repo did, it was not a directional
               ablation. This is the gate that rejected `trohrbaugh` (1.14–4.59).
  rank1E       fraction of ΔW's energy in the first singular component.
  cos          |cos(v0, u0ᵀW_base)|. Asks whether ΔW even has the *shape* of a
               projection −λ·r̂r̂ᵀW. This is what rejected `trohrbaugh` (0.17–0.46).
  λ_eff        how many times the edit removes the direction. ≈1.0 is a clean single
               removal. Overshoot is a real failure mode and it is not rare:
               Blackfrost-AI/…-ABLITERATED-BF16 measures 2.599 — past the DeepSeek-V4
               regime (2.43) where the component is not removed but *inverted*.
  coherence    |cos| between the dominant directions of DIFFERENT modules. High means
               one global direction (Ektome ≥0.9995); low means the edit is per-layer
               and there is no single axis to extract.

READ s0/s1 BEFORE cos. When s0 ≈ s1 the "dominant direction" u0 is arbitrary, so its
cos against anything is noise dressed as a measurement. A candidate with cos = 0.99
and s0/s1 = 1.2 is not projection-shaped; it is undetermined.

Baseline for "is this cos meaningful": two random unit vectors in R^d have
|cos| ≈ 1/sqrt(d) — 0.0140 at d=5120 (the 27B), 0.0198 at d=2560 (flash-next). The
floor is reported per tensor from that tensor's own width; quoting the 27B's number
for a narrower model makes a noise-level cos look meaningful.

Gated repos need a read token (HF_TOKEN, or ~/.cache/huggingface/token). Without one a
gated side answers 401/403 and you get an AccessError naming the repo — not, as before,
a KeyError naming a tensor.

Quantized checkpoints are read, not skipped: modelopt-NVFP4, compressed-tensors NVFP4
and F8_E4M3 are dequantized on the fly (see the layout note above `E2M1`). A tensor
whose recipe differs between the two sides is marked `format_mismatch` and kept out of
the aggregate, because a cross-precision difference measures requantization noise as
much as it measures the edit.

    python3 tools/probe_rank1_candidates.py --preflight pf.json --save-direction d.npy \\
        --against /opt/refusal/refusal_direction_flashnext.npy --json results.json

Pin revisions with --rev when you publish a number. HuggingFace `main` moves, and a
measurement against a moving target is not reproducible; the resolved SHA is recorded as
`revision_sha` from the `x-revision` header.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pathlib
import struct
import sys
import urllib.error
import urllib.request

import numpy as np

UA = {"User-Agent": "qwen38-rank1-probe/1.0"}

BASE_DEFAULT = "Qwen/Qwen3.8-27B"
# The three measured for §2 of the README. The verdicts there came from this code.
CANDIDATES_DEFAULT = [
    "Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored",
    "trohrbaugh/Qwen3.8-27B-heretic-ara",
    "orwelian84/Qwen3.8-27B-OBLITERATUS-Advanced",
]

# Modules that WRITE TO THE RESIDUAL STREAM — the only ones where a refusal direction
# can live. Layer 20 is a linear-attention layer, 31/47/63 are full-attention
# (`full_attention_interval: 4`, so layers 3, 7, … 63). Sampling both kinds and
# spreading across depth catches per-layer edits that a single probe would miss.
MODULES_DEFAULT = [
    "model.language_model.layers.20.linear_attn.out_proj.weight",
    "model.language_model.layers.31.self_attn.o_proj.weight",
    "model.language_model.layers.47.self_attn.o_proj.weight",
    "model.language_model.layers.63.mlp.down_proj.weight",
]

# Reference floor for the 27B the tool was written against. Do NOT print this for a
# model of another width: the floor is 1/sqrt(dim), so for flash-next (dim 2560) it is
# 0.0198, not 0.0140, and quoting the wrong one makes a noise-level cos look meaningful.
RANDOM_COS_5120 = 1.0 / np.sqrt(5120)  # 0.01398


def random_cos_floor(dim: int) -> float:
    """|cos| between two random unit vectors in R^dim. Anything at this level is
    orthogonal, not related — so it is the floor a measured cos has to clear."""
    return float(1.0 / np.sqrt(dim))


class AccessError(RuntimeError):
    """The repo exists but this token may not read it (gated, or not accepted yet).

    Distinct from "the tensor is not there": the old code swallowed the 401/403 into
    `None` and the failure surfaced three frames later as a `KeyError` naming a tensor,
    which reads like a wrong module name and sends you looking for a typo.
    """


@functools.lru_cache(maxsize=1)
def _hf_token() -> str:
    """Read token from HF_TOKEN, else the huggingface_hub cache file. Empty if neither.

    The public sides of a pair need no token, so an absent token is not an error here —
    it only becomes one when a gated side answers 401/403.
    """
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    if not tok:
        p = pathlib.Path.home() / ".cache" / "huggingface" / "token"
        if p.is_file():
            tok = p.read_text(encoding="utf-8").strip()
    return tok


def _open(url: str, rng: tuple[int, int] | None = None):
    h = dict(UA)
    tok = _hf_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180)


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    with _open(url, rng) as r:
        return r.read()


def revision_sha(repo: str, rev: str) -> str | None:
    """The commit a revision string resolves to.

    Publish numbers against this, not against `main`: orcarouter's repos were modified on
    27-10, after the 03-09 extraction, so a measurement pinned to the branch name cannot
    be re-derived.

    Read from `GET /api/models/<repo>/revision/<rev>`, not from an `x-revision` response
    header — huggingface.co does not send that header on `resolve/` (measured 17-11: the
    resolve response carries only an ETag, and it is not the commit). The API echoes a
    SHA that is already pinned, so the field is always the resolved commit.
    """
    try:
        raw = _get(f"https://huggingface.co/api/models/{repo}/revision/{rev}")
        return json.loads(raw).get("sha")
    except Exception:  # noqa: BLE001 — a missing pin is reported, never fatal
        return None


def _resolve(repo: str, fname: str, rev: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"


@functools.lru_cache(maxsize=64)
def _index(repo: str, rev: str):
    """tensor -> shard. Returns None for single-shard repos (no index file).

    Only a genuine 404 means "no index". A 401/403 is a missing grant and must not be
    reported as a missing tensor.
    """
    try:
        raw = _get(_resolve(repo, "model.safetensors.index.json", rev))
        return json.loads(raw)["weight_map"]
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise AccessError(
                f"{repo}@{rev}: HTTP {e.code}, sin acceso con este token. Es un repo "
                f"gated: pide acceso en https://huggingface.co/{repo} (o usa un token de "
                f"una cuenta que ya lo tenga) y vuelve a correr."
            ) from None
        if e.code == 404:
            return None  # de verdad no hay index: monoshard
        raise


@functools.lru_cache(maxsize=256)
def _header(repo: str, shard: str, rev: str):
    """The safetensors header: u64 length, then that many bytes of JSON. A few KB."""
    url = _resolve(repo, shard, rev)
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    return json.loads(_get(url, (8, 8 + n - 1))), 8 + n


# --- quantized layouts -------------------------------------------------------------
# Two families show up in the flash-next NVFP4 checkpoints and they are NOT the same
# convention, which is the whole reason this section is explicit and unit-tested:
#
#   modelopt (RadixArk):           weight U8 + weight_scale F8_E4M3 + weight_scale_2 F32
#                                  value = e2m1(code) * block_scale * scale_2
#   compressed-tensors (orcarouter): weight_packed U8 + weight_scale F8_E4M3
#                                  + weight_global_scale F32
#                                  value = e2m1(code) * (block_scale / global_scale)
#
# The two agree because the formats are reciprocals of each other: the official
# modelopt -> compressed-tensors converter writes `weight_global_scale = 1 /
# weight_scale_2` (compressed_tensors/entrypoints/convert/converters/modelopt_nvfp4.py),
# and compressed-tensors' own `_dequantize` divides by it
# (quantization/lifecycle/forward_helpers.py). Multiplying on the compressed-tensors
# side — the obvious guess — is off by the square of the global scale.
#
# Packing and block size are shared: two E2M1 values per byte, LOW NIBBLE FIRST, one
# E4M3 block scale per 16 elements along the last dim (vLLM modelopt.py: "two e2m1
# values per byte (low nibble first), one e4m3 scale per 16 values and a per-tensor
# fp32 scale").
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
NVFP4_GROUP = 16


def _e2m1(codes: np.ndarray) -> np.ndarray:
    """4-bit codes -> floats. Bit 3 is the sign, bits 0-2 the magnitude (vLLM builds
    its lookup table as cat([0,.5,1,1.5,2,3,4,6], -same))."""
    mag = E2M1[codes.astype(np.uint8) & 0x07]
    return np.where(codes.astype(np.uint8) & 0x08, -mag, mag)


def _unpack_fp4(packed: np.ndarray) -> np.ndarray:
    """[..., n/2] uint8 -> [..., n] floats, low nibble first (lo, hi, lo, hi, ...)."""
    lo = packed & 0x0F
    hi = packed >> 4
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return _e2m1(out)


def _e4m3_table() -> np.ndarray:
    """FP8 E4M3 as a 256-entry byte->float table.

    Decoded by hand rather than via `numpy.float8_e4m3fn` because the numpy in this
    house (2.4.6) does not expose any float8 dtype — measured, not assumed. Format:
    sign, 4-bit exponent bias 7, 3-bit mantissa, no infinities, and E4M3's only NaN is
    the all-ones mantissa at max exponent (so 0x7F is NaN and 0x7E is the 448 max).
    """
    t = np.empty(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0x0F, b & 0x07
        if e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6          # subnormal
        elif e == 15 and m == 7:
            t[b] = float("nan")
        else:
            t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


FP8_E4M3 = _e4m3_table()


def _fp8(raw: bytes) -> np.ndarray:
    """F8_E4M3 buffer -> float64."""
    return FP8_E4M3[np.frombuffer(raw, dtype=np.uint8)]


def _bf16(raw: bytes) -> np.ndarray:
    """numpy has no bfloat16: widen by placing the 16 bits in the high half of a float32."""
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _apply_scale(vals: np.ndarray, scale: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """Multiply by a per-tensor, per-output-channel or grouped (along last dim) scale."""
    if scale is None:
        return vals
    if scale.size == 1:
        return vals * scale.reshape(-1)[0]
    s = scale
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if s.shape[-1] == 1:
        return vals * s.reshape(out_dim, 1)
    nblk = s.shape[-1]
    if in_dim % nblk:
        raise RuntimeError(f"scale blocks {nblk} does not divide in_dim {in_dim}")
    g = in_dim // nblk
    return (vals.reshape(out_dim, nblk, g) * s.reshape(out_dim, nblk, 1)).reshape(out_dim, in_dim)


def _stem(key: str) -> str:
    """Base name that the scale tensors hang off.

    Scales are named after the weight, not after the module: `...down_proj.weight_scale`
    sits beside `...down_proj.weight`, and `...weight_packed` beside `...weight_scale`.
    So only the `_packed` suffix comes off — stripping `.weight` too would look for
    `...down_proj_scale`, which no checkpoint has ever contained.
    """
    return key[: -len("_packed")] if key.endswith("_packed") else key


def quant_kind(hdr: dict, key: str) -> tuple[str, dict]:
    """Classify one tensor from its header: a format label plus the parts it needs.

    The label is what `format_mismatch` compares, so it names the *recipe*, not just the
    dtype: two sides that are both "4 bits" but scaled by reciprocal conventions are not
    a paired measurement.
    """
    meta = hdr[key]
    dt = meta["dtype"]
    st = _stem(key)
    has = lambda suf: f"{st}{suf}" in hdr  # noqa: E731
    none = {"scale": None, "global": None, "mode": "none"}

    if dt == "U8":
        glob, mode = (f"{st}_global_scale", "div") if has("_global_scale") else (
            (f"{st}_scale_2", "mul") if has("_scale_2") else (None, "none"))
        label = {"div": "ct-NVFP4/g16", "mul": "modelopt-NVFP4/g16"}.get(mode, "U8/unscaled")
        return label, {"scale": f"{st}_scale" if has("_scale") else None, "global": glob, "mode": mode}

    if dt == "F8_E4M3":
        sc = f"{st}_scale" if has("_scale") else None
        if sc is None:
            return "F8_E4M3/unscaled", none
        sshape = hdr[sc]["shape"]
        out_dim, in_dim = meta["shape"][0], meta["shape"][-1]
        if len(sshape) == 0 or (len(sshape) == 1 and sshape[0] == out_dim) or (len(sshape) >= 2 and sshape[-1] == 1):
            label = "F8_E4M3/per-channel"
        elif len(sshape) >= 2 and sshape[-1] and in_dim % sshape[-1] == 0:
            label = f"F8_E4M3/g{in_dim // sshape[-1]}"
        else:
            label = "F8_E4M3/other"
        glob, mode = (f"{st}_global_scale", "div") if has("_global_scale") else (None, "none")
        return label, {"scale": sc, "global": glob, "mode": mode}

    return dt, none


def resolve_key(repo: str, key: str, rev: str) -> str | None:
    """The physical key this side uses for a logical module.

    modelopt keeps the FP4 payload in `...weight`; compressed-tensors calls it
    `...weight_packed`. Asking for one name and getting a KeyError on the side that
    spells it the other way would look like "this module was not edited".
    """
    wm = _index(repo, rev)
    shard = "model.safetensors" if wm is None else wm.get(key) or wm.get(f"{key}_packed")
    if shard is None:
        for k in (wm or {}):
            if k == f"{key}_packed":
                shard = wm[k]
                break
    if shard is None:
        return None
    hdr, _ = _header(repo, shard, rev)
    for cand in (key, f"{key}_packed"):
        if cand in hdr:
            return cand
    return None


def _load_range(repo: str, shard: str, rev: str, hdr: dict, start: int, key: str):
    a, b = hdr[key]["data_offsets"]
    raw = _get(_resolve(repo, shard, rev), (start + a, start + b - 1))
    return raw, hdr[key]["dtype"]


def _scale_of(repo, shard, rev, hdr, start, parts):
    """The effective scale array for a quantized tensor, global scale already folded in
    (multiplied for modelopt, divided for compressed-tensors — see the note above)."""
    if not parts["scale"]:
        return None
    sraw, sdt = _load_range(repo, shard, rev, hdr, start, parts["scale"])
    sc = _fp8(sraw) if sdt == "F8_E4M3" else _bf16(sraw) if sdt == "BF16" else np.frombuffer(sraw, np.float32).astype(np.float64)
    sc = sc.reshape(hdr[parts["scale"]]["shape"])
    if parts["global"]:
        graw, gdt = _load_range(repo, shard, rev, hdr, start, parts["global"])
        g = np.frombuffer(graw, np.float32 if gdt == "F32" else np.float64).reshape(-1)
        sc = sc * g[0] if parts["mode"] == "mul" else sc / g[0]
    return sc


def tensor(repo: str, key: str, rev: str = "main", dtype=np.float64) -> np.ndarray:
    """One tensor as a float array [out, in], dequantized if needed.

    Only the byte ranges asked for cross the wire: the tensor, and its scale tensors.
    `key` is the logical name; `resolve_key` finds how this side spells it.
    """
    phys = resolve_key(repo, key, rev)
    if phys is None:
        raise KeyError(f"{key} not in {repo}")
    wm = _index(repo, rev)
    shard = "model.safetensors" if wm is None else (wm.get(phys) or wm.get(key))
    hdr, start = _header(repo, shard, rev)
    kind, parts = quant_kind(hdr, phys)
    shape = hdr[phys]["shape"]
    out_dim, in_dim = shape[0], shape[-1]

    if kind in ("BF16", "F16", "F32"):
        raw, dt = _load_range(repo, shard, rev, hdr, start, phys)
        if dt == "BF16":
            arr = _bf16(raw)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
        else:
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float64)
        return arr.reshape(shape).astype(dtype, copy=False)

    if kind.endswith("NVFP4/g16"):
        # The header declares the PACKED shape (out, in/2): the payload is two E2M1 codes
        # per byte. Taking shape[-1] as the input width halves every expert.
        raw, dt = _load_range(repo, shard, rev, hdr, start, phys)
        if dt != "U8":
            raise RuntimeError(f"{phys}: expected U8 packed FP4, header says {dt}")
        in_dim *= 2
        vals = _unpack_fp4(np.frombuffer(raw, dtype=np.uint8).reshape(out_dim, in_dim // 2))
        sc = _scale_of(repo, shard, rev, hdr, start, parts)
        if sc is not None:
            vals = _apply_scale(vals, sc, out_dim, in_dim)
        return vals.reshape(out_dim, in_dim).astype(dtype, copy=False)

    if kind.startswith("F8_E4M3"):
        raw, dt = _load_range(repo, shard, rev, hdr, start, phys)
        vals = _fp8(raw).reshape(out_dim, in_dim)
        sc = _scale_of(repo, shard, rev, hdr, start, parts)
        if sc is not None:
            vals = _apply_scale(vals, sc, out_dim, in_dim)
        return vals.reshape(shape).astype(dtype, copy=False)

    raise RuntimeError(f"dtype {kind} unsupported by this reader")


SHA_LIMIT_DEFAULT = 64 * 1024 * 1024  # bytes of a tensor hashed by `describe`


def describe(repo: str, key: str, rev: str, sha_limit: int = SHA_LIMIT_DEFAULT) -> dict:
    """Pre-flight record for one tensor on one side: what it actually is, by range.

    Run this BEFORE measuring. Two of the limitations that decide the verdict are read
    here and not inferred: whether the two sides are the same precision at all, and
    whether a writer changed recipe between them.

    `sha256` covers the first `sha_bytes` of the tensor's range, not the whole tensor:
    the fused expert block is 1.7 GB and the embedding 1.3 GB per side, and the
    pre-flight decides dtype pairing, not content. `bytes` is the full length, so the
    record stays re-derivable — same revision, same range, same hash.
    """
    phys = resolve_key(repo, key, rev)
    if phys is None:
        return {"repo": repo, "rev": rev, "key": key, "present": False}
    wm = _index(repo, rev)
    shard = "model.safetensors" if wm is None else (wm.get(phys) or wm.get(key))
    hdr, start = _header(repo, shard, rev)
    meta, kind, parts = hdr[phys], *quant_kind(hdr, phys)
    a, b = meta["data_offsets"]
    n = min(b - a, sha_limit)
    raw = _get(_resolve(repo, shard, rev), (start + a, start + a + n - 1))
    logical = list(meta["shape"])
    if kind.endswith("NVFP4/g16"):
        logical[-1] *= 2          # the header declares the packed (out, in/2) shape
    return {
        "repo": repo, "rev": rev, "key": key, "physical_key": phys, "shard": shard,
        "present": True, "dtype": meta["dtype"], "shape": list(meta["shape"]),
        "logical_shape": logical, "bytes": b - a, "format": kind,
        "sha256": hashlib.sha256(raw).hexdigest(), "sha_bytes": n,
        "scales": {k: hdr[k]["dtype"] for k in (parts["scale"], parts["global"]) if k and k in hdr},
    }


def _phys_hdr(repo: str, key: str, rev: str):
    """(header, physical key) for a logical module, or (None, None) if absent here."""
    phys = resolve_key(repo, key, rev)
    if phys is None:
        return None, None
    wm = _index(repo, rev)
    shard = "model.safetensors" if wm is None else (wm.get(phys) or wm.get(key))
    return _header(repo, shard, rev)[0], phys


def _fmt_of(repo: str, key: str, rev: str) -> str:
    hdr, phys = _phys_hdr(repo, key, rev)
    return "ABSENT" if hdr is None else quant_kind(hdr, phys)[0]


def _nbytes(repo: str, key: str, rev: str) -> int:
    hdr, phys = _phys_hdr(repo, key, rev)
    if hdr is None:
        return 0
    a, b = hdr[phys]["data_offsets"]
    return b - a


ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "I32": 4, "I64": 8}


def _logical_shape(repo: str, key: str, rev: str) -> list[int] | None:
    """Shape the tensor has as a matrix, packed FP4 widths expanded."""
    hdr, phys = _phys_hdr(repo, key, rev)
    if hdr is None:
        return None
    shape = list(hdr[phys]["shape"])
    if quant_kind(hdr, phys)[0].endswith("NVFP4/g16"):
        shape[-1] *= 2
    return shape


def expert_slice(repo: str, key: str, rev: str, i: int) -> np.ndarray:
    """One expert's 2D slice of a fused 3D writer, read by byte range.

    A fused MoE weight (`mlp.experts.down_proj`, shape [n_experts, out, in]) is a single
    safetensors blob: whole it is 1.7 GB per side, and the edit is per expert. The
    experts are contiguous inside the blob, so only the asked-for slice crosses the wire.
    """
    hdr, phys = _phys_hdr(repo, key, rev)
    if hdr is None:
        raise KeyError(f"{key} not in {repo}")
    kind = quant_kind(hdr, phys)[0]
    if kind not in ("BF16", "F16", "F32"):
        raise RuntimeError(
            f"{key}: fused tensor in {kind} — the quantized checkpoints keep one key per "
            f"expert (…experts.0.down_proj.weight), so probe those directly"
        )
    wm = _index(repo, rev)
    shard = "model.safetensors" if wm is None else (wm.get(phys) or wm.get(key))
    _, start = _header(repo, shard, rev)
    shape = hdr[phys]["shape"]
    per = int(np.prod(shape[1:])) * ITEMSIZE[kind]
    a, _ = hdr[phys]["data_offsets"]
    raw = _get(_resolve(repo, shard, rev), (start + a + i * per, start + a + (i + 1) * per - 1))
    if kind == "BF16":
        arr = _bf16(raw)
    elif kind == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    else:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float64)
    return arr.reshape(shape[1:])


def top_singular(dW: np.ndarray, iters: int = 200):
    """(u0, v0, s0, s1, rank1_energy) by power iteration + deflation.

    Full SVD of a [5120, 5120] float64 matrix is wasteful when only the top two
    singular values matter, and this is the same method the extractor uses — so the
    numbers here are comparable with what actually ships in the directions file.
    """
    rng = np.random.default_rng(0)  # fixed: two runs on the same weights must agree
    # The seed carries dW's dtype. A float64 seed upcasts every `dW @ v`, so a float32 ΔW
    # (the BIG path, embed_tokens) was copied to 5 GB of float64 on each of the 800
    # iterations: 25 min for the embedding measured against 3 min at float32. Same seed,
    # same sequence — only the width of the arithmetic changes, and only on the BIG path.
    v = rng.standard_normal(dW.shape[1]).astype(dW.dtype, copy=False)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = dW @ v
        nu = np.linalg.norm(u)
        if nu < 1e-30:
            return None
        u /= nu
        v = dW.T @ u
        nv = np.linalg.norm(v)
        if nv < 1e-30:
            return None
        v /= nv
    s0 = float(u @ dW @ v)
    dW1 = dW - s0 * np.outer(u, v)
    v2 = rng.standard_normal(dW.shape[1]).astype(dW.dtype, copy=False)
    v2 /= np.linalg.norm(v2)
    for _ in range(iters):
        u2 = dW1 @ v2
        n2 = np.linalg.norm(u2)
        if n2 < 1e-30:
            break
        u2 /= n2
        v2 = dW1.T @ u2
        n2 = np.linalg.norm(v2)
        if n2 < 1e-30:
            break
        v2 /= n2
    s1 = float(abs(u2 @ dW1 @ v2)) if np.linalg.norm(dW1) > 1e-30 else 0.0
    fro2 = float((dW ** 2).sum())
    return u, v, s0, s1, (s0 * s0 / fro2 if fro2 > 0 else 0.0)


BIG = 40_000_000  # above this many elements, work in float32: embed_tokens is 636M


def _numbers(Wb: np.ndarray, Wa: np.ndarray):
    """Rank-1 read of the edit between two aligned matrices: (delta, numbers, u0).

    `numbers` carries either `untouched`, an `error`, or the measurement. The caller adds
    the provenance (which repos, which formats) around it.
    """
    dW = Wa - Wb
    delta = float(np.linalg.norm(dW) / max(np.linalg.norm(Wb), 1e-30))
    if np.linalg.norm(dW) < 1e-12:
        # Byte-identical. Not "no ablation found" — this module was simply not touched,
        # which is itself a fact worth printing (see README on the vision tower).
        return delta, {"delta": 0.0, "untouched": True}, None
    got = top_singular(dW)
    if got is None:
        return delta, {"delta": delta, "error": "power iteration did not converge"}, None
    u0, v0, s0, s1, rank1E = got
    # Does ΔW look like −λ·r̂r̂ᵀW? Then v0 should align with u0ᵀW_base.
    w = Wb.T @ u0
    nw = np.linalg.norm(w)
    cos = float(abs(v0 @ (w / nw))) if nw > 1e-30 else float("nan")
    # λ_eff: how many times over the edit removes the component. Sign matters —
    # a positive s0 with this construction would mean the edit ADDS the direction.
    denom = float(u0 @ Wb @ v0)
    lam_eff = float(-s0 / denom) if abs(denom) > 1e-30 else float("nan")
    return delta, {
        "delta": delta, "untouched": False,
        "s0": s0, "s1": s1, "s0_over_s1": (s0 / s1 if s1 > 1e-30 else float("inf")),
        "rank1E": rank1E, "cos": cos, "lam_eff": lam_eff,
        "dim": int(Wb.shape[0]), "random_cos_floor": random_cos_floor(Wb.shape[0]),
        "compute_dtype": str(dW.dtype),
    }, u0


# The expert indices the 03-09 extraction sampled (verify_fused_and_noise.py): first,
# two early, the middle, the last. Deterministic, so a re-run gives the same number.
EXPERT_SAMPLE_DEFAULT = (0, 1, 7, 255, 511)


def _probe_fused(base_repo, abl_repo, key, base_rev, abl_rev, fmt, experts):
    """Fused MoE writer: one row per sampled expert, pooled into the module row.

    The edit is per expert, so the module's number is the WORST expert (min s0/s1, min
    rank1E, min cos, max λ_eff): a fused block where 511 experts are clean and one is
    noise is not a clean rank-1 edit.
    """
    n = _logical_shape(base_repo, key, base_rev)[0]
    idx = sorted({i for i in experts if 0 <= i < n} | {n - 1})
    subs, u0 = [], None
    for i in idx:
        delta, nums, ui = _numbers(expert_slice(base_repo, key, base_rev, i),
                                   expert_slice(abl_repo, key, abl_rev, i))
        if nums.get("error"):
            return {"key": key, "fused": True, "n_experts": n, "sampled": idx, **fmt, **nums}
        if nums.get("untouched"):
            continue        # this expert was not edited; the ones that were still count
        if u0 is None:
            u0 = ui
        subs.append({"expert": i, **nums})
    if not subs:
        return {"key": key, "fused": True, "n_experts": n, "sampled": idx,
                "delta": 0.0, "untouched": True, **fmt}
    worst = {k: (min(s[k] for s in subs) if k != "lam_eff" else max(s[k] for s in subs))
             for k in ("s0_over_s1", "rank1E", "cos", "lam_eff")}
    return {
        "key": key, "fused": True, "n_experts": n, "sampled": idx,
        "delta": float(np.mean([s["delta"] for s in subs])), "untouched": False,
        **worst, "experts": subs,
        "dim": subs[0]["dim"], "random_cos_floor": subs[0]["random_cos_floor"],
        "compute_dtype": subs[0]["compute_dtype"], **fmt,
        "u0": None, "_u0": u0,
    }


def probe_module(base_repo, abl_repo, key, base_rev, abl_rev, experts=EXPERT_SAMPLE_DEFAULT):
    # The two sides may spell the same module differently (`weight` vs `weight_packed`)
    # and may store it in different precisions. Both facts are recorded, not smoothed over.
    shape_b = _logical_shape(base_repo, key, base_rev)
    shape_a = _logical_shape(abl_repo, key, abl_rev)
    fmt = {"format_base": _fmt_of(base_repo, key, base_rev), "format_abl": _fmt_of(abl_repo, key, abl_rev)}
    fmt["format_mismatch"] = fmt["format_base"] != fmt["format_abl"]
    if shape_b is None or shape_a is None:
        missing = "base" if shape_b is None else "abl"
        return {"key": key, "error": f"module absent on the {missing} side", **fmt}
    if shape_b != shape_a:
        return {"key": key, "error": f"shape {shape_b} vs {shape_a}", **fmt}
    if len(shape_b) == 3:
        return _probe_fused(base_repo, abl_repo, key, base_rev, abl_rev, fmt, experts)
    Wb = tensor(base_repo, key, base_rev, dtype=np.float32 if _nbytes(base_repo, key, base_rev) > BIG else np.float64)
    Wa = tensor(abl_repo, key, abl_rev, dtype=np.float32 if _nbytes(abl_repo, key, abl_rev) > BIG else np.float64)
    delta, nums, u0 = _numbers(Wb, Wa)
    if u0 is None:
        return {"key": key, **nums, **fmt}
    return {"key": key, **nums, **fmt, "u0": None, "_u0": u0}


def paired_rows(rows):
    """Rows that carry a number: measured, edited, and SAME PRECISION on both sides.

    A tensor whose recipe changed between the two repos is a cross-precision difference:
    its s0/s1 mixes the edit with requantization noise, so it cannot enter an aggregate
    that claims to measure geometry. It is still reported, per tensor, just not pooled.
    """
    return [r for r in rows if not r.get("error") and not r.get("untouched") and not r.get("format_mismatch")]


def verdict(rows):
    """A candidate is usable only if EVERY paired probed module is a clean rank-1 edit."""
    ok = paired_rows(rows)
    if not ok:
        return "NO PAIRED DATA (every module changed format between the two sides)"
    if min(r["s0_over_s1"] for r in ok) < 5:
        return "REJECTED (no dominant direction)"
    if min(r["cos"] for r in ok) < 0.9:
        return "REJECTED (not projection-shaped)"
    if max(r["lam_eff"] for r in ok) > 2.0:
        return "CAUTION (over-ablated: inverts the component)"
    return "CLEAN rank-1"


def coherence(us: list[np.ndarray]):
    """|cos| between the dominant directions of DIFFERENT modules, same-width ones only.

    `embed_tokens` is a right-vector branch: its u0 lives in R^vocab (248320 here), and
    dot-ing it with a residual-stream u0 (R^2560) is a ValueError, not a measurement.
    Pairs of different widths are skipped; if nothing pairs, there is no coherence.
    """
    cs = [abs(float(us[a] @ us[b])) for a in range(len(us)) for b in range(a + 1, len(us))
          if us[a].size == us[b].size]
    if len(cs) < 1:
        return None
    return {"min": min(cs), "median": float(np.median(cs))}


def canon_sign(v: np.ndarray) -> np.ndarray:
    """Fix a vector's global sign deterministically: the largest-|component| entry positive.

    A singular vector comes out of power iteration with an arbitrary sign, so a "signed
    cos" against anything is meaningless until both sides agree on a rule. This rule is
    applied to the measured direction AND to the reference vector, so the sign of the
    result means something and the number is re-derivable.
    """
    v = np.asarray(v, dtype=np.float64)
    return -v if v[int(np.argmax(np.abs(v)))] < 0 else v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--base-rev", default="main", help="pin this when publishing a number")
    ap.add_argument("--abl", action="append", default=None, help="repeatable; default = the three from §2")
    ap.add_argument("--rev", action="append", default=None, help="revision per --abl, in order")
    ap.add_argument("--module", action="append", default=None, help="repeatable; default = 4 residual writers")
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--preflight", default=None,
                    help="write the per-tensor, per-side dtype/shape/bytes/SHA record here, BEFORE measuring")
    ap.add_argument("--save-direction", default=None, metavar="OUT.NPY",
                    help="write the paired direction (u0 of --direction-module, float32, normalized) here")
    ap.add_argument("--direction-module", default="model.language_model.layers.0.linear_attn.out_proj.weight")
    ap.add_argument("--against", action="append", default=None, metavar="REF.NPY",
                    help="signed cos of the saved direction against this .npy; repeatable")
    ap.add_argument("--experts", default=None, metavar="0,1,7,255,511",
                    help="expert indices to sample inside a fused 3D MoE writer (default: the 03-09 sample)")
    args = ap.parse_args()

    cands = args.abl or CANDIDATES_DEFAULT
    revs = args.rev or []
    mods = args.module or MODULES_DEFAULT
    experts = tuple(int(x) for x in args.experts.split(",")) if args.experts else EXPERT_SAMPLE_DEFAULT
    out = {"base": args.base, "base_rev": args.base_rev,
           "base_revision_sha": revision_sha(args.base, args.base_rev), "candidates": {}}

    print(f"base: {args.base}@{args.base_rev} -> {out['base_revision_sha']}")

    if args.preflight:
        pf = {"base": [describe(args.base, k, args.base_rev) for k in mods]}
        for i, abl in enumerate(cands):
            rev = revs[i] if i < len(revs) else "main"
            pf[abl] = [describe(abl, k, rev) for k in mods]
        with open(args.preflight, "w", encoding="utf-8") as f:
            json.dump(pf, f, indent=2)
        print(f"preflight written: {args.preflight}")
        for k in mods:
            b = next(x for x in pf["base"] if x["key"] == k)
            for abl in cands:
                a = next(x for x in pf[abl] if x["key"] == k)
                flag = "" if b.get("format") == a.get("format") else "  <-- FORMAT MISMATCH"
                print(f"  {k.replace('model.language_model.layers.', 'L'):40} "
                      f"{b.get('format', '?'):24} vs {a.get('format', '?'):24}{flag}")
        print()

    for i, abl in enumerate(cands):
        rev = revs[i] if i < len(revs) else "main"
        print(f"=== {abl}@{rev} -> {revision_sha(abl, rev)}")
        rows = []
        for key in mods:
            try:
                r = probe_module(args.base, abl, key, args.base_rev, rev, experts=experts)
            except Exception as e:  # noqa: BLE001
                r = {"key": key, "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            short = key.replace("model.language_model.layers.", "L").replace(".weight", "")
            if r.get("fused"):
                short += f" [{len(r.get('sampled', []))}/{r.get('n_experts')} experts]"
            if r.get("error"):
                print(f"  {short:<34} ERROR: {r['error']}")
            elif r.get("untouched"):
                print(f"  {short:<34} byte-identical to base (not edited)")
            else:
                mm = " [FORMAT MISMATCH]" if r.get("format_mismatch") else ""
                print(f"  {short:<34} s0/s1={r['s0_over_s1']:8.2f}  rank1E={r['rank1E']:.4f}  "
                      f"cos={r['cos']:.4f}  lam_eff={r['lam_eff']:.4f}  delta={r['delta']:.5f}"
                      f"  floor={r['random_cos_floor']:.4f}{mm}")

        us = [r["_u0"] for r in rows if r.get("_u0") is not None]
        coh = coherence(us)
        if coh:
            print(f"  {'coherence between modules':<34} min={coh['min']:.4f}  median={coh['median']:.4f}"
                  f"   -> {'ONE global axis' if coh['min'] > 0.9 else 'per-layer, no single axis'}")
        v = verdict(rows)
        paired = paired_rows(rows)
        agg = {"n_paired": len(paired),
               "min_s0_over_s1": min((r["s0_over_s1"] for r in paired), default=None),
               "max_s0_over_s1": max((r["s0_over_s1"] for r in paired), default=None),
               "n_format_mismatch": sum(1 for r in rows if r.get("format_mismatch"))}
        print(f"  paired tensors: {agg['n_paired']} (mismatch excluded: {agg['n_format_mismatch']})"
              f"   s0/s1 range: {agg['min_s0_over_s1']} .. {agg['max_s0_over_s1']}")
        print(f"  VERDICT: {v}\n")

        # The paired direction only means something if BOTH sides were the same recipe:
        # a u0 read off a cross-precision ΔW (Par B's out_proj is BF16 on one side and
        # FP8 on the other) carries requantization noise in its top direction.
        direction = None
        for r in rows:
            if (r.get("key") == args.direction_module and r.get("_u0") is not None
                    and not r.get("format_mismatch")):
                direction = canon_sign(r["_u0"] / np.linalg.norm(r["_u0"]))
        if args.save_direction and direction is not None:
            np.save(args.save_direction, direction.astype(np.float32))
            print(f"  direction written: {args.save_direction} (float32, dim {direction.size}, normalized)")
        elif args.save_direction:
            print(f"  direction NOT saved: {args.direction_module} absent or format_mismatch on this pair")
        cos_against = {}
        for ref in args.against or []:
            live = canon_sign(np.load(ref)).astype(np.float64)
            if direction is None:
                cos_against[ref] = None
                print(f"  cos vs {ref}: n/a (no direction measured for {args.direction_module})")
            elif live.size != direction.size:
                cos_against[ref] = None
                print(f"  cos vs {ref}: n/a (dim {live.size} != {direction.size})")
            else:
                cos_against[ref] = float(live @ direction)
                print(f"  cos_direction_signed vs {ref}: {cos_against[ref]:+.6f}"
                      f"   (floor {random_cos_floor(direction.size):.4f})")

        for r in rows:
            r.pop("_u0", None)
            r.pop("u0", None)
        out["candidates"][abl] = {"rev": rev, "revision_sha": revision_sha(abl, rev), "modules": rows,
                                  "coherence": coh, "verdict": v, "aggregate_paired": agg,
                                  "direction_module": args.direction_module,
                                  "cos_direction_signed": cos_against,
                                  "sign_rule": "largest-|component| entry positive, applied to both vectors"}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
