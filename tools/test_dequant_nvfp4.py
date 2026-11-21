#!/usr/bin/env python3
"""Does the dequantization path produce the right floats?

    python3 tools/test_dequant_nvfp4.py

The rank-1 verdicts this tool prints for a quantized checkpoint come out of this path,
so a wrong scale convention here becomes a wrong s0/s1 there and — for DGX-324 — a wrong
call on whether the served refusal direction is geometry or requantization noise. Hence
a test with hand-built quantized input and hand-computed expected floats.

Two things are checked, not one:

  * the E2M1 nibble unpacking and the 16-wide block scale, and
  * that the two NVFP4 conventions are applied in the RIGHT DIRECTION. modelopt
    multiplies by `weight_scale_2`; compressed-tensors DIVIDES by `weight_global_scale`
    (the converter between them writes `weight_global_scale = 1 / weight_scale_2`).
    Applying the same operator to both is the plausible mistake, and the last test
    asserts the mistake is actually caught rather than merely absent.
"""

from __future__ import annotations

import io
import json
import struct
import sys
import urllib.error

import numpy as np

import probe_rank1_candidates as P

# --------------------------------------------------------------------------- helpers
# A minimal in-memory safetensors: 8-byte header length, JSON header, then the blobs.
# Serving it through a fake `_get` exercises header parsing, key resolution and the
# dequant together, which is the path that actually runs against HuggingFace.

def build(files: dict[str, dict[str, tuple[str, list[int], np.ndarray]]]) -> dict[str, bytes]:
    """files: {repo: {key: (dtype, shape, raw_array)}} -> {url: bytes} for the fake _get."""
    out: dict[str, bytes] = {}
    for repo, tensors in files.items():
        hdr, blobs, off = {}, [], 0
        for key, (dt, shape, arr) in tensors.items():
            raw = arr.tobytes()
            hdr[key] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
            blobs.append(raw)
            off += len(raw)
        hj = json.dumps(hdr, separators=(",", ":")).encode()
        body = struct.pack("<Q", len(hj)) + hj + b"".join(blobs)
        url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
        out[url] = body
        out[f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"] = json.dumps(
            {"weight_map": {k: "model.safetensors" for k in tensors}}).encode()
    return out


def serve(store: dict[str, bytes]):
    def fake_get(url, rng=None):
        if url not in store:
            raise urllib.error.HTTPError(url, 404, "not found", {}, io.BytesIO(b""))
        data = store[url]
        if rng:
            lo, hi = rng
            return data[lo:hi + 1]
        return data
    return fake_get


def reset_caches():
    for fn in (P._index, P._header, P._hf_token):
        fn.cache_clear()


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
CODES = np.arange(16, dtype=np.uint8)          # one full block: every code once
EXPECT_E2M1 = np.array([s * E2M1[c & 7] for c, s in
                        ((c, -1 if c & 8 else 1) for c in CODES)], dtype=np.float64).reshape(1, 16)

fails: list[str] = []


def check(name: str, got, want, tol=1e-6):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape:
        fails.append(f"{name}: shape {got.shape} != {want.shape}")
        return
    bad = np.max(np.abs(got - want)) if got.size else 0.0
    scale = max(1.0, float(np.max(np.abs(want))))
    if bad / scale > tol:
        fails.append(f"{name}: max abs error {bad:.3e} (tol {tol}) — got {got.ravel()[:4]}, want {want.ravel()[:4]}")
    else:
        print(f"  ok  {name}")


def encode_e4m3(v: float) -> int:
    """Encode a float that is EXACTLY representable in FP8 E4M3, from the format spec
    (sign, 4-bit exponent bias 7, 3-bit mantissa). Independent of the decoder under test,
    and it refuses anything that would need rounding so a test cannot pass by luck."""
    import math
    if v == 0.0:
        return 0x00
    s = 0x80 if v < 0 else 0x00
    a = abs(v)
    if a < 2.0 ** -6:                                   # subnormal range
        m = round(a / (2.0 ** -9))
        assert abs(m * 2.0 ** -9 - a) < 1e-15, f"{v} not exactly representable"
        return s | m
    e = math.floor(math.log2(a)) + 7
    m = round((a / 2.0 ** (e - 7) - 1.0) * 8)
    if m == 8:
        e, m = e + 1, 0
    assert abs((1.0 + m / 8.0) * 2.0 ** (e - 7) - a) < 1e-9, f"{v} not exactly representable"
    return s | (e << 3) | m


def fp8(vals) -> np.ndarray:
    return np.array([encode_e4m3(float(v)) for v in np.asarray(vals, np.float64).ravel()], dtype=np.uint8)


def pack(codes: np.ndarray) -> np.ndarray:
    """Low nibble first: element 2i in bits 0-3, element 2i+1 in bits 4-7."""
    c = codes.reshape(-1, 2)
    return (c[:, 0] | (c[:, 1] << 4)).astype(np.uint8)


# --------------------------------------------------------------------------- tests
def test_fp8_table():
    """E4M3 against the documented corners, not against itself.

    The numpy in this house exposes no float8 dtype, so the decoder is a hand-built
    table; these are the values that catch a wrong bias or a wrong subnormal rule."""
    check("E4M3 max is 448", P.FP8_E4M3[0x7E], 448.0)
    check("E4M3 min normal", P.FP8_E4M3[0x08], 2.0 ** -6)
    check("E4M3 min subnormal", P.FP8_E4M3[0x01], 2.0 ** -9)
    check("E4M3 zero", P.FP8_E4M3[0x80], 0.0)
    if not np.isnan(P.FP8_E4M3[0x7F]):
        fails.append("0x7F should be the E4M3 NaN")
    else:
        print("  ok  E4M3 0x7F is NaN")


def test_e2m1_lut():
    """The 4-bit table and its sign bit."""
    check("e2m1 table", P._e2m1(CODES), EXPECT_E2M1.reshape(-1))


def test_nibble_order():
    """Low nibble first, interleaved — the order vLLM's loader uses."""
    packed = pack(np.array([1, 4, 2, 15], dtype=np.uint8))   # -> [0.5, 2.0, 1.0, -6.0]
    check("nibble order", P._unpack_fp4(packed.reshape(1, 2)), np.array([[0.5, 2.0, 1.0, -6.0]]))


def test_modelopt_nvfp4():
    """modelopt: value = e2m1 * block_scale * weight_scale_2.

    The declared shape is the PACKED one — [1, 8] bytes for 16 codes, as RadixArk really
    declares `...down_proj.weight` [2560, 320] for 640 inputs. Reading shape[-1] as the
    input width halves every expert, so it is asserted here rather than assumed.
    """
    repo = "test/modelopt"
    store = build({repo: {
        "w.weight": ("U8", [1, 8], pack(CODES)),
        "w.weight_scale": ("F8_E4M3", [1, 1], fp8([2.0])),
        "w.weight_scale_2": ("F32", [], np.array([3.0], dtype=np.float32)),
    }})
    reset_caches(); P._get = serve(store)
    kind, _ = P.quant_kind(_hdr(repo)[0], "w.weight")
    if kind != "modelopt-NVFP4/g16":
        fails.append(f"modelopt kind: got {kind}")
    got = P.tensor(repo, "w.weight")
    check("modelopt NVFP4", got, EXPECT_E2M1 * 2.0 * 3.0)
    if got.shape != (1, 16):
        fails.append(f"packed shape not expanded: got {got.shape}, want (1, 16)")
    else:
        print("  ok  declared packed shape expands to the logical one")


def test_compressed_tensors_nvfp4():
    """compressed-tensors: value = e2m1 * (block_scale / weight_global_scale).

    Both repos are served at once with the SAME underlying weights — modelopt with
    scale_2 = 3, compressed-tensors with global_scale = 1/3 — so the two conventions are
    compared against each other, not just against a hand-computed number. If the
    division were written as a multiplication the ct side comes out 1/9 of the other.
    """
    store = build({
        "test/ct": {
            "w.weight_packed": ("U8", [1, 8], pack(CODES)),
            "w.weight_scale": ("F8_E4M3", [1, 1], fp8([2.0])),
            "w.weight_global_scale": ("F32", [], np.array([1.0 / 3.0], dtype=np.float32)),
        },
        "test/modelopt": {
            "w.weight": ("U8", [1, 8], pack(CODES)),
            "w.weight_scale": ("F8_E4M3", [1, 1], fp8([2.0])),
            "w.weight_scale_2": ("F32", [], np.array([3.0], dtype=np.float32)),
        },
    })
    reset_caches(); P._get = serve(store)
    kind, _ = P.quant_kind(_hdr("test/ct")[0], "w.weight_packed")
    if kind != "ct-NVFP4/g16":
        fails.append(f"ct kind: got {kind}")
    got = P.tensor("test/ct", "w.weight")     # asked by LOGICAL name, found as _packed
    check("compressed-tensors NVFP4 (divides)", got, EXPECT_E2M1 * 6.0)
    check("the two conventions agree on the same weights",
          got, P.tensor("test/modelopt", "w.weight"))


def test_wrong_convention_is_caught():
    """Guard on the guard: multiplying where the format divides must fail the test."""
    repo = "test/ct"
    store = build({repo: {
        "w.weight_packed": ("U8", [1, 8], pack(CODES)),
        "w.weight_scale": ("F8_E4M3", [1, 1], fp8([2.0])),
        "w.weight_global_scale": ("F32", [], np.array([1.0 / 3.0], dtype=np.float32)),
    }})
    reset_caches(); P._get = serve(store)
    wrong = EXPECT_E2M1 * 2.0 * (1.0 / 3.0)   # the multiplication mistake
    right = EXPECT_E2M1 * 6.0
    if np.allclose(wrong, right):
        fails.append("the mistaken convention is indistinguishable — test proves nothing")
    else:
        print("  ok  mistaken convention differs from the correct one")


def test_fp8_per_channel():
    """F8_E4M3 with a per-output-channel BF16 scale (how orcarouter stores out_proj)."""
    repo = "test/fp8"
    vals = np.array([[1.0, -2.0, 0.5, 6.0], [1.5, -1.5, 448.0, 0.0]], dtype=np.float64)
    store = build({repo: {
        "w.weight": ("F8_E4M3", [2, 4], fp8(vals)),
        "w.weight_scale": ("BF16", [2, 1],
                           (np.array([2.0, 0.5], dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)),
    }})
    reset_caches(); P._get = serve(store)
    kind, _ = P.quant_kind(_hdr(repo)[0], "w.weight")
    if kind != "F8_E4M3/per-channel":
        fails.append(f"fp8 kind: got {kind}")
    check("F8_E4M3 per-channel", P.tensor(repo, "w.weight"), vals * np.array([[2.0], [0.5]]))


def test_plain_bf16_still_works():
    """The path this tool has always used must not have moved."""
    repo = "test/plain"
    want = np.array([[1.0, -0.5, 2.5, 0.0]], dtype=np.float64)
    store = build({repo: {"w.weight": ("BF16", [1, 4],
                                       (want.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))}})
    reset_caches(); P._get = serve(store)
    check("BF16 unchanged", P.tensor(repo, "w.weight"), want)


def _hdr(repo: str):
    """The header dict for a repo in the currently-served store."""
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    n = struct.unpack("<Q", P._get(url, (0, 7)))[0]
    return json.loads(P._get(url, (8, 8 + n - 1))), 8 + n


def test_access_error_is_not_a_keyerror():
    """A gated side must say so, not surface as a missing tensor."""
    def fake_get(url, rng=None):
        raise urllib.error.HTTPError(url, 403, "gated", {}, io.BytesIO(b""))
    reset_caches(); P._get = fake_get
    try:
        P._index("orcarouter/x", "main")
        fails.append("403 did not raise")
    except P.AccessError as e:
        if "orcarouter/x" not in str(e):
            fails.append(f"AccessError does not name the repo: {e}")
        else:
            print("  ok  403 raises AccessError naming the repo")
    except Exception as e:  # noqa: BLE001
        fails.append(f"403 raised {type(e).__name__}, expected AccessError")
    # and a real 404 still means "monoshard", not "no access"
    def only404(url, rng=None):
        raise urllib.error.HTTPError(url, 404, "nope", {}, io.BytesIO(b""))
    reset_caches(); P._get = only404
    if P._index("any/repo", "main") is not None:
        fails.append("404 should mean monoshard (None)")
    else:
        print("  ok  404 still means monoshard")


def test_cos_floor_follows_dim():
    """The floor is 1/sqrt(dim): flash-next (2560) is 0.0198, not the 27B's 0.0140."""
    check("floor at 2560", P.random_cos_floor(2560), 1.0 / np.sqrt(2560))
    check("floor at 5120", P.random_cos_floor(5120), P.RANDOM_COS_5120)


def test_canon_sign_is_deterministic():
    """A signed cos is meaningless until both sides agree on the global sign.

    The contract is not "flip this one": it is that the result is idempotent and its
    largest-|component| entry is positive, whichever way the input came in.
    """
    v = np.array([1.0, -9.0, 2.0])
    c = P.canon_sign(v)
    check("canon_sign makes the largest entry positive", c, np.array([-1.0, 9.0, -2.0]))
    check("canon_sign is idempotent", P.canon_sign(c), c)
    check("canon_sign ignores the input's starting sign", P.canon_sign(-v), c)


def test_fused_expert_slice_and_row():
    """A fused 3D MoE writer: read one expert by range, and pool the sampled experts.

    `mlp.experts.down_proj` is [n_experts, out, in] in one blob. Whole it is 1.7 GB per
    side and the edit is per expert, so the tool reads slices and reports the worst
    sampled expert. Two things are checked: the byte-range arithmetic of the slice, and
    that an untouched expert does not make the module look untouched.
    """
    base, abl = "test/fused-base", "test/fused-abl"
    rng = np.random.default_rng(3)
    n, o, i = 3, 6, 8
    W = rng.standard_normal((n, o, i))
    r = rng.standard_normal(o); r /= np.linalg.norm(r)
    d = rng.standard_normal(i); d /= np.linalg.norm(d)
    Wa = W.copy()
    Wa[1] -= 1.0 * np.outer(r, d) * float(np.linalg.norm(W[1]))   # only expert 1 is edited
    bf = lambda a: (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    store = build({base: {"e.down_proj": ("BF16", [n, o, i], bf(W))},
                   abl: {"e.down_proj": ("BF16", [n, o, i], bf(Wa))}})
    reset_caches(); P._get = serve(store)
    # what crosses the wire is BF16-rounded, so that is the expected value — bit-exact
    stored = bf(W).astype(np.uint32) << 16
    check("expert_slice returns the asked expert",
          P.expert_slice(base, "e.down_proj", "main", 2),
          stored[2].view(np.float32).astype(np.float64), tol=0)
    row = P.probe_module(base, abl, "e.down_proj", "main", "main", experts=(0, 1, 2))
    if not row.get("fused"):
        fails.append(f"fused module not reported as fused: {row}")
    elif [e["expert"] for e in row.get("experts", [])] != [1]:
        fails.append(f"edited experts reported as {row.get('experts')}, expected only expert 1")
    else:
        print("  ok  fused row pools only the edited expert (0 and 2 untouched)")
    if row.get("s0_over_s1", 0) < 100:
        fails.append(f"a rank-1 edit in BF16 should still dominate, got s0/s1 = {row.get('s0_over_s1')}")
    else:
        print(f"  ok  fused rank-1 edit dominates (s0/s1 = {row['s0_over_s1']:.0f}, "
              f"the residue is BF16 rounding)")


def test_coherence_skips_other_widths():
    """embed_tokens' u0 (R^vocab) must not crash coherence against residual u0s (R^2560).

    The run passes embed_tokens as a module (right-vector branch, reported apart), so
    main() hands coherence() vectors of two different widths. The old pairwise dot raised
    ValueError on the first mismatched pair — outside the per-module try, so the whole
    candidate died after all its measurements.
    """
    a = np.zeros(2560); a[0] = 1.0
    b = np.zeros(2560); b[1] = 1.0
    wide = np.zeros(248320); wide[7] = 1.0
    c = P.coherence([a, b, wide])
    if c is None or abs(c["min"]) > 1e-12:
        fails.append(f"mixed-width coherence wrong: {c}")
    else:
        print("  ok  coherence pairs the same-width vectors and skips the vocab-width one")
    if P.coherence([wide]) is not None:
        fails.append("a single vector has no coherence to report")
    else:
        print("  ok  one vector -> no coherence")


def test_top_singular_keeps_the_matrices_dtype():
    """A float32 ΔW must stay float32 through the power iteration.

    The seed used to be float64, so every `dW @ v` upcast the matrix: on the BIG path
    (embed_tokens, 636M elements) each of the 800 iterations copied 5 GB of float64 — 25
    min for one tensor, against 3 min at the matrix's own width. Same seed and same draw,
    so the number is the same number; only the width of the arithmetic changed.
    """
    dW = np.random.default_rng(3).standard_normal((96, 64)).astype(np.float32)
    u, v, s0, s1, e = P.top_singular(dW)
    if u.dtype != np.float32 or v.dtype != np.float32:
        fails.append(f"top_singular widened a float32 ΔW to {u.dtype}/{v.dtype}")
    else:
        print("  ok  top_singular keeps a float32 ΔW at float32")
    check("its s0 is the float64 iteration's", s0, P.top_singular(dW.astype(np.float64))[2], tol=1e-4)


def main() -> int:
    print("dequant path:")
    for t in (test_fp8_table, test_e2m1_lut, test_nibble_order, test_modelopt_nvfp4,
              test_compressed_tensors_nvfp4, test_wrong_convention_is_caught,
              test_fp8_per_channel, test_plain_bf16_still_works):
        t()
    print("reader guards:")
    for t in (test_access_error_is_not_a_keyerror, test_cos_floor_follows_dim,
              test_canon_sign_is_deterministic, test_fused_expert_slice_and_row,
              test_coherence_skips_other_widths, test_top_singular_keeps_the_matrices_dtype):
        t()
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
