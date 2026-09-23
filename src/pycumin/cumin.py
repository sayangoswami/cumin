#!/usr/bin/env python3
"""
cumin.py — low-memory syncmer containment mapping with window voting
==========================================================================

The parts that work, extracted from the order-vs-content experiment. Ordered
pair features, gap strata and the mode-comparison machinery are gone: the
ablation showed they lose ~10 positives for every 1 rescued, at 6x the index.

Method
------
  anchors    open syncmers -- a k-mer is an anchor iff its minimal s-mer starts
             at offset t. Selection depends ONLY on the k-mer's own content, so
             a read and the reference agree exactly at shared positions: no
             window context, no read-boundary artefact, no error cascade.
             Anchor survival under substitutions is exactly (1-eps)^k.

  density    1 / ((k-s+1) * downsample). The hash downsample is also
             content-only, so it preserves the agreement property.

  index      reference tiled into overlapping windows of `win` bp, step win/2.
             Every anchor is filed under the two windows containing it.
             Sorted uint64 keys + CSR. ~6.6 bytes per entry.

  query      one searchsorted, ragged gather, vote per window, take the max.
             score = best_window_votes / n_unique_query_anchors.
             Read scored forward and reverse-complemented, better kept, so the
             index is single-strand.

  build      entries are radix-partitioned on the feature's top byte into 256
             on-disk buckets as they are produced (or, in --high-mem mode,
             sorted in a single in-memory pass -- see _core.Index).

Measured: 3.1 Gbp in 2.44 GB (0.79 B/base), AUROC 0.985, 96.9% correct locus,
~0.7 ms/read, ~5.5 min single-threaded build.

The anchor/index/query engine itself (syncmer selection, the radix-bucket or
in-memory build, and windowed voting) is implemented in C++ (see ``_core``,
built from ``src/index.cpp``/``src/cumin.h``); this module is the thin CLI
and evaluation layer on top of it.

Usage
-----
  cumin.py build --ref gut.fa.gz --index gut.idx
  cumin.py map   --index gut.idx --reads sample.fq.gz --out hits.tsv
  cumin.py eval  --index gut.idx --pos pos.fq.gz --neg neg.fq.gz --out results/
  cumin.py demo                       # synthetic end-to-end check

Read headers for eval are parsed as  <refname>!<start>!<end>!<strand>
e.g.  @Rep_817_C_0!30866!31389!+ qs:f:9.9340
"""

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np

from . import _core


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─────────────────────────── i/o ────────────────────────────────────────────

def _open(path):
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "rt")


def read_fasta(path, max_bp=None):
    name, chunks, total = None, [], 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(chunks)
                    yield name, seq
                    total += len(seq)
                    if max_bp and total >= max_bp:
                        return
                name, chunks = line[1:].strip().split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def read_reads(path, limit=None):
    with _open(path) as fh:
        first = fh.readline()
        if not first:
            return
        fh.seek(0)
        n = 0
        if first.startswith("@"):
            while True:
                h = fh.readline()
                if not h:
                    return
                seq = fh.readline().strip()
                fh.readline()
                fh.readline()
                yield h[1:].strip(), seq
                n += 1
                if limit and n >= limit:
                    return
        else:
            name, chunks = None, []
            for line in fh:
                if line.startswith(">"):
                    if name is not None:
                        yield name, "".join(chunks)
                        n += 1
                        if limit and n >= limit:
                            return
                    name, chunks = line[1:].strip(), []
                else:
                    chunks.append(line.strip())
            if name is not None:
                yield name, "".join(chunks)


def parse_header(h):
    """'Rep_817_C_0!30866!31389!+ ...' -> (ref, start, end, strand) or None."""
    parts = h.split()[0].rsplit("!", 3)
    if len(parts) != 4:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), parts[3]
    except ValueError:
        return None


# ─────────────────────────── driver helpers ─────────────────────────────────

def build_index(ref_path, tmpdir="/tmp", batch=20_000_000, chunk=32_000_000, max_ref_bp=None, **anchor_kwargs):
    """anchor_kwargs: k, s, t, downsample, win, max_occ_pct, low_mem (see _core.Index)."""
    ix = _core.Index(tmpdir=tmpdir, batch=batch, chunk=chunk, **anchor_kwargs)
    bp = anchors = nrec = 0
    for name, seq in read_fasta(ref_path, max_ref_bp):
        bp += len(seq)
        anchors += ix.add_record(name, seq)
        nrec += 1
        del seq
        if nrec % 20000 == 0:
            log(f"  ... {nrec:,} records, {bp/1e9:.2f} Gbp")
    if bp == 0:
        sys.exit("no reference sequence loaded")
    log(f"  {bp:,} bp, {anchors:,} anchors (density 1/{bp/max(anchors,1):.1f})")
    ix.build()
    log(f"  {ix.n_entries:,} entries, occ cutoff {ix.occ_cutoff:,}, "
        f"{ix.n_dropped:,} repeat features suppressed, {ix.nbytes()/1e9:.2f} GB "
        f"({ix.nbytes()/max(bp,1):.2f} B/base)")
    return ix


def score_reads(ix, reads, label, max_qbp=None):
    """
    Returns ids, scores, n_anchors, locus_ok (-1 = unknown), mean query length.

    max_qbp truncates each read to its first max_qbp bases. Adaptive sampling
    decides on the leading fragment, so sweeping this is the latency curve:
    how few bases are needed before the keep/reject call can be made.

    eval/demo have every read available upfront (unlike Aligner.map_reads'
    live per-read MinKNOW stream), so this uses query_batch() -- parallel
    across reads -- instead of one query() call at a time.
    """
    headers, seqs = [], []
    for h, seq in reads:
        if max_qbp:
            seq = seq[:max_qbp]
        if len(seq) < ix.p.k + 5:
            continue
        headers.append(h)
        seqs.append(seq)

    results = ix.query_batch(seqs)

    ids = [h.split()[0] for h in headers]
    sc = np.array([r.score for r in results])
    na = np.array([r.n_anchors for r in results])
    lo = np.empty(len(results), np.int8)
    for i, (h, r) in enumerate(zip(headers, results)):
        truth = parse_header(h)
        if truth is not None and r.window >= 0:
            rn, wstart = ix.window_locus(r.window)
            lo[i] = 1 if (rn == truth[0] and wstart < truth[2]
                          and wstart + ix.p.win > truth[1]) else 0
        else:
            lo[i] = -1

    mq = float(np.mean([len(s) for s in seqs])) if seqs else 0.0
    log(f"  {label}: scored {len(ids):,} reads (mean {mq:.0f} bp)")
    return ids, sc, na, lo, mq


def auroc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    a = np.concatenate([p, n])
    order = np.argsort(a, kind="stable")
    sv, r = a[order], np.empty(len(a))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n)))


# ─────────────────────────── subcommands ────────────────────────────────────

def _anchor_kwargs(a):
    return dict(k=a.k, s=a.s, t=a.t, downsample=a.downsample, win=a.win,
                max_occ_pct=a.max_occ_pct, low_mem=not a.high_mem)


def cmd_build(a):
    ix = build_index(a.ref, a.tmpdir, a.batch, a.chunk, a.max_ref_bp, **_anchor_kwargs(a))
    ix.save(a.index)
    log(f"wrote {a.index}")


def cmd_map(a):
    ix = _core.Index()
    ix.load(a.index)
    log(f"loaded {a.index}: {ix.n_entries:,} entries, {ix.nbytes()/1e9:.2f} GB, {ix.p}")
    n, t0 = 0, time.time()

    # Batched (query_batch, parallel across reads) like cumin's native `map`
    # CLI (main.cpp) -- all reads in the file are available upfront here,
    # unlike Aligner.map_reads' live per-read MinKNOW stream, so there's no
    # reason to query one at a time. BATCH_SZ matches main.cpp's.
    BATCH_SZ = 4096

    def flush(fh, ids, seqs):
        for h, seq, r in zip(ids, seqs, ix.query_batch(seqs)):
            ref, ws = ix.window_locus(r.window) if r.window >= 0 else ("*", -1)
            fh.write(f"{h.split()[0]}\t{len(seq)}\t{r.score:.6f}\t{r.n_anchors}\t{r.strand}\t"
                     f"{ref}\t{ws}\t{'keep' if r.score > a.threshold else 'reject'}\n")

    with open(a.out, "w") as fh:
        fh.write("read_id\tquery_bp\tscore\tn_anchors\tstrand\tref\t"
                 "window_start\tdecision\n")
        ids, seqs = [], []
        for h, seq in read_reads(a.reads, a.max_reads):
            if a.max_query_bp:
                seq = seq[:a.max_query_bp]
            if len(seq) < ix.p.k + 5:
                continue
            ids.append(h)
            seqs.append(seq)
            n += 1
            if len(seqs) >= BATCH_SZ:
                flush(fh, ids, seqs)
                ids, seqs = [], []
        if seqs:
            flush(fh, ids, seqs)
    dt = time.time() - t0
    log(f"{n:,} reads in {dt:.1f}s ({1000*dt/max(n,1):.2f} ms/read) -> {a.out}")


def cmd_eval(a):
    if a.index:
        ix = _core.Index()
        ix.load(a.index)
        log(f"loaded {a.index}: {ix.n_entries:,} entries, {ix.nbytes()/1e9:.2f} GB")
    else:
        ix = build_index(a.ref, a.tmpdir, a.batch, a.chunk, a.max_ref_bp, **_anchor_kwargs(a))

    pid, P, Pa, Plo, mq = score_reads(ix, read_reads(a.pos, a.max_reads),
                                      "positives", a.max_query_bp)
    nid, N, Na, Nlo, _ = score_reads(ix, read_reads(a.neg, a.max_reads),
                                     "negatives", a.max_query_bp)

    known = Plo >= 0
    print()
    print("=" * 72)
    print("SYNCMER CONTAINMENT + WINDOW VOTING")
    print("=" * 72)
    print(f"  {ix.p}")
    print(f"  query length          {mq:.0f} bp"
          f"{' (truncated)' if a.max_query_bp else ''}")
    print(f"  AUROC                 {auroc(P, N):.4f}")
    print(f"  median positive       {np.median(P):.4f}   "
          f"(sub-equiv error {1 - np.median(P)**(1/ix.p.k):.4f})")
    print(f"  median negative       {np.median(N):.4f}")
    print(f"  anchors per read      {np.mean(Pa):.1f}")
    print(f"  correct locus         "
          f"{Plo[known].mean() if known.any() else float('nan'):.4f}")
    print()
    print(f"  {'specificity':>12s} {'threshold':>10s} {'sensitivity':>12s} {'n_neg above':>12s}")
    for sp in (0.90, 0.95, 0.99, 0.999):
        thr = float(np.quantile(N, sp))
        above = int((N > thr).sum())
        flag = "  (few negatives, indicative only)" if above < 100 else ""
        print(f"  {sp:12.3f} {thr:10.4f} {float((P > thr).mean()):12.4f} "
              f"{above:12,d}{flag}")
    print()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.tsv", "w") as fh:
        fh.write("label\tread_id\tscore\tn_anchors\tlocus_ok\n")
        for r, v, k_, l_ in zip(pid, P, Pa, Plo):
            fh.write(f"pos\t{r}\t{v:.6f}\t{k_}\t{l_}\n")
        for r, v, k_, l_ in zip(nid, N, Na, Nlo):
            fh.write(f"neg\t{r}\t{v:.6f}\t{k_}\t{l_}\n")
    log(f"wrote {out/'scores.tsv'}")


def cmd_demo(a):
    rng = np.random.default_rng(0)
    log("demo: synthetic reference and 10% error reads")
    refs, conserved = [], None
    for i in range(4):
        seq = "".join(rng.choice(list("ACGT"), 400_000))
        if i == 0:
            conserved = seq[1000:9000]
        else:
            seq = seq[:1000] + conserved + seq[9000:]
        refs.append((f"Rep_{i}_C_0", seq))

    def mutate(s, eps):
        out = []
        for ch in s:
            r = rng.random()
            if r < eps * 0.5:
                out.append(rng.choice(list("ACGT")))
            elif r < eps * 0.75:
                continue
            elif r < eps:
                out.append(ch); out.append(rng.choice(list("ACGT")))
            else:
                out.append(ch)
        return "".join(out)

    ix = _core.Index(tmpdir=a.tmpdir, batch=a.batch, chunk=a.chunk, **_anchor_kwargs(a))
    bp = anchors = 0
    for nm, sq in refs:
        bp += len(sq)
        anchors += ix.add_record(nm, sq)
    ix.build()
    log(f"  {bp:,} bp, {anchors:,} anchors, {ix.n_entries:,} entries")

    pos, neg = [], []
    for _ in range(300):
        nm, sq = refs[int(rng.integers(0, len(refs)))]
        st = int(rng.integers(0, len(sq) - 4000))
        pos.append((f"{nm}!{st}!{st+4000}!+", mutate(sq[st:st + 4000], 0.10)))
    for j in range(300):
        neg.append((f"decoy_{j}", mutate("".join(rng.choice(list("ACGT"), 4000)), 0.10)))

    _, P, Pa, Plo, mq = score_reads(ix, pos, "positives", a.max_query_bp)
    _, N, _, _, _ = score_reads(ix, neg, "negatives", a.max_query_bp)
    thr = float(np.quantile(N, 0.95))
    known = Plo >= 0
    print()
    print(f"  query length     {mq:.0f} bp")
    print(f"  AUROC            {auroc(P, N):.4f}")
    print(f"  median positive  {np.median(P):.4f}  "
          f"(sub-equiv error {1 - np.median(P)**(1/ix.p.k):.4f})")
    print(f"  median negative  {np.median(N):.4f}")
    print(f"  sens @ 95% spec  {float((P > thr).mean()):.4f}  (thr {thr:.4f})")
    print(f"  correct locus    {Plo[known].mean():.4f}")
    print(f"  anchors/read     {np.mean(Pa):.1f}")
    print()
    q = ix.query(pos[0][1][:a.max_query_bp] if a.max_query_bp else pos[0][1])
    log(f"single-read query check: score={q.score:.4f} strand={q.strand} "
        f"locus={ix.window_locus(q.window) if q.window >= 0 else None}")


# ─────────────────────────── cli ────────────────────────────────────────────

def add_query_args(ap):
    ap.add_argument("--max-query-bp", type=int, default=None,
                    help="truncate each read to its first N bases before "
                         "querying; sweep it (150/250/450/...) for the "
                         "sensitivity-vs-decision-latency curve")


def add_anchor_args(ap):
    ap.add_argument("--k", type=int, default=15, help="anchor k-mer size")
    ap.add_argument("--s", type=int, default=8, help="syncmer s-mer size")
    ap.add_argument("--t", type=int, default=0, help="syncmer offset (0 = open)")
    ap.add_argument("--downsample", type=int, default=2,
                    help="density = 1/((k-s+1)*downsample)")
    ap.add_argument("--win", type=int, default=4000, help="reference window (bp)")
    ap.add_argument("--max-occ-pct", type=float, default=99.9,
                    help="drop features above this occurrence percentile")
    ap.add_argument("--high-mem", action="store_true",
                    help="sort the whole reference's anchors in memory in one pass "
                         "instead of streaming through on-disk radix buckets")
    ap.add_argument("--tmpdir", default="/tmp", help="scratch for radix buckets (low-mem mode)")
    ap.add_argument("--batch", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=32_000_000,
                    help="max bp of one record processed at a time")
    ap.add_argument("--max-ref-bp", type=int, default=None,
                    help="cap reference size (default: no cap)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[3],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build and save an index")
    b.add_argument("--ref", required=True)
    b.add_argument("--index", required=True)
    add_anchor_args(b)
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("map", help="classify reads against an index")
    m.add_argument("--index", required=True)
    m.add_argument("--reads", required=True)
    m.add_argument("--out", default="hits.tsv")
    m.add_argument("--threshold", type=float, default=0.15,
                   help="keep/reject cut; set it from an eval run")
    m.add_argument("--max-reads", type=int, default=None)
    add_query_args(m)
    m.set_defaults(func=cmd_map)

    e = sub.add_parser("eval", help="positives vs negatives, with a specificity sweep")
    e.add_argument("--index", help="prebuilt index; otherwise pass --ref")
    e.add_argument("--ref")
    e.add_argument("--pos", required=True)
    e.add_argument("--neg", required=True)
    e.add_argument("--out", default="eval_out")
    e.add_argument("--max-reads", type=int, default=50000)
    add_query_args(e)
    add_anchor_args(e)
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("demo", help="synthetic end-to-end check")
    add_query_args(d)
    add_anchor_args(d)
    d.set_defaults(func=cmd_demo)

    a = ap.parse_args()
    if a.cmd == "eval" and not (a.index or a.ref):
        sys.exit("eval needs --index or --ref")
    a.func(a)


if __name__ == "__main__":
    main()
