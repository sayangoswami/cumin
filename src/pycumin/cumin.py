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
             on-disk buckets as they are produced. Processing buckets 0..255 in
             order and concatenating yields a globally sorted array, so only one
             bucket (~n/256) is ever sorted in RAM, once. No full argsort.

Measured: 3.1 Gbp in 2.44 GB (0.79 B/base), AUROC 0.985, 96.9% correct locus,
~0.7 ms/read, ~5.5 min single-threaded build.

Usage
-----
  cumin.py build --ref gut.fa.gz --index gut.idx.npz
  cumin.py map   --index gut.idx.npz --reads sample.fq.gz --out hits.tsv
  cumin.py eval  --index gut.idx.npz --pos pos.fq.gz --neg neg.fq.gz --out results/
  cumin.py demo                       # synthetic end-to-end check

Read headers for eval are parsed as  <refname>!<start>!<end>!<strand>
e.g.  @Rep_817_C_0!30866!31389!+ qs:f:9.9340

numpy only.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

M_ANCHOR = np.uint64(0x9E3779B97F4A7C15)
M_DOWN = np.uint64(0xC2B2AE3D27D4EB4F)

_B = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    _B[ord(_c)] = _i
    _B[ord(_c.lower())] = _i


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─────────────────────────── sequence ───────────────────────────────────────

def encode(seq: str) -> np.ndarray:
    return _B[np.frombuffer(seq.encode(), np.uint8)]


def revcomp_codes(codes: np.ndarray) -> np.ndarray:
    out = codes[::-1].copy()
    v = out < 4
    out[v] = 3 - out[v]
    return out


def mix64(x):
    """splitmix64 finalizer, vectorised."""
    x = np.asarray(x, np.uint64).copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _roll(codes, m):
    """m-mer base-4 values over `codes`; returns (values int64, bad bool)."""
    n = len(codes)
    if n < m:
        return np.empty(0, np.int64), np.empty(0, bool)
    L = n - m + 1
    c = codes.astype(np.int64)
    valid = codes < 4
    np.putmask(c, ~valid, 0)
    vals = np.zeros(L, np.int64)
    bad = np.zeros(L, bool)
    for j in range(m):
        vals = vals * 4 + c[j:j + L]
        bad |= ~valid[j:j + L]
    return vals, bad


def syncmer_anchors(codes, k, s, t, downsample):
    """Open syncmers. Returns (positions int64, anchor_hash uint64)."""
    n = len(codes)
    w = k - s + 1
    nk = n - k + 1
    if nk <= 0:
        return np.empty(0, np.int64), np.empty(0, np.uint64)

    sv, sbad = _roll(codes, s)
    sh = mix64(np.where(sbad, np.int64(0), sv).astype(np.uint64))
    sh[sbad] = np.uint64(0xFFFFFFFFFFFFFFFF)       # never the minimum

    need = nk + w - 1
    if len(sh) < need:
        return np.empty(0, np.int64), np.empty(0, np.uint64)
    sel = np.lib.stride_tricks.sliding_window_view(sh[:need], w).argmin(axis=1) == t

    kv, kbad = _roll(codes, k)
    sel &= ~kbad[:nk]
    kh = mix64(kv[:nk].astype(np.uint64) * M_ANCHOR)
    if downsample > 1:
        sel &= (mix64(kh ^ M_DOWN) % np.uint64(downsample)) == np.uint64(0)

    pos = np.nonzero(sel)[0].astype(np.int64)
    return pos, kh[pos]


def read_anchors(seq, p):
    """Anchor hashes for a read, forward and reverse-complement."""
    codes = encode(seq)
    _, fwd = syncmer_anchors(codes, p.k, p.s, p.t, p.downsample)
    _, rev = syncmer_anchors(revcomp_codes(codes), p.k, p.s, p.t, p.downsample)
    return fwd, rev


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


# ─────────────────────────── index ──────────────────────────────────────────

class Params:
    """Anchor parameters. Stored with the index so query cannot drift."""

    FIELDS = ("k", "s", "t", "downsample", "win", "max_occ_pct")

    def __init__(self, k=15, s=8, t=0, downsample=2, win=4000, max_occ_pct=99.9):
        self.k, self.s, self.t = k, s, t
        self.downsample, self.win, self.max_occ_pct = downsample, win, max_occ_pct
        if s >= k:
            sys.exit("s must be smaller than k")
        if not 0 <= t <= k - s:
            sys.exit(f"t must be in [0, {k - s}]")

    @property
    def density(self):
        return 1.0 / ((self.k - self.s + 1) * self.downsample)

    def to_json(self):
        return json.dumps({f: getattr(self, f) for f in self.FIELDS})

    @staticmethod
    def from_json(txt):
        return Params(**json.loads(txt))

    def __str__(self):
        return (f"k={self.k} s={self.s} t={self.t} downsample={self.downsample} "
                f"win={self.win} density~1/{1/self.density:.0f}")


class Index:
    """
    Anchors filed under overlapping reference windows.

    ufeat   uint64  sorted unique anchor hashes
    starts  int64   offset of each key's run in wsorted
    cnt     int32   run length
    wsorted int32   window ids, grouped by key
    """

    NB = 256
    SHIFT = np.uint64(56)

    # ---- construction ----------------------------------------------------

    def __init__(self, params, tmpdir="/tmp", batch=20_000_000, chunk=32_000_000):
        self.p = params
        self.step = params.win // 2
        self.batch, self.chunk = batch, chunk
        self.rec_names, self.rec_first = [], []
        self._base = 0
        self._pf, self._pw, self._pending = [], [], 0
        self.n_entries = 0
        self.dir = Path(tmpdir) / f"sm_buckets_{int(time.time())}_{id(self)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = [(open(self.dir / f"f{i:03d}.bin", "wb"),
                     open(self.dir / f"w{i:03d}.bin", "wb")) for i in range(self.NB)]

    def add_record(self, name, seq):
        """
        Index one reference record, chunked so a long chromosome does not need
        ~8 bytes/base of transient arrays at once. Returns the anchor count.
        """
        L = len(seq)
        nwin = max(1, L // self.step + 1)
        rec = len(self.rec_names)
        self.rec_names.append(name)
        self.rec_first.append(self._base)
        base = self._base
        self._base += nwin

        pad = self.p.k + 4096
        step = max(self.chunk, pad * 4)
        n_anchor, start = 0, 0
        while start < L:
            end = min(L, start + step)
            lo, hi = max(0, start - pad), min(L, end + pad)
            pos, ah = syncmer_anchors(encode(seq[lo:hi]),
                                      self.p.k, self.p.s, self.p.t, self.p.downsample)
            if len(pos):
                pos = pos + lo
                keep = (pos >= start) & (pos < end)
                pos, ah = pos[keep], ah[keep]
                n_anchor += len(pos)
                if len(pos):
                    self._file(pos, ah, base, nwin)
            start = end
        if self._pending >= self.batch:
            self._flush()
        return n_anchor

    def _file(self, pos, ah, base, nwin):
        w0 = pos // self.step
        for shift in (0, -1):
            ww = w0 + shift
            ok = (ww >= 0) & (ww < nwin)
            if not ok.any():
                continue
            self._pf.append(ah[ok])
            self._pw.append((ww[ok] + base).astype(np.int32))
            self._pending += int(ok.sum())

    def _flush(self):
        if not self._pf:
            return
        f = np.concatenate(self._pf)
        w = np.concatenate(self._pw)
        self._pf, self._pw, self._pending = [], [], 0
        b = (f >> self.SHIFT).astype(np.uint8)
        o = np.argsort(b, kind="stable")
        f, w, bs = f[o], w[o], b[o]
        del o, b
        edges = np.searchsorted(bs, np.arange(self.NB + 1))
        for i in range(self.NB):
            a, z = int(edges[i]), int(edges[i + 1])
            if z > a:
                f[a:z].tofile(self._fh[i][0])
                w[a:z].tofile(self._fh[i][1])
        self.n_entries += len(f)

    def _bucket(self, i):
        f = np.fromfile(self.dir / f"f{i:03d}.bin", dtype=np.uint64)
        w = np.fromfile(self.dir / f"w{i:03d}.bin", dtype=np.int32)
        if len(f) == 0:
            return f, w
        o = np.argsort(f, kind="stable")
        return f[o], w[o]

    @staticmethod
    def _runs(f):
        m = np.empty(len(f), bool)
        m[0] = True
        np.not_equal(f[1:], f[:-1], out=m[1:])
        st = np.flatnonzero(m)
        return st, np.diff(np.append(st, len(f)))

    def build(self, sample_buckets=16):
        self._flush()
        for a, b in self._fh:
            a.close(); b.close()
        self._fh = []

        # occurrence cutoff. Buckets partition by feature value, so per-bucket
        # counts are exact global counts and a sample suffices for a percentile.
        if self.p.max_occ_pct >= 100:
            self.occ_cutoff = np.iinfo(np.int64).max
        else:
            samp, stride = [], max(1, self.NB // sample_buckets)
            for i in range(0, self.NB, stride):
                f, _ = self._bucket(i)
                if len(f):
                    samp.append(self._runs(f)[1])
                del f
            self.occ_cutoff = (max(1, int(np.percentile(np.concatenate(samp),
                                                        self.p.max_occ_pct)))
                               if samp else np.iinfo(np.int64).max)

        uf, cn, ws, dropped = [], [], [], 0
        for i in range(self.NB):
            f, w = self._bucket(i)
            if len(f) == 0:
                continue
            st, cnt = self._runs(f)
            keep = cnt <= self.occ_cutoff
            dropped += int((~keep).sum())
            if keep.any():
                uf.append(f[st[keep]])
                cn.append(cnt[keep].astype(np.int32))
                ws.append(w[np.repeat(keep, cnt)])
            del f, w, st, cnt, keep
            (self.dir / f"f{i:03d}.bin").unlink(missing_ok=True)
            (self.dir / f"w{i:03d}.bin").unlink(missing_ok=True)

        self.n_dropped = dropped
        if uf:
            self.ufeat = np.concatenate(uf); uf.clear()
            self.cnt = np.concatenate(cn); cn.clear()
            self.wsorted = np.concatenate(ws); ws.clear()
            self.starts = np.concatenate(([0], np.cumsum(self.cnt.astype(np.int64))[:-1]))
        else:
            self.ufeat = np.empty(0, np.uint64)
            self.cnt = np.empty(0, np.int32)
            self.wsorted = np.empty(0, np.int32)
            self.starts = np.empty(0, np.int64)
        self.rec_first = np.asarray(self.rec_first, np.int64)
        try:
            self.dir.rmdir()
        except OSError:
            pass
        return self

    # ---- persistence -----------------------------------------------------

    def save(self, path):
        np.savez(path, ufeat=self.ufeat, starts=self.starts, cnt=self.cnt,
                 wsorted=self.wsorted, rec_first=self.rec_first,
                 rec_names=np.array(self.rec_names, dtype=object),
                 params=np.array(self.p.to_json()),
                 occ_cutoff=np.array(self.occ_cutoff),
                 n_entries=np.array(self.n_entries))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        ix = Index.__new__(Index)
        ix.p = Params.from_json(str(z["params"]))
        ix.step = ix.p.win // 2
        for f in ("ufeat", "starts", "cnt", "wsorted", "rec_first"):
            setattr(ix, f, z[f])
        ix.rec_names = list(z["rec_names"])
        ix.occ_cutoff = int(z["occ_cutoff"])
        ix.n_entries = int(z["n_entries"])
        ix.n_dropped = 0
        return ix

    def nbytes(self):
        return sum(a.nbytes for a in (self.ufeat, self.starts, self.cnt, self.wsorted))

    def window_locus(self, wid):
        """Window id -> (record name, window start bp)."""
        r = int(np.searchsorted(self.rec_first, wid, "right") - 1)
        return self.rec_names[r], int((wid - self.rec_first[r]) * self.step)

    # ---- query -----------------------------------------------------------

    def vote(self, anchors):
        """
        One orientation. Returns (score, window_id, n_unique_anchors).
        score = best_window_votes / n_unique_anchors.
        """
        if len(anchors) == 0 or len(self.ufeat) == 0:
            return 0.0, -1, 0
        q = np.unique(anchors)
        lo = np.clip(np.searchsorted(self.ufeat, q), 0, len(self.ufeat) - 1)
        hit = self.ufeat[lo] == q
        if not hit.any():
            return 0.0, -1, len(q)
        sel = lo[hit]
        st = self.starts[sel]
        cnt = self.cnt[sel].astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            return 0.0, -1, len(q)
        off = np.repeat(np.cumsum(cnt) - cnt, cnt)
        gidx = np.repeat(st, cnt) + (np.arange(total) - off)
        wins, wc = np.unique(self.wsorted[gidx], return_counts=True)
        b = int(wc.argmax())
        return float(wc[b]) / len(q), int(wins[b]), len(q)

    def query(self, seq):
        """
        Both orientations, better kept.
        Returns (score, window_id, strand, n_unique_anchors).
        """
        fwd, rev = read_anchors(seq, self.p)
        sf, wf, nf = self.vote(fwd)
        sr, wr, nr = self.vote(rev)
        if sf >= sr:
            return sf, wf, "+", nf
        return sr, wr, "-", nr


# ─────────────────────────── driver helpers ─────────────────────────────────

def build_index(ref_path, params, tmpdir, batch, chunk, max_ref_bp=None):
    ix = Index(params, tmpdir, batch, chunk)
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
    """
    ids, sc, na, lo, qlen = [], [], [], [], []
    for h, seq in reads:
        if max_qbp:
            seq = seq[:max_qbp]
        if len(seq) < ix.p.k + 5:
            continue
        qlen.append(len(seq))
        s_, wid, _, nq = ix.query(seq)
        ids.append(h.split()[0])
        sc.append(s_)
        na.append(nq)
        truth = parse_header(h)
        if truth is not None and wid >= 0:
            rn, wstart = ix.window_locus(wid)
            lo.append(1 if (rn == truth[0] and wstart < truth[2]
                            and wstart + ix.p.win > truth[1]) else 0)
        else:
            lo.append(-1)
    mq = float(np.mean(qlen)) if qlen else 0.0
    log(f"  {label}: scored {len(ids):,} reads (mean {mq:.0f} bp)")
    return ids, np.array(sc), np.array(na), np.array(lo, np.int8), mq


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

def cmd_build(a):
    p = Params(a.k, a.s, a.t, a.downsample, a.win, a.max_occ_pct)
    log(f"building: {p}")
    ix = build_index(a.ref, p, a.tmpdir, a.batch, a.chunk, a.max_ref_bp)
    ix.save(a.index)
    log(f"wrote {a.index}")


def cmd_map(a):
    ix = Index.load(a.index)
    log(f"loaded {a.index}: {ix.n_entries:,} entries, {ix.nbytes()/1e9:.2f} GB, {ix.p}")
    n, t0 = 0, time.time()
    with open(a.out, "w") as fh:
        fh.write("read_id\tquery_bp\tscore\tn_anchors\tstrand\tref\t"
                 "window_start\tdecision\n")
        for h, seq in read_reads(a.reads, a.max_reads):
            if a.max_query_bp:
                seq = seq[:a.max_query_bp]
            if len(seq) < ix.p.k + 5:
                continue
            s_, wid, strand, nq = ix.query(seq)
            ref, ws = ix.window_locus(wid) if wid >= 0 else ("*", -1)
            fh.write(f"{h.split()[0]}\t{len(seq)}\t{s_:.6f}\t{nq}\t{strand}\t"
                     f"{ref}\t{ws}\t{'keep' if s_ > a.threshold else 'reject'}\n")
            n += 1
    dt = time.time() - t0
    log(f"{n:,} reads in {dt:.1f}s ({1000*dt/max(n,1):.2f} ms/read) -> {a.out}")


def cmd_eval(a):
    if a.index:
        ix = Index.load(a.index)
        log(f"loaded {a.index}: {ix.n_entries:,} entries, {ix.nbytes()/1e9:.2f} GB")
    else:
        p = Params(a.k, a.s, a.t, a.downsample, a.win, a.max_occ_pct)
        log(f"building: {p}")
        ix = build_index(a.ref, p, a.tmpdir, a.batch, a.chunk, a.max_ref_bp)

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

    p = Params(a.k, a.s, a.t, a.downsample, a.win, a.max_occ_pct)
    ix = Index(p, a.tmpdir, a.batch, a.chunk)
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
          f"(sub-equiv error {1 - np.median(P)**(1/p.k):.4f})")
    print(f"  median negative  {np.median(N):.4f}")
    print(f"  sens @ 95% spec  {float((P > thr).mean()):.4f}  (thr {thr:.4f})")
    print(f"  correct locus    {Plo[known].mean():.4f}")
    print(f"  anchors/read     {np.mean(Pa):.1f}")
    print()
    q = ix.query(pos[0][1][:a.max_query_bp] if a.max_query_bp else pos[0][1])
    log(f"single-read query check: score={q[0]:.4f} strand={q[2]} "
        f"locus={ix.window_locus(q[1]) if q[1] >= 0 else None}")


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
    ap.add_argument("--tmpdir", default="/tmp", help="scratch for radix buckets")
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