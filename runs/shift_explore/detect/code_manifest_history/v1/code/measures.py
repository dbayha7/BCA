"""EXPLORATORY detection rehearsal (plan section 4.4), NO BCA: the JOIN of one signal's scores with the labels, and
the measures. The join runs only on two hashed files: the scores record (written by signals.score_signal, which never
reads a label) and the labels record (written by labels.py, which never reads a signal). Both are verified against
their sidecars and the file and content SHA-256 their records name, and both must name the same pair sets (the same
pairs content SHA-256), which are loaded and verified too. Nothing here fits anything.

Notation per set T: for each state (pair id) the score at the command s(cmd), at its 80% cap s(cap), at the recorded
command s(rec) (T1, T2) and at the 81% knee command s(knee) (T1); labels at the command: past80 (unsupported), harmful,
over = over_pref. The paired penalty of the consumer (change A1): W(x) = [s(x) - s(rec)]_+ at the same state.

  level                    the signal's split-conformal 95% level on T4: the ceil((n + 1) 0.95)-th smallest T4 score
                           (n = 5,000), unless a level is passed (Stage 2 passes its tau). flag = score > level.
  widening_T_past80        share of past-80% command pairs of T with s(cmd) > s(cap)        (plan: paired widening)
  widening_T_harmful       the same over harmful command pairs           (Stage 2's direction mark, T1 and T3)
  widening_T_harmless_past80  the same over past-80% pairs that are not harmful (reported)
  widening_T_harmful_paired   share of harmful pairs with W(cmd) > W(cap) (T1, T2; reported)
  size_T_raw               median over harmful command pairs of (s(cmd) - s(cap)) / over       (value-unit signals)
  size_T_paired            median over harmful command pairs of (W(cmd) - W(cap)) / over (T1, T2; Stage 2's size mark)
  knee_rise_T1_past80 / _harmful   share with s(knee) > s(cap) (check (e)'s lesson: does it rise just past 80%?)
  knee_rise_T1_harmful_paired      share with W(knee) > W(cap)
  ranking_T12, ranking_T3  P(score of a random harmful pair > score of a random harmless pair), every pair (every
                           role) of T1 and T2 together / of T3; ties count one half (plan: ranking score)
  ranking_T1, ranking_T2, ranking_T3a, ranking_T3b   the same per set (reported)
  ranking_T12_among_past80, ranking_T3_among_past80  the same among past-80% pairs only: harm against mere novelty
  rate_above_level_T4      share of T4 rows flagged (about 5% at the signal's own level by construction)
  catch_rate_T_harmful     share of harmful pairs of T flagged                  (plan: catch rate, T1, T2, T3, T3a, T3b)
  catch_rate_T_past80      share of past-80% command pairs of T flagged; catch_rate_T3a_past80 is the fallback
                           diagnostic's bar (strong-current proposals along the actor's own rollouts: distance 35%,
                           exact one-step check 67%)
  false_flag_T             share of harmless pairs of T flagged (every role); false_flag_T_<role> per role
  flag_rate_past80         share of past-80% command pairs of T1 and T3 together flagged; selectivity = this in the
                           stalling world minus the same in the no-stall world (pooled phase); flag_rate_past80_T per set
  n_*                      counts
A measure over an empty group is None.
"""

import math

import numpy as np

import common as C
import pairs as P

LEVEL = 0.95


def conformal_level(scores, level=LEVEL):
    """The ceil((n + 1) * level)-th smallest score (+inf when that exceeds n)."""
    s = np.sort(np.asarray(scores, np.float64))
    k = int(math.ceil((len(s) + 1) * level))
    return float(s[k - 1]) if 1 <= k <= len(s) else math.inf


def auroc(score, positive):
    """P(score of a random positive > score of a random negative), ties one half (average ranks); None when a class
    is empty."""
    from scipy.stats import rankdata

    positive = np.asarray(positive, bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(score, np.float64))
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _share(x):
    x = np.asarray(x, bool)
    return float(x.mean()) if x.size else None


def _median(x):
    x = np.asarray(x, np.float64)
    return float(np.median(x)) if x.size else None


def load_join(scores_record_path, labels_record_path):
    """The verified join: (pairs arrays, scores, labels arrays, scores record, labels record, pairs record)."""
    srec = C.read_json(scores_record_path)
    lrec = C.read_json(labels_record_path)
    if srec is None or lrec is None:
        raise SystemExit("refused: the join needs both the scores record and the labels record")
    if srec.get("kind") != "detect-scores" or lrec.get("kind") != "detect-labels":
        raise SystemExit("refused: not a scores record and a labels record")
    if srec.get("reads_labels") is not False or lrec.get("reads_signals") is not False:
        raise SystemExit("refused: a record does not state that it was written blind to the other side")
    if srec["pairs_content_sha256"] != lrec["pairs_content_sha256"]:
        raise SystemExit("refused: the scores and the labels were computed on different pair sets")
    sdir = C.Path(scores_record_path).parent
    ldir = C.Path(labels_record_path).parent
    scores = C.load_npz(sdir / srec["file"], srec["file_sha256"], srec["content_sha256"])["scores"]
    lab = C.load_npz(ldir / lrec["file"], lrec["file_sha256"], lrec["content_sha256"])
    pdir = C.Path(srec["pairs_dir"])
    prec = C.read_json(pdir / P.RECORD)
    if prec is None or prec["content_sha256"] != srec["pairs_content_sha256"]:
        raise SystemExit("refused: the pair sets the scores name are not on disk")
    arrays = C.load_npz(pdir / P.FILE, prec["file_sha256"], prec["content_sha256"])
    if len(scores) != len(arrays["set"]) or any(len(v) != len(scores) for v in lab.values()):
        raise SystemExit("refused: scores, labels and pairs differ in length")
    return arrays, scores, lab, srec, lrec, prec


def compute(arrays, scores, lab, *, units="value", level=None):
    """The measures (module docstring) from joined arrays."""
    s = np.asarray(scores, np.float64)
    if np.any(np.isnan(s)):
        raise ValueError("NaN scores")
    v = P.view(arrays)
    harmful = np.asarray(lab["harmful"], bool)
    past = np.asarray(lab["unsupported"], bool)
    over = np.asarray(lab["over_pref"], np.float64)
    t4 = arrays["set"] == "T4"
    own_level = conformal_level(s[t4])
    lev = own_level if level is None else float(level)
    flag = s > lev
    value = units == "value"
    m = dict(level=lev, own_level_T4=own_level, level_rule=("given" if level is not None else
                                                            "split-conformal 95% on T4: ceil((n + 1) 0.95)-th smallest"),
             units=units, rate_above_level_T4=_share(flag[t4]), n_T4=int(t4.sum()))
    groups = {"T1": ("T1", None), "T2": ("T2", None), "T3": ("T3", None), "T3a": ("T3", "actor"),
              "T3b": ("T3", "controller")}
    for g, (st, src) in groups.items():
        ix = v[st]
        ic, ip = ix["command"], ix["cap"]
        keep = np.ones(len(ic), bool) if src is None else (arrays["t3_source"][ic] == src)
        ic, ip = ic[keep], ip[keep]
        pa, ha, ov = past[ic], harmful[ic], over[ic]
        if np.any(harmful[ip]) or np.any(past[ip]):
            raise ValueError("a cap pair is labelled past 80% or harmful")
        wid = s[ic] > s[ip]
        m[f"n_pairs_{g}"] = int(len(ic))
        m[f"n_past80_{g}"] = int(pa.sum())
        m[f"n_harmful_{g}"] = int(ha.sum())
        m[f"widening_{g}_past80"] = _share(wid[pa])
        m[f"widening_{g}_harmful"] = _share(wid[ha])
        m[f"widening_{g}_harmless_past80"] = _share(wid[pa & ~ha])
        m[f"size_{g}_raw"] = _median((s[ic][ha] - s[ip][ha]) / ov[ha]) if value else None
        m[f"catch_rate_{g}_harmful"] = _share(flag[ic][ha])
        m[f"catch_rate_{g}_past80"] = _share(flag[ic][pa])
        m[f"flag_rate_past80_{g}"] = m[f"catch_rate_{g}_past80"]
        rows = np.concatenate([ix[r][keep] for r in P.SET_ROLES[st]])
        m[f"false_flag_{g}"] = _share(flag[rows][~harmful[rows]])
        for r in P.SET_ROLES[st]:
            rr = ix[r][keep]
            m[f"false_flag_{g}_{r}"] = _share(flag[rr][~harmful[rr]])
        m[f"ranking_{g}"] = auroc(s[rows], harmful[rows])
        m[f"ranking_{g}_among_past80"] = auroc(s[rows][past[rows]], harmful[rows][past[rows]])
        if "recorded" in ix:
            ir = ix["recorded"]
            W = lambda i: np.maximum(s[i] - s[ir], 0.0)  # noqa: E731
            wc, wp = W(ic), W(ip)
            m[f"widening_{g}_harmful_paired"] = _share((wc > wp)[ha])
            m[f"widening_{g}_past80_paired"] = _share((wc > wp)[pa])
            m[f"size_{g}_paired"] = _median((wc[ha] - wp[ha]) / ov[ha]) if value else None
            if "knee" in ix:
                ik = ix["knee"]
                m[f"knee_rise_{g}_past80"] = _share((s[ik] > s[ip])[pa])
                m[f"knee_rise_{g}_harmful"] = _share((s[ik] > s[ip])[ha])
                m[f"knee_rise_{g}_harmful_paired"] = _share((W(ik) > wp)[ha])
                m[f"knee_rise_{g}_past80_paired"] = _share((W(ik) > wp)[pa])
    r12 = np.concatenate([np.concatenate([v[st][r] for r in P.SET_ROLES[st]]) for st in ("T1", "T2")])
    m["ranking_T12"] = auroc(s[r12], harmful[r12])
    m["ranking_T12_among_past80"] = auroc(s[r12][past[r12]], harmful[r12][past[r12]])
    i13 = np.concatenate([v["T1"]["command"], v["T3"]["command"]])
    m["flag_rate_past80"] = _share(flag[i13][past[i13]])
    m["n_past80_T1_T3"] = int(past[i13].sum())
    return m


def measures(scores_path, labels_path, *, level=None, extra_levels=None):
    """Join one signal's hashed scores with the hashed labels and compute the measures (module docstring).
    scores_path / labels_path: the records (<signal>.json, labels.json). level: None for the signal's own 95% level
    on T4, or Stage 2's tau. extra_levels: {name: level} reported beside (flag rates only). Returns a dict holding
    the measures and the SHA-256 of everything joined."""
    arrays, scores, lab, srec, lrec, prec = load_join(scores_path, labels_path)
    m = compute(arrays, scores, lab, units=srec.get("units", "value"), level=level)
    extra = {}
    for name, lv in (extra_levels or {}).items():
        mm = compute(arrays, scores, lab, units=srec.get("units", "value"), level=lv)
        extra[name] = {k: mm[k] for k in mm if k.startswith(("catch_rate", "false_flag", "flag_rate", "rate_above"))
                       or k == "level"}
    return dict(label=C.LABEL if not str(srec.get("label", "")).startswith("EXPLORATORY/STAGE 2") else srec["label"],
                kind="detect-measures", signal=srec["signal"], rep=srec["rep"], world=srec["world"],
                measures=m, extra_levels=extra, joined=dict(
                    scores_record_sha256=C.file_sha256(scores_path), scores_content_sha256=srec["content_sha256"],
                    labels_record_sha256=C.file_sha256(labels_path), labels_content_sha256=lrec["content_sha256"],
                    pairs_content_sha256=prec["content_sha256"], signal_sha256=srec.get("signal_sha256")))
