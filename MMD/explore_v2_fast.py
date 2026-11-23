"""Helpers for explore_v2_fast.ipynb: compare v2_fast runs over sigma and look inside the sums.

Every number comes from run_v2_fast folders with output_version >= 2 (they store the three
sums). Per sequence, with N GT and M generated events (both polarities):

    a = S_gg / N^2        GT with GT          (same for every generator)
    b = S_pp / M^2        prediction with itself
    c = S_gp / (N M)      GT with prediction
    prob_distance^2 = a + b - 2c,   similarity = c / sqrt(a b)

The point cloud recomputes the same three sums event by event inside a short window, so
one can see which events make each term large.
"""
import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .benchmark_io import Reference, benchmark
from .events import Events, prepare_events
from .kernel_v2_fast import PairPlan

HERE = Path(__file__).resolve().parent
METRICS = {"similarity": ("Similarity to GT (cosine)", "higher is better"),
           "prob_distance": ("MMD, normalized (shape)", "lower is better"),
           "count_relative_to_empty": ("Count distance / empty prediction", "lower is better"),
           "rank": ("Rank per sequence by similarity", "1 = best")}
# Explicit colour scales: dropdown buttons pass them to plotly.js unchanged.
SEQUENTIAL = [[0.0, "#440154"], [0.25, "#3b528b"], [0.5, "#21918c"], [0.75, "#5ec962"],
              [1.0, "#fde725"]]
DIVERGING = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]  # blue < 0 < red
TERMS = {"a": "GT – GT   a = S_gg / N²", "b": "prediction – prediction   b = S_pp / M²",
         "c": "GT – prediction   c = S_gp / (N M)"}


# ----------------------------------------------------------------------------- run results

def discover_runs(dataset, split, root=None):
    """v2_fast run folders of one dataset/split that store the sums, one per (σs, σt).

    When several folders share a (σs, σt), the one with most finished rows is kept.
    """
    import pandas as pd

    root = Path(root) if root else HERE / "runs"
    found = []
    for config_file in sorted(root.glob("*/config.json")):
        config = json.loads(config_file.read_text())
        if (config.get("dataset"), config.get("split")) != (dataset, split):
            continue
        if config.get("engine") != "v2_fast" or config.get("output_version", 1) < 2:
            continue
        folder = config_file.parent
        found.append(dict(run=folder.name, folder=folder,
                          sigma_spatial_px=float(config["sigma_spatial_px"]),
                          sigma_temporal_ms=float(config["sigma_temporal_ms"]),
                          cutoff_sigma=config.get("cutoff_sigma"),
                          time_window_s=config.get("time_window_s"),
                          rows=len(list((folder / "rows").glob("*/*.json")))))
    runs = pd.DataFrame(found, columns=["run", "folder", "sigma_spatial_px",
                                        "sigma_temporal_ms", "cutoff_sigma", "time_window_s",
                                        "rows"])
    if runs.empty:
        return runs
    runs = runs.sort_values("rows", ascending=False)
    dropped = runs[runs.duplicated(["sigma_spatial_px", "sigma_temporal_ms"])]
    for name in dropped.run:
        print(f"Skipping {name}: another folder with the same sigmas has more rows")
    return (runs.drop_duplicates(["sigma_spatial_px", "sigma_temporal_ms"])
            .sort_values(["sigma_temporal_ms", "sigma_spatial_px"]).reset_index(drop=True))


def runs_from_folders(folders, dataset, split):
    """The run folders you list, checked: same dataset/split, v2_fast with the three sums,
    and one folder per (σs, σt). Relative paths are taken from the repository root."""
    import pandas as pd

    found = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_absolute():
            folder = HERE.parent / folder
        config_file = folder / "config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"No run in {folder} (config.json missing)")
        config = json.loads(config_file.read_text())
        if (config.get("dataset"), config.get("split")) != (dataset, split):
            raise ValueError(f"{folder.name} is {config.get('dataset')}/{config.get('split')}, "
                             f"not {dataset}/{split}")
        if config.get("engine") != "v2_fast" or config.get("output_version", 1) < 2:
            raise ValueError(f"{folder.name} has no stored sums: rerun it with the current "
                             "run_v2_fast into a new folder")
        found.append(dict(run=folder.name, folder=folder,
                          sigma_spatial_px=float(config["sigma_spatial_px"]),
                          sigma_temporal_ms=float(config["sigma_temporal_ms"]),
                          cutoff_sigma=config.get("cutoff_sigma"),
                          time_window_s=config.get("time_window_s"),
                          rows=len(list((folder / "rows").glob("*/*.json")))))
    runs = pd.DataFrame(found)
    if runs.empty:
        raise ValueError("RUN_FOLDERS is empty")
    twins = runs[runs.duplicated(["sigma_spatial_px", "sigma_temporal_ms"], keep=False)]
    if len(twins):
        raise ValueError(f"Same sigmas in more than one folder: {list(twins.run)}; keep one")
    return runs.sort_values(["sigma_temporal_ms", "sigma_spatial_px"]).reset_index(drop=True)


def load_scores(runs):
    """One row per (run, sequence, generator) with status ok, plus a, b, c and the rank."""
    import pandas as pd

    rows = []
    for run in runs.itertuples():
        for path in sorted(Path(run.folder, "rows").glob("*/*.json")):
            row = json.loads(path.read_text())
            if row.get("status") != "ok" or row.get("sum_gt_gt_on") is None:
                continue
            n, m = float(row["n_gt"]), float(row["n_gen"])
            rows.append(dict(
                run=run.run, sigma_spatial_px=run.sigma_spatial_px,
                sigma_temporal_ms=run.sigma_temporal_ms, sequence=row["sequence"],
                generator=row["generator"], n_gt=n, n_gen=m,
                similarity=row["similarity"], prob_distance=row["prob_distance"],
                count_relative_to_empty=row["count_relative_to_empty"],
                a=(row["sum_gt_gt_on"] + row["sum_gt_gt_off"]) / n ** 2,
                b=(row["sum_gen_gen_on"] + row["sum_gen_gen_off"]) / m ** 2,
                c=(row["sum_gt_gen_on"] + row["sum_gt_gen_off"]) / (n * m)))
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    scores["rank"] = scores.groupby(["run", "sequence"]).similarity.rank(ascending=False)
    scores["b_over_a"] = scores.b / scores.a
    scores["c_over_a"] = scores.c / scores.a
    return scores


def check_terms(scores):
    """Largest relative gap between prob_distance^2 and a + b - 2c (should be ~1e-12)."""
    rebuilt = scores.a + scores.b - 2 * scores.c
    return float((rebuilt - scores.prob_distance ** 2).abs().max()
                 / (scores.a + scores.b).min())


def sigma_label(sigma_spatial_px, sigma_temporal_ms):
    return f"σ {sigma_spatial_px:g} px / {sigma_temporal_ms:g} ms"


def metric_table(scores):
    """Table 1: mean over sequences of each metric, per generator (rows) and sigma (columns)."""
    import pandas as pd

    frame = scores.assign(sigma=[sigma_label(s, t) for s, t in
                                 zip(scores.sigma_spatial_px, scores.sigma_temporal_ms)])
    order = list(dict.fromkeys(frame.sort_values(["sigma_temporal_ms", "sigma_spatial_px"])
                               .sigma))
    table = frame.pivot_table(index="generator", columns="sigma", aggfunc="mean",
                              values=["similarity", "prob_distance", "count_relative_to_empty",
                                      "rank"])
    columns = [(metric, sigma) for metric in ("similarity", "rank", "prob_distance",
                                              "count_relative_to_empty") for sigma in order]
    table = table.reindex(columns=pd.MultiIndex.from_tuples(columns))
    table.insert(0, ("sequences", ""), frame.groupby("generator").sequence.nunique())
    return table.sort_values(("similarity", order[0]), ascending=False)


def terms_table(scores):
    """Table 2: mean over sequences of a, b, c and a + b - 2c, per generator and sigma."""
    frame = scores.assign(sigma=[sigma_label(s, t) for s, t in
                                 zip(scores.sigma_spatial_px, scores.sigma_temporal_ms)],
                          distance_squared=scores.a + scores.b - 2 * scores.c)
    table = frame.groupby(["sigma", "generator"])[
        ["a", "b", "c", "distance_squared", "b_over_a", "c_over_a"]].mean()
    return table.rename(columns={"a": "a  GT–GT", "b": "b  pred–pred", "c": "c  GT–pred",
                                 "distance_squared": "a + b − 2c", "b_over_a": "b / a",
                                 "c_over_a": "c / a"})


# ----------------------------------------------------------------------------- plots

def _mean_and_spread(scores, value):
    grouped = scores.groupby(["sigma_temporal_ms", "generator", "sigma_spatial_px"])[value]
    return grouped.mean(), grouped.std().fillna(0.0)


def metric_vs_sigma(scores):
    """Plot 1: metric (dropdown) against σ spatial; one line per generator, error = std."""
    import plotly.graph_objects as go

    fig = go.Figure()
    generators = list(scores.groupby("generator").similarity.mean()
                      .sort_values(ascending=False).index)
    temporals = sorted(scores.sigma_temporal_ms.unique())
    keys = [(metric, st) for metric in METRICS for st in temporals]
    owners = []
    for metric, st in keys:
        mean, spread = _mean_and_spread(scores, metric)
        for g in generators:
            if (st, g) not in mean.index:
                continue
            owners.append((metric, st))
            m, s = mean.loc[(st, g)], spread.loc[(st, g)]
            fig.add_trace(go.Scatter(
                x=list(m.index), y=list(m.values), mode="lines+markers", name=g,
                error_y=dict(type="data", array=list(s.values), visible=True),
                visible=(metric, st) == keys[0], legendgroup=g,
                hovertemplate=f"{g}<br>σ %{{x:g}} px<br>{metric} %{{y:.4g}}<extra></extra>"))
    buttons = []
    for metric, st in keys:
        title, hint = METRICS[metric]
        label = title if len(temporals) == 1 else f"{title} · σt {st:g} ms"
        buttons.append(dict(label=label, method="update", args=[
            {"visible": [owner == (metric, st) for owner in owners]},
            {"title.text": f"{label} ({hint}) vs spatial σ, mean ± std over sequences",
             "yaxis.title.text": metric,
             "yaxis.autorange": "reversed" if metric == "rank" else True}]))
    first = buttons[0]["args"][1]
    fig.update_layout(
        title=dict(text=first["title.text"]), xaxis_title="spatial σ (px)",
        yaxis_title=first["yaxis.title.text"], height=480, hovermode="closest",
        updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left", y=1.15,
                          yanchor="top", showactive=True)],
        margin=dict(t=110))
    return fig


def terms_vs_sigma(scores):
    """Plot 2: a, b, c against σ spatial, one panel each; absolute (log) or relative to a."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(TERMS.values()),
                        horizontal_spacing=0.07)
    generators = list(scores.groupby("generator").similarity.mean()
                      .sort_values(ascending=False).index)
    temporals = sorted(scores.sigma_temporal_ms.unique())
    st = temporals[0]
    data = scores[scores.sigma_temporal_ms == st]
    views = {"absolute": ("a", "b", "c"), "relative": ("a_over_a", "b_over_a", "c_over_a")}
    data = data.assign(a_over_a=1.0)
    owners = []
    for view, columns in views.items():
        for col, column in enumerate(columns, 1):
            # a depends on GT only: one line.
            lines = ["GT"] if col == 1 else generators
            for g in lines:
                part = data if g == "GT" else data[data.generator == g]
                if col == 1:
                    part = part.drop_duplicates(["sigma_spatial_px", "sequence"])
                mean = part.groupby("sigma_spatial_px")[column].mean()
                fig.add_trace(go.Scatter(
                    x=list(mean.index), y=list(mean.values), mode="lines+markers", name=g,
                    legendgroup=g, showlegend=col == 2 or (col == 1 and g == "GT"),
                    line=dict(color="black", dash="dot") if g == "GT" else None,
                    visible=view == "absolute",
                    hovertemplate=f"{g}<br>σ %{{x:g}} px<br>%{{y:.4g}}<extra></extra>"),
                    row=1, col=col)
                owners.append(view)
    axes = {f"yaxis{'' if i == 1 else i}.type": None for i in (1, 2, 3)}
    buttons = [
        dict(label="Absolute (log scale)", method="update",
             args=[{"visible": [o == "absolute" for o in owners]},
                   {**{k: "log" for k in axes},
                    "title.text": f"The three terms vs spatial σ (σt {st:g} ms), mean over "
                                  "sequences"}]),
        dict(label="Relative to GT–GT (b / a, c / a)", method="update",
             args=[{"visible": [o == "relative" for o in owners]},
                   {**{k: "linear" for k in axes},
                    "title.text": f"b / a and c / a vs spatial σ (σt {st:g} ms): b/a > 1 = "
                                  "prediction more clustered than GT; c/a = overlap"}])]
    fig.update_layout(title=dict(text=buttons[0]["args"][1]["title.text"]), height=460,
                      updatemenus=[dict(buttons=buttons, direction="down", x=0,
                                        xanchor="left", y=1.2, yanchor="top")],
                      margin=dict(t=120))
    for i in (1, 2, 3):
        fig.update_xaxes(title_text="spatial σ (px)", row=1, col=i)
        fig.update_yaxes(type="log", row=1, col=i)
    if len(temporals) > 1:
        print(f"Plot 2 shows σt = {st:g} ms; other σt values: {temporals[1:]}")
    return fig


# ----------------------------------------------------------------------------- event windows

def _npz_member(path, name):
    """Memory-map one member of an uncompressed .npz; None if it is compressed."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{name}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, "rb") as stream:
        stream.seek(info.header_offset)
        local = stream.read(30)
        start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") \
            + int.from_bytes(local[28:30], "little")
        stream.seek(start)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            return None
        offset = stream.tell()
    if fortran or len(shape) != 1:
        return None
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def _prediction_window(path, start, stop):
    """Raw prediction events with start <= t < stop, reading only that slice if possible."""
    arrays = {k: _npz_member(path, k) for k in "txyp"}
    if any(a is None for a in arrays.values()):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in "txyp"}
    lo, hi = np.searchsorted(arrays["t"], [start, stop])
    return {k: np.asarray(v[lo:hi]) for k, v in arrays.items()}


def load_window(run_folder, sequence, generator, start_s, duration_ms, region=None,
                gt_root=None, pred_root=None):
    """GT and prediction events of one finished pair in [start, start + duration).

    start_s=None centres the window in the scored interval. By default the GT folder and
    prediction file are the ones the run recorded for this pair; gt_root / pred_root
    (the same layout as --data-root / --pred-root) read them from elsewhere instead.
    Uses the GT folder and prediction file recorded in the pair's row, the same alignment
    checks as the runner, and reads only the window from disk. region = (x, y, w, h) in
    the scored frame keeps a smaller patch.
    """
    run_folder = Path(run_folder)
    if not run_folder.is_absolute():
        run_folder = HERE.parent / run_folder
    row = json.loads((run_folder / "rows" / sequence / f"{generator}.json").read_text())
    config = json.loads((run_folder / "config.json").read_text())
    if row.get("status") != "ok":
        raise ValueError(f"{sequence}/{generator} is {row.get('status')}: {row.get('detail')}")
    source, prediction = Path(row["gt_source"]), Path(row["prediction"])
    if gt_root is not None:
        source = Path(gt_root) / config["dataset"] / config["split"] / sequence
    if pred_root is not None:
        prediction = (Path(pred_root) / config["dataset"] / config["split"] / generator
                      / sequence / "events.npz")
    reference = Reference(source, config.get("allow_benchmark_crop", False))
    metadata = reference.align(prediction, argparse.Namespace(**config), sequence)
    if start_s is None:
        start_s = max(0.0, (metadata["duration"] - duration_ms / 1000) / 2)
    start, stop = float(start_s), float(start_s) + duration_ms / 1000
    if not 0 <= start < stop <= metadata["duration"]:
        raise ValueError(f"Window must lie inside [0, {metadata['duration']:.3f}) s")
    crop_x, crop_y, width, height = metadata["crop"]
    _, (full_width, full_height) = reference.geometry

    origin = metadata["absolute_t0"]
    times = np.load(source / "ev_t.npy", mmap_mode="r")
    lo, hi = np.searchsorted(times, [origin + start, origin + stop])
    raw = {k: np.asarray(np.load(source / f"ev_{k}.npy", mmap_mode="r")[lo:hi]) for k in "xytp"}
    gt = prepare_events(raw, full_width, full_height, unit="s", origin=origin,
                        polarity="benchmark")
    keep = ((gt.x >= crop_x) & (gt.x < crop_x + width)
            & (gt.y >= crop_y) & (gt.y < crop_y + height))
    gt = Events(gt.x[keep].astype(np.int64) - crop_x, gt.y[keep].astype(np.int64) - crop_y,
                gt.t[keep], gt.p[keep])

    generated = prepare_events(_prediction_window(prediction, start, stop), width, height,
                               unit="s", polarity="signed")
    if region is not None:
        x0, y0, w, h = region
        gt, generated = (Events(e.x[k] - x0, e.y[k] - y0, e.t[k], e.p[k]) for e, k in
                         ((e, (e.x >= x0) & (e.x < x0 + w) & (e.y >= y0) & (e.y < y0 + h))
                          for e in (gt, generated)))
    return dict(gt=gt, generated=generated, start_s=start, stop_s=stop, sequence=sequence,
                generator=generator, width=width if region is None else region[2],
                height=height if region is None else region[3])


def row_sums(rows, cols, scales, cutoff_sigma=6.0, block_size=2048, chunk_size=8192):
    """For every row point, sum_j k(row, col_j) over column points (time-sorted (t, x, y))."""
    out = np.zeros(len(rows))
    if not len(rows) or not len(cols):
        return out
    plan = PairPlan(rows, cols, scales, cutoff_sigma, tile_px=32)
    for tile_id in plan.tiles():
        owned, near = plan.members(tile_id)
        left, right = plan.rows[owned], plan.cols[near]
        for r0, r1, c0, c1 in plan.blocks(owned, near, block_size, chunk_size, 1.0):
            exponent = np.subtract.outer(left[r0:r1, 0], right[c0:c1, 0]) ** 2
            for axis in (1, 2):
                exponent += np.subtract.outer(left[r0:r1, axis], right[c0:c1, axis]) ** 2
            out[owned[r0:r1]] += np.exp(-0.5 * exponent).sum(axis=1)
    return out


def event_terms(window, sigma_spatial_px, sigma_temporal_ms, cutoff_sigma=6.0):
    """Per event of a window: its share of each sum, and the window's a, b, c.

    gt_gt[i] = sum_j k(g_i, g_j), gen_gen[j] = sum_j' k(p_j, p_j'), gt_gen / gen_gt = the
    other stream around each event (same polarity only). difference = GT density minus
    prediction density (each normalized) at the event: > 0 where prediction is missing
    events, < 0 where it has extra ones; a + b - 2c equals sum_i difference(g_i) / N -
    sum_j difference(p_j) / M.
    """
    gt, gen = window["gt"], window["generated"]
    scales = np.array([sigma_temporal_ms / 1000, sigma_spatial_px, sigma_spatial_px])
    out = {key: np.zeros(len(e.t)) for key, e in
           (("gt_gt", gt), ("gt_gen", gt), ("gen_gen", gen), ("gen_gt", gen))}
    for polarity in (1, -1):
        g, p = gt.p == polarity, gen.p == polarity
        gp = np.column_stack((gt.t[g], gt.x[g], gt.y[g])).astype(np.float64)
        pp = np.column_stack((gen.t[p], gen.x[p], gen.y[p])).astype(np.float64)
        out["gt_gt"][g] = row_sums(gp, gp, scales, cutoff_sigma)
        out["gt_gen"][g] = row_sums(gp, pp, scales, cutoff_sigma)
        out["gen_gen"][p] = row_sums(pp, pp, scales, cutoff_sigma)
        out["gen_gt"][p] = row_sums(pp, gp, scales, cutoff_sigma)
    n, m = max(len(gt.t), 1), max(len(gen.t), 1)
    a, b = out["gt_gt"].sum() / n ** 2, out["gen_gen"].sum() / m ** 2
    c = out["gt_gen"].sum() / (n * m)
    out["gt_difference"] = out["gt_gt"] / n - out["gt_gen"] / m
    out["gen_difference"] = out["gen_gt"] / n - out["gen_gen"] / m
    out.update(a=a, b=b, c=c, distance=math.sqrt(max(0.0, a + b - 2 * c)),
               similarity=c / math.sqrt(a * b) if a > 0 and b > 0 else float("nan"))
    return out


def _compact(values):
    """Plain list with 4 significant digits (dropdown buttons carry these as JSON)."""
    return [float(f"{v:.4g}") for v in np.asarray(values, dtype=np.float64)]


def point_cloud(window, sigmas_px=(1, 2, 5), sigma_temporal_ms=1.0, max_points=15000,
                cutoff_sigma=6.0, seed=0):
    """Plot 3: the window in (x, t, y); dropdown colours events by one term at one σ."""
    import plotly.graph_objects as go

    gt, gen = window["gt"], window["generated"]
    rng = np.random.default_rng(seed)

    def shown(n):
        return np.arange(n) if n <= max_points else np.sort(rng.choice(n, max_points, False))

    show_gt, show_gen = shown(len(gt.t)), shown(len(gen.t))
    terms = {s: event_terms(window, s, sigma_temporal_ms, cutoff_sigma) for s in sigmas_px}
    grey = "rgba(150,150,150,0.35)"
    views = (("GT – GT", "gt_gt", None, SEQUENTIAL),
             ("prediction – prediction", None, "gen_gen", SEQUENTIAL),
             ("GT – prediction", "gt_gen", "gen_gt", SEQUENTIAL),
             ("difference (red: missing, blue: extra)", "gt_difference", "gen_difference",
              DIVERGING))
    buttons = []
    for s in sigmas_px:
        t = terms[s]
        numbers = (f"window: a {t['a']:.3g} · b {t['b']:.3g} · c {t['c']:.3g} · "
                   f"distance {t['distance']:.3g} · similarity {t['similarity']:.3f}")
        for label, gt_key, gen_key, scale in views:
            colors, lo, hi = [], [], []
            values = [t[k][idx] for k, idx in ((gt_key, show_gt), (gen_key, show_gen)) if k]
            top = max((np.abs(v).max() for v in values if len(v)), default=1.0) or 1.0
            for key, idx in ((gt_key, show_gt), (gen_key, show_gen)):
                colors.append(_compact(t[key][idx]) if key else grey)
                lo.append(-float(top) if scale is DIVERGING else 0.0)
                hi.append(float(top))
            buttons.append(dict(label=f"σ {s:g} px · {label}", method="update", args=[
                {"marker.color": colors, "marker.colorscale": [scale, scale],
                 "marker.cmin": lo, "marker.cmax": hi,
                 "marker.showscale": [bool(gt_key), not gt_key]},
                {"title.text": f"{window['sequence']} / {window['generator']}, "
                               f"{window['start_s']:.4f}–{window['stop_s']:.4f} s · "
                               f"σ {s:g} px / {sigma_temporal_ms:g} ms · {label}<br>"
                               f"<sup>{numbers}</sup>"}]))
    first = buttons[0]["args"]
    fig = go.Figure()
    for index, (events, idx, name, symbol) in enumerate(
            ((gt, show_gt, f"GT ({len(gt.t):,})", "circle"),
             (gen, show_gen, f"{window['generator']} ({len(gen.t):,})", "diamond"))):
        fig.add_trace(go.Scatter3d(
            x=events.x[idx], y=(events.t[idx] - window["start_s"]) * 1000, z=events.y[idx],
            mode="markers", name=name,
            marker=dict(size=2, symbol=symbol, color=first[0]["marker.color"][index],
                        colorscale=first[0]["marker.colorscale"][index],
                        cmin=first[0]["marker.cmin"][index],
                        cmax=first[0]["marker.cmax"][index],
                        showscale=first[0]["marker.showscale"][index],
                        colorbar=dict(title=dict(text="value"), x=1.02 + 0.08 * index)),
            customdata=events.p[idx],
            hovertemplate=(f"{name}<br>x %{{x}} · y %{{z}} · t %{{y:.3f}} ms · "
                           f"polarity %{{customdata}}<br>value %{{marker.color:.3g}}"
                           "<extra></extra>")))
    fig.update_layout(
        title=dict(text=first[1]["title.text"]), height=720,
        scene=dict(xaxis=dict(title=dict(text="x (px)")), yaxis=dict(title=dict(text="t (ms)")),
                   zaxis=dict(title=dict(text="y (px)"), autorange="reversed"),
                   aspectmode="manual", aspectratio=dict(x=1.4, y=1.4, z=1.0)),
        updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left", y=1.08,
                          yanchor="top")],
        legend=dict(x=0, y=0), margin=dict(t=120))
    return fig, terms
