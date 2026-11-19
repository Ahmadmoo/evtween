"""Compare normalized distributions on a common support."""
import math

import numpy as np
from scipy.special import rel_entr


def count_error(n_gt, n_gen, epsilon):
    absolute = abs(n_gt - n_gen)
    return {
        "n_gt": n_gt,
        "n_gen": n_gen,
        "absolute_count_difference": absolute,
        "relative_count_error": absolute / max(n_gt, epsilon),
    }


def _aligned_probabilities(gt, generated):
    support = np.union1d(gt.ids, generated.ids)
    p = np.zeros(support.size, dtype=np.float64)
    q = np.zeros_like(p)
    p[np.searchsorted(support, gt.ids)] = gt.probabilities()
    q[np.searchsorted(support, generated.ids)] = generated.probabilities()
    return p, q


def divergences(gt, generated, config):
    """Return selected scores and explicit statuses for inapplicable scores."""
    if (gt.grid, gt.name) != (generated.grid, generated.name):
        raise ValueError("Histograms must share their grid and distribution")
    result = {
        "grid_bins": math.prod(gt.shape),
        "gt_nonzero_bins": int(gt.ids.size),
        "gen_nonzero_bins": int(generated.ids.size),
        "histogram_bytes": gt.nbytes + generated.nbytes,
    }
    nonempty = gt.n_events > 0 and generated.n_events > 0
    if nonempty and {"jsd", "kl"}.intersection(config.comparisons):
        p, q = _aligned_probabilities(gt, generated)
        log_scale = math.log(2) if config.log_base == "2" else 1.0
        if "jsd" in config.comparisons:
            middle = (p + q) * 0.5
            jsd = float(0.5 * (rel_entr(p, middle) + rel_entr(q, middle)).sum())
            jsd = min(math.log(2), max(0.0, jsd))
            result["jsd"] = jsd / (math.log(2) if config.normalize_jsd else log_scale)
        if "kl" in config.comparisons:
            epsilon = config.kl_epsilon
            if epsilon:
                mass = epsilon / result["grid_bins"]
                p = (1 - epsilon) * p + mass
                q = (1 - epsilon) * q + mass
            result["kl_gt_gen"] = max(0.0, float(rel_entr(p, q).sum()) / log_scale)
            if config.reverse_kl:
                result["kl_gen_gt"] = max(0.0, float(rel_entr(q, p).sum()) / log_scale)
    if "wasserstein" in config.comparisons:
        value, status = wasserstein(gt, generated, config.max_spatial_transport_pairs)
        result.update(wasserstein=value, wasserstein_status=status)
    return result


def wasserstein(gt, generated, max_spatial_pairs):
    if gt.name not in ("temporal", "spatial"):
        return None, "not_applicable"
    if not gt.n_events or not generated.n_events:
        return None, "empty_stream"
    grid = gt.grid
    if gt.name == "temporal":
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(
            grid.temporal_positions(gt.ids), grid.temporal_positions(generated.ids),
            gt.probabilities(), generated.probabilities(),
        )), "ok"
    if gt.ids.size * generated.ids.size > max_spatial_pairs:
        return None, "spatial_transport_limit"
    from scipy.stats import wasserstein_distance_nd

    value = wasserstein_distance_nd(
        grid.spatial_positions(gt.ids), grid.spatial_positions(generated.ids),
        gt.probabilities(), generated.probabilities(),
    )
    return float(value), "ok"
