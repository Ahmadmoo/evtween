"""Diagnostic plots from shared histograms."""
import numpy as np


def plot_comparison(path, gt, generated, title):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize

    fig = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    fig.suptitle(title, fontsize=15)
    labels = ("Ground truth", "Generated")
    colors = ("#246c91", "#dc6544")
    grid = gt["spatial"].grid

    norm = Normalize(0, max(h["spatial"].probabilities().max(initial=0)
                            for h in (gt, generated)) or 1.0)
    edges_x = np.minimum(np.arange(grid.nx + 1) * grid.spatial_bin_size, grid.width)
    edges_y = np.minimum(np.arange(grid.ny + 1) * grid.spatial_bin_size, grid.height)
    for ax, h, label in zip(axes[0, :2], (gt, generated), labels):
        histogram = h["spatial"]
        values = np.zeros(histogram.shape)
        values.flat[histogram.ids] = histogram.probabilities()
        image = ax.pcolormesh(edges_x, edges_y, values, cmap="magma", norm=norm)
        ax.set(title=label, xlabel="x (px)", ylabel="y (px)", ylim=(grid.height, 0), aspect="equal")
    fig.colorbar(image, ax=list(axes[0, :2]), label="Probability per spatial bin", shrink=0.8)

    temporal_ax = axes[0, 2]
    styles = (("solid", 2.5), ("dashed", 1.3))
    for h, label, color, (style, width) in zip((gt, generated), labels, colors, styles):
        histogram = h["temporal"]
        axis_x = grid.temporal_positions(histogram.ids)
        temporal_ax.vlines(axis_x, 0, histogram.probabilities(), color=color,
                           linestyles=style, linewidth=width, label=label)
    temporal_ax.set(title="Temporal activity", xlabel="Time (s)", ylabel="Probability per time bin",
                    xlim=(grid.start_s, grid.stop_s))
    temporal_ax.legend(fontsize=8)

    projections = []
    for h in (gt, generated):
        histogram = h["spatiotemporal"]
        time = histogram.ids // (grid.nx * grid.ny)
        x = histogram.ids % grid.nx
        ids, inverse = np.unique(time * grid.nx + x, return_inverse=True)
        probabilities = np.bincount(inverse, weights=histogram.probabilities())
        projections.append((ids, probabilities))
    norm = Normalize(0, max(prob.max(initial=0) for _, prob in projections) or 1.0)
    for ax, (ids, prob), label in zip(axes[1, :2], projections, labels):
        x = ids % grid.nx
        px = (edges_x[x] + edges_x[x + 1]) / 2
        dots = ax.scatter(grid.temporal_positions(ids // grid.nx), px, c=prob, s=8,
                          linewidths=0, cmap="viridis", norm=norm)
        ax.set(title=f"{label} · x–time projection", xlabel="Time (s)", ylabel="x (px)",
               xlim=(grid.start_s, grid.stop_s), ylim=(0, grid.width))
    fig.colorbar(dots, ax=list(axes[1, :2]), label="Probability summed across y", shrink=0.8)

    polarity_ax = axes[1, 2]
    for i, (h, label, color) in enumerate(zip((gt, generated), labels, colors)):
        histogram = h["polarity"]
        prob = np.zeros(2)
        prob[histogram.ids] = histogram.probabilities()
        polarity_ax.bar(np.arange(2) + 0.32 * (i - 0.5), prob, width=0.32, label=label, color=color)
    polarity_ax.set(title="Polarity", ylabel="Probability", ylim=(0, 1),
                    xticks=[0, 1], xticklabels=["OFF (−1)", "ON (+1)"])
    polarity_ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
