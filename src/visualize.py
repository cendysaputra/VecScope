"""Plotly figures: colored vector strips, similarity heatmap, 2D scatter (PCA / UMAP)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Red = positive, blue = negative, white ≈ 0 (Jay Alammar's "Illustrated Word2vec" look).
# Plotly's "RdBu" runs red→blue (low→high), so reverse it.
STRIP_COLORSCALE = "RdBu_r"


def vector_strips(labels: list[str], vectors: np.ndarray, title: str = "", height_per_row: int = 46) -> go.Figure:
    """One colored row per vector; each cell is one dimension."""
    vectors = np.asarray(vectors, dtype=np.float32)
    limit = float(np.abs(vectors).max()) if vectors.size else 1.0
    dims = list(range(vectors.shape[1])) if vectors.ndim == 2 else []
    fig = go.Figure(
        go.Heatmap(
            z=vectors,
            x=dims,
            y=labels,
            colorscale=STRIP_COLORSCALE,
            zmid=0,
            zmin=-limit,
            zmax=limit,
            xgap=1,
            ygap=4,
            colorbar=dict(title="value", thickness=12),
            hovertemplate="%{y}<br>dim %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or None,
        height=max(180, 70 + height_per_row * len(labels)),
        margin=dict(l=10, r=10, t=40 if title else 10, b=30),
        xaxis=dict(title="dimension", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        plot_bgcolor="white",
    )
    return fig


def similarity_heatmap(row_labels: list[str], sim: np.ndarray, col_labels: list[str] | None = None, title: str = "") -> go.Figure:
    col_labels = col_labels or row_labels
    fig = go.Figure(
        go.Heatmap(
            z=sim,
            x=col_labels,
            y=row_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(sim, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(title="cosine", thickness=12),
        )
    )
    fig.update_layout(
        title=title or None,
        height=max(320, 60 + 42 * len(row_labels)),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def project_2d(vectors: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    """Reduce to 2D. PCA is the default (deterministic); UMAP needs ``umap-learn``."""
    vectors = np.asarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("UMAP needs `pip install umap-learn`.") from exc
        reducer = umap.UMAP(n_components=2, n_neighbors=max(2, min(15, n - 1)), metric="cosine", random_state=random_state)
        return reducer.fit_transform(vectors)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=random_state).fit_transform(vectors)


def scatter_2d(
    labels: list[str],
    points: np.ndarray,
    groups: list[str] | None = None,
    symbols: dict[str, str] | None = None,
    title: str = "",
    axis_prefix: str = "PC",
) -> go.Figure:
    """Labelled 2D scatter. ``groups`` colors (and optionally shapes) the points."""
    groups = groups or ["words"] * len(labels)
    fig = go.Figure()
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        fig.add_trace(
            go.Scatter(
                x=points[idx, 0],
                y=points[idx, 1],
                mode="markers+text",
                name=g,
                text=[labels[i] for i in idx],
                textposition="top center",
                marker=dict(size=12, symbol=(symbols or {}).get(g, "circle"), line=dict(width=1, color="white")),
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f})<extra>" + g + "</extra>",
            )
        )
    fig.update_layout(
        title=title or None,
        height=520,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis_title=f"{axis_prefix} 1",
        yaxis_title=f"{axis_prefix} 2",
        legend=dict(orientation="h", y=-0.12),
    )
    return fig
