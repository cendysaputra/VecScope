from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Per-mode palette
PALETTES = {
    "light": {
        "diverging": [
            [0.0, "#0d366b"], [0.2, "#2a78d6"], [0.4, "#9ec5f4"], [0.5, "#f0efec"],
            [0.6, "#f2aaa6"], [0.8, "#e34948"], [1.0, "#8a1c1c"],
        ],
        "sequential": [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]],
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ink": "#52514e",
        "surface": "#f9f9f7",
    },
    "dark": {
        "diverging": [
            [0.0, "#b7d3f6"], [0.2, "#5598e7"], [0.4, "#256abf"], [0.5, "#383835"],
            [0.6, "#b8393a"], [0.8, "#e66767"], [1.0, "#f6b8b5"],
        ],
        "sequential": [[0.0, "#0d366b"], [0.5, "#3987e5"], [1.0, "#cde2fb"]],
        "categorical": ["#3987e5", "#d95926", "#199e70"],
        "ink": "#c3c2b7",
        "surface": "#121211",
    },
}


def palette(mode: str) -> dict:
    return PALETTES.get(mode, PALETTES["light"])


def strip_limit(vectors: np.ndarray) -> float:
    # Robust color range
    return float(np.percentile(np.abs(vectors), 99)) or 1.0


def diverging_hex(values: np.ndarray, mode: str = "light") -> list[str]:
    limit = strip_limit(values)
    pos = np.clip((np.asarray(values) / limit + 1) / 2, 0, 1)
    return sample_colorscale(palette(mode)["diverging"], pos.tolist(), colortype="rgb")


def _layout(fig: go.Figure, title: str, height: int, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=44 if title else 8, b=8),
        **kw,
    )
    return fig


def vector_strips(labels: list[str], vectors: np.ndarray, title: str = "", height_per_row: int = 44, mode: str = "light") -> go.Figure:
    vectors = np.asarray(vectors, dtype=np.float32)
    limit = strip_limit(vectors)
    fig = go.Figure(
        go.Heatmap(
            z=vectors,
            x=np.arange(vectors.shape[1]),
            y=labels,
            colorscale=palette(mode)["diverging"],
            zmid=0,
            zmin=-limit,
            zmax=limit,
            xgap=1 if vectors.shape[1] <= 100 else 0,
            ygap=6,
            colorbar=dict(thickness=10, outlinewidth=0, len=0.9),
            hovertemplate="%{y}<br>dim %{x}: %{z:.3f}<extra></extra>",
        )
    )
    return _layout(
        fig, title, max(170, 80 + height_per_row * len(labels)),
        xaxis=dict(title="dimension", showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )


def similarity_heatmap(
    row_labels: list[str],
    sim: np.ndarray,
    col_labels: list[str] | None = None,
    title: str = "",
    mode: str = "light",
    sequential: bool = False,
) -> go.Figure:
    pal = palette(mode)
    scale = dict(colorscale=pal["sequential"], zmin=float(sim.min()), zmax=float(sim.max())) if sequential else dict(
        colorscale=pal["diverging"], zmid=0, zmin=-1, zmax=1
    )
    fig = go.Figure(
        go.Heatmap(
            z=sim,
            x=col_labels or row_labels,
            y=row_labels,
            texttemplate="%{z:.2f}",
            xgap=2,
            ygap=2,
            hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(title="cosine", thickness=10, outlinewidth=0),
            **scale,
        )
    )
    return _layout(
        fig, title, max(300, 70 + 44 * len(row_labels)),
        xaxis=dict(showgrid=False), yaxis=dict(autorange="reversed", showgrid=False),
    )


def project_2d(vectors: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
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
    mode: str = "light",
) -> go.Figure:
    pal = palette(mode)
    groups = groups or ["words"] * len(labels)
    symbols = symbols or {}
    fig = go.Figure()
    for k, g in enumerate(dict.fromkeys(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        # Fixed slot order
        color = pal["categorical"][k] if k < len(pal["categorical"]) else None
        fig.add_trace(
            go.Scatter(
                x=points[idx, 0],
                y=points[idx, 1],
                mode="markers+text",
                name=g,
                text=[labels[i] for i in idx],
                textposition="top center",
                textfont=dict(color=pal["ink"], size=12),
                marker=dict(size=11, color=color, symbol=symbols.get(g, "circle"), line=dict(width=2, color=pal["surface"])),
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f})<extra>" + g + "</extra>",
            )
        )
    return _layout(
        fig, title, 540,
        xaxis_title=f"{axis_prefix} 1",
        yaxis_title=f"{axis_prefix} 2",
        legend=dict(orientation="h", y=-0.14),
    )
