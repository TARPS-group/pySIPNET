"""Interactive run dashboard backed by Plotly.

Usage::

    from pysipnet.viz import dashboard

    fig = dashboard(result)
    fig.show()          # opens in browser
    fig.show("notebook")  # renders inline in Jupyter

Requires the optional ``viz`` dependency group::

    pip install pysipnet[viz]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from pysipnet.result import SIPNETResult

# Climate variables shown in the top 2×2 input panel.
_CLIM_PANELS = [
    ("tair", "Air Temperature (°C)", 1, 1),
    ("par", "PAR (mol m⁻²)", 1, 2),
    ("precip", "Precipitation (mm)", 2, 1),
    ("vpd", "VPD (Pa)", 2, 2),
]

# Flux outputs (row 3, full-width).  ET is a water flux but shown alongside
# C fluxes; unit differences are noted in the subplot title.
_FLUX_COLS: dict[str, str] = {
    "nee": "NEE",
    "gpp": "GPP",
    "evapotranspiration": "ET",
    "ra": "Rₐ",
    "rh": "Rₕ",
}

# Pool outputs (row 4, full-width).
_POOL_COLS: dict[str, str] = {
    "plant_wood_c": "Plant Wood C",
    "plant_leaf_c": "Plant Leaf C",
    "soil_c": "Soil C",
    "litter_c": "Litter C",
    "soil_water": "Soil Water",
}


def dashboard(
    result: SIPNETResult,
    *,
    show_cum_nee: bool = False,
) -> go.Figure:
    """Build an interactive Plotly dashboard for a single SIPNET run.

    The figure has four sections arranged vertically:

    * **Climate inputs** (2 × 2 grid): air temperature, PAR, precipitation, VPD
    * **Fluxes**: NEE, GPP, ET, Rₐ, Rₕ — optionally cumulative NEE
    * **Pools**: plant wood C, plant leaf C, soil C, litter C, soil water

    Columns absent from the timeseries (e.g. ``litter_c`` without
    ``LITTER_POOL=1``) are silently skipped.

    Parameters
    ----------
    result:
        A completed :class:`~pysipnet.result.SIPNETResult`.
    show_cum_nee:
        If ``True``, add a cumulative NEE trace to the flux panel.
        Default is ``False``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure.  Call ``.show()`` to open in a browser or
        ``.show("notebook")`` for inline Jupyter display.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for pysipnet.viz.dashboard(). "
            "Install with: pip install pysipnet[viz]"
        ) from exc

    if result.timeseries.empty:
        raise ValueError(
            "result.timeseries is empty — the SIPNET run may have failed "
            f"(returncode={result.returncode})."
        )

    ts = result.timeseries
    clim = result.climate.data

    # Fractional-year x-axis shared across all panels.
    x_ts = ts["year"] + (ts["day"] - 1) / 365
    x_clim = clim["year"] + (clim["day"] - 1) / 365

    flux_cols = dict(_FLUX_COLS)
    if show_cum_nee and "cum_nee" in ts.columns:
        flux_cols["cum_nee"] = "Cumulative NEE"

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
        subplot_titles=[
            "Air Temperature (°C)",
            "PAR (mol m⁻²)",
            "Precipitation (mm)",
            "VPD (Pa)",
            "Fluxes (g C m⁻² per timestep · ET in cm)",
            "Carbon & Water Pools (g C m⁻² · soil water in cm)",
        ],
        vertical_spacing=0.07,
        row_heights=[0.18, 0.18, 0.32, 0.32],
    )

    # ── Climate inputs ────────────────────────────────────────────────────────
    for col, label, row, col_idx in _CLIM_PANELS:
        if col not in clim.columns:
            continue
        fig.add_trace(
            go.Scatter(x=x_clim, y=clim[col], mode="lines", name=label, showlegend=False),
            row=row,
            col=col_idx,
        )

    # ── Fluxes ────────────────────────────────────────────────────────────────
    for col, label in flux_cols.items():
        if col not in ts.columns:
            continue
        fig.add_trace(
            go.Scatter(x=x_ts, y=ts[col], mode="lines", name=label),
            row=3,
            col=1,
        )

    # ── Pools ─────────────────────────────────────────────────────────────────
    for col, label in _POOL_COLS.items():
        if col not in ts.columns:
            continue
        fig.add_trace(
            go.Scatter(x=x_ts, y=ts[col], mode="lines", name=label),
            row=4,
            col=1,
        )

    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_xaxes(title_text="Year", row=4, col=1)

    fig.update_layout(
        title_text="SIPNET Run Dashboard",
        height=950,
        template="plotly_white",
        legend={"tracegroupgap": 20},
    )

    return fig
