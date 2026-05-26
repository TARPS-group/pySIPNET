"""Interactive run dashboard backed by Plotly.

Usage::

    from pysipnet.viz import dashboard

    fig = dashboard(result)
    fig.show()           # opens in browser
    fig.show("notebook") # renders inline in Jupyter

Requires the optional ``viz`` dependency group::

    pip install pysipnet[viz]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from pysipnet.result import SIPNETResult

# ── Panel definitions ──────────────────────────────────────────────────────────

_CLIM_PANELS: list[tuple[str, str, int, int]] = [
    ("tair", "Air Temperature (°C)", 3, 1),
    ("par", "PAR (mol m⁻²)", 3, 2),
    ("precip", "Precipitation (mm)", 4, 1),
    ("vpd", "VPD (Pa)", 4, 2),
]

_FLUX_COLS: dict[str, str] = {
    "nee": "NEE",
    "gpp": "GPP",
    "evapotranspiration": "ET",
    "ra": "Rₐ",
    "rh": "Rₕ",
}

_POOL_COLS: dict[str, str] = {
    "plant_wood_c": "Wood C (stem)",  # aboveground wood; roots tracked separately
    "plant_leaf_c": "Leaf C",
    "coarse_root_c": "Coarse Root C",
    "fine_root_c": "Fine Root C",
    "soil_c": "Soil C",
    "litter_c": "Litter C",
    "soil_water": "Soil Water",
}

_TH_BG = "#e8eef4"  # table header background
_ROW_A = "#f9fafb"  # odd-group row fill
_ROW_B = "#ffffff"  # even-group row fill


# ── Layout geometry ────────────────────────────────────────────────────────────

# These must stay in sync with the make_subplots call in dashboard().
_NROWS = 6
_ROW_WEIGHTS = [0.07, 0.28, 0.10, 0.10, 0.225, 0.225]  # relative heights (sum=1)
_VERT_SPACING = 0.04  # fraction of figure height


def _row_bounds() -> tuple[list[float], list[float]]:
    """Return (tops, bottoms) in paper coords for each row, from top to bottom.

    Derived analytically from *_ROW_WEIGHTS* and *_VERT_SPACING* so that
    section-header annotations are always placed correctly regardless of
    figure height.
    """
    n_gaps = _NROWS - 1
    available = 1.0 - n_gaps * _VERT_SPACING
    total_w = sum(_ROW_WEIGHTS)

    tops, bottoms = [], []
    y = 1.0
    for w in _ROW_WEIGHTS:
        h = (w / total_w) * available
        tops.append(y)
        bottoms.append(y - h)
        y = y - h - _VERT_SPACING

    return tops, bottoms


# ── Table helpers ──────────────────────────────────────────────────────────────


def _provenance_table(result: SIPNETResult) -> go.Table:
    """Compact run-provenance table (4 rows, no scrollbar)."""
    import plotly.graph_objects as go

    prov = result.provenance
    try:
        preset = str(prov.preset)
        rid = str(prov.run_id)
        run_id = (rid[:10] + "…") if len(rid) > 11 else rid
        binary = (
            prov.binary_path.name if hasattr(prov.binary_path, "name") else str(prov.binary_path)
        )
        status = "✓ Success" if prov.returncode == 0 else f"✗ Failed (returncode {prov.returncode})"
    except Exception:
        preset = run_id = binary = status = "N/A"

    keys = ["Preset", "Run ID", "Binary", "Status"]
    vals = [preset, run_id, str(binary), status]
    colors = [_ROW_A if i % 2 == 0 else _ROW_B for i in range(len(keys))]

    return go.Table(
        header=dict(
            values=["<b>Property</b>", "<b>Value</b>"],
            fill_color=_TH_BG,
            font=dict(size=12, color="#333"),
            align="left",
            height=28,
        ),
        cells=dict(
            values=[keys, vals],
            fill_color=[colors, colors],
            font=dict(size=11, color="#333"),
            align="left",
            height=24,
        ),
        columnwidth=[120, 450],
    )


def _param_table(result: SIPNETResult) -> go.Table:
    """Parameter summary table grouped by domain; only non-None fields shown."""
    import plotly.graph_objects as go

    group_col: list[str] = []
    param_col: list[str] = []
    value_col: list[str] = []
    row_colors: list[str] = []

    try:
        params_dict = result.parameters.model_dump()
        fill = _ROW_A
        for group_name, group_dict in params_dict.items():
            if not isinstance(group_dict, dict):
                continue
            label = group_name.replace("_", " ").title()
            first = True
            for pname, pval in group_dict.items():
                if pval is None:
                    continue
                group_col.append(f"<b>{label}</b>" if first else "")
                param_col.append(pname.replace("_", " "))
                value_col.append(f"{pval:.4g}" if isinstance(pval, float) else str(pval))
                row_colors.append(fill)
                first = False
            if not first:
                fill = _ROW_B if fill == _ROW_A else _ROW_A
    except Exception:
        pass

    if not group_col:
        group_col = ["(no parameters)"]
        param_col = [""]
        value_col = [""]
        row_colors = [_ROW_B]

    return go.Table(
        header=dict(
            values=["<b>Group</b>", "<b>Parameter</b>", "<b>Value</b>"],
            fill_color=_TH_BG,
            font=dict(size=12, color="#333"),
            align="left",
            height=28,
        ),
        cells=dict(
            values=[group_col, param_col, value_col],
            fill_color=[row_colors, row_colors, row_colors],
            font=dict(size=11, color="#333"),
            align=["left", "left", "right"],
            height=22,
        ),
        columnwidth=[120, 200, 80],
    )


# ── Dashboard ──────────────────────────────────────────────────────────────────


def dashboard(
    result: SIPNETResult,
    *,
    show_cum_nee: bool = False,
) -> go.Figure:
    """Build an interactive Plotly dashboard for a single SIPNET run.

    The figure has six sections arranged vertically:

    * **Run Configuration**: provenance table (preset, run ID, binary, status)
      and grouped parameter table (non-``None`` fields only)
    * **Climate Inputs** (2 × 2 grid): air temperature, PAR, precipitation, VPD
    * **Model Outputs**: flux panel (NEE, GPP, ET, Rₐ, Rₕ) with a
      per-panel legend and variable-selector dropdown; pool panel (plant wood C,
      plant leaf C, soil C, litter C, soil water) with a separate per-panel legend

    Columns absent from ``result.outputs`` (e.g. ``litter_c`` when
    ``LITTER_POOL=0``) are silently skipped.

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
        just evaluate it as the last expression in a Jupyter cell.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for pysipnet.viz.dashboard(). "
            "Install with: pip install pysipnet[viz]"
        ) from exc

    if result.outputs.empty:
        raise ValueError(
            "result.outputs is empty — the SIPNET run may have failed "
            f"(returncode={result.provenance.returncode})."
        )

    ts = result.outputs
    clim = result.climate.data

    x_ts = ts["year"] + (ts["day"] - 1) / 365
    x_clim = clim["year"] + (clim["day"] - 1) / 365

    flux_cols = dict(_FLUX_COLS)
    if show_cum_nee and "cum_nee" in ts.columns:
        flux_cols["cum_nee"] = "Cumulative NEE"

    # ── Compute section-boundary paper coordinates ─────────────────────────────
    #
    # tops[i] / bottoms[i] are in Plotly paper coords (0–1, y=1 is the top).
    # The gap between rows i and i+1 runs from bottoms[i] down to tops[i+1].
    # Section dividers sit at the midpoint of each inter-section gap.

    tops, bottoms = _row_bounds()

    # Gap midpoints used for section-header annotations and separator lines.
    # Config section:  rows 0-1 (provenance + params)
    # Climate section: rows 2-3 (2×2 climate grid)
    # Outputs section: rows 4-5 (fluxes + pools)
    gap_config_climate = (bottoms[1] + tops[2]) / 2  # between rows 2 and 3
    gap_climate_output = (bottoms[3] + tops[4]) / 2  # between rows 4 and 5

    # ── Layout ────────────────────────────────────────────────────────────────

    fig = make_subplots(
        rows=_NROWS,
        cols=2,
        specs=[
            [{"type": "table", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
        row_heights=_ROW_WEIGHTS,
        vertical_spacing=_VERT_SPACING,
        subplot_titles=[
            # Table rows: blank (section headers added as annotations below)
            "",
            "",
            # Climate panels
            "Air Temperature (°C)",
            "PAR (mol m⁻²)",
            "Precipitation (mm)",
            "VPD (Pa)",
            # Output panels
            "Fluxes  (g C m⁻² per timestep · ET in cm)",
            "Carbon & Water Pools  (g C m⁻² · soil water in cm)",
        ],
    )

    # ── Provenance and parameter tables ───────────────────────────────────────

    fig.add_trace(_provenance_table(result), row=1, col=1)
    fig.add_trace(_param_table(result), row=2, col=1)

    # ── Climate inputs ────────────────────────────────────────────────────────

    for col, label, row, col_idx in _CLIM_PANELS:
        if col not in clim.columns:
            continue
        fig.add_trace(
            go.Scatter(x=x_clim, y=clim[col], mode="lines", name=label, showlegend=False),
            row=row,
            col=col_idx,
        )

    # ── Fluxes (legend2) ──────────────────────────────────────────────────────

    flux_trace_indices: dict[str, int] = {}
    for col, label in flux_cols.items():
        if col not in ts.columns:
            continue
        flux_trace_indices[label] = len(fig.data)
        fig.add_trace(
            go.Scatter(x=x_ts, y=ts[col], mode="lines", name=label, legend="legend2"),
            row=5,
            col=1,
        )

    # ── Pools (legend3) ───────────────────────────────────────────────────────

    pool_trace_indices: dict[str, int] = {}
    for col, label in _POOL_COLS.items():
        if col not in ts.columns:
            continue
        pool_trace_indices[label] = len(fig.data)
        fig.add_trace(
            go.Scatter(x=x_ts, y=ts[col], mode="lines", name=label, legend="legend3"),
            row=6,
            col=1,
        )

    fig.update_xaxes(title_text="Year", row=5, col=1)
    fig.update_xaxes(title_text="Year", row=6, col=1)

    # ── Variable-selector dropdowns (flux and pool panels) ───────────────────

    def _selector_buttons(trace_indices: dict[str, int], all_label: str) -> list[dict]:
        idxs = list(trace_indices.values())
        labels = list(trace_indices.keys())
        n = len(idxs)
        return [
            dict(
                label=all_label,
                method="restyle",
                args=[{"visible": [True] * n}, idxs],
            ),
        ] + [
            dict(
                label=lbl,
                method="restyle",
                args=[{"visible": [j == i for j in range(n)]}, idxs],
            )
            for i, lbl in enumerate(labels)
        ]

    flux_buttons = _selector_buttons(flux_trace_indices, "All fluxes")
    pool_buttons = _selector_buttons(pool_trace_indices, "All pools")

    # ── X-axis tick formatting ────────────────────────────────────────────────
    #
    # The x-axis encodes fractional years (e.g. 2020 + (doy-1)/365).  Force
    # ticks at whole-year boundaries so labels read "2020" not "2020.27".
    # For runs shorter than one year, fall back to auto-ticks with two decimal
    # places (fractional year is meaningful at that scale).

    year_min = int(ts["year"].min())
    year_max = int(ts["year"].max())
    if year_max > year_min:
        fig.update_xaxes(
            tickmode="linear",
            tick0=float(year_min),
            dtick=1,
            tickformat="d",
        )
    else:
        fig.update_xaxes(tickformat=".2f")

    # ── Section headers and separator lines ───────────────────────────────────
    #
    # Each separator is a pair of thin horizontal rules framing a bold label
    # in the gap between sections.  Positions are computed from _row_bounds()
    # so they track the subplot geometry exactly.

    _HEADER_FONT = dict(size=12, color="#555")
    _RULE_COLOR = "#d0d7de"

    def _section_header(text: str, y: float) -> None:
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            xref="paper",
            yref="paper",
            line=dict(color=_RULE_COLOR, width=1),
        )
        fig.add_annotation(
            text=f"<b>{text}</b>",
            x=0,
            y=y,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            font=_HEADER_FONT,
            showarrow=False,
            bgcolor="white",
            borderpad=2,
        )

    # "Run Configuration" — label sits on the top edge of the figure (y=1.0)
    _section_header("RUN CONFIGURATION", tops[0])

    # "Climate Inputs" — sits on the separator line between rows 2 and 3
    _section_header("CLIMATE INPUTS", gap_config_climate)

    # "Model Outputs" — sits on the separator line between rows 4 and 5
    _section_header("MODEL OUTPUTS", gap_climate_output)

    # ── Layout, legends, dropdown ─────────────────────────────────────────────

    # Flux legend: anchor to centre of row 5 (fluxes).
    # Pool legend: anchor to centre of row 6 (pools).
    flux_legend_y = (tops[4] + bottoms[4]) / 2
    pool_legend_y = (tops[5] + bottoms[5]) / 2

    fig.update_layout(
        title_text="SIPNET Run Dashboard",
        height=2300,
        template="plotly_white",
        margin=dict(t=60, r=160, b=50, l=60),
        legend2=dict(
            x=1.01,
            y=flux_legend_y,
            xanchor="left",
            yanchor="middle",
            title=dict(text="<b>Fluxes</b>", font=dict(size=11)),
            font=dict(size=11),
            bordercolor="#ccc",
            borderwidth=1,
        ),
        legend3=dict(
            x=1.01,
            y=pool_legend_y,
            xanchor="left",
            yanchor="middle",
            title=dict(text="<b>Pools</b>", font=dict(size=11)),
            font=dict(size=11),
            bordercolor="#ccc",
            borderwidth=1,
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.0,
                y=tops[4],
                xanchor="right",
                yanchor="bottom",
                buttons=flux_buttons,
                showactive=True,
                bgcolor="#f8f9fa",
                bordercolor="#ccc",
                font=dict(size=11),
            ),
            dict(
                type="dropdown",
                direction="down",
                x=1.0,
                y=tops[5],
                xanchor="right",
                yanchor="bottom",
                buttons=pool_buttons,
                showactive=True,
                bgcolor="#f8f9fa",
                bordercolor="#ccc",
                font=dict(size=11),
            ),
        ],
    )

    return fig
