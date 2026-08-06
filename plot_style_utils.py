"""
plot_style_utils.py
=====================
Small shared helpers for consistent, readable axis formatting across the
project's plotting scripts.
"""

import matplotlib.ticker as mticker


def clean_decimal_axis(ax, axis="both", nbins=5, use_offset=True):
    """
    Reduce tick count and switch to scientific/offset notation on an axis
    with narrow-range, many-decimal values which otherwise tend to visually
    overlap.

    Parameters
    ----------
    ax    : matplotlib Axes
    axis  : "x", "y", or "both" (default) -- which axis/axes to clean up
    nbins : target number of ticks (passed to MaxNLocator; the actual
            count may differ slightly since MaxNLocator picks "nice"
            round numbers rather than exactly nbins)
    use_offset : if True, applies scientific notation with a shared offset
            (e.g. "+2.9e-2") rather than full decimals on every tick
    """
    axes_to_do = ("x", "y") if axis == "both" else (axis,)

    for a in axes_to_do:
        axis_obj = ax.xaxis if a == "x" else ax.yaxis
        axis_obj.set_major_locator(mticker.MaxNLocator(nbins=nbins))

        if use_offset:
            formatter = mticker.ScalarFormatter(useOffset=True, useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 3))  # switch to sci notation outside this range
            axis_obj.set_major_formatter(formatter)
