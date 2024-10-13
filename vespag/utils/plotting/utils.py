from typing import Union

import matplotlib as mpl


def label_bars(
    ax: mpl.axes.Axes, digits: int = 3, fontsize: Union[str, int] = "small"
) -> None:
    for c in ax.containers:
        ax.bar_label(
            c,
            fmt=f"%.{digits}f",
            label_type="center",
            fontsize=fontsize,
            color="white",
        )


def change_width(ax: mpl.axes.Axes, new_value: float) -> None:
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * 0.5)
