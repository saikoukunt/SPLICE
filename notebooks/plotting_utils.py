import matplotlib.pyplot as plt


def axisMove(xd, yd, ax=None):
    """
        ax = axisMove(xd, yd; ax=nothing)

    Move an axis within a figure.

        Parameters
        ----------
        xd : float
            How much to move horizontally. Units are scaled figure units, from
            0 to 1 (with 1 meaning the full width of the figure)
        yd : float
            How much to move vertically. Units are scaled figure units, from
            0 to 1 (with 1 meaning the full height of the figure)
        ax : matplotlib.axes.Axes, optional
            If left as the default (None), works on the current axis;
            otherwise should be an axis object to be modified.
    """

    if ax == None:
        ax = plt.gca()

    x, y, w, h = ax.get_position().bounds

    x += xd
    y += yd

    ax.set_position([x, y, w, h])
    return ax


def axisWidthChange(factor, lock="c", ax=None):
    """
    Changes the width of the current axes by a scalar factor.

    Parameters
    ----------
    factor : float
        The scalar value by which to change the width, for example
        0.8 (to make them narrower) or 1.5 (to make them wider)
    lock : str, optional
        Which part of the axis to keep fixed. "c", the default does
        the changes around the middle; "l" means keep the left edge fixed
        "r" means keep the right edge fixed
    ax : matplotlib.axes.Axes, optional
        If left as the default (None), works on the current axis;
        otherwise should be an axis object to be modified.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axis object
    """
    if ax is None:
        ax = plt.gca()
    x, y, w, h = ax.get_position().bounds

    if lock == "l":
        pass
    elif lock == "c" or lock == "m":
        x = x + w * (1 - factor) / 2
    elif lock == "r":
        x = x + w * (1 - factor)
    else:
        raise ValueError("I don't know lock type ", lock)

    w = w * factor
    ax.set_position([x, y, w, h])

    return ax


def axisHeightChange(factor, lock="c", ax=None):
    """
    Changes the height of the current axes by a scalar factor.

    Parameters
    ----------
    factor : float
        The scalar value by which to change the height, for example
        0.8 (to make them shorter) or 1.5 (to make them taller)
    lock : str, optional
        Which part of the axis to keep fixed. "c", the default does
        the changes around the middle; "b" means keep the bottom edge fixed
        "t" means keep the top edge fixed
    ax : matplotlib.axes.Axes, optional
        If left as the default (None), works on the current axis;
        otherwise should be an axis object to be modified.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axis object

    """
    if ax == None:
        ax = plt.gca()

    x, y, w, h = ax.get_position().bounds

    if lock == "b":
        pass
    elif lock == "c" or lock == "m":
        y = y + h * (1 - factor) / 2
    elif lock == "t":
        y = y + h * (1 - factor)
    else:
        raise ValueError("I don't know lock type ", lock)

    h = h * factor
    ax.set_position([x, y, w, h])

    return ax
