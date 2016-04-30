"""Functions to plot directed connectivity
"""
from __future__ import print_function

# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: Simplified BSD


from itertools import cycle
from functools import partial

import numpy as np

from mne.viz.utils import plt_show
from mne.externals.six import string_types
from mne.fixes import tril_indices, normalize_colors
from mne.viz.circle import _plot_connectivity_circle_onpick


# copied from mne.viz.circle to add optional `plot_names` argument
def plot_connectivity_circle(con, node_names, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', node_edgecolor='black',
                             linewidth=1.5, colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, padding=6.,
                             fig=None, subplot=111, interactive=True,
                             node_linewidth=2., plot_names=True, show=True):
    """Visualize connectivity as a circular graph.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.labri.fr/perso/nrougier/coding/.

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strength=abs(con))
        are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, the minimum angle between any
        two nodes is used as the width.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.
    node_linewidth : float
        Line with for nodes.
    plot_names : bool
        Draw node names if True.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        # widths correspond to the minimum angle between two nodes
        dist_mat = node_angles[None, :] - node_angles[:, None]
        dist_mat[np.diag_indices(n_nodes)] = 1e9
        node_width = np.min(np.abs(dist_mat))
    else:
        node_width = node_width * np.pi / 180

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [plt.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indices = tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # get the colormap
    if isinstance(colormap, string_types):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True, axisbg=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additonal space if requested
    plt.ylim(0, 10 + padding)

    # Remove the black axes border which may obscure the labels
    axes.spines['polar'].set_visible(False)

    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        con_thresh = 0.

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initalize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                           float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                         float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth, alpha=1.)
        axes.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.0
    bars = axes.bar(node_angles, height, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=node_linewidth,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # Draw node labels
    if plot_names:
        angles_deg = 180 * node_angles / np.pi
        for name, angle_rad, angle_deg in zip(node_names, node_angles,
                                              angles_deg):
            if angle_deg >= 270:
                ha = 'left'
            else:
                # Flip the label, so text is always upright
                angle_deg += 180
                ha = 'right'

            axes.text(angle_rad, 10.4, name, size=fontsize_names,
                      rotation=angle_deg, rotation_mode='anchor',
                      horizontalalignment=ha, verticalalignment='center',
                      color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    if colorbar:
        norm = normalize_colors(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    # Add callback for interaction
    if interactive:
        callback = partial(_plot_connectivity_circle_onpick, fig=fig,
                           axes=axes, indices=indices, n_nodes=n_nodes,
                           node_angles=node_angles)

        fig.canvas.mpl_connect('button_press_event', callback)

    plt_show(show)
    return fig, axes


def _plot_connectivity_matrix_nodename(x, y, con, node_names):
    x = int(round(x) - 2)
    y = int(round(y) - 2)
    if x < 0 or y < 0 or x >= len(node_names) or y >= len(node_names):
        return ''
    return '%s --> %s: %.2f' % (node_names[x], node_names[y],
                                con[y + 2, x + 2])


def plot_connectivity_matrix(con, node_names, indices=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, fig=None, subplot=111,
                             show_names=True):
    """Visualize connectivity as a matrix.

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    show_names : bool
        Enable or disable display of node names in the plot. The names are
        always displayed in the status bar when hovering over them.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt

    n_nodes = len(node_names)

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [plt.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices must be provided if con.ndim == 1')
        tmp = np.zeros((n_nodes, n_nodes)) * np.nan
        for ci in zip(con, *indices):
            tmp[ci[1:]] = ci[0]
        con = tmp
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # remove diagonal (do not show node's self-connectivity)
    con = con.copy()
    np.fill_diagonal(con, np.nan)

    # get the colormap
    if isinstance(colormap, string_types):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, axisbg=facecolor)

    axes.spines['bottom'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)

    tmp = np.empty((n_nodes + 4, n_nodes + 4)) * np.nan
    tmp[2:-2, 2:-2] = con
    con = tmp

    h = axes.imshow(con, cmap=colormap, interpolation='nearest', vmin=vmin,
                    vmax=vmax)

    nodes = np.empty((n_nodes + 4, n_nodes + 4, 4)) * np.nan
    for i in range(n_nodes):
        nodes[i + 2, 0, :] = node_colors[i]
        nodes[i + 2, -1, :] = node_colors[i]
        nodes[0, i + 2, :] = node_colors[i]
        nodes[-1, i + 2, :] = node_colors[i]
    axes.imshow(nodes, interpolation='nearest')

    if colorbar:
        cb = plt.colorbar(h, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    # Draw node labels
    if show_names:
        for i, name in enumerate(node_names):
            axes.text(-1, i + 2, name, size=fontsize_names,
                      rotation=0, rotation_mode='anchor',
                      horizontalalignment='right', verticalalignment='center',
                      color=textcolor)
            axes.text(i + 2, len(node_names) + 4, name, size=fontsize_names,
                      rotation=90, rotation_mode='anchor',
                      horizontalalignment='right', verticalalignment='center',
                      color=textcolor)

    axes.format_coord = partial(_plot_connectivity_matrix_nodename, con=con,
                                node_names=node_names)

    return fig, axes


def plot_connectivity_inoutcircles(con, seed, node_names, facecolor='black',
                                   textcolor='white', colormap='hot',
                                   title=None, fontsize_suptitle=14, fig=None,
                                   subplot=(121, 122), **kwargs):
    """Visualize effective connectivity with two circular graphs, one for
    incoming, and one for outgoing connections.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.loria.fr/~rougier/coding/recipes.html

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    seed : int | str
        Index or name of the seed node. Connections towards and from that node
        are displayed. The seed can be changed by clicking on a node in
        interactive mode.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    colormap : str | (str, str)
        Colormap to use for coloring the connections. Can be a tuple of two
        strings, in which case the first colormap is used for incoming, and the
        second colormap for outgoing connections.
    title : str
        The figure title.
    fontsize_suptitle : int
        Font size to use for title.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : (int, int) | (3-tuple, 3-tuple)
        Location of the two subplots for incoming and outgoing connections.
        E.g. 121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    **kwargs :
        The remaining keyword-arguments will be passed directly to
        plot_connectivity_circle.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes_in : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    axes_out : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt

    n_nodes = len(node_names)

    if any(isinstance(seed, t) for t in string_types):
        try:
            seed = node_names.index(seed)
        except ValueError:
            from difflib import get_close_matches
            close = get_close_matches(seed, node_names)
            raise ValueError('{} is not in the list of node names. Did you '
                             'mean {}?'.format(seed, close))

    if seed < 0 or seed >= n_nodes:
        raise ValueError('seed={} is not in range [0, {}].'
                         .format(seed, n_nodes - 1))

    if type(colormap) not in (tuple, list):
        colormap = (colormap, colormap)

    # Default figure size accomodates two horizontally arranged circles
    if fig is None:
        fig = plt.figure(figsize=(8, 4), facecolor=facecolor)

    index_in = (np.array([seed] * n_nodes),
                np.array([i for i in range(n_nodes)]))
    index_out = index_in[::-1]

    fig, axes_in = plot_connectivity_circle(con[seed, :].ravel(), node_names,
                                            indices=index_in,
                                            colormap=colormap[0], fig=fig,
                                            subplot=subplot[0],
                                            title='incoming', **kwargs)

    fig, axes_out = plot_connectivity_circle(con[:, seed].ravel(), node_names,
                                             indices=index_out,
                                             colormap=colormap[1], fig=fig,
                                             subplot=subplot[1],
                                             title='outgoing', **kwargs)

    if title is not None:
        plt.suptitle(title, color=textcolor, fontsize=fontsize_suptitle)

    return fig, axes_in, axes_out
