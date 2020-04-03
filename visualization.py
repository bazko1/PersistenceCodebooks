import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial._plotutils import _held_figure


def plot_bow(bow, height=None, ax=None):
    if not ax:
        ax = plt.axes()

    bins = list(range(len(bow)))
    colors = matplotlib.cm.cividis(np.linspace(0, 0.9, len(bow)))

    ax.bar(bins, bow, color=colors)
    if height:
        ax.set_ylim(0,height)


# Copied from scipy.spatial._plotutils and modified
def plot_voronoi(kmeans, plot_range=(1, 1), ax=None, **kw):
    if ax is None:
        ax = plt.axes()

    plot_range = np.array(plot_range)
    vor = Voronoi(kmeans.cluster_centers_)

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], 'x', markersize=point_size, color = (0,0,0))

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * \
                2  # to guarantee that coords > 1

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))

    ax.set_xlim(0, plot_range[0])
    ax.set_ylim(0, plot_range[1])

    return ax
