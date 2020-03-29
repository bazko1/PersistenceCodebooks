import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_persistence_bow(pd, bow):
    pred_clusters = bow.predict([pd])[0]
    bins = list(range(bow.n_clusters))
    histogram = np.bincount(pred_clusters, minlength=bow.n_clusters)
    colors = matplotlib.cm.cividis(np.linspace(0, 0.9, bow.n_clusters))

    plt.bar(bins, histogram, color=colors)

def plot_voronoi(bow):
    return voronoi_plot_2d(Voronoi(bow.cluster_centers_))
