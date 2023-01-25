# ---------------------------------------------------------------------------#
# AUTHOR: Alberto M. Esmoris Pena                                            #
#                                                                            #
# Useful plots in the context of VirtuaLearn3D project                       #
# ---------------------------------------------------------------------------#


# ----------------- #
# ---  IMPORTS  --- #
# ----------------- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np


# ---  POINT CLOUD VIEW  --- #
# -------------------------- #
def plot_2dviews(
    X,
    figsize=(16, 9),
    nrows=1,
    ncols=3,
    c='black',
    s=16,
    cmap='jet',
    ppp=None
):
    """Plot the 2d views of a point cloud (x, y), (x, z), (y, z)
    :param X: The structure space of the point cloud (x, y, z)
    :param figsize: The size of the figure
    :param nrows: The number of subplot rows.
    :param ncols: The number of subplot columns.
    :param c: The color specification. Either one color for all points or a
        vector of colors, one component per point.
    :param s: The point size.
    :param cmap: If using a vector of colors, specify the color map to be used
    :param ppp: Points per plot. If None, all points will be plotted. If
        given, ppp at most will be plotted.
    """
    # Prepare plot context
    fig = plt.figure(figsize=figsize)
    # Do (x, y) view subplot
    ax = fig.add_subplot(nrows, ncols, 1)
    plot_2dviews_subplot(
        ax, X[:, 0], X[:, 1],
        c=c, s=s, cmap=cmap, xlabel='$x$', ylabel='$y$', ppp=ppp
    )
    # Do (x, z) view subplot
    ax = fig.add_subplot(nrows, ncols, 2)
    plot_2dviews_subplot(
        ax, X[:, 0], X[:, 2],
        c=c, s=s, cmap=cmap, xlabel='$x$', ylabel='$z$', ppp=ppp
    )
    # Do (y, z) view subplot
    ax = fig.add_subplot(nrows, ncols, 3)
    plot_2dviews_subplot(
        ax, X[:, 1], X[:, 2],
        c=c, s=s, cmap=cmap, xlabel='$y$', ylabel='$z$', ppp=ppp
    )
    # Configure the context
    fig.tight_layout()
    # Show
    plt.show()


def plot_2dviews_subplot(
    ax, x, y,
    c='black', s='16', cmap='jet', xlabel='?', ylabel='?', ppp=None
):
    # Find points per plot
    if ppp is not None:
        h = int(np.ceil(len(x)/ppp))
        x, y = x[::h], y[::h]
        if isinstance(c, list) or isinstance(c, np.ndarray):
            c = c[::h]
    # Plot the points
    ax.scatter(x, y, c=c, s=s, cmap=cmap)
    # Configure plot
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.axis('equal')
    ax.grid('both')
    ax.set_axisbelow(True)


# ---  FEATURES VISTUALIZATION  --- #
# --------------------------------- #
def plot_features_histogram(
    F,
    figsize=(16, 9),
    ncols=4,
    mucolor='tab:red',
    sigmacolor='tab:orange',
    barcolor='tab:blue',
    features=None
):
    """Plot the histogram of the matrix representing a given feature space.
    :param F: The feature space matrix such that the rows are the points or
        samples and the columns are the features. Thus, F[i, j] is the j-th
        feature of the i-th point/sample.
    :param ncols: The number of subplot columns.
    :param mucolor: The color for the line representing the mean value.
    :param sigmacolor: The color for the line representing the median value.
    :param barcolor: The color for the bars of the histogram.
    :param features: The name for each feature (can be None).
    """
    # Prepare plot context
    nfeatures = F.shape[1]
    nrows = int(np.ceil(nfeatures/ncols))
    fig = plt.figure(figsize=figsize)
    # Plot each histogram in its subplot context
    for i in range(nfeatures):
        Fi = F[:, i]
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.hist(Fi, color=barcolor)
        ax.axvline(np.mean(Fi), color=mucolor, lw=2, label='$\\mu$')
        ax.axvline(
            np.median(Fi), color=sigmacolor, lw=2, ls='--', label='$\\sigma$'
        )
        if features is not None:
            ax.set_title(features[i], fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid('both')
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')
    # Configure plot
    fig.tight_layout()
    # Show
    plt.show()


def plot_features_2Dscatter_color(
    F,
    L,
    figsize=(16, 9),
    ncols=4,
    features=None,
    s=16,
    title='2D scatter color',
    cmap='Set1',
    ppp=None
):
    """Plot the 2D scatter plots for all different combinations of pairs of
        features, representing the label with the color.
    :param F: The feature space matrix such that the rows are the points or
        samples and the columns are the features. Thus, F[i, j] is the j-th
        feature of the i-th point/sample.
    :param L: The label for each point (the class it belongs too).
    :param ncols: The number of subplot columns.
    :param features: The name for each feature (can be None).
    :param s: The point size.
    :param title: The title for the entire plot.
    :param cmap: Specify the color map to represent the labels.
    :param ppp: Points per plot. If None, all points will be plotted. If given,
        ppp at most will be plotted.
    """
    # Prepare plot context
    nfeatures = F.shape[1]
    nplots = int(np.math.factorial(nfeatures)/2/np.math.factorial(nfeatures-2))
    nrows = int(np.ceil(nplots/ncols))
    indices = np.arange(nfeatures)  # Index for each feature
    pairs = [  # All combinations of pairs of indices (no order, no repetition)
        (indices[i], indices[j])
        for i in range(nfeatures)
        for j in range(i+1, nfeatures)
    ]
    fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=20)
    h = int(np.ceil(len(F)/ppp)) if ppp is not None else 1
    # Plot each 2D scatter color subplot
    for i in range(nplots):
        pair = pairs[i]
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.scatter(
            F[::h, pair[0]],  # x-axis feature
            F[::h, pair[1]],  # y-axis feature
            c=L[::h],  # Color from label
            s=s,  # Point size
            cmap=cmap
        )
        if features is not None:
            ax.set_xlabel(features[pair[0]], fontsize=18)
            ax.set_ylabel(features[pair[1]], fontsize=18)
        ax.tick_params(axis='both', which='both', labelsize=16)
        ax.axis('equal')
        ax.grid('both')
        ax.set_axisbelow(True)
    # Configure the plot context
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    # Show
    plt.show()


def plot_feature_class_histogram(
    L,
    F,
    LNames=None,
    FName='Feature',
    clipRatio=0,
    numFeatureBins=10,
    figsize=(16, 14),
    ncols=3,
    mucolor='tab:red',
    mediancolor='tab:orange',
    barcolor='tab:blue'
):
    """Plot one histogram per class showing the distribution of the given
        feature
    :param L: The class for each point such that L[i] contains the label/class
        for i-th point
    :param F: The value of the feature for each point such that F[i] contains
        the value of the feature for the i-th point
    :param clipRatio: If > 0, then feature values are clipped to the interval
        [Qa, Qb] where Qa is the clipRatio quantile and Qb is the (1-clipRatio)
        quantile
    :param numFeatureBins: How many bins use to discretize the feature domain
    :param ncols: The number of subplot columns (if None it will be determined
        automatically)
    :param mucolor: The color for the line representing the mean value
    :param mediancolor: The color for the line representing the median value
    :param barcolor: The color for the bars of the histogram
    """
    # Prepare plot
    C = np.unique(L)
    nplots = len(C)
    ncols = int(np.sqrt(nplots)) if ncols is None else ncols
    nrows = int(np.ceil(nplots / ncols))
    fig = plt.figure(figsize=(16, 14))
    # Plot each histogram
    for i in range(nplots):  # For each i-th label/class ...
        # Extract feature per class
        Ci = C[i]  # The current class
        FCi = F[np.argwhere(L == Ci).flatten()]  # Feature per current class
        mu = np.mean(FCi)
        median = np.median(FCi)
        # Clip feature
        if clipRatio > 0:
            a, b = np.quantile(FCi, (clipRatio, 1-clipRatio))
            FCi = np.clip(FCi, a, b)
        # Prepare subplot context
        ax = fig.add_subplot(nrows, ncols, i+1)
        # Do subplot
        ax.hist(FCi, bins=numFeatureBins)
        ax.axvline(mu, color=mucolor, lw=2, label='$\\mu$')
        ax.axvline(median, color=mediancolor, lw=2, label='$M_e$')
        # Configure subplot
        LName = LNames[Ci] if LNames is not None else 'L{L}'.format(L=i)
        ax.set_title(
            '{FName} of {LName}'.format(FName=FName, LName=LName), fontsize=16
        )
        ax.tick_params(axis='both', which='both', labelsize=14)
        ax.grid('both')
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')
    # Configure plot
    fig.tight_layout()
    # Show plot
    plt.show()


def plot_features_per_class_hist2d(
    L,
    F,
    LNames=None,
    FNames=None,
    numFeatureBins=10,
    plot_unfiltered=True,
    plot_filtered=True,
    symlog=False,
    figsize=(16, 24)
):
    """Plot the 2D histogram representing the features per label.
    :param L: The vector of labels such that L[i] is the label for the i-th
        point
    :param F: The matrix of features such that F[i, j] is the i-th value
        of the j-th feature
    :param LNames: If not None, it must specify the text name for each label
    :param FNames: If not None, it must specify the name for each feature
    :param numFeatureBins: How many bins use to discretize the feature domain
    :param plot_unfiltered: True to plot the unfiltered 2D histogram
    :param plot_filtered: True to plot a quartile-filtered version of the
        2D histogram
    :param symlog: Whether to use symmetric logarithmic scale (True) or not
        (False) for the plot
    """
    # Prepare plot
    nFeatures = F.shape[1]
    nPlotsPerFeature = plot_unfiltered + plot_filtered
    nRows = nFeatures*nPlotsPerFeature
    fig = plt.figure(figsize=figsize)
    # Do each subplot
    subplotIdx = 1
    for j in range(nFeatures):  # For each feature
        # Extract feature name
        FName = '{j}-th feature'.format(j=j) if FNames is None else FNames[j]
        # Build unfiltered and filtered subplots, as requested
        axu, axf = None, None
        if plot_unfiltered:  # Do unfiltered subplot
            axu = fig.add_subplot(nRows, 1, subplotIdx)  # Subplot context
            _plot_features_per_class_hist2d(  # Do the subplot itself
                L,
                F[:, j],
                LNames=LNames,
                FName=FName,
                numFeatureBins=numFeatureBins,
                suffix='',
                ax=axu,
                fig=fig,
                symlog=symlog
            )
            subplotIdx += 1
        if plot_filtered:  # Do quartile-filtered subplot
            axf = fig.add_subplot(nRows, 1, subplotIdx)  # Subplot context
            Fj = F[:, j]  # Extract j-th feature
            Q1, Q3 = np.quantile(Fj, [0.25, 0.75])  # Get quartiles
            IQR = Q3 - Q1  # Compute interquartilic range
            FjMin, FjMax = Q1-3*IQR/2, Q1+3*IQR/2  # Define clip interval
            Fj = np.clip(Fj, FjMin, FjMax)  # Clip by quartile-filter
            _plot_features_per_class_hist2d(  # Do the subplot itself
                L,
                Fj,
                LNames=LNames,
                FName=FName,
                numFeatureBins=numFeatureBins,
                suffix=' with Quartile Filter',
                ax=axf,
                fig=fig,
                symlog=symlog
            )
            subplotIdx += 1
    # Configure plot
    fig.tight_layout()
    # Show
    plt.show()


def _plot_features_per_class_hist2d(
    L,
    Fj,
    LNames=None,
    FName=None,
    numFeatureBins=None,
    suffix='',
    fig=None,
    ax=None,
    symlog=False
):
    """Plot the 2D histogram representing the given feature per label.
    :param L: The vector of labels such that L[i] is the label for the it-h
        point
    :param Fj: The vector of j-th feature values such that Fj[i] is the value
        of the j-th feature for the i-th point
    :param LNames: If not None, it must specify the text name for each label
    :param FName: The name of the feature to be plotted
    :param numFeatureBins: How many bins use to discretize the feature domain
    :param suffix: Suffix for the title of the subplot
    """
    # Set the 2D histogram's title
    ax.set_title(
        '{FName} per class{suffix}'.format(FName=FName, suffix=suffix),
        fontsize=20
    )
    # Determine color normalization
    norm = None
    if symlog:
        norm = mpl.colors.SymLogNorm(
            linthresh=np.min(np.abs(Fj[Fj != 0])) / 10
        )
    # Build range for label bins
    labelBins = range(len(np.unique(L))+1)
    # The 2D histogram plot itself
    hist2d, xbins, ybins, im = ax.hist2d(
        L,  # The labels
        Fj,  # The feature
        bins=(labelBins, numFeatureBins),  # Number of bins per dimension
        cmap='jet',
        norm=norm
    )
    # Configure histogram ticks
    if LNames is not None:
        ax.set_xticks(
            range(len(LNames)), LNames, horizontalalignment='left'
        )
        ax.tick_params(
            axis='x', which='both', labelsize=16, labelrotation=45.0
        )
        ax.tick_params(axis='y', which='both', labelsize=16)
    else:
        ax.tick_params(axis='both', which='both', labelsize=16)
    # Add text values to the 2D histogram
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            # Prepare text value
            x = hist2d[j, i]
            unit = ''
            if x >= 1000 and x < 1000000:
                x = x // 1000
                unit = 'K'
            elif x >= 1000000:
                x = x // 1e6
                unit = 'M'
            x = '{x:.0f}{unit}'.format(x=x, unit=unit)
            # Plot text value
            txt = ax.text(
                xbins[j]+0.5*(xbins[j+1]-xbins[j]),
                ybins[i]+0.5*(ybins[i+1]-ybins[i]),
                x,
                color='white',
                ha='center',
                va='center',
                fontweight='bold',
                fontsize=20
            )
            txt.set_path_effects([
                mpe.withStroke(linewidth=2, foreground='black')
            ])
    # Add colorbar
    cb = fig.colorbar(im)
    cb.ax.tick_params(axis='both', which='both', labelsize=16)


# ---  PLOT MODEL EVALUATION  --- #
# ------------------------------- #
def plot_models_by_score(
    model_names,
    model_scores,
    model_stdevs=None,
    best_th=0.9,
    good_th=0.8,
    acceptable_th=0.7,
    figsize=(16, 10),
    title='Model comparison by score',
    xlabel='Model',
    ylabel='Score'
):
    """Plot a comparison between mutliple models based on a given score
    :param model_names: The name for each model.
    :param model_scores: The score for each model.
    :param model_stdevs: The standard deviation of the score for each model.
        It can be None if no standard deviations are known.
    :param best_th: The threshold such that models with a score above this
        value are considered as the best models.
    :param good_th: The threshold such that models with a score above this
        value are considered as good models.
    :param acceptable_th: The threshold such that models with a score above
        this value are considered as acceptable models.
    :param figsize: Define the size of the figure.
    :param title: The title for the plot.
    :param ylabel: The label for the y-axis.
    :param xlabel: The label for the x-axis.
    """
    # Prepare plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1)
    color = []
    if best_th is not None and \
            good_th is not None and \
            acceptable_th is not None:
        relop = lambda a, b: a >= b
        if best_th < good_th:
            relop = lambda a, b: a <= b
        for score in model_scores:
            if relop(score, best_th):
                color.append('green')
            elif relop(score, good_th):
                color.append('darkcyan')
            elif relop(score, acceptable_th):
                color.append('skyblue')
            else:
                color.append('gray')
    else:
        color = 'tab:blue'
    # Plot bars
    ax.bar(
        model_names, model_scores,
        color=color, yerr=model_stdevs,
        edgecolor='black', linewidth=2,
        error_kw={
            'ecolor': 'red',
            'capsize': 10,
            'markeredgewidth': 3
        }
    )
    # Plot thresholds
    if best_th is not None:
        ax.axhline(
            best_th, color='green', lw=3, ls='-', zorder=0, label='best'
        )
    if good_th is not None:
        ax.axhline(
            good_th, color='darkcyan', lw=3, ls='-', zorder=0, label='good'
        )
    if acceptable_th is not None:
        ax.axhline(
            acceptable_th, color='skyblue', lw=3,
            ls='-', zorder=0, label='acceptable'
        )
    # Format plot
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid('both')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='both', labelsize=16, labelrotation=60.0)
    ax.legend(loc='upper right', fontsize=16, bbox_to_anchor=(1.2, 1.0))
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    # Show plot
    plt.show()
