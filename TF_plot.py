import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Create preliminary plots


def plot_TF(x, xerr, y, yerr, slope, slope_err, intercept, intercept_err, residuals, sc, binned_s, figure_name, legend):
    def model_lin(params, x):
        a, b = params
        return a * x + b


    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7.5), sharey=False)
    plt.subplots_adjust(top=0.96, bottom=0.09, right=0.98, left=0.06, wspace=0.12, hspace=0.05)

    axs[0].errorbar(x, y, yerr, xerr, color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[0].plot(x, y, color='blueviolet', ls='none', marker='.', alpha=0.8, zorder=1, label=legend)
    axs[0].plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[0].plot([1.7, 3], model_lin([slope, intercept], np.array([1.7, 3])), 'r',
                label=r'M = ({2:0.2f} $\pm$ {3:0.2f})  $-$ ({0:0.2f} $\pm$ {1:0.2f})'.format(-1 * slope, slope_err,
                                                                                             intercept,
                                                                                             intercept_err) + r'$\log_{10}(W)$',
                zorder=3)

    axs[0].set_xlim([min(x) - 0.2, max(x) + 0.2])
    axs[0].set_ylim([max(y) + 0.5, min(y) - 1.2])

    axs[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    axs[0].grid('visible', which='both', color='0.65', linestyle=':')
    axs[0].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=15, labelpad=2)
    axs[0].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)

    poly = np.poly1d(sc)
    means_logW, bin_edges_logW, logw_avg = stats.binned_statistic(x, x, 'mean', bins=5)

    axs[1].plot(x, residuals,
                '.', color='darkgray', alpha=1, markersize='3', label='Residuals')
    axs[1].plot(means_logW, binned_s, 's', color='dimgrey', label='Binned Average Intrinsic Scatter', zorder=3)

    axs[1].plot(np.linspace(1.7, 3.1, 100), poly(np.linspace(1.7, 3.1, 100)), 'r',
                label=r'$\xi$ = {0:0.2f}'.format(sc[0]) + r'$\log_{10}(W)^2$' + '{0:0.2f}'.format(
                    sc[1]) + r'$\log_{10}(W)$' + ' + {0:0.2f}'.format(sc[2]),
                zorder=2)

    axs[1].set_xlim([min(x) - 0.05, max(x) + 0.05])
    axs[1].set_ylim([-0.01, max(residuals) + 0.5])

    axs[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    axs[1].grid('visible', which='both', color='0.65', linestyle=':')
    axs[1].set_ylabel('Scatter (vega mags)', fontfamily='serif', fontsize=15, labelpad=2)
    axs[1].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)
    # ======================================================================================================================

    plt.savefig(figure_name)
