import matplotlib.pyplot as plt
import numpy as np
from kapteyn import kmpfit


def morphology_plots(Data, mag, mag_err, scatter, figure_name):

    def model_lin(params, x):
        a, b = params
        return a * x + b

    def residuals_lin(params, data):
        x, xerr, y, yerr = np.array(data)
        a, b = params
        w = yerr ** 2 + (a ** 2 * xerr ** 2) + (scatter[0] * x ** 2 + scatter[1] * x ** 1 +
                                                scatter[2]) ** 2
        return (y - model_lin(params, x)) / np.sqrt(w)

    # Get the three morphological sub-samples
    morph_groups = Data.groupby('Morph_Type')

    Sa = morph_groups.get_group('Sa')
    Sb = morph_groups.get_group('Sb')
    Sc = morph_groups.get_group('Sc')

    # fit TF relation to each group using intrinsic scatter from previous FULL fit
    fit_sa = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sa')['log_W'], morph_groups.get_group('Sa')['log_W_err'],
                                 morph_groups.get_group('Sa')[mag],
                                 morph_groups.get_group('Sa')[mag_err]])
    fit_sa.fit(params0=[-9, 1])

    fit_sb = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sb')['log_W'], morph_groups.get_group('Sb')['log_W_err'],
                                 morph_groups.get_group('Sb')[mag],
                                 morph_groups.get_group('Sb')[mag_err]])
    fit_sb.fit(params0=[-9, 1])

    fit_sc = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sc')['log_W'], morph_groups.get_group('Sc')['log_W_err'],
                                 morph_groups.get_group('Sc')[mag],
                                 morph_groups.get_group('Sc')[mag_err]])
    fit_sc.fit(params0=[-9, 1])

    params = [fit_sa.params, fit_sb.params, fit_sc.params]

    # set figure parameters
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig1, ax = plt.subplots(ncols=1, nrows=3, figsize=(7.5, 11), sharex=True)
    plt.subplots_adjust(top=0.98, bottom=0.06, right=0.95, left=0.1, wspace=0.0, hspace=0.0)

    # Plot the Sc sample
    ax[2].errorbar(np.array(Sc['log_W']), np.array(Sc[mag]),
                   xerr=np.array(Sc['log_W_err']), yerr=np.array(Sc[mag_err]),
                   color='grey', ls='none', markersize=0,
                   elinewidth=0.8, alpha=0.9, zorder=0)

    ax[2].plot(np.array(Sc['log_W']), np.array(Sc[mag]), color='blue', ls='none',
               marker='.', alpha=1, label='Sc sub-sample',
               zorder=1)

    ax[2].plot([1.7, 3], model_lin([params[2][0], params[2][1]], np.array([1.7, 3])), 'k',
               label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * params[2][0],
                                                         params[2][1]) + r'$\log_{10}(W)$', zorder=3)

    ax[1].errorbar(np.array(Sb['log_W']), np.array(Sb[mag]),
                   xerr=np.array(Sb['log_W_err']), yerr=np.array(Sb[mag_err]),
                   color='grey', ls='none', markersize=0,
                   elinewidth=0.8, alpha=0.9, zorder=0)

    ax[1].plot(np.array(Sb['log_W']), np.array(Sb[mag]), color='green', ls='none',
               marker='.', alpha=1, label='Sb sub-sample',
               zorder=1)

    ax[1].plot([1.7, 3], model_lin([params[1][0], params[1][1]], np.array([1.7, 3])), 'k',
               label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * params[1][0],
                                                         params[1][1]) + r'$\log_{10}(W)$', zorder=3)

    ax[0].errorbar(np.array(Sa['log_W']), np.array(Sa[mag]),
                   xerr=np.array(Sa['log_W_err']), yerr=np.array(Sa[mag_err]),
                   color='grey', ls='none', markersize=0,
                   elinewidth=0.8, alpha=0.9, zorder=0)

    ax[0].plot(np.array(Sa['log_W']), np.array(Sa[mag]), color='red',
               ls='none',
               marker='.', alpha=1, label='Sa sub-sample',
               zorder=1)

    ax[0].plot([1.7, 3], model_lin([params[0][0], params[0][1]], np.array([1.7, 3])), 'k',
               label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * params[0][0],
                                                         params[0][1]) + r'$\log_{10}(W)$', zorder=3)

    ax[0].set_xlim([1.85, 3.1])
    ax[0].set_ylim([-18.5, -27])
    ax[1].set_ylim([-18.5, -27])
    ax[2].set_ylim([-18.5, -27])

    ax[0].set_yticks([-20, -22, -24, -26])
    ax[0].set_yticklabels([-20, -22, -24, -26], fontsize=12)

    ax[1].set_yticks([-20, -22, -24, -26])
    ax[1].set_yticklabels([-20, -22, -24, -26], fontsize=12)

    ax[2].set_yticks([-20, -22, -24, -26])
    ax[2].set_yticklabels([-20, -22, -24, -26], fontsize=12)

    ax[2].set_xticks([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    ax[2].set_xticklabels([2.0, 2.2, 2.4, 2.6, 2.8, 3.0], fontsize=12)

    ax[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    ax[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    ax[2].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)

    ax[0].grid(b=True, which='both', color='0.65', linestyle=':')
    ax[1].grid(b=True, which='both', color='0.65', linestyle=':')
    ax[2].grid(b=True, which='both', color='0.65', linestyle=':')

    ax[2].set_xlabel('$\log_{10}(W)$', fontsize=15, labelpad=2)

    ax_invis = fig1.add_subplot(111, frameon=False)
    ax_invis.set_xticks([])
    ax_invis.set_yticks([])
    ax_invis.set_ylabel(r'Absolute Magnitude (vega mags)', fontfamily='serif', fontsize=15, labelpad=32)


    # calculate the number of galaxies in EVENLY SPACED BINS for each sub-sampe
    count_sa, edges = np.histogram(Sa['log_W'], bins=8, range=(2.1, 2.9), density=True)
    count_sb, edges = np.histogram(Sb['log_W'], bins=8, range=(2.1, 2.9), density=True)
    count_sc, edges = np.histogram(Sc['log_W'], bins=8, range=(2.1, 2.9), density=True)



    # calculate the percentage of the sample in each bin for each sub-sample
    sa_hist = []
    sb_hist = []
    sc_hist = []
    for j in range(len(edges) - 1):
        sa_hist.append(count_sa[j] / (count_sa[j] + count_sb[j] + count_sc[j]))
        sb_hist.append(count_sb[j] / (count_sa[j] + count_sb[j] + count_sc[j]))
        sc_hist.append(count_sc[j] / (count_sa[j] + count_sb[j] + count_sc[j]))


    mini_sa = fig1.add_subplot(939, frameon=True)
    mini_sa.set_ylim([0, 75])
    mini_sa.yaxis.tick_right()
    mini_sa.set_yticks([10, 30, 50, 70])
    mini_sa.set_yticklabels([10, 30, 50, 70], fontsize=9)
    mini_sa.set_xticks([2.1, 2.3, 2.5, 2.7, 2.9])
    mini_sa.set_xticklabels([2.1, 2.3, 2.5, 2.7, 2.9], fontsize=9)
    mini_sa.xaxis.tick_top()
    mini_sa.bar(edges[:-1] + 5 * 0.0125, np.array(sa_hist) * 100, bottom=None, align='edge', width=0.075, alpha=1,
                color='red', label='Sa')
    # mini_sa.yaxis.set_label_position("right")
    mini_sa.set_ylabel('% Sa galaxies', fontsize=9)
    # mini_sa.legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)

    mini_sb = fig1.add_subplot(9, 3, 18, frameon=True)
    mini_sb.set_ylim([0, 75])
    mini_sb.yaxis.tick_right()
    mini_sb.set_yticks([10, 30, 50, 70])
    mini_sb.set_yticklabels([10, 30, 50, 70], fontsize=9)
    mini_sb.set_xticks([2.1, 2.3, 2.5, 2.7, 2.9])
    mini_sb.set_xticklabels([2.1, 2.3, 2.5, 2.7, 2.9], fontsize=9)
    mini_sb.xaxis.tick_top()
    mini_sb.bar(edges[:-1] + 5 * 0.0125, np.array(sb_hist) * 100, bottom=None, align='edge', width=0.075, alpha=1,
                color='green', label='Sb')
    mini_sb.set_ylabel('% Sb galaxies', fontsize=9)

    mini_sc = fig1.add_subplot(9, 3, 27, frameon=True)
    mini_sc.set_ylim([0, 75])
    mini_sc.yaxis.tick_right()
    mini_sc.set_yticks([10, 30, 50, 70])
    mini_sc.set_yticklabels([10, 30, 50, 70], fontsize=9)
    mini_sc.set_xticks([2.1, 2.3, 2.5, 2.7, 2.9])
    mini_sc.set_xticklabels([2.1, 2.3, 2.5, 2.7, 2.9], fontsize=9)
    mini_sc.xaxis.tick_top()
    mini_sc.bar(edges[:-1] + 5 * 0.0125, np.array(sc_hist) * 100, bottom=None, align='edge', width=0.075, alpha=1,
                color='blue', label='Sc')
    mini_sc.set_ylabel('% Sc galaxies', fontsize=9)

    plt.savefig(figure_name)
