import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages
import scipy.special as special
import matplotlib.pylab as plt
from kapteyn import kmpfit
import warnings

warnings.filterwarnings("ignore")


def bias_calc(FigNamestr, M_star, alpha, scatter_coefficients, Data, Mtot_W1, Mtot_err_W1, addition, LF_exclude,
              CF_exclude, slope, intercept, no_galaxies):
    def luminosity_function(mags, phi_star):
        return (0.4 * np.log(10)) * phi_star * (10 ** (0.4 * (M_star - mags))) ** (1 + alpha) * np.exp(
            -10 ** (0.4 * (M_star - mags)))

    def completeness_function(mags, y_F, n):
        return 1 / (np.exp((mags - y_F) / n) + 1)

    def Chi2_LF(bins, hist, errs):
        # Array of Phi to test
        phi_star = np.linspace(min(hist), 2 * max(hist), 100)
        chi2 = []
        for i in range(len(phi_star)):
            chi2.append(np.sum(((luminosity_function(bins, phi_star[i]) - hist) / errs) ** 2))

        if list(chi2).index(min(chi2)) == 0:
            phi_star = np.linspace(phi_star[list(chi2).index(min(chi2))] / 2, phi_star[list(chi2).index(min(chi2)) + 1],
                                   100)

        elif list(chi2).index(min(chi2)) == 99:
            phi_star = np.linspace(phi_star[list(chi2).index(min(chi2)) - 1], phi_star[list(chi2).index(min(chi2))] * 2,
                                   100)


        else:
            phi_star = np.linspace(phi_star[list(chi2).index(min(chi2)) - 1], phi_star[list(chi2).index(min(chi2)) + 1],
                                   100)

        chi2 = []

        for i in range(len(phi_star)):
            chi2.append(np.sum(((luminosity_function(bins, phi_star[i]) - hist) / errs) ** 2))

        return [phi_star[list(chi2).index(min(chi2))], min(chi2)]

    def Chi2_CF(bins, hist, errs, i):
        # Array of Phi to test
        y_F = np.linspace(-26, -16, 100)
        n = np.linspace(0.05, 1, 100)

        chi2 = []
        for i in range(len(y_F)):
            chi2_row = []
            for j in range(len(n)):
                chi2_row.append(np.sum(((completeness_function(bins, y_F[i], n[j]) - hist) / errs) ** 2))

            chi2.append(chi2_row)

        chi2 = np.vstack(chi2)
        index = np.where(chi2 == np.amin(chi2))

        if index[0][0] == 0:
            y_F = np.linspace(y_F[index[0][0]], y_F[index[0][0] + 1], 100)

        elif index[0][0] == 99:
            y_F = np.linspace(y_F[index[0][0] - 1], y_F[index[0][0]], 100)

        else:
            y_F = np.linspace(y_F[index[0][0] - 1], y_F[index[0][0] + 1], 100)

        if index[1][0] == 0:
            n = np.linspace(n[index[1][0]], n[index[1][0] + 1], 100)

        elif index[1][0] == 99:
            n = np.linspace(n[index[1][0] - 1], n[index[1][0]], 100)

        else:
            n = np.linspace(n[index[1][0] - 1], n[index[1][0] + 1], 100)

        chi2 = []
        for i in range(len(y_F)):
            chi2_row = []
            for j in range(len(n)):
                chi2_row.append(np.sum(((completeness_function(bins, y_F[i], n[j]) - hist) / errs) ** 2))

            chi2.append(chi2_row)

        chi2 = np.vstack(chi2)
        index = np.where(chi2 == np.amin(chi2))

        return [np.amin(chi2), y_F[index[0]], n[index[1]]]

    def model_lin(params, x):
        a, b = params
        return a * x + b

    def model_quad(params, x):
        a, b, c = params
        return a * x ** 2 + b * x + c

    def residuals_lin2(params, data):
        x, xerr, y, yerr = np.array(data)
        a, b = params
        w = yerr ** 2 + (a ** 2 * xerr ** 2)
        return (y - model_lin(params, x)) / np.sqrt(w)

    iteration = 0

    # Extract Data:
    log_W = Data['log_W']

    bins_2 = np.histogram(np.array(log_W), 203)[1]
    bins_residuals_centres = bins_2 + (bins_2[1] - bins_2[0]) / 2
    scatter_c3 = model_quad(scatter_coefficients, bins_residuals_centres)

    # Make empty arrays for corrected absolute magnitudes and errors
    U_M_tot_corrected = np.array([0.00000] * no_galaxies)
    U_M_err_corrected = np.array([0.00000] * no_galaxies)
    bias_full_list = np.array([0.00000] * no_galaxies)

    Groups = Data.groupby("Cluster ID")
    Ids = list(Groups.groups.keys())

    # This is where all arrays to be filled go
    bin_edges = [[None] * 35 for i in range(len(Ids))]
    bin_centres = [[None] * 35 for i in range(len(Ids))]
    hist_raw = [[None] * 35 for i in range(len(Ids))]
    hist_raw_errs = [[None] * 35 for i in range(len(Ids))]
    hist_completeness = [[None] * 35 for i in range(len(Ids))]
    hist_completeness2 = [[None] * 35 for i in range(len(Ids))]
    hist_completeness_errs = [[None] * 35 for i in range(len(Ids))]
    bin_width = [None] * 35
    phi_star = [None] * 35
    y_F_bestfit = [None] * 35
    n_bestfit = [None] * 35
    dist_avg = []

    # Create figure which will be used to plot the histograms
    fig1, ax = plt.subplots(ncols=2, nrows=2, figsize=(22, 22), sharey=False, sharex=False)
    plt.subplots_adjust(top=0.98, bottom=0.04, right=0.98, left=0.08, wspace=0.17, hspace=0.1)
    axs = ax.flatten()
    index_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                     26, 27, 28, 29, 30]
    with PdfPages(FigNamestr) as pdf:
        for i in index_to_plot:
            axs[0].cla()
            axs[1].cla()
            axs[2].cla()
            axs[3].cla()
            group = Groups.get_group(Ids[i])
            Id = Ids[i]

            # Get arrays for the absolute magnitude for the group being considered
            M_err_group = np.reshape(np.array(Mtot_err_W1)[list(group.index)], (len(group),))
            M_tot_group = np.reshape(np.array(Mtot_W1)[list(group.index)], (len(group),))

            # Calculate the number of bins in the histogram
            n = int(np.ceil(math.log(len(group), 2))) + addition[i]

            # Calculate the average distance to the cluster
            dist_avg.append(np.average(group['L_dist']))

            # Get a list of the the bin edges
            prelim_bins = np.histogram(M_tot_group, int(np.ceil(n)))[1]
            bin_width[i] = (prelim_bins[1] - prelim_bins[0])

            prelim_bins2 = np.insert(prelim_bins, [0, len(prelim_bins)],
                                     [prelim_bins[0] - bin_width[i], prelim_bins[-1] + bin_width[i]])

            bin_edges[i] = np.insert(prelim_bins2, [0, len(prelim_bins2)],
                                     [prelim_bins2[0] - bin_width[i], prelim_bins2[-1] + bin_width[i]])

            bin_centres[i] = 0.5 * np.array(bin_edges[i][1:] + bin_edges[i][:-1])
            M1 = bin_edges[i][:-1]
            M2 = bin_edges[i][1:]

            # Make an empty list to be filled with all the integrals (each integral as a row)
            P_jk = []
            for j in range(len(M_tot_group)):
                u1 = (M1 - M_tot_group[j]) / (np.sqrt(2) * 20 * M_err_group[j])
                u2 = (M2 - M_tot_group[j]) / (np.sqrt(2) * 20 * M_err_group[j])
                P_j = (special.erf(u2) - special.erf(u1)) / 2
                P_jk.append(P_j)

            hist_raw[i] = [np.sum(np.vstack(P_jk)[:, k]) for k in range(np.shape(P_jk)[1])]
            hist_raw_errs[i] = [np.sum(np.vstack(P_jk)[:, k] * (1 - np.vstack(P_jk)[:, k])) / 2 for k in
                                range(np.shape(P_jk)[1])]

            axs[0].bar(bin_centres[i], hist_raw[i], width=bin_width[i] - 0.01, color='dodgerblue', alpha=0.6,
                       yerr=hist_raw_errs[i],
                       label='{}'.format(Id), ecolor='k', capsize=5, zorder=0)
            axs[0].bar(bin_centres[i], hist_raw[i], width=bin_width[i] - 0.01, color='blue', alpha=0.3, zorder=1)

            b = np.delete(bin_centres[i], LF_exclude[i])
            h = np.delete(hist_raw[i], LF_exclude[i])
            e = np.delete(hist_raw_errs[i], LF_exclude[i])

            max_index = list(h).index(max(h))

            if max_index > 2:
                if i == 11 or i == 12 or i == 30:
                    phi_best = Chi2_LF(b[0:max_index - 1], h[0:max_index - 1], e[0:max_index - 1])[0]
                elif i == 1:
                    phi_best = Chi2_LF(b[0:max_index - 0], h[0:max_index - 0], e[0:max_index - 0])[0]
                else:
                    phi_best = Chi2_LF(b[0:max_index - 2], h[0:max_index - 2], e[0:max_index - 2])[0]
                    chi_best1 = Chi2_LF(b[0:max_index - 2], h[0:max_index - 2], e[0:max_index - 2])[1]

            elif max_index == 0:
                phi_best = Chi2_LF(b, h, e)[0]
                chi_best1 = Chi2_LF(b, h, e)[1]

            else:
                phi_best = Chi2_LF(b[0:max_index], h[0:max_index], e[0:max_index])[0]
                chi_best1 = Chi2_LF(b[0:max_index], h[0:max_index], e[0:max_index])[1]

            phi_star[i] = phi_best

            axs[0].plot(np.linspace(min(bin_edges[i] - 3), max(bin_edges[i] + 1), 100),
                        luminosity_function(np.linspace(min(bin_edges[i] - 3), max(bin_edges[i] + 1), 100),
                                            phi_star[i]),
                        color='red', linewidth=2,
                        label='Luminosity' + '\n' + 'Function' + '\n' + r'$\Phi * = ${0:0.3f}'.format(phi_star[i]),
                        zorder=2)

            if luminosity_function(bin_edges[i][-1], phi_star[i]) > 2 * max(hist_raw[i]):
                if i == 3 or i == 6 or i == 18 or i == 28:
                    axs[0].set_ylim([0, 1.6 * max(hist_raw[i])])

                    axs[0].set_yticks(
                        [0, 0.476 * max(hist_raw[i]), 0.933 * max(hist_raw[i]), max(hist_raw[i]) * 1.6])
                    axs[0].set_yticklabels(
                        [str(0), str(round(0.467 * max(hist_raw[i]), 1)), str(round(0.933 * max(hist_raw[i]), 1)),
                         str(round(max(hist_raw[i]) * 1.6, 1))])
                elif i == 30 and i == 19:
                    axs[0].set_ylim([0, 1.1 * max(hist_raw[i])])

                    axs[0].set_yticks(
                        [0, 0.476 * max(hist_raw[i]), 0.933 * max(hist_raw[i]), max(hist_raw[i]) * 1.1])
                    axs[0].set_yticklabels(
                        [str(0), str(round(0.467 * max(hist_raw[i]), 1)), str(round(0.933 * max(hist_raw[i]), 1)),
                         str(round(max(hist_raw[i]) * 1.1, 1))])


                else:
                    axs[0].set_ylim([0, 1.4 * max(hist_raw[i])])

                    axs[0].set_yticks(
                        [0, 0.476 * max(hist_raw[i]), 0.933 * max(hist_raw[i]), max(hist_raw[i]) * 1.4])
                    axs[0].set_yticklabels(
                        [str(0), str(round(0.467 * max(hist_raw[i]), 1)), str(round(0.933 * max(hist_raw[i]), 1)),
                         str(round(max(hist_raw[i]) * 1.4, 1))])

            else:
                if i == 3 or i == 6 or i == 18 or i == 28:
                    axs[0].set_ylim([0, 1.6 * max(hist_raw[i])])
                    axs[0].set_yticks([0 / 2, luminosity_function(bin_edges[i][-1], phi_star[i]) / 2,
                                       luminosity_function(bin_edges[i][-1], phi_star[i])])
                    axs[0].set_yticklabels(
                        [str(0), str(round(luminosity_function(bin_edges[i][-1], phi_star[i]) / 2, 1)),
                         str(round(luminosity_function(bin_edges[i][-1], phi_star[i]), 1))])

                elif i == 30 and i == 19:
                    axs[0].set_ylim([0, 1.1 * max(hist_raw[i])])
                    axs[0].set_yticks([0 / 2, luminosity_function(bin_edges[i][-1], phi_star[i]) / 2,
                                       luminosity_function(bin_edges[i][-1], phi_star[i])])
                    axs[0].set_yticklabels(
                        [str(0), str(round(luminosity_function(bin_edges[i][-1], phi_star[i]) / 2, 1)),
                         str(round(luminosity_function(bin_edges[i][-1], phi_star[i]), 1))])

                else:
                    axs[0].set_ylim([0, 1.4 * max(hist_raw[i])])
                    axs[0].set_yticks([0 / 2, luminosity_function(bin_edges[i][-1], phi_star[i]) / 2,
                                       luminosity_function(bin_edges[i][-1], phi_star[i])])
                    axs[0].set_yticklabels(
                        [str(0), str(round(luminosity_function(bin_edges[i][-1], phi_star[i]) / 2, 1)),
                         str(round(luminosity_function(bin_edges[i][-1], phi_star[i]), 1))])

            if i == 28 or i == 18:
                axs[0].set_xlim([min(bin_edges[i]) - 0.8, max(bin_edges[i])])

            else:
                axs[0].set_xlim([min(bin_edges[i]), max(bin_edges[i])])

            axs[0].legend(ncol=1, loc="upper left", frameon=False,
                          facecolor='white', framealpha=1, fontsize=20)

            axs[0].set_ylabel('Number of Galaxies', fontfamily='serif', fontsize=21, labelpad=2)
            axs[0].set_xlabel('Absolute Magnitude (Vega mags), M', fontfamily='serif', fontsize=20, labelpad=7)

            hist_completeness[i] = hist_raw[i] / luminosity_function(bin_centres[i], phi_star[i])
            hist_completeness_errs[i] = hist_raw_errs[i] / luminosity_function(bin_centres[i], phi_star[i])

            axs[1].bar(bin_centres[i], hist_completeness[i], width=bin_width[i] - 0.01, color='darkorchid', alpha=0.2,
                       yerr=hist_completeness_errs[i], ecolor='k', capsize=5, zorder=1)
            axs[1].bar(bin_centres[i], hist_completeness[i], width=bin_width[i] - 0.01, color='mediumpurple', alpha=0.8,
                       zorder=0)

            hist_completeness[i][np.where(hist_completeness[i] > 1)[0]] = [1]
            b = np.delete(bin_centres[i], CF_exclude[iteration][i])
            c = np.delete(hist_completeness[i], CF_exclude[iteration][i])
            e = np.delete(hist_raw_errs[i], CF_exclude[iteration][i])

            chi2 = Chi2_CF(b, c, e, i)
            y_F_bestfit[i] = chi2[1]
            n_bestfit[i] = chi2[2]

            axs[1].plot(np.linspace(bin_edges[i][0], bin_edges[i][-1], 100),
                        completeness_function(np.linspace(bin_edges[i][0], bin_edges[i][-1], 100), chi2[1],
                                              chi2[2]),
                        'r', label='Completeness' + '\n' + 'Function' + '\n' + r'$M_F = ${0:0.3f}'.format(
                    y_F_bestfit[i][0]) + '\n' + '$\eta = ${0:0.3f}'.format(n_bestfit[i][0]))

            axs[1].legend(loc="upper right", frameon=False,
                          facecolor='white', framealpha=1, fontsize=20)
            axs[1].set_ylim([0, 1.5])
            axs[1].set_xlim([bin_edges[i][0], bin_edges[i][-1]])

            axs[1].set_yticks([0, 0.5, 1, 1.7])

            axs[1].set_ylabel(r'Completeness ($c(y)$)', fontfamily='serif', fontsize=21, labelpad=5)
            axs[1].set_xlabel('Absolute Magnitude (Vega mags), M', fontfamily='serif', fontsize=20, labelpad=7)

            # ==================================================================================================================
            g = group.nsmallest(len(group), 'log_W', keep='all')
            group_ordered = g.reset_index().set_index(np.arange(min(group.index), max(group.index) + 1), drop=False)

            # Get log_W for the group ONLY:
            log_W_group = np.array(group['log_W'])
            log_W_group_err = np.array(group['log_W_err'])

            # This is an empty array that will be filled by n arrays
            sample = []

            for n in range(30000):
                sample_init = []
                W_array = []
                W_array_err = []
                for b in range(len(bins_2) - 1):

                    W = log_W_group[np.where(np.logical_and(log_W_group >= bins_2[b], log_W_group <= bins_2[b + 1]))[0]]
                    W_err = log_W_group_err[
                        np.where(np.logical_and(log_W_group >= bins_2[b], log_W_group <= bins_2[b + 1]))[0]]

                    if len(W) != 0:
                        for j, w in enumerate(W):
                            W_array.append(w)
                            W_array_err.append(W_err[j])
                            M = model_lin([slope, intercept], w)
                            s = np.random.normal(loc=0.0, scale=scatter_c3[b], size=None)
                            sample_init.append(M + s)

                random_numbers = np.random.random(len(W_array))

                completeness_values = completeness_function(sample_init, y_F_bestfit[i], n_bestfit[i])

                sample_filtered = []

                # Make a loop to test whether the point in the sample passes the completness filter
                for j in range(len(sample_init)):
                    if random_numbers[j] < completeness_values[j]:
                        sample_filtered.append(sample_init[j])

                    else:
                        sample_filtered.append(None)

                sample.append(sample_filtered)

            sample = np.vstack(sample)

            # make a loop to run over each COLUMN of the array and then turn each column into its own array
            bias = []
            for k in range(np.shape(sample)[1]):
                # k is the index of galaxy for which the bias is being calculated

                # Get the array of magnitudes
                array_mags = sample[:, k]

                # run a loop over each of these elements
                bias_array = []
                for m in range(len(array_mags)):

                    # m is the index of the iteration
                    if array_mags[m] != None:
                        # bias_array.append(array_mags[m] - model(Universal_FitObj.params, log_W[k]))
                        bias_array.append(array_mags[m] - model_lin([slope, intercept], W_array[k]))

                bias.append(np.average(bias_array))

            bias = np.array(bias)
            s = np.isnan(bias)
            bias[s] = 0.0

            M_err_group = np.reshape(np.array(Mtot_err_W1)[group_ordered['index'].values.tolist()],
                                     (len(group_ordered),))
            M_tot_group = np.reshape(np.array(Mtot_W1)[group_ordered['index'].values.tolist()], (len(group_ordered),))

            U_M_tot_corrected[group_ordered['index'].values.tolist()] = [sum(x) for x in
                                                                         zip(M_tot_group, list(-1 * bias))]
            U_M_err_corrected[group_ordered['index'].values.tolist()] = [np.sqrt(sum(x)) for x in
                                                                         zip(M_err_group ** 2,
                                                                             list((-0.03 * bias) ** 2))]

            bias_full_list[group_ordered['index'].values.tolist()] = [bias]

            axs[2].plot(W_array, bias, 'b.', markersize=10, label='{} Bias'.format(Id))
            axs[2].set_ylim(max(bias) + 0.05, min(bias) - 0.05)

            axs[2].grid('visible', which='both', color='0.65', linestyle=':')
            axs[2].legend(loc="upper right", frameon=False, facecolor='white', framealpha=1, fontsize=20)
            axs[2].set_ylabel(r'Bias (Vega mags)', fontfamily='serif', fontsize=21, labelpad=2)
            axs[2].set_xlabel(r'$\log_{10}(W)$', fontfamily='serif', fontsize=21, labelpad=5)

            # Perform 2 bivariate fits: to the corrected and uncorrected GROUP samples (for plotting only, no functional form needed)
            FitObj = kmpfit.Fitter(residuals=residuals_lin2,
                                   data=[W_array, np.zeros(len(W_array)), M_tot_group, np.ones(len(W_array))])
            FitObj.fit(params0=[-9, 1])

            axs[3].plot(W_array, M_tot_group, 'b.', markersize=10, label='{} biased'.format(Id))
            axs[3].plot([max(W_array) + 0.03, min(W_array) - 0.03],
                        model_lin(FitObj.params, np.array([max(W_array) + 0.03, min(W_array) - 0.03])),
                        color='blue', ls='--',
                        label='M = {0:.2f} {1:.2f}'.format(FitObj.params[1], FitObj.params[0]) + r'$\log_{10}(W)$')

            FitObj = kmpfit.Fitter(residuals=residuals_lin2, data=[W_array, np.zeros(len(W_array)), np.array(
                [sum(x) for x in zip(M_tot_group, list(-1 * bias))]), np.ones(len(W_array))])
            FitObj.fit(params0=[-8, 1])

            axs[3].plot(W_array, [sum(x) for x in zip(M_tot_group, list(-1 * bias))], 'r.', markersize=10,
                        label='{} de-biased'.format(Id))
            axs[3].plot([max(W_array) + 0.03, min(W_array) - 0.03],
                        model_lin(FitObj.params, np.array([max(W_array) + 0.03, min(W_array) - 0.03])), color='red',
                        ls='--', alpha=0.8,
                        label='M = {0:.2f} {1:.2f}'.format(FitObj.params[1], FitObj.params[0]) + r'$\log_{10}(W)$')

            for hehe, w in enumerate(W_array):
                axs[3].plot([w, w], [[sum(x) for x in zip(M_tot_group, list(-1 * bias))][hehe], M_tot_group[hehe]],
                            'k', ls=':')

            axs[3].set_ylim(max([sum(x) for x in zip(M_tot_group, list(-1 * bias))]) + 0.2,
                            min(model_lin(FitObj.params, np.array([min(W_array) - 0.02])),
                                min([sum(x) for x in zip(M_tot_group, list(-1 * bias))])) - 0.7)
            axs[3].set_xlim(min(W_array) - 0.03, max(W_array) + 0.03)

            axs[3].grid('visible', which='both', color='0.65', linestyle=':')
            axs[3].legend(ncol=1, loc="upper left", frameon=False, facecolor='white', framealpha=1, fontsize=21)
            axs[3].set_ylabel(r'Absolute Magnitude (Vega mags)', fontfamily='serif', fontsize=21, labelpad=2)
            axs[3].set_xlabel(r'$\log_{10}(W)$', fontfamily='serif', fontsize=21, labelpad=5)

            pdf.savefig()

    return U_M_tot_corrected, U_M_err_corrected, bias_full_list
