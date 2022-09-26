import numpy as np
import math
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import scipy.special as special
import matplotlib.pylab as plt
from kapteyn import kmpfit
import warnings

warnings.filterwarnings("ignore")


def bias_calc(FigNamestr, Data, mag, mag_err, slope, intercept, scatter, addition, LF_exclude, CF_exclude, iterations):
    M_star = -24.27 + 5 * np.log10(0.72)
    alpha = -1.40

    # Define functions used in bias correction
    def luminosity_function(phi_star, m):
        return (0.4 * np.log(10)) * phi_star * (10 ** (0.4 * (M_star - m))) ** (1 + alpha) * np.exp(
            -10 ** (0.4 * (M_star - m)))

    def residuals_LF(phi_star, data):
        m, h, herr = np.array(data)
        return (h - luminosity_function(phi_star, m)) / np.sqrt(herr ** 2)

    def completeness_function(params, m):
        y_F, n = params
        return 1 / (np.exp((m - y_F) / n) + 1)

    def residuals_CF(params, data):
        m, h, herr = np.array(data)
        y_F, n = params
        return (h - completeness_function(params, m)) / np.sqrt(herr ** 2)


    def model_lin(params, x):
        a, b = params
        return a * x + b

    def residuals_lin(params, data):
        x, xerr, y, yerr = np.array(data)
        a, b = params
        w = yerr ** 2 + (a ** 2 * xerr ** 2) + (scatter[0] * x ** 2 + scatter[1] * x ** 1 +
                                                scatter[2]) ** 2
        return (y - model_lin(params, x)) / np.sqrt(w)

    def model_quad(params, x):
        a, b, c = params
        return a * x ** 2 + b * x + c


    # Create the figure that results are saved into.
    fig1, ax = plt.subplots(ncols=2, nrows=2, figsize=(22, 22), sharey=False, sharex=False)
    plt.subplots_adjust(top=0.98, bottom=0.04, right=0.98, left=0.08, wspace=0.17, hspace=0.1)
    axs = ax.flatten()

    # Group the Data according to cluster ID.
    Groups = Data.groupby("Cluster_ID")
    Ids = list(Groups.groups.keys())

    index = range(len(Ids))

    # Create empty series to add bias correction paramemets too
    phi = []
    eta = []
    M_F = []
    Names = []
    dist = []
    bias_list = []
    with PdfPages(FigNamestr) as pdf:
            for i in index:
                # Clear the figure.
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                axs[3].cla()

                # Get the cluster and ID
                cluster = Groups.get_group(Ids[i])
                ID = Ids[i]
                print(ID)

                # ------------------------------------------------------------------------------------------------------
                # CONSTRUCT ABSOLUTE MAG HISTOGRAM AND FIT LF
                # ------------------------------------------------------------------------------------------------------

                M = cluster[mag]
                M_err = cluster[mag_err]

                # Calculate the approximate number of bins in the histogram
                n_approx_1 = int(np.ceil(int(np.ceil(math.log(len(M), 2))) + addition[i]))
                n_approx = int(np.round(1 + 3.322*np.log10(len(M))) + addition[i])
                ### CHANGE HERE

                # Get an array of the bin edges
                bins = np.histogram(M, n_approx)[1]
                width = bins[1] - bins[0]
                bins = np.insert(bins, [0, len(bins)], [bins[0]-width, bins[-1]+width])

                # define the bin centers
                mags = bins[:-1] + width/2

                # run through loop of each galaxy in the sample and add the histogram bin heights together.
                histLF = np.zeros(len(bins)-1)
                histLF_errs = np.zeros(len(bins) - 1)
                for j, Mj in enumerate(M):
                    hist_dist = 1/2*(special.erf((bins[1:] - Mj)/(np.sqrt(2)*20*M_err.values.tolist()[j])) - special.erf(
                        (bins[:-1] - Mj)/(np.sqrt(2)*20*M_err.values.tolist()[j])))
                    histLF += hist_dist
                    histLF_errs += hist_dist * (1-hist_dist)

                # Plot the histogram
                axs[0].bar(mags, histLF, width=width - 0.01, color='dodgerblue', alpha=0.6,
                           yerr=histLF_errs,
                           label='{}'.format(ID), ecolor='k', capsize=5, zorder=0)
                axs[0].bar(mags, histLF, width=width - 0.01, color='blue', alpha=0.3, zorder=1)

                # Remove any bins that "throw" LF fitting
                mLF = np.delete(mags, LF_exclude[i])
                hLF = np.delete(histLF, LF_exclude[i])
                herrLF = np.delete(histLF_errs, LF_exclude[i])

                # print(LF_exclude[i])



                # Find the peak of the histogram
                peak = np.where(hLF == max(hLF))[0][0]

                if peak>3:
                    peak=peak-1

                # print(peak)

                # Fit the LF upto this peak
                LF_fit = kmpfit.Fitter(residuals=residuals_LF, data=[mLF[0:peak], hLF[0:peak], herrLF[0:peak]])
                LF_fit.fit(params0=[max(hLF)])

                # plot the LF over the histogram
                axs[0].plot(np.linspace(min(bins - 3), max(bins + 1), 100),luminosity_function(LF_fit.params[0],np.linspace(min(bins - 3), max(bins + 1), 100)),
                            color='red', linewidth=2,
                            label='Luminosity' + '\n' + 'Function' + '\n' + r'$\Phi * = ${0:0.3f}'.format(LF_fit.params[0]),
                            zorder=2)

                # format panel 0 of the figure.
                axs[0].set_xlim([min(bins), max(bins)])
                axs[0].set_ylim([0, 1.5*(max(histLF))])
                axs[0].legend(ncol=1, loc="upper left", frameon=False,facecolor='white', framealpha=1, fontsize=20)
                axs[0].set_ylabel('Number of Galaxies', fontfamily='serif', fontsize=21, labelpad=2)
                axs[0].set_xlabel('Absolute Magnitude (Vega mags), M', fontfamily='serif', fontsize=20, labelpad=7)

                # ------------------------------------------------------------------------------------------------------
                # CONSTRUCT COMPLETENESS HISTOGRAMS AND FIT CF
                # ------------------------------------------------------------------------------------------------------

                # Construct histograms
                histCF = histLF / luminosity_function(LF_fit.params[0], mags)
                histCF_errs = np.sqrt((histLF_errs/histLF) ** 2 + ((np.ones(len(histLF)) * LF_fit.xerror[0])/luminosity_function(LF_fit.params[0], mags)) ** 2)

                # Plot histograms
                axs[1].bar(mags, histCF, width=width - 0.01, color='darkorchid',
                           alpha=0.2,
                           yerr=histCF_errs, ecolor='k', capsize=5, zorder=1)
                axs[1].bar(mags, histCF, width=width - 0.01, color='mediumpurple',
                           alpha=0.8,
                           zorder=0)

                # For fitting purposes set all histogram bin heights greater than 1 equal to 1 (cf has a maximum of 1)
                histCF[np.where(histCF > 1)[0]] = [1]

                # Remove any bins that "throw" CF fitting

                mCF = np.delete(mags, CF_exclude[i])
                hCF = np.delete(histCF, CF_exclude[i])
                herrCF = np.delete(histCF_errs, CF_exclude[i])

                # Fit the CF
                CF_fit = kmpfit.Fitter(residuals=residuals_CF, data=[mCF, hCF, herrCF])
                CF_fit.fit(params0=[mags[int(np.ceil(len(mags)/2))], 0.5])

                # Plot the completeness function over the histogram
                axs[1].plot(np.linspace(bins[0], bins[-1], 100),
                            completeness_function(CF_fit.params, np.linspace(bins[0], bins[-1], 100)),
                            'r', label='Completeness' + '\n' + 'Function' + '\n' + r'$M_F = ${0:0.3f}'.format(
                        CF_fit.params[0]) + '\n' + '$\eta = ${0:0.3f}'.format(CF_fit.params[1]))


                # Format panel 2
                axs[1].legend(loc="upper right", frameon=False,
                              facecolor='white', framealpha=1, fontsize=20)
                axs[1].set_ylim([0, 1.5])
                axs[1].set_xlim([bins[0], bins[-1]])

                axs[1].set_ylabel(r'Completeness ($c(y)$)', fontfamily='serif', fontsize=21, labelpad=5)
                axs[1].set_xlabel('Absolute Magnitude (Vega mags), M', fontfamily='serif', fontsize=20, labelpad=7)


                # Append correction values onto lists
                phi.append(LF_fit.params[0])
                eta.append(CF_fit.params[1])
                M_F.append(CF_fit.params[0])
                Names.append(ID)
                dist.append(np.mean(cluster['L_dist']))

                # ------------------------------------------------------------------------------------------------------
                # RUN MCMC OVER ITERATIONS
                # ------------------------------------------------------------------------------------------------------
                logW = cluster['log_W']
                logW_err = cluster['log_W_err']

                # This is an empty array that the differences calculated from each mock will be stacked into.
                vertical_stack_diff = []
                for n in range(iterations):
                    np.random.seed(n+37)
                    # np.random.seed(n)

                    # create the mock based on the preliminary fit.
                    M_predicted = (slope*logW + intercept).values.tolist()
                    mock = M_predicted + np.random.normal(np.zeros(len(logW)), model_quad(scatter, logW), np.shape(model_quad(scatter, logW)))

                    # generate completeness values
                    c_vals = completeness_function(CF_fit.params, mock)

                    # generate random numbers for filter
                    filter_vals = np.random.uniform(0, 1, np.shape(mock))

                    # cut all elements where the filter is not passes
                    for k in range(len(c_vals)):
                        if filter_vals[k] > c_vals[k]:
                            mock[k] = None


                    # Calculate the difference between the mock and the predicted absolute magnitude and append onto larger array for each iteration
                    diff = mock - M_predicted

                    # each ROW corresponds to a single mock, each COLUMN corresponds to a single galaxy.
                    vertical_stack_diff.append(diff)

                # for every galaxy in the sample, calculate the bias
                bias = []
                for g in range(len(logW)):
                    hold = np.vstack(vertical_stack_diff)[:,g]
                    if len(hold[~np.isnan(hold)]) == 0:
                        bias.append(0)
                    else:
                        bias.append(np.sum(hold[~np.isnan(hold)]) / len(hold[~np.isnan(hold)]))

                M_group = M - bias
                M_err_c = np.sqrt(M_err ** 2 + (0.1*np.abs(bias)) **2)


                # Plot the bias as a function of logW
                axs[2].plot(logW, bias, 'b.', markersize=10, label='{} Bias'.format(ID))
                axs[2].set_ylim(max(bias) + 0.05, min(bias) - 0.05)

                axs[2].grid('visible', which='both', color='0.65', linestyle=':')
                axs[2].legend(loc="upper right", frameon=False, facecolor='white', framealpha=1, fontsize=20)
                axs[2].set_ylabel(r'Bias (Vega mags)', fontfamily='serif', fontsize=21, labelpad=2)
                axs[2].set_xlabel(r'$\log_{10}(W)$', fontfamily='serif', fontsize=21, labelpad=5)


                # Fit lines to the corrected and uncorrected data (for plotting purposes only)
                fit_uncorrected = kmpfit.Fitter(residuals=residuals_lin,
                                       data=[logW, np.zeros(len(logW)), M, M_err])
                fit_uncorrected.fit(params0=[-9, 1])

                axs[3].plot(logW, M, 'b.', markersize=10, label='{} biased'.format(ID))
                axs[3].plot([max(logW) + 0.03, min(logW) - 0.03],
                            model_lin(fit_uncorrected.params, np.array([max(logW) + 0.03, min(logW) - 0.03])),
                            color='blue', ls='--',
                            label='M = {0:.2f} {1:.2f}'.format(fit_uncorrected.params[1], fit_uncorrected.params[0]) + r'$\log_{10}(W)$')



                fit_corrected = kmpfit.Fitter(residuals=residuals_lin,
                                                data=[logW, np.zeros(len(logW)), M_group, M_err_c])
                fit_corrected.fit(params0=[-9, 1])

                axs[3].plot(logW, M_group, 'r.', markersize=10,
                            label='{} de-biased'.format(ID))
                axs[3].plot([max(logW) + 0.03, min(logW) - 0.03],
                            model_lin(fit_corrected.params, np.array([max(logW) + 0.03, min(logW) - 0.03])), color='red',
                            ls='--', alpha=0.8,
                            label='M = {0:.2f} {1:.2f}'.format(fit_corrected.params[1], fit_corrected.params[0]) + r'$\log_{10}(W)$')

                for line, w in enumerate(logW):
                    axs[3].plot([w, w], [M_group.iloc[line], M.iloc[line]],
                                'k', ls=':')

                # if ID == "Coma":
                #     for coma_index in range(len(cluster)):
                #         axs[3].annotate(cluster['2MASS_Name'].iloc[coma_index], (logW.iloc[coma_index], M_group.iloc[coma_index]))


                axs[3].set_ylim(max(M_group) + 0.2,
                                min(model_lin(fit_corrected.params, np.array([min(logW) - 0.02])),
                                    min(M_group)) - 0.7)
                axs[3].set_xlim(min(logW) - 0.03, max(logW) + 0.03)

                axs[3].grid('visible', which='both', color='0.65', linestyle=':')
                axs[3].legend(ncol=1, loc="upper left", frameon=False, facecolor='white', framealpha=1, fontsize=21)
                axs[3].set_ylabel(r'Absolute Magnitude (Vega mags)', fontfamily='serif', fontsize=21, labelpad=2)
                axs[3].set_xlabel(r'$\log_{10}(W)$', fontfamily='serif', fontsize=21, labelpad=5)

                pdf.savefig()

                if i==0:
                    M_corrected = M_group
                    M_corrected_err = M_err_c
                    bias_list.append(bias)

                else:
                    M_corrected = pd.concat([M_corrected,M_group], axis=0)
                    M_corrected_err = pd.concat([M_corrected_err, M_err_c], axis=0)
                    bias_list.append(bias)

    output = pd.DataFrame(M_corrected)
    output['M_BiasCorrected_err'] = M_corrected_err
    output['bias'] = sum(bias_list, [])
    output.sort_index(axis=0, inplace=True)

    bias_correction = pd.DataFrame(Names, columns=['Clusters'])
    bias_correction['Group_Distance'] = dist
    bias_correction['phi_star_W1'] = phi
    bias_correction['M_f_W1'] = M_F
    bias_correction['n_W1'] = eta

    bias_correction.to_csv('Figures/Incompleteness_Correction_prams.csv', sep='\t', na_rep='na', float_format='%.4f',
                               index=False)



    return output



