import matplotlib.pyplot as plt
from bivariate_fitting import *


def morphological_correction(Data, mag_to_use, magerr_to_use, scatter_coefficient):

    def model_lin(params, x):
        a, b = params
        return a * x + b

    def residuals_lin(params, data):
        x, xerr, y, yerr = np.array(data)
        a, b = params
        w = yerr ** 2 + (a ** 2 * xerr ** 2) + (scatter_coeffs[0] * x ** 2 + scatter_coeffs[1] * x ** 1 +
                                                scatter_coeffs[2]) ** 2
        return (y - model_lin(params, x)) / np.sqrt(w)

    scatter_coeffs = scatter_coefficient
    morphtype_groups = Data.groupby('Morph_Type')

    MT = list(morphtype_groups.groups.keys())

    Sa_group = morphtype_groups.get_group(MT[0])
    Sb_group = morphtype_groups.get_group(MT[1])
    Sc_group = morphtype_groups.get_group(MT[2])

    # Perform bivariate fit using intrinsic scatter from WHOLE sample (above)
    params_i = [-9, 1]
    FitObjSa_b = kmpfit.Fitter(residuals=residuals_lin, data=[np.array(Sa_group['log_W']), np.array(Sa_group['log_W_err']), np.array(Sa_group[mag_to_use]), np.array(Sa_group[magerr_to_use])])
    FitObjSa_b.fit(params0=params_i)

    FitObjSb_b = kmpfit.Fitter(residuals=residuals_lin,
                               data=[np.array(Sb_group['log_W']), np.array(Sb_group['log_W_err']),
                                     np.array(Sb_group[mag_to_use]), np.array(Sb_group[magerr_to_use])])
    FitObjSb_b.fit(params0=params_i)

    FitObjSc_b = kmpfit.Fitter(residuals=residuals_lin,
                               data=[np.array(Sc_group['log_W']), np.array(Sc_group['log_W_err']),
                                     np.array(Sc_group[mag_to_use]), np.array(Sc_group[magerr_to_use])])
    FitObjSc_b.fit(params0=params_i)

    slope_sa_b, intercept_sa_b = FitObjSa_b.params
    slope_sb_b, intercept_sb_b = FitObjSb_b.params
    slope_sc, intercept_sc = FitObjSc_b.params

    print('slope sa before = ', slope_sa_b, 'intercept sa before = ', intercept_sa_b)
    print('slope sb before = ', slope_sb_b, 'intercept sb before = ', intercept_sb_b)
    print('slope sc = ', slope_sc, 'intercept sc = ', intercept_sc)

    # Adjust Sa to align with Sc
    dy_sa = intercept_sc - intercept_sa_b
    dm_sa = slope_sc - slope_sa_b
    Sa_Mtot_corrected = np.add(np.array(Sa_group[mag_to_use]) + dy_sa, dm_sa * np.array(Sa_group['log_W']))

    sa_diff = np.abs(Sa_Mtot_corrected - np.array(Sa_group[mag_to_use])+ np.array(Sa_group['Bias_IT1']))

    for loop_sa_i in range(len(sa_diff)):
        if sa_diff[loop_sa_i] > 0.1*np.abs(np.array(Sa_group['mag_W1'] - (5. * np.log10(Sa_group['L_dist'] * 10 ** 6) - 5.))[loop_sa_i]):
            print(Sa_group['log_W'])

    # Adjust Sa to align with Sc
    dy_sb = intercept_sc - intercept_sb_b
    dm_sb = slope_sc - slope_sb_b
    Sb_Mtot_corrected = np.add(np.array(Sb_group[mag_to_use]) + dy_sb, dm_sb * np.array(Sb_group['log_W']))

    sb_diff = np.abs(Sb_Mtot_corrected - np.array(Sb_group[mag_to_use]) + np.array(Sb_group['Bias_IT1']))


    for loop_sb_i in range(len(sb_diff)):
        if sb_diff[loop_sb_i] > 0.1*np.abs(np.array(Sb_group['mag_W1'] - (5. * np.log10(Sb_group['L_dist'] * 10 ** 6) - 5.))[loop_sb_i]):
            print(Sb_group['log_W'])



    # Make array with all corrected magnitudes correctly placed
    Mtot_W1_morphcorr = np.array([0.00000] * len(Data))

    Mtot_W1_morphcorr[np.array(Sa_group.index)] = Sa_Mtot_corrected
    Mtot_W1_morphcorr[np.array(Sb_group.index)] = Sb_Mtot_corrected
    Mtot_W1_morphcorr[np.array(Sc_group.index)] = np.array(Sc_group[mag_to_use])

    slope_corr, intercept_corr, residuals_corr, scatter_coefficients_corr, rms_corr, slope_corr_err, intercept_corr_err = bivariate_fit(np.array(Data['log_W']),
                                                                                          np.array(
                                                                                              Data['log_W_err']),
                                                                                          Mtot_W1_morphcorr, np.array(Data[magerr_to_use]), -7, 1, 100)

    scatter_coeffs = scatter_coefficients_corr

    FitObjSa_a = kmpfit.Fitter(residuals=residuals_lin,
                               data=[np.array(Sa_group['log_W']), np.array(Sa_group['log_W_err']),
                                     Sa_Mtot_corrected, np.array(Sa_group[magerr_to_use])])
    FitObjSa_a.fit(params0=params_i)

    FitObjSb_a = kmpfit.Fitter(residuals=residuals_lin,
                               data=[np.array(Sb_group['log_W']), np.array(Sb_group['log_W_err']),
                                     Sb_Mtot_corrected, np.array(Sb_group[magerr_to_use])])
    FitObjSb_a.fit(params0=params_i)

    slope_sa_a, intercept_sa_a = FitObjSa_a.params
    slope_sb_a, intercept_sb_a = FitObjSb_a.params

    print('slope sa after = ', slope_sa_a, 'intercept sa after = ', intercept_sa_a)
    print('slope sb after = ', slope_sb_a, 'intercept sb after = ', intercept_sb_a)

    # Print the adjustments:
    print('dy_sa = ', dy_sa, 'dy_sb=', dy_sb, 'dm_sa = ', dm_sa, 'dm_sb = ', dm_sb)


    # Make figures
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['mathtext.fontset'] = 'cm'


    fig1, axs = plt.subplots(ncols=1, nrows=3, figsize=(7.5, 10), sharex=True)
    plt.subplots_adjust(top=0.98, bottom=0.06, right=0.97, left=0.12, wspace=0.0, hspace=0.0)

    axs[2].errorbar(np.array(Sc_group['log_W']), np.array(Sc_group[mag_to_use]),
                    xerr=np.array(Sc_group['log_W_err']), yerr=np.array(Sc_group[magerr_to_use]),
                    color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[2].plot(np.array(Sc_group['log_W']), np.array(Sc_group[mag_to_use]), color='blue', ls='none',
                marker='.', alpha=1, label='Sc galaxies',
                zorder=1)
    # axs.plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[2].plot([1.7, 3], model_lin([slope_sc, intercept_sc], np.array([1.7, 3])), 'k',
                label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1* slope_sc,
                                                           intercept_sc) + r'$\log_{10}(W_{F50}))$', zorder=3)

    axs[1].errorbar(np.array(Sb_group['log_W']), np.array(Sb_group[mag_to_use]),
                    xerr=np.array(Sb_group['log_W_err']), yerr=np.array(Sb_group[magerr_to_use]),
                    color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[1].plot(np.array(Sb_group['log_W']), np.array(Sb_group[mag_to_use]), color='green', ls='none',
                marker='.', alpha=1, label='Sb galaxies',
                zorder=1)
    # axs.plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[1].plot([1.7, 3], model_lin([slope_sb_b, intercept_sb_b], np.array([1.7, 3])), 'k',
                label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope_sb_b,
                                                          intercept_sb_b) + r'$\log_{10}(W_{F50}))$', zorder=3)

    axs[0].errorbar(np.array(Sa_group['log_W']), np.array(Sa_group[mag_to_use]),
                    xerr=np.array(Sa_group['log_W_err']), yerr=np.array(Sa_group[magerr_to_use]),
                    color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[0].plot(np.array(Sa_group['log_W']), np.array(Sa_group[mag_to_use]), color='red',
                ls='none',
                marker='.', alpha=1, label='Sa galaxies',
                zorder=1)
    # axs.plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[0].plot([1.7, 3], model_lin([slope_sa_b, intercept_sa_b], np.array([1.7, 3])), 'k',
                label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope_sa_b,
                                                          intercept_sa_b) + r'$\log_{10}(W_{F50}))$', zorder=3)

    axs[0].set_xlim([1.8, 3])
    axs[0].set_ylim([-17, min(np.array(Sa_group[mag_to_use])) - 0.8])
    axs[1].set_ylim([-17,
                     min(np.array(Sb_group[mag_to_use])) - 0.8])
    axs[2].set_ylim([-17,
                     min(np.array(Sc_group[mag_to_use])) - 0.8])

    axs[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    axs[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    axs[2].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)

    axs[0].grid(b=True, which='both', color='0.65', linestyle=':')
    axs[1].grid(b=True, which='both', color='0.65', linestyle=':')
    axs[2].grid(b=True, which='both', color='0.65', linestyle=':')

    # axs[0].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=13, labelpad=2)
    # axs[1].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=13, labelpad=2)
    # axs[2].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=13, labelpad=2)
    axs[2].set_xlabel('$\log_{10}(W_{F50}))$', fontsize=17, labelpad=2)

    ax_invis = fig1.add_subplot(111, frameon=False)
    ax_invis.set_xticks([])
    ax_invis.set_yticks([])
    ax_invis.set_ylabel(r'Absolute Magnitude (vega mags), $\rm M_{\rm Tot}$', fontfamily='serif', fontsize=15, labelpad=40)

    plt.savefig('Stacked_Morphology_Before.pdf')

    fig2, axs = plt.subplots(ncols=1, nrows=3, figsize=(7.5, 10), sharex=True)
    plt.subplots_adjust(top=0.98, bottom=0.06, right=0.97, left=0.12, wspace=0.0, hspace=0.0)

    axs[2].errorbar(np.array(Sc_group['log_W']), np.array(Sc_group[mag_to_use]),
                    xerr=np.array(Sc_group['log_W_err']), yerr=np.array(Sc_group[magerr_to_use]),
                    color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[2].plot(np.array(Sc_group['log_W']), np.array(Sc_group[mag_to_use]), color='blue', ls='none',
                marker='.', alpha=1, label='Sc galaxies',
                zorder=1)
    # axs.plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[2].plot([1.7, 3], model_lin([slope_sc, intercept_sc], np.array([1.7, 3])), 'k',
                label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope_sc,
                                                          intercept_sc) + r'$\log_{10}(W_{F50}))$', zorder=3)

    axs[1].errorbar(np.array(Sb_group['log_W']), Sb_Mtot_corrected,
                    xerr=np.array(Sb_group['log_W_err']), yerr=np.array(Sb_group[magerr_to_use]),
                    color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[1].plot(np.array(Sb_group['log_W']), Sb_Mtot_corrected, color='green', ls='none',
                marker='.', alpha=1, label='Sb galaxies',
                zorder=1)
    # axs.plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[1].plot([1.7, 3], model_lin([slope_sb_a, intercept_sb_a], np.array([1.7, 3])), 'k',
                label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope_sb_a,
                                                          intercept_sb_a) + r'$\log_{10}(W_{F50}))$', zorder=3)

    axs[0].errorbar(np.array(Sa_group['log_W']), Sa_Mtot_corrected,
                    xerr=np.array(Sa_group['log_W_err']), yerr=np.array(Sa_group[magerr_to_use]),
                    color='grey', ls='none', markersize=0,
                    elinewidth=0.8, alpha=0.9, zorder=0)

    axs[0].plot(np.array(Sa_group['log_W']), Sa_Mtot_corrected, color='red',
                ls='none',
                marker='.', alpha=1, label='Sa galaxies',
                zorder=1)
    # axs.plot(x, y, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

    axs[0].plot([1.7, 3], model_lin([slope_sa_a, intercept_sa_a], np.array([1.7, 3])), 'k',
                label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope_sa_a,
                                                          intercept_sa_a) + r'$\log_{10}(W_{F50}))$', zorder=3)

    axs[0].set_xlim([1.8, 3])
    axs[0].set_ylim([-17, min(Sa_Mtot_corrected) - 0.8])
    axs[1].set_ylim([-17,
                     min(Sb_Mtot_corrected) - 0.8])
    axs[2].set_ylim([-17,
                     min(np.array(Sc_group[mag_to_use])) - 0.8])

    axs[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    axs[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
    axs[2].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)

    axs[0].grid(b=True, which='both', color='0.65', linestyle=':')
    axs[1].grid(b=True, which='both', color='0.65', linestyle=':')
    axs[2].grid(b=True, which='both', color='0.65', linestyle=':')

    # axs[0].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=13, labelpad=2)
    # axs[1].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=13, labelpad=2)
    # axs[2].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=13, labelpad=2)
    axs[2].set_xlabel('$\log_{10}(W_{F50}))$', fontsize=17, labelpad=2)

    ax_invis = fig2.add_subplot(111, frameon=False)
    ax_invis.set_xticks([])
    ax_invis.set_yticks([])
    ax_invis.set_ylabel(r'Absolute Magnitude (vega mags), $\rm M_{\rm Tot}$', fontfamily='serif', fontsize=15, labelpad=40)

    plt.savefig('Stacked_Morphology_After.pdf')


    return slope_corr, slope_corr_err, intercept_corr, intercept_corr_err, residuals_corr, scatter_coefficients_corr, rms_corr, Mtot_W1_morphcorr

