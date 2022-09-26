import numpy as np
from kapteyn import kmpfit
import pandas as pd


def morhology(Data, scatter):
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

    # fit TF relation to each group using intrinsic scatter from previous FULL fit
    fit_sa = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sa')['log_W'], morph_groups.get_group('Sa')['log_W_err'],
                                 morph_groups.get_group('Sa')['M_biascorrected'],
                                 morph_groups.get_group('Sa')['M_biascorrected_err']])
    fit_sa.fit(params0=[-9, 1])

    fit_sb = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sb')['log_W'], morph_groups.get_group('Sb')['log_W_err'],
                                 morph_groups.get_group('Sb')['M_biascorrected'],
                                 morph_groups.get_group('Sb')['M_biascorrected_err']])
    fit_sb.fit(params0=[-9, 1])

    fit_sc = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sc')['log_W'], morph_groups.get_group('Sc')['log_W_err'],
                                 morph_groups.get_group('Sc')['M_biascorrected'],
                                 morph_groups.get_group('Sc')['M_biascorrected_err']])
    fit_sc.fit(params0=[-9, 1])

    parameters_before = [fit_sa.params, fit_sb.params, fit_sc.params]

    # Calculate the Sa and Sb corrections
    sa_correction = (fit_sc.params[1] - fit_sa.params[1]) + (fit_sc.params[0] - fit_sa.params[0]) * \
                    morph_groups.get_group('Sa')['log_W']
    sb_correction = (fit_sc.params[1] - fit_sb.params[1]) + (fit_sc.params[0] - fit_sb.params[0]) * \
                    morph_groups.get_group('Sb')['log_W']

    M_Sa = morph_groups.get_group('Sa')['M_biascorrected'] + sa_correction
    M_Sa_err = np.sqrt(morph_groups.get_group('Sa')['M_biascorrected_err'] ** 2 + (0.1 * np.abs(sa_correction)) ** 2)

    M_Sb = morph_groups.get_group('Sb')['M_biascorrected'] + sb_correction
    M_Sb_err = np.sqrt(morph_groups.get_group('Sb')['M_biascorrected_err'] ** 2 + (0.1 * np.abs(sb_correction)) ** 2)

    # Refit the Sa and Sb samples to make sure the correction is working properly
    fit_sa = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sa')['log_W'], morph_groups.get_group('Sa')['log_W_err'],
                                 M_Sa, M_Sa_err])
    fit_sa.fit(params0=[-9, 1])

    fit_sb = kmpfit.Fitter(residuals=residuals_lin,
                           data=[morph_groups.get_group('Sb')['log_W'], morph_groups.get_group('Sb')['log_W_err'],
                                 M_Sb, M_Sb_err])
    fit_sb.fit(params0=[-9, 1])

    # Recombine the samples
    recombined = pd.DataFrame(pd.concat([M_Sa, M_Sb, morph_groups.get_group('Sc')['M_biascorrected']]),
                              columns=['M_tot_corrected'])
    recombined['M_tot_corrected_err'] = pd.concat(
        [M_Sa_err, M_Sb_err, morph_groups.get_group('Sc')['M_biascorrected_err']])
    recombined.sort_index(axis=0, inplace=True)


    return recombined
