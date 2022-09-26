import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from astroML.plotting import scatter_contour


def analysis(Data, slope, intercept, mag, mag_err, polynomials):

    def model_lin(params, x):
        a, b = params
        return a * x + b

    # Create figure 1 to plot scatter contours
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Plot the various scatter contributions
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(7, 7), sharey=False)
    plt.subplots_adjust(top=0.98, bottom=0.09, right=0.95, left=0.1, wspace=0.1, hspace=0.05)

    x = np.linspace(2.1, 2.85, 1000)
    axs.plot(x, np.polyval(polynomials[3], np.array(x)), ls='--', label=r'$\xi$' , zorder=2, color='r')
    axs.plot(x, np.polyval(polynomials[0], np.array(x)), label=r'$p_{\sigma}$')
    axs.plot(x, np.polyval(polynomials[1], np.array(x)), label=r'$p_{\Delta_M}$')
    axs.plot(x, np.polyval(polynomials[2], np.array(x)), label=r'$p_{\Delta_{HI}}$')

    axs.text(2.22, 1.345, ' = {0:0.2f}'.format(polynomials[3][0]) + r'$\log_{10}(W)^2$' + '- {0:0.2f}'.format(-1*polynomials[3][1]) + r'$\log_{10}(W)$' + ' + {0:0.2f}'.format(polynomials[3][2]), fontsize=9)
    axs.text(2.22, 1.28, ' = {0:0.2f}'.format(polynomials[0][0]) + r'$\log_{10}(W)^2$' + ' - {0:0.2f}'.format(-1*polynomials[0][1]) + r'$\log_{10}(W)$' + '+ {0:0.2f}'.format(polynomials[0][2]), fontsize=9)
    axs.text(2.22, 1.22, ' = {0:0.2f}'.format(polynomials[1][0]) + r'$\log_{10}(W)^2$' + ' - {0:0.2f}'.format(-1*polynomials[1][1]) + r'$\log_{10}(W)$' + '+ {0:0.2f}'.format(polynomials[1][2]), fontsize=9)
    axs.text(2.22, 1.158, ' = {0:0.2f}'.format(polynomials[2][0]) + r'$\log_{10}(W)^3$' + '+ {0:0.2f}'.format(polynomials[2][1]) + r'$\log_{10}(W)^2$' + ' - {0:0.2f}'.format(-1*polynomials[2][2]) + r'$\log_{10}(W)$' + '+ {0:0.2f}'.format(polynomials[2][3]), fontsize=9)

    axs.set_ylabel('Scatter (vega mags)', fontfamily='serif', fontsize=15, labelpad=2)
    axs.set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)

    axs.set_ylim([0, 1.41])
    axs.set_xlim([2.1, 2.85])

    axs.legend(loc='upper left', frameon=False, facecolor='white', framealpha=1)
    axs.grid('visible', which='both', color='0.65', linestyle=':')
    plt.savefig('Figures/Scatter_contributions.pdf')
    plt.clf()
    plt.close()

    # Plot scatter contours against independent variables
    fig = plt.figure(tight_layout=True, figsize=(7, 12))
    axs = fig.add_subplot(211)

    # Calculate the residuals
    scatter = -1 * (Data[mag] - model_lin([slope, intercept], Data['log_W']))
    ###
    scatter_contour(Data['Axis_Ratio'], scatter, threshold=9, log_counts=True, ax=axs,
                    histogram2d_args=dict(bins=17),
                    plot_args=dict(marker=',', linestyle='none', color='black'),
                    contour_args=dict(cmap=plt.cm.BuPu_r))

    axs.plot([0.1, 1], [0, 0], c='black')
    axs.set_xlim([0.1, 0.95])
    axs.grid('visible', which='both', color='0.65', linestyle=':')

    axs.set_xlabel(r'Axis Ratio $\left(\frac{b}{a}\right)$, measured in the W1 Band')
    axs.set_ylabel(r'Residuals (vega mags)')
    ###
    axs2 = fig.add_subplot(212)
    scatter_contour(Data['Redshift'], scatter, threshold=7, log_counts=True, ax=axs2,
                    histogram2d_args=dict(bins=22),
                    plot_args=dict(marker=',', linestyle='none', color='black'),
                    contour_args=dict(cmap=plt.cm.BuPu_r))

    axs2.plot([0, 0.035], [0, 0], c='black')
    axs2.set_xlim([0, 0.035])
    axs2.grid('visible', which='both', color='0.65', linestyle=':')
    axs2.set_xlabel(r'Redshift, $z$')
    axs2.set_ylabel(r'Residuals (vega mags)')

    plt.savefig('Figures/ScatterContours.pdf')


    # Perform "test" calculations of distance and H0 using calibrated relation
    c = 3 * 10 ** 5
    q0 = -0.55

    def TF_dist(logW, m):
        return 10 ** ((m - (slope * logW + intercept)) / 5 + 1) / 10 ** 6

    def H0(logW, m, z):
        v = c * z * (1 + 1 / 2 * (1 - q0) * z)
        Dl = 10 ** ((m - (slope * logW + intercept)) / 5 + 1) / 10 ** 6
        return v / Dl


    # Plot of H0 against redshift using contours
    fig2 = plt.figure(tight_layout=True, figsize=(7, 7))
    axs2 = fig2.add_subplot(111)

    scatter_contour(Data['Redshift'], H0(Data['log_W'], Data['magW1_kg'], Data['Redshift']), threshold=15,
                    log_counts=True, ax=axs2,
                    histogram2d_args=dict(bins=12),
                    plot_args=dict(marker=',', linestyle='none', color='black'),
                    contour_args=dict(cmap=plt.cm.BuPu_r))
    axs2.set_xlabel('Redshift, z')
    axs2.set_ylabel(r'$H_0$ (km/s/Mpc)')
    plt.savefig('Figures/H0_Contours.pdf')
    plt.clf()
    plt.close()

    fig3, axs3 = plt.subplots(ncols=1, nrows=1, figsize=(7, 7), sharey=False)
    plt.subplots_adjust(top=0.98, bottom=0.09, right=0.97, left=0.11, wspace=0.1, hspace=0.05)

    axs3.plot(Data['log_W'], TF_dist(Data['log_W'], Data['magW1_kg']), color='indigo', ls='none', marker='.', markersize=4,
             alpha=1)
    axs3.grid('visible', which='both', color='0.65', linestyle=':')
    axs3.set_xlabel(r'$\log_{10}(W)$')
    axs3.set_ylabel(r'Luminosity Distance (Predicted by the calibrated TF relation)')
    plt.savefig('Figures/L_Dist vs log(W).pdf')

    fig4, axs4 = plt.subplots(ncols=1, nrows=1, figsize=(7, 7), sharey=False)
    plt.subplots_adjust(top=0.98, bottom=0.09, right=0.97, left=0.11, wspace=0.1, hspace=0.05)

    axs4.plot(Data['Axis_Ratio'], TF_dist(Data['log_W'], Data['magW1_kg']), color='indigo', ls='none', marker='.',
             markersize=4, alpha=1)
    axs4.plot(Data['Axis_Ratio'], np.mean(TF_dist(Data['log_W'], Data['magW1_kg'])) * np.ones(len(Data)), 'k')

    axs4.grid('visible', which='both', color='0.65', linestyle=':')
    axs4.set_xlabel(r'Axis Ratio $\left(\frac{b}{a}\right)$, measured in the W1 Band')
    axs4.set_ylabel(r'Luminosity Distance (Predicted by the calibrated TF relation)')
    plt.savefig('Figures/L_Dist vs AxisRatio.pdf')
    plt.clf()
    plt.close()