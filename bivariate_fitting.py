import numpy as np
from kapteyn import kmpfit



def bivariate_fit(log_W, log_W_err, Mtot_W1, Mtot_err_W1, slope_g, intercept_g, i_max):
    def model_lin(params, x):
        a, b = params
        return a * x + b

    def residuals_lin(params, data):
        x, xerr, y, yerr = np.array(data)
        a, b = params
        w = yerr ** 2 + (a ** 2 * xerr ** 2) + (scatter_coeffs[0] * x ** 2 + scatter_coeffs[1] * x ** 1 +
                                                scatter_coeffs[2]) ** 2
        return (y - model_lin(params, x)) / np.sqrt(w)

    # Set the intrinsic scatter to 0
    scatter_coeffs = [0, 0, 0]

    # Set the initial TF parameters (guess)
    params_i = [slope_g, intercept_g]

    # Set the initial Slope and Intercept (for calculating the diff 1st time)
    slope_fit = slope_g
    intercept_fit = intercept_g

    diff = 1
    i = 0
    # Perform the initial fit
    while diff > 10e-6:
        # Fit the parameters
        FitObj = kmpfit.Fitter(residuals=residuals_lin, data=[log_W, log_W_err, Mtot_W1, Mtot_err_W1])
        FitObj.fit(params0=params_i)

        # Calculate the diff
        diff = np.abs(slope_fit - FitObj.params[0])
        # print('diff=', diff)

        # Set slope and intercept as new values
        intercept_fit = FitObj.params[1]
        slope_fit = FitObj.params[0]

        intercept_fit_err = FitObj.stderr[1]
        slope_fit_err = FitObj.stderr[0]
        # print('slope=', slope, 'intercept=', intercept)

        # Calculate the residuals
        residuals = np.abs(Mtot_W1 - model_lin([slope_fit, intercept_fit], np.array(log_W)))

        # Calculate the total scatter
        rms = np.sqrt(np.sum(np.square(residuals))/(len(residuals) - 1))

        # Fit models to each scatter component.
        p_res = np.polyfit(log_W, residuals, deg=2)
        p_mag = np.polyfit(log_W, Mtot_err_W1, deg=2)
        p_HI = np.polyfit(log_W, np.array(log_W_err) * -1 * slope_fit, deg=3)

        # Calculate polynomial for the intrinsic scatter
        p_tot = np.polyfit(log_W, np.sqrt(
            np.polyval(p_res, np.array(log_W)) ** 2 - np.polyval(p_mag, np.array(log_W)) ** 2 - np.polyval(p_HI,
                                                                                                           np.array(
                                                                                                               log_W)) ** 2),
                           deg=2)

        i += 1

        if i > i_max:
            diff = 10e-8

        scatter_coeffs = p_tot

    print('iterations to convergence = ', i)

    return (slope_fit, intercept_fit, residuals, scatter_coeffs, rms, slope_fit_err, intercept_fit_err)
