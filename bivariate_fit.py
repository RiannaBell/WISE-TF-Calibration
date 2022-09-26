import numpy as np
from kapteyn import kmpfit
from scipy import stats


def bivariate(logW, logW_err, M, M_err, i_max, polynomials):
    def model_lin(params, x):
        a, b = params
        return a * x + b

    def residuals_lin(params, data):
        x, xerr, y, yerr = np.array(data)
        a, b = params
        w = yerr ** 2 + (a ** 2 * xerr ** 2) + (sc[0] * x ** 2 + sc[1] * x ** 1 +
                                                sc[2]) ** 2
        return (y - model_lin(params, x)) / np.sqrt(w)

    # Initialise intrinsic scatter as 0.
    sc = [0, 0, 0]

    # Initialise guess for slope and intercept
    params_i = [-10, 1]

    # Initialise while loop params
    diff = 1
    i = 0

    # Set up while loop for iterative fitting
    while diff > 10e-8:
        # Fit the TF parameters
        FitObj = kmpfit.Fitter(residuals=residuals_lin, data=[logW, logW_err, M, M_err])
        FitObj.fit(params0=params_i)

        # Calculate the change in slope
        diff = np.abs(params_i[0] - FitObj.params[0])

        # reset initial params to be previous iteration
        params_i = FitObj.params

        # Calculate the residuals around the fitted TF relation
        residuals = np.abs(M - model_lin([FitObj.params[0], FitObj.params[1]], np.array(logW)))

        # Calculate the total rms scatter
        rms = np.sqrt(np.sum(np.square(residuals)) / (len(residuals) - 1))

        # Fit polynomial models to each scatter component.
        p_res = np.polyfit(logW, residuals, deg=2)
        p_mag = np.polyfit(logW, M_err, deg=2)
        p_HI = np.polyfit(logW, np.array(logW_err) * -1 * FitObj.params[0], deg=3)

        # Calculate polynomial for the intrinsic scatter using sum of squares
        p_intrinsic = np.polyfit(logW, np.sqrt(
            np.polyval(p_res, np.array(logW)) ** 2 - np.polyval(p_mag, np.array(logW)) ** 2 - np.polyval(p_HI, np.array(
                logW)) ** 2), deg=2)

        i += 1
        if i > i_max:
            diff = 10e-8

        sc = p_intrinsic

    intrinsic_s = np.sqrt(stats.binned_statistic(logW, residuals, 'mean', bins=5)[0] ** 2 - stats.binned_statistic(logW, np.polyval(p_mag, np.array(logW)),
                                                                        'mean', bins=5)[0] ** 2 - stats.binned_statistic(logW,
                                                                     np.polyval(p_HI, np.array(logW)),
                                                                     'mean', bins=5)[0] ** 2)
    print('iterations to convergence = ', i)

    if polynomials == 'True':
        return (FitObj.params[0], FitObj.stderr[0], FitObj.params[1], FitObj.stderr[1], residuals, sc, rms, intrinsic_s, [p_res, p_mag, p_HI, p_intrinsic])

    else:
        return (FitObj.params[0],FitObj.stderr[0], FitObj.params[1], FitObj.stderr[1], residuals, sc, rms, intrinsic_s)
