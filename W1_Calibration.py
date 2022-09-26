import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy.coordinates import SkyCoord
import astropy.units as u
from bivariate_fit import bivariate
from TF_plot import plot_TF
from Incompleteness_Bias_Correction import bias_calc
from Morphological_Correction import morhology
from Morphological_Distributions import morphology_plots
from results_analysis import analysis


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', 4)

# =====================================================================================================================
# Functions
# =====================================================================================================================

# Define the functions
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


# =====================================================================================================================
# ORGANISE WISE Data
# =====================================================================================================================
Photometry = pd.read_csv("../Input files/photometry.cvs", sep="\t")
Derived = pd.read_csv("../Input files/derived.cvs", sep="\t")
Data = pd.read_csv("../Input files/WISETFR.cvs", sep="\t")

WISE_Data = pd.DataFrame(Photometry["wxscdesig"])
WISE_Data.columns = ["2MASS_Name"]
WISE_Data['NedName'] = Data['NedName']
WISE_Data['Cluster_ID'] = Data["ClusterID"]

WISE_Data['Ra'] = Photometry['Ra']
WISE_Data['Dec'] = Photometry['Dec']
###
WISE_Data['Redshift'] = Photometry['zcmb']
WISE_Data['Morph_Type'] = Data['Morphology']
WISE_Data['Axis_Ratio'] = Photometry['ba']
WISE_Data['log_W'] = Data['logW']
WISE_Data['log_W_err'] = Data['logW_err']
WISE_Data['L_dist'] = Photometry['Dmpc']

WISE_Data['Flux_W1_mJy'] = Photometry['ftot_1']
WISE_Data['Flux_W1_err'] = Photometry['ftoterr_1']
WISE_Data['Flux_W2_mJy'] = Photometry['ftot_2']
WISE_Data['Flux_W2_err'] = Photometry['ftoterr_2']
# WISE_Data['mag_test'] = Derived['w1']
# WISE_Data['mag_test_err'] = Derived['w1err']
# WISE_Data['mag_test_err'] = Photometry['W1_best_err']


# =====================================================================================================================
# Save Positions to Separate file for Galactic extinction.
# =====================================================================================================================
Position_Data = WISE_Data[["Ra", "Dec"]]
Position_Data.columns = ["|   ra     ", "|   dec    |"]
Position_Data.to_csv('../Input Files/Position_Data.txt', sep='\t', header=True, index=False)

data = Table([list(WISE_Data["Ra"]), list(WISE_Data["Dec"])], names=['ra', 'dec'])
ascii.write(data, '../Input Files/Ra_Dec.tbl', overwrite=True, format='ipac')

# =====================================================================================================================
# Perform K-correction.
# =====================================================================================================================

# Import the K-correction files
Sa_Kcorr = pd.read_csv('../Input files/Sa.Kcorrection.txt', skiprows=35, sep='\s+')
Sb_Kcorr = pd.read_csv('../Input files/Sb.Kcorrection.txt', skiprows=35, sep='\s+')
Sc_Kcorr = pd.read_csv('../Input files/Sc.Kcorrection.txt', skiprows=35, sep='\s+')

# group WISE Data by morph type
morph_groups = WISE_Data.groupby('Morph_Type')

# apply k-correction to each group and convert to apparent magnitudes
magW1_sa = []
magW1_sa_err = []
for index, row in morph_groups.get_group('Sa').iterrows():
    # Find the value of the k-correction
    z_diff = np.abs(Sa_Kcorr['z'] - row['Redshift'])
    min_index = np.where(z_diff == min(z_diff))[0][0]

    # Apply the k-correction to the total flux
    f_corrected = row['Flux_W1_mJy'] * Sa_Kcorr['fW1'].iloc[min_index]
    f_err = row['Flux_W1_err'] * Sa_Kcorr['fW1'].iloc[min_index]

    # Calculate the apparent magnitude and append onto list.
    magW1_sa.append(-2.5 * np.log10(f_corrected * 10 ** (-3)) + 2.5 * np.log10(309.54))
    magW1_sa_err.append(np.sqrt(0.002 ** 2 + (2.5 / np.log(10)) ** 2 * (f_err / f_corrected) ** 2))

# Do the same for Sb and Sc
magW1_sb = []
magW1_sb_err = []
for index, row in morph_groups.get_group('Sb').iterrows():
    # Find the value of the k-correction
    z_diff = np.abs(Sb_Kcorr['z'] - row['Redshift'])
    min_index = np.where(z_diff == min(z_diff))[0][0]

    # Apply the k-correction to the total flux
    f_corrected = row['Flux_W1_mJy'] * Sb_Kcorr['fW1'].iloc[min_index]
    f_err = row['Flux_W1_err'] * Sb_Kcorr['fW1'].iloc[min_index]

    # Calculate the apparent magnitude and append onto list.
    magW1_sb.append(-2.5 * np.log10(f_corrected * 10 ** (-3)) + 2.5 * np.log10(309.54))
    magW1_sb_err.append(np.sqrt(0.002 ** 2 + (2.5 / np.log(10)) ** 2 * (f_err / f_corrected) ** 2))

magW1_sc = []
magW1_sc_err = []
for index, row in morph_groups.get_group('Sc').iterrows():
    # Find the value of the k-correction
    z_diff = np.abs(Sc_Kcorr['z'] - row['Redshift'])
    min_index = np.where(z_diff == min(z_diff))[0][0]

    # Apply the k-correction to the total flux
    f_corrected = row['Flux_W1_mJy'] * Sc_Kcorr['fW1'].iloc[min_index]
    f_err = row['Flux_W1_err'] * Sc_Kcorr['fW1'].iloc[min_index]

    # Calculate the apparent magnitude and append onto list.
    magW1_sc.append(-2.5 * np.log10(f_corrected * 10 ** (-3)) + 2.5 * np.log10(309.54))
    magW1_sc_err.append(np.sqrt(0.002 ** 2 + (2.5 / np.log(10)) ** 2 * (f_err / f_corrected) ** 2))

# Create single list of magnitudes
magsW1_k = magW1_sa + magW1_sb + magW1_sc
magsW1_err_k = magW1_sa_err + magW1_sb_err + magW1_sc_err

# Combine groups and add magnitudes to df
Data_Test = pd.concat([morph_groups.get_group('Sa'), morph_groups.get_group('Sb'), morph_groups.get_group('Sc')])
Data_Test['magW1_k'] = magsW1_k
Data_Test['magW1_err_k'] = magsW1_err_k
WISE_Data = Data_Test.sort_index(axis=0)

# =====================================================================================================================
# Perform galactic extinction correction.
# =====================================================================================================================
# Import Galactic Extinction corrections
EBV_new = pd.read_csv("../Input files/extinction_edited.csv", sep=",")

# Apply correction and append onto df
WISE_Data['magW1_kg'] = WISE_Data['magW1_k'] - 0.189 * EBV_new['E_B_V_SandF']
WISE_Data['magW1_err_kg'] = np.sqrt(WISE_Data['magW1_err_k'] ** 2 + (0.189 * EBV_new['stdev_E_B_V_SandF']) ** 2)

# Remove Outliers here
outliers = ['2MASXJ00410364+3143576', '2MASXJ12163954+4605147', '2MASXJ12574995+2939154', '2MASXJ12562853+2717280']
index_2remove = [np.where(outliers[i] == WISE_Data["2MASS_Name"])[0][0] for i in range(len(outliers))]
WISE_Data = WISE_Data.drop(labels=index_2remove, axis=0, inplace=False).reset_index()
WISE_Data = WISE_Data.drop('index', axis=1)

# =====================================================================================================================
# Make sample distribution figure
# =====================================================================================================================
Groups = WISE_Data.groupby("Cluster_ID")

# Get an array of cluster ID's
Ids = list(Groups.groups.keys())

plt.rcParams.update({'font.size': 8})
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='hammer')  # aitoff
plt.subplots_adjust(top=0.95, bottom=0.02, right=0.95, left=0.08)

colours = ['k', 'grey', 'mediumpurple', 'firebrick', 'hotpink', 'mediumseagreen', 'lightseagreen', 'steelblue', 'r',
           'maroon', 'royalblue',
           'mediumblue', 'mediumpurple', 'darkorange', 'darkorchid', 'purple', 'mediumvioletred', 'hotpink', 'k',
           'grey', 'darkorchid', 'firebrick',
           'darkorange', 'forestgreen', 'mediumseagreen', 'r', 'forestgreen', 'dodgerblue', 'royalblue',
           'mediumblue', 'b', 'darkorchid', 'purple', 'mediumvioletred', 'hotpink']

r_thing = [-0.38, 0.15, -0.15, 0.04, 0.04,
           -0.45, -0.02, 0.05, 0.09, -0.03,
           -0.22, 0.02, -0.48, -0.35, 0.2,
           -0.43, 0.2, 0.1, -0.2, 0.09,
           -0.2, 0.1, -0.4, -0.0, 0.,
           -0.15, -0.41, 0.04, 0.06, -0.6, -1.45]
d_thing = [-0.07, 0.05, -0.14, -0.07, 0.08,
           0.06, 0.05, -0.04, -0.03, 0.025,
           -0.18, 0.02, -0.04, 0.05, -0.00,
           -0.05, 0.05, -0.03, -0.13, -0.03,
           -0.07, -0.03, 0.06, 0.04, 0.03,
           -0.18, 0.045, -0.06, -0.04, 0.06, 0.08]

for i, Id in enumerate(Ids):
    group = Groups.get_group(Id)
    ra = np.average(group['Ra'])
    dec = np.average(group['Dec'])

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    c_gal = c.galactic
    r = c_gal.l.radian

    if r > np.pi:
        r -= 2. * np.pi
    d = c_gal.b

    size = round(len(group))
    ax.text(r + r_thing[i], d.radian + d_thing[i], r'{}'.format(Id), fontsize=9, color=colours[i])
    ax.plot(r, d.radian, 'o', markersize=size / 3, color=colours[i], markeredgecolor=colours[i])

ax.grid(visible=True, which='both', color='0.65', linestyle=':')
ax.set_ylabel('Galactic Latitude', fontsize=10, labelpad=1)
ax.set_xlabel('Galactic Longitude', fontsize=10, labelpad=+4)
plt.savefig('Figures/SampleDistribution.pdf')
plt.clf()
plt.close()

# =====================================================================================================================
# Perform preliminary fit.
# =====================================================================================================================
# Calculate and save the absolute magnitudes
WISE_Data['MW1_nocorr'] = WISE_Data['magW1_kg'] - (5. * np.log10(WISE_Data['L_dist'] * 10 ** 6) - 5.)

# Perform preliminary fit
slope_prelim, slope_prelim_err, intercept_prelim, intercept_prelim_err, res_array_prelim, scatter_coeffs_prelim, rms_prelim, intrinsic_binned = bivariate(
    WISE_Data['log_W'], WISE_Data['log_W_err'], WISE_Data['MW1_nocorr'], WISE_Data['magW1_err_kg'], 500, "False")
results_prelim = [slope_prelim, slope_prelim_err, intercept_prelim, intercept_prelim_err, scatter_coeffs_prelim,
                  rms_prelim]

# plot preliminary results
plot_TF(WISE_Data['log_W'], WISE_Data['log_W_err'], WISE_Data['MW1_nocorr'], WISE_Data['magW1_err_kg'], slope_prelim,
        slope_prelim_err, intercept_prelim, intercept_prelim_err, res_array_prelim, scatter_coeffs_prelim,
        intrinsic_binned, 'Figures/PreliminaryFit_W1.pdf', 'W1 Band - Raw Sample')

# =====================================================================================================================
# Incompleteness Bias Correction
# =====================================================================================================================
bin_adjustments = [-1, 3, 0, 0, 1,
                   0, -1, -1, 0, 0,
                   2, 0, 0, -1, 0,
                   0, 6, 0, 1, 0,
                   3, -1, 1, 0, 0,
                   0, 0, -1, 0, 1, 0]

#reduce 2 from 9
# 0,7, 8
LF_adjustments = [[0], [0, 1, 2], [], [0], [0], [0, 1],
                  [], [0], [], [], [0], [],
                  [0,1], [], [0], [0], [0, 1], [0],
                  [0], [1], [0], [0], [0, 1], [0],
                  [], [], [], [], [], [0, 1], []]

### back to nothign for 3
CF_adjustments = [[], [], [], [3, -1], [3, 4], [2, 3],
                  [], [-2], [-3], [], [3, 4], [],
                  [], [1], [], [3], [4], [2, 3],
                  [], [], [], [2], [3, 4], [4],
                  [3], [3], [4, 5], [], [], [4, -2], []]


bias_correction = bias_calc("Figures/Bias_W1.pdf", WISE_Data, "MW1_nocorr", "magW1_err_kg", slope_prelim,
                            intercept_prelim, scatter_coeffs_prelim,
                            bin_adjustments, LF_adjustments, CF_adjustments, 100000)

# Add the corrected data onto the df.
WISE_Data['M_biascorrected'] = bias_correction['MW1_nocorr']
WISE_Data['M_biascorrected_err'] = bias_correction['M_BiasCorrected_err']
WISE_Data['bias'] = bias_correction['bias']

# drop galaxies with extremely large bias corrections
drop = []
for index, row in WISE_Data.iterrows():
    if np.abs(row['bias']) > np.abs(0.05 * row['MW1_nocorr']):
        drop.append(index)

# drop = []
# for index, row in WISE_Data.iterrows():
#     if np.abs(row['bias']) >= np.abs(0.5 * row['MW1_nocorr']):
#         drop.append(index)

Cut_Data = WISE_Data.drop(index=drop, inplace=False)
print(len(Cut_Data))
# WISE_Data = WISE_Data.reset_index(drop=True, inplace=False)


# refit the TF relation to the debiased sample
slope_debiased, slope_debiased_err, intercept_debiased, intercept_debiased_err, res_array_debiased, scatter_coeffs_debiased, rms_debiased, intrinsic_binned_debiased = bivariate(
    Cut_Data['log_W'], Cut_Data['log_W_err'], Cut_Data['M_biascorrected'], Cut_Data['M_biascorrected_err'], 500,
    "False")

results_debiased = [slope_debiased, slope_debiased_err, intercept_debiased, intercept_debiased_err,
                    scatter_coeffs_debiased, rms_debiased]

plot_TF(Cut_Data['log_W'], Cut_Data['log_W_err'], Cut_Data['M_biascorrected'], Cut_Data['M_biascorrected_err'],
        slope_debiased, slope_debiased_err, intercept_debiased, intercept_debiased_err, res_array_debiased,
        scatter_coeffs_debiased, intrinsic_binned_debiased, 'Figures/Debiased_Fit.pdf', 'W1 Band - Bias Corrected Sample')

# apply the morphological correction
morphological_correction = morhology(Cut_Data, scatter_coeffs_debiased)

# Add the corrected data onto the df.
Cut_Data['M_FINAL'] = morphological_correction['M_tot_corrected']
Cut_Data['M_Final_err'] = morphological_correction['M_tot_corrected_err']

# create morphological analysis figures
morphology_plots(Cut_Data, 'MW1_nocorr', 'magW1_err_kg', scatter_coeffs_prelim,
                 'Figures/Morphological_Distributions.pdf')

# Refit the TF relation to the fully-corrected sample
slope_final, slope_final_err, intercept_final, intercept_final_err, res_array_final, scatter_coeffs_final, rms_final, intrinsic_binned_final, polynomials = bivariate(
    Cut_Data['log_W'], Cut_Data['log_W_err'], Cut_Data['M_FINAL'], Cut_Data['M_Final_err'], 500, "True")
results_final = [slope_final, slope_final_err, intercept_final, intercept_final_err, scatter_coeffs_final, rms_final]

plot_TF(Cut_Data['log_W'], Cut_Data['log_W_err'], Cut_Data['M_FINAL'], Cut_Data['M_Final_err'], slope_final,
        slope_final_err, intercept_final, intercept_final_err, res_array_final, scatter_coeffs_final,
        intrinsic_binned_final, 'Figures/Final_Fit.pdf', 'W1 Band - Fully Calibrated Sample')

# Create figures to analyse results
analysis(Cut_Data, slope_final, intercept_final, 'M_FINAL', 'M_FINAL_err', polynomials)

# Combine data into a single table (including galaxies dropped after bias correction)
M_final = Cut_Data['M_FINAL']
M_final_err = Cut_Data['M_Final_err']

M_final = pd.concat([M_final, pd.Series(['nan'] * len(drop), index=drop)])
M_final.sort_index(axis=0, inplace=True)

M_final_err = pd.concat([M_final_err, pd.Series(['nan'] * len(drop), index=drop)])
M_final_err.sort_index(axis=0, inplace=True)

WISE_Data['M_FINAL'] = M_final
WISE_Data['M_FINAL_err'] = M_final_err

WISE_Data.to_csv('Figures/Data.csv', sep='\t', na_rep='na', float_format='%.4f',
                 index=False)

# Save results to text file
results = pd.DataFrame([results_prelim, results_debiased, results_final],
                       columns=['slope', 'slope_err', 'intercept', 'intercept_err', 'scatter coefficients',
                                'rms scatter'])
results.to_csv('Figures/Results.csv', sep='\t', na_rep='na', float_format='%.4f',
               index=True)
print(results)

exit()
print(results_debiased)

exit()

exit()

m = -2.5 * np.log10(WISE_Data['mag_W1_check'] * 10 ** (-3)) + 2.5 * np.log10(309.54)
M = (m - (5. * np.log10(WISE_Data['L_dist'] * 10 ** 6) - 5.)).values.tolist()

m2 = -2.5 * np.log10(WISE_Data['Flux_W1_mJy'] * 10 ** (-3)) + 2.5 * np.log10(309.54)
M2 = (m2 - (5. * np.log10(WISE_Data['L_dist'] * 10 ** 6) - 5.)).values.tolist()

plt.plot(WISE_Data['log_W'], M, 'r.')
plt.plot(WISE_Data['log_W'], M2, 'b.')

# for i, txt in enumerate(WISE_Data['2MASS_Name'].iloc[0:31]):
#     plt.annotate(txt, (WISE_Data['log_W'].iloc[i], M[i]))

plt.gca().invert_yaxis()
plt.show()
