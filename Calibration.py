import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy.coordinates import SkyCoord
import astropy.units as u
from kapteyn import kmpfit
import matplotlib.pylab as plt
import itertools
from bivariate_fitting import bivariate_fit
from IncompletenessBias_func import bias_calc
from morphological_correction import morphological_correction


# =====================================================================================================================
# ORGANISE WISE Data
# =====================================================================================================================

Photometry = pd.read_csv("Raw WISE Data/WISE_Photometry.tbl", sep="\s+", skiprows=[0, 1, 2], header=None)
Photometry_Header = pd.read_csv("Raw WISE Data/WISE_Photometry.tbl", sep="|", header=None, nrows=1)

Derived = pd.read_csv("Raw WISE Data/WISE_Derived.tbl", sep="\s+", skiprows=[i for i in range(0, 27)], header=None)
Derived_Header = pd.read_csv("Raw WISE Data/WISE_Derived.tbl", sep="|", skiprows=[i for i in range(0, 24)], header=None,
                             nrows=1)

Data = pd.read_csv("Raw WISE Data/WISETFR.txt", sep="\s+", skiprows=[0], header=None)
Data_Header = pd.read_csv("Raw WISE Data/WISETFR.txt", sep="\s+", nrows=1, header=None)

list_3 = list(Data_Header.iloc[0])
list_1 = list(Photometry_Header.iloc[0])
list_2 = list(Derived_Header.iloc[0])

Photometry.columns = list_1
Derived.columns = list_2
Data.columns = list_3

WISE_Data = pd.DataFrame(Photometry["desig"])
WISE_Data.columns = ["2MASS_Name"]
WISE_Data['NedName'] = Data['Nedname']
WISE_Data['Abell_ID'] = Data["clusterid"]

WISE_Data['Ra'] = Photometry['ra']
WISE_Data['Dec'] = Photometry['dec']
WISE_Data['Redshift'] = Photometry['redshift']
WISE_Data['Morph_Type'] = Data['TF_type']
WISE_Data['Axis_Ratio'] = Photometry['ba']
WISE_Data['log_W'] = Data['log(w)']
WISE_Data['log_W_err'] = Data['err_log(w)']
WISE_Data['L_dist'] = Photometry['Dmpc']

WISE_Data['Radius_tot_W1'] = Photometry['Rtot_1']
WISE_Data['Radius_tot_W2'] = Photometry['Rtot_2']

WISE_Data['Flux_W1_mJy'] = Photometry['ftot_1']
WISE_Data['Flux_W1_err'] = Photometry['ftoterr_1']
WISE_Data['Flux_W2_mJy'] = Photometry['ftot_2']
WISE_Data['Flux_W2_err'] = Photometry['ftoterr_2']

WISE_Data.to_csv('Data/WISE_Data.txt', sep='\t', header=True, index=False)

Position_Data = WISE_Data[["Ra", "Dec"]]
Position_Data.columns = ["|   ra     ", "|   dec    |"]
Position_Data.to_csv('Data/Position_Data.txt', sep='\t', header=True, index=False)

data = Table([list(WISE_Data["Ra"]), list(WISE_Data["Dec"])], names=['ra', 'dec'])
ascii.write(data, 'Data/Ra_Dec.tbl', overwrite=True, format='ipac')

# ======================================================================================================================
# K-Correction & Galactic Extinction
# =====================================================================================================================

# Import K-Corrections
K_Sa_Data = pd.read_csv("Raw WISE Data/Sa.Kcorrection.txt", sep="\s+", skiprows=[i for i in range(0, 36)], header=None)
K_Sb_Data = pd.read_csv("Raw WISE Data/Sb.Kcorrection.txt", sep="\s+", skiprows=[i for i in range(0, 36)], header=None)
K_Sc_Data = pd.read_csv("Raw WISE Data/Sc.Kcorrection.txt", sep="\s+", skiprows=[i for i in range(0, 36)], header=None)

K_Sa_Header = pd.read_csv("Raw WISE Data/Sa.Kcorrection.txt", sep="|", skiprows=[i for i in range(0, 35)], nrows=1,
                          usecols=range(1, 33))
list_Sa = []

for i in range(32):
    list_Sa.append(str(list(K_Sa_Header)[i]))

K_Sa_Data.columns = list_Sa
K_Sb_Data.columns = list_Sa
K_Sc_Data.columns = list_Sa

# Import Galactic Extinction corrections
EBV = pd.read_csv("Raw WISE Data/Galactic_Extinction.tbl.txt", sep="\s+", header=None,
                  skiprows=[i for i in range(0, 16)])
EBV_header = pd.read_csv("Raw WISE Data/Galactic_Extinction.tbl.txt", sep="|", skiprows=[i for i in range(0, 13)],
                         nrows=1,
                         usecols=range(2, 19))

list_EBV = []
for i in range(15):
    list_EBV.append(str(list(EBV_header)[i]))

EBV.columns = list_EBV

# Get the Misc Data
misc_data = open("Raw WISE Data/misc.txt", "r")
misc_line = misc_data.readlines()[1]
misc = misc_line.split()

mag_W1 = []
mag_W2 = []
mag_W1err = []
mag_W2err = []
f = []

for i, row in WISE_Data.iterrows():

    if str(row['Morph_Type']) == 'Sa':
        z_list = np.array(K_Sa_Data['z'])
        z_diff = abs(z_list - row['Redshift'])
        z_index = np.where(z_diff == min(abs(z_list - row['Redshift'])))

        K_corr_W1 = np.array(K_Sa_Data['fW1'])[z_index[0][0]]
        K_corr_W2 = np.array(K_Sa_Data['fW2'])[z_index[0][0]]

    if str(row['Morph_Type']) == 'Sb':
        z_list = np.array(K_Sb_Data['z'])
        z_diff = abs(z_list - row['Redshift'])
        z_index = np.where(z_diff == min(abs(z_list - row['Redshift'])))

        K_corr_W1 = np.array(K_Sb_Data['fW1'])[z_index[0][0]]
        K_corr_W2 = np.array(K_Sb_Data['fW2'])[z_index[0][0]]

    if str(row['Morph_Type']) == 'Sc':
        z_list = np.array(K_Sc_Data['z'])
        z_diff = abs(z_list - row['Redshift'])
        z_index = np.where(z_diff == min(abs(z_list - row['Redshift'])))

        K_corr_W1 = np.array(K_Sc_Data['fW1'])[z_index[0][0]]
        K_corr_W2 = np.array(K_Sc_Data['fW2'])[z_index[0][0]]

    flux_W1_k = (row['Flux_W1_mJy'] * K_corr_W1) * 10 ** -3 / float(misc[12])  # in Janskys
    flux_W2_k = (row['Flux_W2_mJy'] * K_corr_W2) * 10 ** -3 / float(misc[13])  # in Janskys

    flux_W1_k_err = (row['Flux_W1_err'] * K_corr_W1) * 10 ** -3 / float(misc[12])  # in Janskys
    flux_W2_k_err = (row['Flux_W2_err'] * K_corr_W2) * 10 ** -3 / float(misc[13])  # in Janskys

    # Calculate the total corrected magnitude
    mag_W1.append(
        float(misc[4]) - 2.5 * np.log10(flux_W1_k) - float(misc[0]) * EBV['E_B_V_SandF       '][i])
    mag_W2.append(
        float(misc[6]) - 2.5 * np.log10(flux_W2_k) - float(misc[1]) * EBV['E_B_V_SandF       '][i])

    mag_W1err.append(
        np.sqrt((float(misc[5]) ** 2 + 1.179 * (row['Flux_W1_err'] / row['Flux_W1_mJy']) ** 2) + (
                float(misc[0]) * EBV['stdev_E_B_V_SandF '][i]) ** 2))
    mag_W2err.append(
        np.sqrt((float(misc[7]) ** 2 + 1.179 * (row['Flux_W2_err'] / row['Flux_W2_mJy']) ** 2) + (
                float(misc[1]) * EBV['stdev_E_B_V_SandF '][i]) ** 2))

# Save the magnitudes to the WISE_Data file
WISE_Data['mag_W1'] = mag_W1
WISE_Data['mag_W1_err'] = mag_W1err
WISE_Data['mag_W2'] = mag_W2
WISE_Data['mag_W2_err'] = mag_W2err


# ======================================================================================================================
# Assign the Galaxies to their groups and save this to the Data file.
# ======================================================================================================================

# Import Master's ID's
Masters_Table4 = pd.read_table("Data/Cluster ID Data/Masters_Table4.dat.txt", sep="\s+")
Masters_Table4.columns = ["ID", "Name", 'HRA', 'MRA', 'SRA', 'DDec', 'AmDec', 'AsDec', 'Morph', 'LogW',
                          'LogW_err', 'mag_I', 'mag_i_Corr', 'Mag_I', 'Mag_I_err', 'Extc', 'Inc', 'Vec', 'Ang_Sep',
                          'Cluster Name', 'spec']

Masters_2008 = pd.read_csv("Data/Cluster ID Data/Masters_2008.txt", delimiter="\s+", header=None)
Masters_2008.columns = ['ID', 'ra', 'dec', 'dist', 'log(w)', 'lgwer', 'M', 'Morph_Corr', 'm', 'm_err', 'inc',
                        'Morph_Type', 'CMB_vel', 'gal_vec', 'Source', 'Abell Id', 'Bias', 'Bias_dist', 'Bias_size',
                        'Cluster_vpec', 'SemiMajorAxis', 'radius', 'Rot_source', 'Width', 'ORC_Corr', 'HI_Corr', 'yeet']

# Import f-o-f rough Ids
Data_IDs = open("Data/Cluster ID Data/Khaled_ClusterIds.txt", "r")
GalaxyName_K = []  # List for the galaxy names as given by the NEW file
Cluster_Id_K = []  # List of the corresponding CORRECT Cluster ID's

# Put rough information into lists
for line in Data_IDs:
    GalaxyName_K.append(str(line.split()[0]))
    Cluster_Id_K.append(str(line.split()[1]))

# Cross match between the galaxies in the WISE_Data file and the Galaxies in rough file
GalaxyName_WD = np.reshape(np.array(WISE_Data['2MASS_Name'].values.tolist()), (888, 1))
Cluster_Id_WD = np.reshape(np.array(WISE_Data['Abell_ID'].values.tolist()), (888, 1))

Cluster_Id_prelim = [];
No_Match_Galaxies = ['No Matches'];
Two_Match_Galaxies = ['Two Matches'];

# For loop to run through each galaxy in ORIGINAL list and find its cluster id.
for i, name in enumerate(GalaxyName_WD):
    index = np.where(np.array(GalaxyName_K) == name)

    if np.shape(index[0])[0] == 1:
        Cluster_Id_prelim.append(Cluster_Id_K[index[0][0]])

    if np.shape(index[0])[0] == 2:
        Two_Match_Galaxies.append(name[0])
        Cluster_Id_prelim.append('Two Matches')

    if np.shape(index[0])[0] == 0:
        No_Match_Galaxies.append(name[0])
        Cluster_Id_prelim.append('No Matches')

# Make galaxy ids from table 4 into a list
GalaxyName_T4 = np.array(Masters_Table4['ID'].values.tolist())
Cluster_Id_T4 = np.array(Masters_Table4['Cluster Name'].values.tolist())

# Make an empty list to put the cluster ID's from Masters2008 in
Cluster_Id_M08 = []

# For every row in Masters2008 identify the galaxy name and identify the cluster ID (from table4)
i = 0
for index, row in Masters_2008.iterrows():

    # Get the Cluster ID's from Masters2008
    GalaxyName_M08 = Masters_2008["ID"][index]

    # Find the index in T4 where the 2008 ID matches the T4 ID.
    T4_name_index = np.where(GalaxyName_T4 == GalaxyName_M08)[0]

    if T4_name_index.size > 0:
        Cluster_Id_M08.append(Cluster_Id_T4[T4_name_index[0]])

    else:
        i += 1
        Cluster_Id_M08.append('Nan')

# Add a column to Masters2008 with the cluster ID's
Masters_2008['Cluster Names'] = Cluster_Id_M08

# Make groups for each identified cluster
ClusterGroups_M08 = Masters_2008.groupby("Cluster Names")

ClusterNames = list(ClusterGroups_M08.groups.keys())
ClusterNames.remove('Nan')

# Make an empty array to fill with the cluster names for each galaxy in WISE
Abell_Id_lists = []

# For each Cluster get the corresponding Abell ID's
for i, name in enumerate(ClusterNames):
    if name != 'Nan':
        # Get the group of galaxies for each cluster name
        Name = ClusterGroups_M08.get_group(name)

        # Identify the Abell ID's of the galaxies with that name
        Ids_inCluster = Name.groupby("Abell Id")

        # Put these Abell ID's into a list for each Cluster
        Abell_Id_lists.append(list(Ids_inCluster.groups.keys()))

# Make an empty array to put the cluster ID for each galaxy in the WISE sample in
WISE_ClusterID = []

for j, row in WISE_Data.iterrows():

    # Get the Abell ID for each galaxy
    Abell_Id = WISE_Data["Abell_ID"][j]

    for i, name in enumerate(ClusterNames):

        # If the Abell ID is in the list for each cluster then add the cluster name to the array
        if Abell_Id in Abell_Id_lists[i] and Abell_Id != 10301:
            WISE_ClusterID.append(name)

    # If the Abell ID is not in any of the lists for the clusters use Khaled's cluster ID
    if Abell_Id not in list(itertools.chain.from_iterable(Abell_Id_lists)):
        # print(Abell_Id, Cluster_Id_prelim[index])
        WISE_ClusterID.append(Cluster_Id_prelim[j])

    # If the ID is 10301 use Khaled's ID's
    if Abell_Id == 10301:
        # print(Abell_Id, Cluster_Id_prelim[j])
        WISE_ClusterID.append(Cluster_Id_prelim[j])

# Add columns for rough ID's and the Identified ID's
WISE_Data["Cluster ID"] = WISE_ClusterID

# Plots to Check if ID's are correct
Groups1 = WISE_Data.groupby("Cluster ID")

# Get an array of cluster ID's
Ids = list(Groups1.groups.keys())

# Make a figure
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

for m, Id in enumerate(Ids):
    group1 = Groups1.get_group(Id)
    ra1 = group1['Ra']
    dec1 = group1['Dec']
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c_gal1 = c1.galactic
    r1 = c_gal1.l.radian
    for k, rval in enumerate(r1):
        if rval > np.pi:
            r1[k] = rval - 2. * np.pi
    d1 = c_gal1.b
    ax.plot(r1, d1.radian, '.', color=colours[m], markersize=3)

ax.grid(visible=True, which='both', color='0.65', linestyle=':')
ax.set_ylabel('Galactic Latitude', fontsize=10, labelpad=1)
ax.set_xlabel('Galactic Longitude', fontsize=10, labelpad=+4)

plt.savefig('EvaluationOfClusterIDs.pdf')
plt.close()

# ======================================================================================================================
# Remove Outliers
# ======================================================================================================================

outliers = ['2MASXJ00410364+3143576', '2MASXJ12163954+4605147']
index_2remove = [np.where(outliers[i] == WISE_Data["2MASS_Name"])[0][0] for i in range(len(outliers))]
WISE_Data = WISE_Data.drop(labels=index_2remove, axis=0, inplace=False).reset_index()
WISE_Data = WISE_Data.drop('index', axis=1)


# ======================================================================================================================
# Preliminary Bivariate Fitting & Intrinsic Scatter Models
# ======================================================================================================================

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


# Calculate and save the absolute magnitudes
Mtot_W11 = (WISE_Data['mag_W1'] - (5. * np.log10(WISE_Data['L_dist'] * 10 ** 6) - 5.)).values.tolist()
Mtot_err_W11 = (WISE_Data['mag_W1_err']).values.tolist()

WISE_Data['M_W1_NoCorr'] = Mtot_W11
WISE_Data['M_W1_NoCorr_err'] = Mtot_err_W11

Mtot_W2 = (WISE_Data['mag_W2'] - (5. * np.log10(WISE_Data['L_dist'] * 10 ** 6) - 5.)).values.tolist()
Mtot_err_W2 = (WISE_Data['mag_W2_err']).values.tolist()

# Get the HI widths as a list
log_W1 = WISE_Data['log_W'].values.tolist()
log_W_err1 = WISE_Data['log_W_err'].values.tolist()

# Perform bivariate fit before corrections
slope1, intercept1, residuals1, scatter_coefficients1, rms1, slope1_err, intercept1_err= bivariate_fit(log_W1, log_W_err1, Mtot_W11, Mtot_err_W11, -9, 1, 100)
print('slope (no corrections) = ', slope1, slope1_err, 'intercept (no corrections) = ', intercept1, intercept1_err, 'rms = ', rms1)

# Create preliminary plots
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'cm'
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7.5), sharey=False)
plt.subplots_adjust(top=0.96, bottom=0.09, right=0.98, left=0.06, wspace=0.1, hspace=0.05)

# ======================================================================================================================
axs[0].plot([1.7, 3], model_lin([slope1, intercept1], np.array([1.7, 3])), 'r',
            label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope1, intercept1) + r'$\log_{10}(W)$', zorder=3)
axs[0].errorbar(log_W1, Mtot_W11, Mtot_err_W11, log_W_err1, color='grey', ls='none', markersize=0,
                elinewidth=0.8, alpha=0.9, zorder=0)

axs[0].plot(log_W1, Mtot_W11, color='blueviolet', ls='none', marker='.', alpha=0.8, zorder=1, label='Raw W1 TF Data')
axs[0].plot(log_W1, Mtot_W11, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

axs[0].set_xlim([min(log_W1) - 0.2, max(log_W1) + 0.2])
axs[0].set_ylim([max(Mtot_W11) + 0.5, min(Mtot_W11) - 0.5])

axs[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
axs[0].grid('visible', which='both', color='0.65', linestyle=':')
axs[0].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=15, labelpad=2)
axs[0].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)

axs[1].plot(log_W1, residuals1,
            '.', color='darkgray', alpha=1, markersize='2', label='Residuals')
poly = np.poly1d(scatter_coefficients1)
axs[1].plot(np.linspace(1.7, 3.1, 100), poly(np.linspace(1.7, 3.1, 100)), 'r',
            label=r'$\xi$ = {0:0.2f}'.format(scatter_coefficients1[0]) + r'$\log_{10}(W)^2$' + '{0:0.2f}'.format(
                scatter_coefficients1[1]) + r'$\log_{10}(W)$' + ' + {0:0.2f}'.format(scatter_coefficients1[2]),
            zorder=2)

axs[1].set_xlim([min(log_W1) - 0.05, max(log_W1) + 0.05])
axs[1].set_ylim([-0.01, max(residuals1) + 0.5])

axs[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
axs[1].grid('visible', which='both', color='0.65', linestyle=':')
axs[1].set_ylabel('Scatter (vega mags)', fontfamily='serif', fontsize=15, labelpad=2)
axs[1].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)
# ======================================================================================================================

plt.savefig('PreliminaryFit_W1.pdf')

# ======================================================================================================================
# Incompleteness Bias Correction
# ======================================================================================================================

# Get the constants
M_star = -24.27 + 5 * np.log10(0.72)
alpha = -1.40

addition = [2, 8, 5, 3, 8, 4, 3, 2, 5, 4, 6, 1, 4, 0, 2, 3, 6, 5, 4, 1, 2, 2, 3, 3, 1, 5, 3, 2, 2, 4, 4]

LF_exclude1 = [[0, 1], [0, 1, 2], [0, 2, 3, 8], [0, 1], [], [0, 1], [1, 2, 3], [0, 3], [0, 1, 2, 3, 4, 6, 7], [],
               [1, 0], [0], [0, 1, 2], [0], [0], [0, 1, 2], [0, 1, 2], [0], [0, 1], [0, 2],
               [0, 1], [0, 2, 3], [0, 1, 2], [0], [0], [], [0], [0], [0], [0], [0, 1, 2]]

CF_exclude1 = [[[0, 1, 2], [0, 2, 3], [0, 2, 3], [0, 1],
                [3, 11, 12], [0, 1, 8, 9], [0, 6, 7], [0, 2, 3, 8],
                [0, 7, 6], [0, 5], [0, 1, 6, 7], [0, 1],
                [0, 1, 2, 6], [0, 2], [0, 1, 2], [0, 1, 2, 6],
                [0, 1, 2], [0, 1, 5, 6, 7], [0, 1], [0, 1, 8],
                [0, 1, 5], [0, 2, 3], [0, 1, 2], [0, 1, 2],
                [0, 3, 4], [0], [0], [0, 3],
                [0, 1], [0, 1], [0, 1]]]


U_M_tot_corrected1, U_M_err_corrected1, bias1 = bias_calc('BiasCorrection_PlotsIT1.pdf', M_star, alpha,
                                                       scatter_coefficients1, WISE_Data, Mtot_W11, Mtot_err_W11,
                                                       addition, LF_exclude1, CF_exclude1, slope1, intercept1, 886)

# Append corrected data onto DF
WISE_Data['M_W1_BiasIT1'] = U_M_tot_corrected1
WISE_Data['M_W1_BiasIT1_err'] = U_M_err_corrected1
WISE_Data['Bias_IT1'] = bias1

# Remove galaxies with bias >10% of the absolute magnitude.
exclude = []
for list_thing in range(len(bias1)):
    if np.abs(bias1[list_thing]) > 0.05*np.abs(U_M_tot_corrected1[list_thing]):
        exclude.append(list_thing)


WISE_Data.drop(labels=exclude, axis=0, inplace=True)
WISE_Data = WISE_Data.reset_index(drop=True)

print('Number of galaxies after bias cut 1', np.shape(WISE_Data))


# Get the HI widths as a list
log_W2 = np.array(WISE_Data['log_W'].values.tolist())
log_W_err2 = np.array(WISE_Data['log_W_err'].values.tolist())

Mtot_W12 = np.array(WISE_Data['M_W1_BiasIT1'].values.tolist())
Mtot_err_W12 = np.array(WISE_Data['M_W1_BiasIT1_err'].values.tolist())

# Perform bivariate fit before corrections
slope2, intercept2, residuals2, scatter_coefficients2, rms2, slope2_err, intercept2_err= bivariate_fit(log_W2, log_W_err2, Mtot_W12, Mtot_err_W12, -9, 1, 100)
print('slope (Bias IT1) = ', slope2, slope2_err, 'intercept (Bias IT1) = ', intercept2, intercept2_err, 'rms = ', rms2)

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7.5), sharey=False)
plt.subplots_adjust(top=0.96, bottom=0.09, right=0.98, left=0.06, wspace=0.1, hspace=0.05)

# ======================================================================================================================
axs[0].plot([1.7, 3], model_lin([slope2, intercept2], np.array([1.7, 3])), 'r',
            label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope2, intercept2) + r'$\log_{10}(W)$', zorder=3)
axs[0].errorbar(log_W2, Mtot_W12, Mtot_err_W12, log_W_err2, color='grey', ls='none', markersize=0,
                elinewidth=0.8, alpha=0.9, zorder=0)

axs[0].plot(log_W2, Mtot_W12, color='blueviolet', ls='none', marker='.', alpha=0.8, zorder=1, label='W1 Band TF Data')
axs[0].plot(log_W2, Mtot_W12, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

axs[0].set_xlim([min(log_W2) - 0.2, max(log_W2) + 0.2])
axs[0].set_ylim([max(Mtot_W12) + 0.5, min(Mtot_W12) - 0.5])
axs[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
axs[0].grid('visible', which='both', color='0.65', linestyle=':')
axs[0].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=15, labelpad=2)
axs[0].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)

axs[1].plot(log_W2, residuals2,
            '.', color='darkgray', alpha=1, markersize='2', label='Residuals')
poly = np.poly1d(scatter_coefficients2)
axs[1].plot(np.linspace(1.7, 3.1, 100), poly(np.linspace(1.7, 3.1, 100)), 'r',
            label=r'$\xi$ = {0:0.2f}'.format(scatter_coefficients2[0]) + r'$\log_{10}(W)^2$' + '{0:0.2f}'.format(
                scatter_coefficients2[1]) + r'$\log_{10}(W)$' + ' + {0:0.2f}'.format(scatter_coefficients2[2]), zorder=2)

axs[1].set_xlim([min(log_W2) - 0.05, max(log_W2) + 0.05])
axs[1].set_ylim([-0.01, max(residuals2) + 0.5])

axs[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
axs[1].grid('visible', which='both', color='0.65', linestyle=':')
axs[1].set_ylabel('Scatter (vega mags)', fontfamily='serif', fontsize=15, labelpad=2)
axs[1].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)
# ======================================================================================================================

plt.savefig('BiasIT1_W1.pdf')


# ======================================================================================================================
# Morphological Correction
# ======================================================================================================================

slope, slope_err, intercept, intercept_err, residuals, scatter_coefficients, rmsM, Mtot_MorphCorrected = morphological_correction(WISE_Data, 'M_W1_BiasIT1', 'M_W1_BiasIT1_err', scatter_coefficients2)

print('slope after morph = ', slope, slope_err, 'intercept after morph = ', intercept, intercept_err, 'rms after = ', rmsM)

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7.5), sharey=False)
plt.subplots_adjust(top=0.96, bottom=0.09, right=0.98, left=0.06, wspace=0.1, hspace=0.05)

# ======================================================================================================================
axs[0].plot([1.7, 3], model_lin([slope, intercept], np.array([1.7, 3])), 'r',
            label=r'M = {1:0.2f} $-$ {0:0.2f}'.format(-1 * slope, intercept) + r'$\log_{10}(W)$', zorder=3)
axs[0].errorbar(log_W2, Mtot_MorphCorrected, Mtot_err_W12, log_W_err2, color='grey', ls='none', markersize=0,
                elinewidth=0.8, alpha=0.9, zorder=0)

axs[0].plot(log_W2, Mtot_MorphCorrected, color='blueviolet', ls='none', marker='.', alpha=0.8, zorder=1, label='Fully-Corrected W1 TF Data')
axs[0].plot(log_W2, Mtot_MorphCorrected, color='mediumpurple', ls='none', marker='.', alpha=0.3, zorder=1)

axs[0].set_xlim([min(log_W2) - 0.2, max(log_W2) + 0.2])
axs[0].set_ylim([max(Mtot_MorphCorrected) + 0.6, min(Mtot_MorphCorrected) - 0.3])
axs[0].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
axs[0].grid('visible', which='both', color='0.65', linestyle=':')
axs[0].set_ylabel('Absolute Magnitude (vega mags), M', fontfamily='serif', fontsize=15, labelpad=2)
axs[0].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)

axs[1].plot(log_W2, residuals,
            '.', color='darkgray', alpha=1, markersize='2', label='Residuals')
poly = np.poly1d(scatter_coefficients)
axs[1].plot(np.linspace(1.7, 3.1, 100), poly(np.linspace(1.7, 3.1, 100)), 'r',
            label=r'$\xi$ = {0:0.2f}'.format(scatter_coefficients[0]) + r'$\log_{10}(W)^2$' + '{0:0.2f}'.format(
                scatter_coefficients[1]) + r'$\log_{10}(W)$' + ' + {0:0.2f}'.format(scatter_coefficients[2]), zorder=2)

axs[1].set_xlim([min(log_W2) - 0.05, max(log_W2) + 0.05])
axs[1].set_ylim([-0.01, max(residuals) + 0.5])

axs[1].legend(loc="upper left", frameon=False, facecolor='white', framealpha=1)
axs[1].grid('visible', which='both', color='0.65', linestyle=':')
axs[1].set_ylabel('Scatter (vega mags)', fontfamily='serif', fontsize=15, labelpad=2)
axs[1].set_xlabel('$\log_{10}(W)$', fontsize=16, labelpad=2)
# ======================================================================================================================

plt.savefig('MorphCorr_W1.pdf')

