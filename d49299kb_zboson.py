# -*- coding: utf-8 -*-
"""
-------------- TITLE --------------
PHYS20161 - Assignment 2 - Z0 Boson.
-----------------------------------

This script will read in csv data files and use them to find the width, mass
and lifetime of a Z0 boson. This is accomplished by:

    1) Reading in the two data files and combining them
    2) Filtering the files for nan, zero and negative values
    3) Iteratively varying the parameters in the chi^2 function to remove
       outliers and find the parameters which minimise chi^2. This is used to
       create a least squares fit.
    4) Lifetime is found using the value for the width

This script also has several additional features which are as follows:

    1) Plots a contour of error ellipses for the parameters and finds their
       errors
    2) Plots a 3-dimentional plot of the parameters against the value for chi^2
    3) Creates plots of the raw data
    4) Creates plots of the valid data and the fit
    5) Finds residuals for the data and the fit

This script was last updated 04/2025.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc


# Global constants:
EE_WIDTH = 83.91e-3  # ee Width in GeV
PI = np.pi
INITIAL_PARAMETERS = [3, 90]  # Initial guess for width, mass (natural units)
HBAR = pc.hbar  # In Joule seconds

# Conversions:
GEV_TO_NB = 0.3894e6  # Conversion from natural units to nanobarns
JOULE_TO_GEV = 1 / 1.6e-10  # Used to convert HBAR into GeV


def read_data():
    """
    Reads in data.

    Reads the data from the provided files and combines them. If the files
    can't be found then the function will return None and code will halt.

    Returns
    -------
    raw_data : array
        The data as read from the input file.

    """
    try:
        raw_data_1 = np.genfromtxt(
            'z_boson_data_1.csv', dtype='float', delimiter=',', skip_header=1)
        raw_data_2 = np.genfromtxt(
            'z_boson_data_2.csv', dtype='float', delimiter=',', skip_header=1)

        # combines the two arrays
        raw_data = np.vstack(
            (raw_data_1, raw_data_2))

        return raw_data

    except FileNotFoundError:
        return False


def data_filter(raw_data):
    """
    Filters the data for unacceptable values.

    Will check that the data has three columns, each for energy, cross-section
    and error. If the data doesnt have 3 columns, function returns None. Next,
    the function filters the raw data by finding the index of any value which
    is not nan or negative and then appending this row to a list of filtered
    data. This list is then converted into a numpy array for later use. If
    the filtered data array is empty (so if all the data is nan or negative
    etc), then the function will return None and the code will halt.

    Parameters
    ----------
    raw_data : array
        The data as read from the input file.

    Returns
    -------
    filtered_data : array
        The data where nan and negative values have been removed.

    """
    # Checks data has three columns
    if raw_data.shape[1] != 3:
        return 0

    # Creates empty list
    filtered_data = []

    # Positive, numerical data will be assigned as 1
    signs = np.sign(raw_data)

    # Data which is positive and numerial
    test_condition = signs > 0

    # Iteratively selects rows where all data passes test condition
    for element in range(len(raw_data)):
        pass_statement = test_condition[element, :].all()
        if pass_statement:
            filtered_data.append(raw_data[element, :])

    # Converts list to np array
    filtered_data = np.asarray(filtered_data)
    if len(filtered_data) == 0:
        return None

    return filtered_data


def distribution_function_nb(energy, z_width, z_mass):
    """
    Defines the Breit-Wigner distribution.

    A function containing the distribution this script will find a fit for.
    Called by other functions to save space and make the scrip more readable.

    Parameters
    ----------
    energy : array
        Energy values in GeV, taken from filtered_data.
    z_width : float
        One of the parameters this script will find, the width of the Z0 boson.
    z_mass : float
        The other parameter this script will find, the mass of the Z0 boson.

    Returns
    -------
    cross_section_fit : float
        An estimated value of the cross-section in nanobarns, as detemined by
        the fit.

    """
    numerator = 12 * PI * energy**2 * EE_WIDTH**2
    denominator = z_mass**2 * ((energy**2 - z_mass**2)**2 + (
        z_mass**2 * z_width**2))

    # Added converstion factor to reurn cross-section in nanobarns
    cross_section_fit = (numerator / denominator) * GEV_TO_NB

    return cross_section_fit


def chi_squared_function(parameters, energy, cross_section, error):
    """
    Chi-squared function.

    This function is minimised to find the optimal parameters. Is also
    evaluated to find the minimised chi-squared and reduced chi-squared values.

    Parameters
    ----------
    parameters : array
        Contains the Z0 width and mass respectively.
    energy : array
        Energy values in GeV.
    cross_section : array
        Cross-section values in nanobarns.
    error : array
        Error values in nanobarns.

    Returns
    -------
    chi_squared : float
        The value of chi-squared for the data.

    """
    # Unpacks parameters
    z_width, z_mass = parameters

    # Array of cross-section values as predicted by the fit
    predicted = distribution_function_nb(energy, z_width, z_mass)

    chi_squared = np.sum(((cross_section - predicted) / error) ** 2)

    return chi_squared


def outlier_filter(data, parameters):
    """
    Filters the data for outliers and finds residuals.

    Finds the index of any cross-section data value which is greater than 3
    standard devations from the fit and deletes that row of data. Produces an
    array with no outliers called 'new_data'. If all points are considered to
    be outliers the function will return None and code will halt. Also will
    find the residuals for plotting purposes.

    Parameters
    ----------
    data : array
        The data where outliers need to be removed.
    parameters : array
        Contains the Z0 width and mass respectively.

    Returns
    -------
    new_data : array
        The data where outliers have been removed.
    residuals : array
        The residuals for each data-point.

    """
    # Unpacks parameters
    z_width, z_mass = parameters

    # Data values for energy, cross_section and error
    energy, cross_section, error = (data[:, 0], data[:, 1], data[:, 2])

    #  Fit based of parameters found from curve_fit. Sigma in nb
    cross_section_fit = distribution_function_nb(energy, z_width, z_mass)

    # Finds the difference between the actual data and the expected value
    difference = abs(cross_section - cross_section_fit)

    # Data must be within 3 standard deviations of the fit to be accepted
    tolerance = 3 * error
    outliers = np.argwhere(difference > tolerance)

    # Deletes outliers and defines a new array with valid data
    new_data = np.delete(data, outliers, 0)

    # If new_data is empty then function returns None
    if len(new_data) == 0:
        return None

    # Defines the residuals
    new_energy, new_cross_section = new_data[:, 0], new_data[:, 1]
    cross_section_fit = distribution_function_nb(new_energy, z_width, z_mass)
    residuals = new_cross_section - cross_section_fit

    return new_data, residuals


def reduced_chi_squared(chi_squared_min, final_data):
    """
    Finds the value for reduced chi squared.

    Parameters
    ----------
    chi_squared_min : float
        The minimum value for chi-squared
    final_data : array
        The data where all outliers have been removed.

    Returns
    -------
    chi_squared_red : float
        The value of reduced chi-squared.

    """
    # Finds the amount of valid data points
    data_points = len(final_data)

    # Number of parameters in this fit
    number_of_parameters = 2

    # Calculates reduced chi squared
    chi_squared_red = chi_squared_min / (
        data_points - number_of_parameters)

    return chi_squared_red


def lifetime_calculation(final_width, percent_width_error):
    """
    Calculates the lifetime of the Z0 boson and its error. If the error is less
    than 1% then it is deemed as negligible (as it will be to a greater
    resolution than the value for the lifetime).

    Parameters
    ----------
    final_width : float
        The final value for the width of the Z0 boson.
    percent_width_error: float
        The perentage error in the width.

    Returns
    -------
    z0_lifetime : float
        The value for the lifetime of the z0 boson.

    """
    # Calculates the lifetime
    z0_lifetime = (HBAR * JOULE_TO_GEV) / final_width

    # Calculates the error
    z0_lifetime_error = z0_lifetime * percent_width_error

    # Sees whether error is negligible or not
    if z0_lifetime_error < (0.01 * z0_lifetime):
        z0_lifetime_error = None

    return z0_lifetime, z0_lifetime_error


def filtered_data_plot(filtered_energy, filtered_cross_section,
                       filtered_error):
    """
    Plots the filtered data.

    This function will make a plot of the filtered data with error bars. Its
    purpose is to visualise the data and any outliers.

    Parameters
    ----------
    filtered_energy : array
        Energy values in GeV, taken from filtered_data.
    filtered_cross_section : array
        Cross-section values in nanobarns, taken from filtered_data.
    filtered_error : array
        Error values in nanobarns, taken from filtered_data.

    Returns
    -------
    None.

    """
    # Creates figure and a_xes
    fig = plt.figure(figsize=(9, 6))
    a_x = fig.add_subplot(111)

    # Plots data and errorbars
    a_x.errorbar(filtered_energy, filtered_cross_section, yerr=filtered_error,
                 fmt='o', c='black', label='Filtered data', capsize=1,
                 markersize=2, elinewidth=1)

    # Plot customisation
    plt.grid(True, alpha=0.4)
    a_x.set_title('Fig 1:  Plot of filtered data with outliers', fontsize=11)
    a_x.set_xlabel('Energy [Gev]')
    a_x.set_ylabel(r'$\sigma$ [nb]')

    plt.savefig(fname='Fig_1_filtered_data.png', dpi=600)
    plt.legend()
    plt.show()


def curve_plot(data, parameters, residuals):
    """
    Plots the distribution curve, data and residuals.

    Parameters
    ----------
    data : array
        The data used for the plot.
    parameters : array
        Contains the Z0 width and mass respectively.
    residuals : array
        The residual for each data-point.

    Returns
    -------
    None.

    """
    # Unpacks parameters
    z_width, z_mass = parameters

    # Defines x, y and error values from data array
    energy, cross_section, error = data[:, 0], data[:, 1], data[:, 2]

    # Defines an x a_xis linspace for plotting purposes
    energy_linspace_1 = np.linspace(np.min(energy) - 1, np.max(energy)
                                    + 1, 1000)

    # Defines the smooth curve
    fit = distribution_function_nb(energy_linspace_1, z_width, z_mass)

    # Creates figure and a_xes
    fig, (a_x1, a_x2) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [3, 1]},
        sharex=True)

    # Plots fit, data and errorbars
    a_x1.plot(energy_linspace_1, fit, c='#f768a1',
              label='Fit using parameters')
    a_x1.errorbar(
        energy, cross_section, yerr=error, fmt='o', label='valid data',
        capsize=1, markersize=2, color='k', ecolor='k', elinewidth=1)

    # Plot 1 customisation
    a_x1.grid(True, alpha=0.4)
    a_x1.set_title(r'Fig 2: $\sigma$ vs Energy', fontsize=11)
    a_x1.set_xlabel('Energy [Gev]')
    a_x1.set_ylabel(r'$\sigma$ [nb]')
    a_x1.legend()

    # Plots residuals and baseline
    a_x2.errorbar(energy, residuals, fmt='o', capsize=1.75,
                  markersize=2.75, c='k', label='Valid data')
    a_x2.axhline(y=0, c='#f768a1', ls='--', label='Baseline')

    # Plot 2 customisation
    a_x2.grid(True, alpha=0.4)
    a_x2.set_ylabel('Residual', fontsize=11)
    a_x2.legend(fontsize='8.5')

    plt.tight_layout()
    plt.savefig(fname='Fig_2_distribution.png', dpi=600)
    plt.show()


def define_meshes(final_width, final_mass, final_energy, final_cross_section,
                  final_error):
    """
    Defines the meshes for width, mass and chi-squared.

    The meshes are used in producing the contour and 3d plots, also in finding
    the parameter errors.

    Parameters
    ----------
    final_width : float
        Final value for the Z0 width.
    final_mass : float
        Final value for the Z0 mass.
    final_energy : array
        Final Energy values in GeV.
    final_cross_section : array
        Final cross-section values in nanobarns.
    final_error : array
        Final error values in nanobarns.

    Returns
    -------
    width_mesh : array
        A mesh for the Z0 width, used for errors and 3d and contour plotting.
    mass_mesh : array
        A mesh for the Z0 mass, used for errors and 3d and contour plotting.
    chi_squared_mesh : array
        A mesh for chi-squared values, used for errors and 3d and contour
        plotting.

    """
    # Produces linspaces for Z0 mass and width
    z_width_values = np.linspace(final_width - 0.05, final_width + 0.05,
                                 len(final_energy))
    z_mass_values = np.linspace(final_mass - 0.05, final_mass + 0.05,
                                len(final_energy))

    width_mesh, mass_mesh = np.meshgrid(z_width_values, z_mass_values)

    # Create an empty array for chi_squared mesh
    chi_squared_mesh = np.zeros_like(width_mesh)

    # Fills the Chi_squared mesh
    for i in range(width_mesh.shape[0]):
        for j in range(width_mesh.shape[1]):
            chi_squared_mesh[i, j] = chi_squared_function([
                width_mesh[i, j], mass_mesh[i, j]], final_energy,
                final_cross_section, final_error)

    return width_mesh, mass_mesh, chi_squared_mesh


def error_function(width_mesh, mass_mesh, chi_squared_mesh, chi_squared_min):
    """
    Uses the contour ellipse plots to find the errors in each parameter.

    Finds the indeces of chi-squared values that are equal to the
    corresponding error ellipse. Next, finds the values of mass and width that
    correspond to these indeces. Finds the minimum and maximum of the width
    and mass arrays, finds the difference and then halves. This is half the
    width and height of the ellipse and is equal to the errors on each value.

    Parameters
    ----------
    width_mesh : array
       A mesh for the Z0 width, used for errors and 3d and contour plotting.
    mass_mesh : array
       A mesh for the Z0 mass, used for errors and 3d and contour plotting.
    chi_squared_mesh : array
       A mesh for chi-squared values, used for errors and 3d and contour
       plotting.
    chi_squared_min : float
        The minimum value of chi-squared for the data.

    Returns
    -------
    error_width: float
        The error in the Z0 boson width.
    error_mass: float
        The error in the Z0 boson mass.
    """
    # Defines the contour of the error ellipse
    first_ellipse = round((chi_squared_min + 2.3), 2)

    width_values = []
    mass_values = []

    # Fills chi squared mesh
    for i in range(width_mesh.shape[0]):
        for j in range(width_mesh.shape[1]):
            if round(chi_squared_mesh[i, j], 2) == first_ellipse:
                width_values.append(width_mesh[i, j])
                mass_values.append(mass_mesh[i, j])

    width_values = np.array(width_values)
    mass_values = np.array(mass_values)

    # Finds min and max values of ellipse contour
    width_max, width_min = np.max(width_values), np.min(width_values)
    mass_max, mass_min = np.max(mass_values), np.min(mass_values)

    # Finds half the width of ellipse (error)
    error_width = 0.5 * (width_max - width_min)
    error_mass = 0.5 * (mass_max - mass_min)

    return error_width, error_mass


def three_dimension_plot(width_mesh, mass_mesh, chi_squared_mesh):
    """
    Creates a 3d plot of the parameters against the chi-squared value.

    Parameters
    ----------
    width_mesh : array
        A mesh for the Z0 width, used for errors and 3d and contour plotting.
    mass_mesh : array
        A mesh for the Z0 mass, used for errors and 3d and contour plotting.
    chi_squared_mesh : array
        A mesh for chi-squared values, used for errors and 3d and contour
        plotting.

    Returns
    -------
    None.

    """
    # Produces figure and a_xes
    fig = plt.figure(figsize=(15, 20))
    a_x = fig.add_subplot(111, projection='3d')

    # Creates the 3d plot
    plot_3d = a_x.plot_surface(width_mesh, mass_mesh, chi_squared_mesh,
                               rstride=1, cstride=1, cmap='RdPu')

    # Plot customisation
    a_x.set_xlabel(r'$\Gamma _Z$ [GeV]', fontsize=10)
    a_x.set_ylabel(r'$m_Z$ [GeV/ c$^2$]', fontsize=10)
    a_x.set_zlabel(r'$\chi ^2$', fontsize=10)
    a_x.set_title(r'Fig 4: Plot of $\Gamma _Z$ and $m_Z$ against $\chi ^2$')
    a_x.clabel(plot_3d, fontsize=11, colors='k')

    # Sets the min and max values for the z a_xis
    a_x.set_zlim(min(chi_squared_mesh.flatten()) - 5,
                 max(chi_squared_mesh.flatten()))

    # Sets the viewing angle
    a_x.view_init(elev=10, azim=120)

    plt.savefig(fname='Fig_4_3d_plot.png', dpi=600)
    plt.show()


def ellipse_plot(chi_squared_min, width_mesh, mass_mesh, chi_squared_mesh,
                 final_parameters):
    """
    Plots the error ellipses and creates a contour plot for chi-squared values.

    Parameters
    ----------
    chi_squared_min : float
        The minimum value of chi-squared for the data.
    width_mesh : array
        A mesh for the Z0 width, used for errors and 3d and contour plotting.
    mass_mesh : array
        A mesh for the Z0 mass, used for errors and 3d and contour plotting.
    chi_squared_mesh : array
        A mesh for chi-squared values, used for errors and 3d and contour
        plotting.
    final_parameters : array
        Contains the final Z0 width and mass respectively.

    Returns
    -------
    None.

    """
    # Unpacks parameters
    final_width, final_mass = final_parameters

    # Creates figure and a_xes
    fig = plt.figure(figsize=(9, 6))
    a_x = fig.add_subplot(111)

    # Defines contour levels and plots them
    contour_levels = [chi_squared_min + 1, chi_squared_min + 2.3,
                      chi_squared_min + 5.99]
    contour_plot = a_x.contour(width_mesh, mass_mesh, chi_squared_mesh,
                               levels=contour_levels, colors=(
                                  '#ae017e', '#7a0177', '#49006a'))

    # Plots the gradient contour
    gradient_plot = a_x.contourf(width_mesh, mass_mesh, chi_squared_mesh,
                                 levels=100, cmap='RdPu', alpha=.4)

    # Plot customisation
    fig.colorbar(gradient_plot, label=r'$\chi ^2$')
    a_x.clabel(contour_plot, fontsize=11, colors='k')
    a_x.plot(final_width, final_mass, 'ko', label="Best fit")
    a_x.set_xlabel(r'$\Gamma _Z$ [GeV]')
    a_x.set_ylabel(r'$m_Z$ [GeV/ c$^2$]')
    a_x.set_title(r'Fig 3: Error ellipses for $\Gamma _{Z}$ vs $m_{Z}$')
    plt.grid(True, alpha=0.4)

    plt.savefig(fname='Fig_3_Error_ellipse.png', dpi=600)
    a_x.legend()
    plt.show()


def fit_iteration(filtered_data, new_parameters):
    """
    Iteratively removes and refits data to find the best parameters.

    This function uses a while loop to find the best parameters by removing
    outliers, refitting to the remaining data and then comparing the new fit
    to the original filtered data to reintroduce all valid data points. This
    while loop will continue until no more outliers are removed (note if all
    data is removed as outliers then the function will return as 0). When this
    condition is met the function will return the data and parameters for this
    fit. The loop also has a counter; If too many iteration attempts are made,
    the function will return as 1 and reject the data as it is too poor to fit
    to the given distribution. A set of data closer to the distribution should
    require less iterations.

    Parameters
    ----------
    filtered_data : array
        The data where nan and negative values have been removed.
    new_parameters : array
        Parameters found from refitting the data.

    Returns
    -------
    prev_data : array
        The data which will be iterated over in order to find an improved
        array.
    new_parameters : array
        Parameters found from refitting the data.

    """
    # Defines number of data in previous data
    prev_data_points = len(filtered_data)

    # Sets initial conditions for the while loop
    prev_data = filtered_data
    new_data_points = prev_data_points + 1
    count = 0

    while new_data_points != prev_data_points and count < 10:

        # New data found by removing outliers using the new parameters
        prev_data = outlier_filter(prev_data, new_parameters)[0]

        # Ensures data isn't entirely outliers
        if prev_data is None:
            return False

        prev_data_points = len(prev_data)

        # Redefines data from the first_data array
        energy, cross_section, error = prev_data.T

        # New data is refitted for more accurate parameter values
        int_parameters = fmin(chi_squared_function, INITIAL_PARAMETERS, args=(
            energy, cross_section, error), disp=False)

        # Filtered data compared to improved fit to return all valid data
        new_data = outlier_filter(filtered_data, int_parameters)[0]

        if new_data is None:
            return False

        energy, cross_section, error = new_data.T

        # Final adjustment of fit parameters to all valid data
        new_parameters = fmin(chi_squared_function, INITIAL_PARAMETERS, args=(
            energy, cross_section, error), disp=False)

        new_data_points = len(new_data)

        prev_data = new_data

        count += 1

        continue

    if count == 10:
        return True

    print(count, 'iterations completed.', new_data_points, ' points remaining.'
          '\n')

    return prev_data, new_parameters


def main():
    """
    Function where the script runs from. All other functions are called from
    here.

    Returns
    -------
    None.

    """
    print('\n---- ASSIGNMENT 2: Z0 BOSON ----\n ')

    print('---- ATTENTION ----\n')
    print('Please ensure any data files adhere to the following format: There'
          ' must be a singular header row. Data must be in three columns with'
          ' energy values in GeV in the first, cross-section in nb in the'
          ' second and cross-section error in the third, also in nb. This code'
          ' may not run as intended if this format isnt followed. \n')

    print('-------------------\n')

    print('Attempting to read data...')

    # Reads data files
    raw_data = read_data()
    if raw_data is False:
        print('ERROR: Data files cannot be found.')
        return

    # Finds and prints amount of data
    raw_data_points = len(raw_data)
    print(f'Data files read successfully. {raw_data_points} raw data'
          ' points found.\n')

    print('Attempting to filter data...')

    # Data is filtered to remove nan and negative values
    filtered_data = data_filter(raw_data)
    if filtered_data is None:
        print('ERROR: There are no acceptable data points.')
        return
    if filtered_data is False:
        print('ERROR: Data is of the wrong dimensions.'
              'There must be three columns.')
        return

    # Finds and prints amount of data remaining and removed
    filtered_data_points = len(filtered_data)
    removed_data_points = raw_data_points - filtered_data_points
    print(f'Data filtered successfully. {filtered_data_points} filtered data'
          f' points remaining. {removed_data_points} have been removed.\n')

    # Defines data as 1d arrays
    energy, cross_section, error = filtered_data.T

    # Produces a plot of the filtered data with error bars
    filtered_data_plot(energy, cross_section, error)

    # Produces initial parameters based off of a minimise chi squared fit
    first_parameters = fmin(chi_squared_function, INITIAL_PARAMETERS, args=(
        energy, cross_section, error), disp=False)

    # Iteratively refits data until a good fit is found
    print('Attempting to find parameters...')
    iteration_check = fit_iteration(filtered_data, first_parameters)

    if iteration_check is True:
        print('ERROR: Too many iterations attempted. Data is too poor to fit.')
        return
    if iteration_check is False:
        print('ERROR: There are no valid data points to plot.')

    # Defines data and parameters from the fit iteration function
    new_data, final_parameters = iteration_check

    # Updates values of residuals using final parameters and plots them
    final_data, final_residuals = outlier_filter(new_data, final_parameters)

    # Redefines data arrays
    energy, cross_section, error = final_data.T

    # Individually defines Z0 width and mass
    final_width, final_mass = final_parameters.T

    print('---- CALCULATED VALUES ---- \n')

    # Calculates min chi squared and reduced chi squared
    chi_squared_min = chi_squared_function(
        final_parameters, energy, cross_section, error)
    red_chi_squared = reduced_chi_squared(chi_squared_min, final_data)

    # Defines meshes for plotting and error purposes
    width_mesh, mass_mesh, chi_squared_mesh = define_meshes(
        final_width, final_mass, energy, cross_section, error)

    # Finds the errors in parameters CHECK THIS +1 OR +2.33
    parameter_error = error_function(
        width_mesh, mass_mesh, chi_squared_mesh, chi_squared_min)
    # individually defines errors in mass and width
    width_error, mass_error = parameter_error[0], parameter_error[1]

    # Finds the percentage error in the width
    percent_width_error = width_error / final_width

    # Calculates lifetime and its uncertainty
    z0_lifetime, z0_lifetime_error = lifetime_calculation(
        final_width, percent_width_error)

    # Print statements

    print(f'The Z\u2080 boson width is {final_width:.4g} \u00B1'
          f' {width_error:.3f}'
          ' GeV.')
    print(f'The Z\u2080 boson mass is {final_mass:.4g} \u00B1 {mass_error:.2f}'
          ' GeV/c\u00B2.')

    # If the error in lifetime is negligible
    if z0_lifetime_error is None:
        print(f'The lifetime of the Z\u2080 boson is {z0_lifetime:.3g}'
              ' seconds. The Error in this value is negligible. \n')
    else:
        print(f'The lifetime of the Z\u2080 boson is {z0_lifetime:.3g} \u00B1'
              f' {z0_lifetime_error:.3g} seconds. \n')

    print(f'The minimum value for χ² is {chi_squared_min:.3f}.')
    print(f'The reduced χ² for this fit is {red_chi_squared:.3f}.\n')

    # Plotting

    print('---- PLOTS LOADING... ----')
    curve_plot(final_data, final_parameters, final_residuals)
    ellipse_plot(chi_squared_min, width_mesh, mass_mesh, chi_squared_mesh,
                 final_parameters)
    three_dimension_plot(width_mesh, mass_mesh, chi_squared_mesh)
    print('Plots loaded successfully.')


if __name__ == '__main__':
    main()
