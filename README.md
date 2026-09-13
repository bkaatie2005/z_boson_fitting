# Non-linear Data Analysis of $Z^0$ Boson Resonance

A script which takes CSV data from particle collisions and finds values for the width, mass and lifetime of a $Z^0$ boson by performing the following:   

  1) Reads in the two data files and combines them
  2) Filters the files for nan, zero and negative values
  3) Iteratively varies the parameters in the chi^2 function to remove outliers and finds the parameters which minimise chi^2. This is used to create a least squares fit.
  4) Finds the Lifetime using the value for the width

This script also has several additional features which are as follows:
  1) Plots a contour of error ellipses for the parameters and finds their
       errors
  2) Plots a 3-dimentional plot of the parameters against the value for chi^2
  3) Creates plots of the raw data
  4) Creates plots of the valid data and the fit
  5) Finds residuals for the data and the fit
