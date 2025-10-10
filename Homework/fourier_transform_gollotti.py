#Import packages
import numpy as np
import cmath
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt

#Reading in .fits file
def importing_file():
    
    """
        This function imports the .fits file for a binary system detected by TESS. It reads in the time/flux/error data and
        isolates the data to a specific epoch after a mask is applied. The function returns the new time/flux/error arrays
        after the mask is applied.
        
        Returns:
            new_times (np.array) = Array of 'times' after mask is applied
            new_fluxes (np.array) = Array of 'fluxes' after mask is applied
            new_ferr (np.array) = Array of 'errors' after mask is applied
    """
    
    #Importing .fits file
    filename = 'tic0000120016.fits'
    
    hdul = fits.open(filename) #Reads in file
    times = hdul[1].data['times']
    fluxes = hdul[1].data['fluxes']
    ferrs = hdul[1].data['ferrs']
    
    #Reducing range, so we only focus on a small section of points
    light_index = (times > 1600) & (times < 1620) #1600 = lower bound, 1610 = upper bound (first epoch)
    
    #Applying mask
    new_times = times[light_index]
    new_fluxes = fluxes[0:len(new_times)]
    new_ferr = ferrs[0:len(new_times)]
    
    return new_times, new_fluxes, new_ferr

#Interpolating values
def interpolation(times, fluxes):
    
    time_interp = np.linspace(1600, 1620, 200) #Lower bound = 1600, Upper bound = 1620
    flux_interp = np.interp(time_interp, times, fluxes) #Interpolated flux values
    
    return time_interp, flux_interp

#Fourier Transform functions
def DFT(fluxes): #finds ck values
    
    """
        This function applies the discrete Fourier Transform to a given set of flux values, and it returns the corresponding
        ck coefficients. This function looks at the real components of the fluxes, so the ck values aren't duplicated. It
        returns the array of ck values after the DFT is implemented.
        
        Parameters:
            fluxes (np.array) = Array of flux values
        
        Constants:
            N (float) = Timestamp (dependent on the length of the flux array)
            N_real (int) = Number of real (non-complex) values in the flux array
        
        Returns:
            c (np.array) = Array of coefficient values for the power spectrum
        
    """
    
    N = len(fluxes) #N is dependent on the length of the flux array
    N_real = N//2 + 1 #If we don't isolate for real values, our ck values will be duplicated
    c = np.zeros(N_real, dtype = 'complex')
    
    for k in range(N_real):
        for n in range(N):
            c[k] += fluxes[n] * cmath.exp(-2j * cmath.pi * k * n/N)
            
    return c

#Find coefficients from Fourier Transform
def finding_coeffs(fluxes):
    
    """
        This function provides the necessary information for plotting the coefficients properly. In order to plot the
        coefficients, we need the real components. The coefficients have both a real and imaginary component, so we need
        to multiply the coefficient by its conjugate in order to be obtain a fully real component value (coeff_squared).
        The k_range will be plotted along the x-axis and will be the same length as the coeff_squared array. Since the 
        coefficient for power spectrum at k = 0 is much higher than all the other components, we ignore the k = 0 value by
        starting the k_range at 1 instead. This function returns the coefficients (both real/imaginary components), 
        coeffs_squared (coefficients with only real components), and k_range for plotting.
        
        Parameters:
            fluxes (np.array) = Array of 'fluxes'
        
        Returns:
            coeffs (np.array) = Array of coefficients containing both the real/imaginary components
            coeffs_squared (np.array) = Array of squared coefficients with just the real component
            k_range (np.array) = Range of k values, spanning from 0 to the length of the coeffs array
    
    """
    
    coeffs = DFT(fluxes)
    
    coeffs_squared = coeffs * np.conjugate(coeffs) #These will be plotted, so every c_k value is a real number
    
    k_range = np.arange(1, len(coeffs_squared) - 1, 1) #Ignores the k = 0 value
    
    return coeffs, coeffs_squared, k_range


#Going to use np.fft.irfft to find the inverse DFT
def inverse_DFT(coeffs, coeffs_squared, k_range):
    
    """
        This function finds the inverse discrete Fourier Transform based on the coefficients calculated in the DFT() and
        finding_coeffs() methods. First, we need to find the max ck^2 value and its corresponding k value, so we can ignore
        ck^2 values that are very small (don't correspond to the period of the binary transit). We only care about the top
        90% of coefficients, so the smallest 10% will be ignored. We then change the smallest 10% values to 0.0, so they
        equate to 1 after passing through the inverse DFT function. In order to calculate the inverse DFT, we are using
        the numpy function fft.irfft(), which takes the real coefficient values as a parameter and returns an array of
        flux values.
        
        Parameters:
            coeffs (np.array) = Array of coefficients containing both the real/imaginary components
            coeffs_squared (np.array) = Array of squared coefficients with just the real component
            k_range (np.array) = Range of k values, spanning from 0 to the length of the coeffs array
        
        Constants:
            ck_max (float) = Max ck^2 value in the power spectrum
            ck_arg_max (int) = Index corresponding to the max ck^2 value
            max_k (int) = k value corresponding to the ck_arg_max index
            
            keep (np.array) = Index corresponding to the top 90% of coefficients
            coeffs_new (np.array) = Copy of coeffs array, but the smallest 10% of coefficients equates to 0.0
        
        Returns:
            new_fluxes (np.array) = Array of fluxes calculated with np.fft.irfft (based on coefficient values)
        
    """
    
    #Finding maximum ck^2 value
    ck_max = np.max(coeffs_squared[1:-1]) #max ck^2 value
    ck_arg_max = np.argmax(coeffs_squared[1:-1]) #Finds the index corresponding to the max ck^2 value
    max_k = k_range[ck_arg_max] #Returns the k value corresponding to the index above
    
    #Isolate data that's within a specific range
    keep = coeffs_squared > 0.1 * ck_max
    coeffs_new = np.copy(coeffs) #Keeps the original coefficient array, so we don't lose the original coefficient data
    
    #Anything that isn't within the given range is assigned a 0.0 value, so the array length doesn't change
    coeffs_new[~keep] = 0.0 
    
    new_fluxes = np.fft.irfft(coeffs_new)
    
    return new_fluxes


#Plotting function
def plotting_DFT():
    
    """
        This function plots the discrete Fourier Transform, power spectrum, and inverse Fourier Transform of the given
        TESS light curves. The blue line represents the original data, while the pink line represents the linearly
        interpolated values.
        
        For the power spectrum and inverse DFT, I am not confident that my interpolated values are correct. 
        
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 5)) 
    
    #Plotting original lightcurve
    times, fluxes, err = importing_file()
    times_interp, fluxes_interp = interpolation(times, fluxes)
    
    ax1.set_title('TIC 0000120016 (Original)', fontsize = 15)
    ax1.set_xlabel('Time (JD)', fontsize = 12)
    ax1.set_ylabel('Flux', fontsize = 12)
    
    ax1.plot(times, fluxes, color = 'blue', label = 'Observed')
    ax1.plot(times_interp, fluxes_interp, color = 'deeppink', label = 'Interpolated')
    
    ax1.legend(loc = 'upper right', fontsize = 12)
    ax1.grid()
    
    #Plotting ck^2 values
    coeffs, coeffs_squared, k_range = finding_coeffs(fluxes)
    coeffs_interp, coeffs_squared_interp, k_range_interp = finding_coeffs(fluxes_interp)
    
    ax2.set_title('DFT of TIC 0000120016', fontsize = 15)
    ax2.set_xlabel('k', fontsize = 12)
    ax2.set_ylabel(r'$c_k^2$', fontsize = 12)
    
    ax2.plot(k_range, coeffs_squared[1:-1], color = 'blue', label = 'Observed')
    ax2.plot(k_range_interp, coeffs_squared_interp[1:-1], color = 'deeppink', label = 'Interpolated') 
    
    ax2.legend(loc = 'upper right', fontsize = 12)
    ax2.grid()
    
    #Plotting inverse DFT
    new_fluxes = inverse_DFT(coeffs, coeffs_squared, k_range)
    new_fluxes_interp = inverse_DFT(coeffs_interp, coeffs_squared_interp, k_range_interp)
    
    ax3.set_title('Inverse DFT of TIC 0000120016', fontsize = 15)
    ax3.set_xlabel('Time (JD)', fontsize = 12)
    ax3.set_ylabel('Flux', fontsize = 12)
    
    ax3.plot(new_fluxes, color = 'blue', label = 'Observed')
    ax3.plot(new_fluxes_interp, color = 'deeppink', label = 'Interpolated')
    
    ax3.legend(loc = 'upper right', fontsize = 12)
    ax3.grid()
    
    plt.show()
    
#Runs methods from the command line    
if __name__ == '__main__':
    
    plotting_DFT()
