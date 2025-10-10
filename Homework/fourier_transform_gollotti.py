#Import packages
import numpy as np
import cmath
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt

#Reading in .fits file
def importing_file():
    
    #Importing .fits file
    filename = 'tic0000120016.fits' #Uploaded to the github repository
    
    hdul = fits.open(filename)
    times = hdul[1].data['times']
    fluxes = hdul[1].data['fluxes']
    ferrs = hdul[1].data['ferrs']
    
    #Reducing range, so we only focus on a small section of points
    light_index = (times > 1600) & (times < 1620) #1600 = lower bound, 1610 = upper bound
    
    #Applying mask
    new_times = times[light_index]
    new_fluxes = fluxes[0:len(new_times)]
    new_ferr = ferrs[0:len(new_times)]
    
    return new_times, new_fluxes, new_ferr

#Interpolating values
def interpolation(times, fluxes):
    
    time_interp = np.linspace(1600, 1620, 200) #Lower bound = 1600, Upper bound = 1620
    flux_interp = np.interp(time_interp, times, fluxes)
    
    return time_interp, flux_interp

#Fourier Transform functions
def DFT(fluxes): #finds ck values
    
    N = len(fluxes) #N is dependent on the length of the flux array
    N_real = N//2 + 1 #If we don't isolate for real values, our ck values will be duplicated
    c = np.zeros(N_real, dtype = 'complex')
    
    for k in range(N_real):
        for n in range(N):
            c[k] += fluxes[n] * cmath.exp(-2j * cmath.pi * k * n/N)
            
    return c

#Find coefficients from Fourier Transform
def finding_coeffs(fluxes):
    
    coeffs = DFT(fluxes)
    
    coeffs_squared = coeffs * np.conjugate(coeffs) #These will be plotted, so every c_k value is a real number
    
    k_range = np.arange(1, len(coeffs_squared) - 1, 1)
    
    return coeffs, coeffs_squared, k_range


#Going to use np.fft.irfft to find the inverse DFT
def inverse_DFT(coeffs, coeffs_squared, k_range):
    
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
#Plotting function
def plotting_DFT():
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5)) 
    
    #Plotting original lightcurve
    times, fluxes, err = importing_file()
    times_interp, fluxes_interp = interpolation(times, fluxes)
    
    ax1.set_title('TIC 0000120016 (Original)', fontsize = 20)
    ax1.set_xlabel('Time (JD)', fontsize = 15)
    ax1.set_ylabel('Flux', fontsize = 15)
    
    ax1.plot(times, fluxes, color = 'blue', label = 'Observed')
    ax1.plot(times_interp, fluxes_interp, color = 'deeppink', label = 'Interpolated')
    
    ax1.legend(loc = 'upper right', fontsize = 15)
    ax1.grid()
    
    #Plotting ck^2 values
    coeffs, coeffs_squared, k_range = finding_coeffs(fluxes)
    coeffs_interp, coeffs_squared_interp, k_range_interp = finding_coeffs(fluxes_interp)
    
    ax2.set_title('DFT of TIC 0000120016', fontsize = 20)
    ax2.set_xlabel('k', fontsize = 15)
    ax2.set_ylabel(r'$c_k^2$', fontsize = 15)
    
    ax2.plot(k_range, coeffs_squared[1:-1], color = 'blue', label = 'Observed')
    ax2.plot(k_range_interp, coeffs_squared_interp[1:-1], color = 'deeppink', label = 'Interpolated')
    
    ax2.legend(loc = 'upper right', fontsize = 15)
    ax2.grid()
    
    #Plotting inverse DFT
    new_fluxes = inverse_DFT(coeffs, coeffs_squared, k_range)
    new_fluxes_interp = inverse_DFT(coeffs_interp, coeffs_squared_interp, k_range_interp)
    
    ax3.set_title('Inverse DFT of TIC 0000120016', fontsize = 20)
    ax3.set_xlabel('Time (JD)', fontsize = 15)
    ax3.set_ylabel('Flux', fontsize = 15)
    
    ax3.plot(new_fluxes, color = 'blue', label = 'Observed')
    ax3.plot(new_fluxes_interp, color = 'deeppink', label = 'Interpolated')
    
    ax3.legend(loc = 'upper right', fontsize = 15)
    ax3.grid()
    
    plt.show()
    
if __name__ == '__main__':
    
    plotting_DFT()
