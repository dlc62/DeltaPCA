# settings:
global min_struct_el
min_struct_el = 7 
max_number_baseline_iterations = 16 # number of iterations in baseline search

# import libraries - requires numpy, scipy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import detrend


def als(x, y, als_lambda=5e6, als_p_weight=3e-6):
    """ asymmetric baseline correction, original algorithm
    Code by Rustam Guliev ~~ https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    Parameters that can be tuned:
    als_lambda  ~ 5e6
    als_p_weight ~ 3e-6
    (found from optimization with random smooth BL)
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2), format='dok')
    D = als_lambda * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L, format='dok')
    for i in range(max_number_baseline_iterations):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    return baseline


def psalsa(x, y, als_lambda=6e7, als_p_weight=1.1e-3):
    """ asymmetric baseline correction with peak screening by amplitudes
    Algorithm by Sergio Oller-Moreno et al.
    Parameters that can be tuned:
    als_lambda  ~ 6e7
    als_p_weight ~ 1.1e-3
    (found from optimization with random 5-point BL)
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2), format='dok')
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L, format='dok')
    peakscreen_amplitude = (np.max(detrend(y)) - np.min(detrend(y)))/8
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * np.exp(-(y-z)/peakscreen_amplitude) * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    return baseline


def derpsalsa(x, y, als_lambda=5e7, als_p_weight=1.5e-3):
    """ asymmetric baseline correction with peak screening by derivatives
    Parameters that can be tuned:
    als_lambda  ~ 5e7
    als_p_weight ~ 1.5e-3
    (found from optimization with random 5-point BL)
    """

    # 0: smooth the spectrum 16 times
    #    with the element of 1/100 of the spectral length:
    zero_step_struct_el = np.int(2*np.round(len(y)/200) + 1)
    y_sm = mollification_smoothing(y, zero_step_struct_el, 16)
    # compute the derivatives:
    y_sm_1d = np.gradient(y_sm)
    y_sm_2d = np.gradient(y_sm_1d)
    # weighting function for the 2nd der:
    y_sm_2d_decay = (np.mean(y_sm_2d**2))**0.5
    weifunc2D = np.exp(-y_sm_2d**2/2/y_sm_2d_decay**2)
    # weighting function for the 1st der:
    y_sm_1d_decay = (np.mean((y_sm_1d-np.mean(y_sm_1d))**2))**0.5
    weifunc1D = np.exp(-(y_sm_1d-np.mean(y_sm_1d))**2/2/y_sm_1d_decay**2)
    
    weifunc = weifunc1D*weifunc2D

    # exclude from screening the edges of the spectrum (optional)
    weifunc[0:zero_step_struct_el] = 1; weifunc[-zero_step_struct_el:] = 1

    # estimate the peak height
    peakscreen_amplitude = (np.max(detrend(y)) - np.min(detrend(y)))/8 # /8 is good, because this is a characteristic height of a tail
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2), format='dok')
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L, format='dok')
    # k = 10 * morphological_noise(y) # above this height the peaks are rejected
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * weifunc * np.exp(-((y-z)/peakscreen_amplitude)**2/2) * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    return baseline


def mollification_smoothing (rawspectrum, struct_el, number_of_mollifications):
    """ Molifier kernel here is defined as in the work of Koch et al.:
        JRS 2017, DOI 10.1002/jrs.5010
        The structure element is in pixels, not in cm-1!
        struct_el should be odd integer >= 3
    """
    mollifier_kernel = np.linspace(-1, 1, num=struct_el)
    mollifier_kernel[1:-1] = np.exp(-1/(1-mollifier_kernel[1:-1]**2))
    mollifier_kernel[0] = 0; mollifier_kernel[-1] = 0
    mollifier_kernel = mollifier_kernel/np.sum(mollifier_kernel)
    denominator = np.convolve(np.ones_like(rawspectrum), mollifier_kernel, 'same')
    smoothline = rawspectrum
    i = 0
    for i in range (number_of_mollifications) :
        smoothline = np.convolve(smoothline, mollifier_kernel, 'same') / denominator
        i += 1
    return smoothline

