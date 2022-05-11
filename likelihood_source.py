#! /usr/bin/env python

__author__ = 'Hazan Daglayan, Simon Vary, Xavier Lambein'
__all__ = ['likelihoodmap',
           'psf_cube',
           'likelihood_value']




import numpy as np
from util import trajectory_mask
import vip_hci as vip
from hciplot import plot_frames
from joblib import Parallel, delayed



def likelihoodmap(residual_cube, angle_list, psfn, planet_position_list, 
                  fwhm=4, norm=1, plot=False,  n_jobs=1):
    '''
    Parallel implementation of the likelihood map generation function. Applies 
    the likelihood function at each pixel.
    
    
    Parameters
    ----------
    residual_cube : numpy ndarray
        Input residual cube (3d array).
    

    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
        
    psfn : numpy ndarray 2d
        2d array with the normalized PSF template
    
    planet_position_list : list which has tuples
        The list of (X,Y) which is the center of aperture.
    
    fwhm : int, optional
        Full width at half maximum for the instrument PSF
    
    norm : int, optional
        1 for Laplacian, 2 for Gaussian 
    
    plot : bool, optional
        Plot the maps
    
    nproc : int, optional
        Number of processes for parallel computing.
        
    Returns
    -------
    fluxmap : 2d ndarray
        Maximum likelihood estimate of the contrast for each pixel in the field
        of view
    likelihoodmap : 2d ndarray
        Likelihood ratio map.
    
    '''

    

    if isinstance(residual_cube, np.ndarray):
        likelihoodmap = np.zeros_like(residual_cube[0])
        fluxmap = np.zeros_like(residual_cube[0])
        if n_jobs == 1:
            for planet_position in zip(planet_position_list):
                flux, likelihood, pos =  likelihood_value(residual_cube, 
                                                          angle_list, psfn, 
                                                          planet_position, 
                                                          fwhm, norm)
                fluxmap[pos[0][0],pos[0][1]] = flux
                likelihoodmap[pos[0][0],pos[0][1]] = likelihood
        else:
            fluxmap_list, likelihoodmap_list, pos_list = zip(
                *Parallel(n_jobs=n_jobs, verbose = True)(
                    delayed(likelihood_value)(residual_cube, angle_list, psfn, 
                                              planet_position, fwhm, norm)
                    for planet_position in zip(planet_position_list)
                )
            )

            for (flux, likelihood, pos) in zip(fluxmap_list, 
                                               likelihoodmap_list, pos_list):
                fluxmap[pos[0][0],pos[0][1]] = flux
                likelihoodmap[pos[0][0],pos[0][1]] = likelihood
    
    else:
        raise TypeError('`residual cube list` must be a numpy (3d) array ')


    if plot:
        plot_frames(likelihoodmap)

    return fluxmap, likelihoodmap

def psf_cube(cube, psfn, angles, rad, theta):
    """Creates a cube filled with zero and a copy of the reference PSF along
    a trajectory."""
    return vip.fm.cube_inject_companions(
        np.zeros_like(cube), psfn, angles,
        1., 0., rad, theta=theta, verbose=False
    )



def likelihood_value(residual_cube, angle_list, psfn, planet_position, fwhm=4, 
                     norm = 1):
    """Computes the log likelihood ratio along a path in a subtracted
    data cube.

    Given a subtracted data cube and a trajectory, finds the flux of a 
    potential planet along that trajectory that minimizes the norm :

        argmin_a  || (R - a * P) / sigma ||_1

    where `a` is the flux, `R` is the residual cube, `P` is the copy of 
    reference PSF along the trajectory, and `sigma` is the standard deviation. 
    
    
    When `a` is negative, because this value is non-physical, the log 
    likelihood ratio is set to zero.

    Parameters
    ----------
    residual_cube : numpy ndarray
        Input residual cube (3d array).
    

    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
        
    psfn : numpy ndarray 2d
        2d array with the normalized PSF template
    
    planet_position_list : list which has tuples
        The list of (X,Y) which is the center of aperture.
    
    fwhm : int, optional
        Full width at half maximum for the instrument PSF
    
    norm : int, optional
        1 for Laplacian, 2 for Gaussian 

    Returns
    -------
    a : float
        Obtained flux for the specified trajectory.
    
    likelihood : float
        The log likelihood ratio for the specified trajectory.
    
    planet_position : tuple
        the assumed planet position (X,Y)  
    """
    y, x = planet_position[0]
    cy, cx = vip.var.frame_center(residual_cube)
    y_cy, x_cx = y - cy, x - cx
    rad, theta = vip.var.cart_to_pol(x_cx, y_cy)
    mask = trajectory_mask(residual_cube.shape, (cy, cx), angle_list, 
                           rad, theta, fwhm)

    P = psf_cube(residual_cube, psfn, angle_list, rad, theta)  

    std = residual_cube.std(axis=0)

    # Normalize R and P by the std, and select only the pixels in the mask
    residual_cube, P = (residual_cube/std)[mask], (P/std)[mask]
    P = P[~np.isnan(residual_cube)]
    residual_cube = residual_cube[~np.isnan(residual_cube)]
    if norm == 1:    
        ais = residual_cube / P
        i_min = np.argmin([np.abs(residual_cube - ais[i] * P).sum() 
                           for i in (range(len(ais)))])
        a = ais[i_min]
        likelihood = np.abs(residual_cube).sum() - np.abs(residual_cube - a * P).sum()
    
    elif norm == 2:
        a = np.sum(residual_cube * P) / np.sum(P ** 2)
        likelihood = (residual_cube**2).sum() - ((residual_cube - a * P)**2).sum()
    
    else:
        raise ValueError("norm should be 1 or 2")
    
    if a <=0:
        return a, 0, planet_position
    else:
        return a, likelihood, planet_position

