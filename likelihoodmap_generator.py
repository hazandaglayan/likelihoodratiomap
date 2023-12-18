import numpy as np
from util import trajectory_mask, psf_cube, pixels_in_annulus
import vip_hci as vip
from hciplot import plot_frames
import multiprocessing
from tqdm import tqdm
from vip_hci.var import frame_center


class LikelihoodRatioMap:
    '''
    Parallel implementation of the likelihoodratio map generation function. Applies 
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
    
    Returns
    -------
    fluxmap : 2d ndarray
        Maximum likelihood estimate of the contrast for each pixel in the field
        of view
    likelihoodmap : 2d ndarray
        Likelihood ratio map.
    
    '''
    def __init__(self, residual_cube, angle_list, psfn, pixels_list=None, fluxmap=None, fwhm=4, norm=1):
        self.residual_cube = residual_cube
        self.angle_list = angle_list
        self.psfn = psfn
        self.fluxmap = fluxmap
        self.fwhm = fwhm
        self.norm = norm
        self.lrmap = np.zeros_like(self.residual_cube[0])
        self.fluxmap_lr = np.zeros_like(self.residual_cube[0])
        if pixels_list is None:
            cy, cx = frame_center(residual_cube)
            nfr, m, n = residual_cube.shape
            pixels = pixels_in_annulus(residual_cube.shape[1:], (cy, cx), 1.5*fwhm, n/2-3)
            pixels_list = []
            for (y,x) in zip(*pixels):
                pixels_list.append((y,x))

        self.pixels_list = pixels_list
        

    def generate(self, plot=False, n_jobs=1):
        '''
        plot : bool, optional
        Plot the maps
    
        nproc : int, optional
        Number of processes for parallel computing.
        '''
        if isinstance(self.residual_cube, np.ndarray):
                        
            pool = multiprocessing.Pool(n_jobs)
            data = [(self.residual_cube, self.angle_list, self.psfn, planet_position, self.fluxmap, self.fwhm, self.norm) for planet_position in self.pixels_list]
            results = list(tqdm(pool.imap(likelihood_value, data), total=len(self.pixels_list)))
            for result in results:
                flux, likelihood, pos = result
                self.fluxmap_lr[pos[0],pos[1]] = flux
                self.lrmap[pos[0],pos[1]] = likelihood
        
        else:
            raise TypeError('`residual cube list` must be a numpy (3d) array ')

        if plot:
            plot_frames(self.lrmap)


def likelihood_value(args):
    
    residual_cube, angle_list, psfn, planet_position, fluxmap, fwhm, norm = args
    
    y, x = planet_position
    cy, cx = vip.var.frame_center(residual_cube)
    y_cy, x_cx = y - cy, x - cx
    rad, theta = vip.var.cart_to_pol(x_cx, y_cy)
    mask = trajectory_mask(residual_cube.shape, (cy, cx), angle_list, 
                        rad, theta, fwhm)

    P = psf_cube(residual_cube, psfn, angle_list, rad, theta)  
    std = residual_cube.std(axis=0)
    norm = norm

    if norm == 1:
        # Normalize R and P by the std, and select only the pixels in the mask
        residual_cube, P = (residual_cube/std)[mask], (P/std)[mask]
        P = P[~np.isnan(residual_cube)]
        residual_cube = residual_cube[~np.isnan(residual_cube)]
        if isinstance(fluxmap, np.ndarray):
            a = fluxmap[y,x]
        else:   
            ais = residual_cube / P
            i_min = np.argmin([np.abs(residual_cube - ais[i] * P).sum() 
                            for i in (range(len(ais)))])
            a = ais[i_min]
        likelihood = np.abs(residual_cube).sum() - np.abs(residual_cube - a * P).sum()
    
    elif norm == 2:
        # Normalize R and P by the std, and select only the pixels in the mask
        residual_cube, P = (residual_cube/(std**2))[mask], (P/(std**2))[mask]
        P = P[~np.isnan(residual_cube)]
        residual_cube = residual_cube[~np.isnan(residual_cube)]
        if isinstance(fluxmap, np.ndarray):
            a = fluxmap[y,x]
        else:
            a = np.sum(residual_cube * P) / np.sum(P ** 2)
        likelihood = (residual_cube**2).sum() - ((residual_cube - a * P)**2).sum()
    
    else:
        raise ValueError("norm should be 1 or 2")
    
    if a <=0:
        return a, 0, planet_position
    else:
        return a, likelihood, planet_position
        