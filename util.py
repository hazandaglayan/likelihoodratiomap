#! /usr/bin/env python

__author__ = 'Xavier Lambein, Simon Vary, Hazan Daglayan'
__all__ = ['trajectory_pixels',
           'trajectory_mask',
           'mask_annulus',
           'pixels_in_annulus']

import numpy as np


def trajectory_pixels(center, angle_list, rad, theta):
    """Compute the pixel locations in a frame of a trajectory.
    
    A trajectory is defined as the path followed by a fixed point in the
    sky of an ADI cube.  For example, companions follow trajectories.
    
    Parameters
    ----------
    center : list of len 2
    angle_list : np.array
        The angles of the ADI cube, in degrees.
    
    rad : float
    theta : float
        The radial coordinates of the initial position of the trajectory.
    
    Returns
    -------
    ys : np.array
    xs : np.array
        A pair of arrays containing the y and x locations of the trajectory, respectively.
    """

    cy = center[0]
    cx = center[1]
    
    ys = cy + np.sin(np.deg2rad(-angle_list + theta)) * rad
    xs = cx + np.cos(np.deg2rad(-angle_list + theta)) * rad
    
    return ys, xs




def trajectory_mask(cube_shape, center, angle_list, rad, theta, fwhm=4):
    """Create a mask along a trajectory.
    
    Given a trajectory starting point, creates a boolean 3D array equal to True
    for a disk of radius FWHM/2 repeated along that path.
    
    SV 12/1/22: 
        split dataset input into: cube_shape, center, angle_list, fwhm=4
    Parameters
    ----------
    cube_shape : tuple
        Shape of cube
    
    center : tuple
        Center of the frame
    
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    
    rad : float
    theta : float
        The starting point of the trajectory.

    fwhm : int, optional
        Full width at half maximum for the instrument PSF
    
    Returns
    -------
    np.array
        The mask, a 3D boolean array.
    """
    mask = np.zeros(cube_shape, bool)
    pixels = trajectory_pixels(center, angle_list, rad, theta)
    
    radius = fwhm/2
    r2 = radius*radius
    
    yy, xx = np.ogrid[:cube_shape[1], :cube_shape[2]]
    for i, (y, x) in enumerate(zip(*pixels)):
        mask[i] = ( (xx - x)**2 + (yy - y)**2 <= r2 )
    return mask

def mask_annulus(shape, center, inner_radius, outer_radius):
    """
    Create boolean matrix which has TRUE for the specified annulus

    Parameters
    ----------
    shape : tuple
        Shape of the frame
    center : tuple
        Center of frame.
    inner_radius : int or float
        inner radius of the annulus
    outer_radius : int or float
        outer radius of the annulus

    """
    cy, cx = center

    ys, xs = np.indices(shape)
    return ((ys - cy )**2 + (xs - cx )**2 <= (outer_radius)**2) &\
           ((ys - cy )**2 + (xs - cx )**2 >= inner_radius**2)

def pixels_in_annulus(shape, center, inner_radius, outer_radius):
    """
    Compute the pixels in the annulus.

    Parameters
    ----------
    shape : tuple
        Shape of the frame
    center : tuple
        Center of frame.
    inner_radius : int or float
        inner radius of the annulus
    outer_radius : int or float
        outer radius of the annulus

    Returns
    -------
    pixels_list : list which has tuples
        The list of pixels (X,Y) in the annulus

    """
    ys, xs = np.indices(shape)
    mask = mask_annulus(shape, center, inner_radius, outer_radius)
    pixels_list = []
    for pixel in zip(ys[mask], xs[mask]):
        pixels_list.append(pixel)
    return pixels_list

