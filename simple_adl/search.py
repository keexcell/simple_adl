#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Sidney Mau, Kabelo Tsiane"

import sys
import os
import glob
import yaml
import numpy as np
import healpy as hp
import scipy.interpolate

import fitsio as fits

import simple_adl.survey
import simple_adl.isochrone
from simple_adl.coordinate_tools import distanceModulusToDistance, angsep
from IPython.core.debugger import set_trace

#-------------------------------------------------------------------------------

def cut_isochrone_path(g, r, g_err, r_err, isochrone, mag_max, radius=0.1, return_all=False):
    """
    Cut to identify objects within isochrone cookie-cutter.
    """
    if np.all(isochrone.stage == 'Main'):
        # Dotter case
        index_transition = len(isochrone.stage)
    else:
        # Other cases
        index_transition = np.nonzero(isochrone.stage >= isochrone.hb_stage)[0][0] # + 1 

    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]

    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))

     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))

    cut = np.logical_or(cut_1, cut_2)

    mag_bins = np.arange(17., mag_max+0.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = np.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = np.median(np.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

def write_output(results_dir, nside, pix_nside_select, best_ra_peak, best_dec_peak, best_r_peak, best_distance_modulus, 
                n_obs_peak, n_obs_half_peak, n_model_peak, 
                best_sig_peak, mc_source_id, mode, outfile):
    if os.path.exists(f'{results_dir}/{outfile}') and 'sim_pop' in outfile:    # better that you delete a sim pop results file manually
        print(f'Files {outfile} already processed')
        return
    #saving only the best
    data = np.array([best_sig_peak, best_ra_peak, best_dec_peak, best_distance_modulus, best_r_peak, n_obs_peak, n_obs_half_peak, n_model_peak, mc_source_id]) 
    f = open(os.path.join(results_dir,outfile), 'ab')
    if os.stat(os.path.join(results_dir,outfile)).st_size == 0:
        np.savetxt(f, [data], fmt="%.2f", delimiter=',', header="SIG,RA,DEC,MODULUS,R,N_OBS,N_OBS_HALF,N_MODEL,MC_SOURCE_ID", comments="")
    else:
         np.savetxt(f, [data], fmt="%.2f", delimiter=',')
    f.close()
    return

def search_by_distance(survey, region, distance_modulus, iso_sel, extension=None, verbose=True):
    """
    Idea: 
    Send a data extension that goes to faint magnitudes, e.g., g < 24.
    Use the whole region to identify hotspots using a slightly brighter 
    magnitude threshold, e.g., g < 23, so not susceptible to variations 
    in depth. Then compute the local field density using a small annulus 
    around each individual hotspot, e.g., radius 0.3 to 0.5 deg.
    """
    if (len(region.data[iso_sel]) == 0):
        return [], [], [], [], [], [], [], []

    ra_peak_array = []
    dec_peak_array = []
    r_peak_array = []
    sig_peak_array = []
    distance_modulus_array = []
    n_obs_peak_array = []
    n_obs_half_peak_array = []
    n_model_peak_array = []

    region.density = region.characteristic_density(iso_sel, verbose=verbose)
    x_peak_array, y_peak_array, angsep_peak_array = region.find_peaks(iso_sel)
    for x_peak, y_peak, angsep_peak in zip(x_peak_array, y_peak_array, angsep_peak_array):
        # Aperture fitting
        if verbose: print('Fitting aperture to hotspot...')
        ra_peaks, dec_peaks, r_peaks, sig_peaks, n_obs_peaks, n_obs_half_peaks, n_model_peaks, density = region.fit_aperture(iso_sel, x_peak, y_peak, angsep_peak, verbose=verbose, extension=extension)
        
        ra_peak_array.append(ra_peaks)
        dec_peak_array.append(dec_peaks)
        r_peak_array.append(r_peaks)
        sig_peak_array.append(sig_peaks)
        distance_modulus_array.append(distance_modulus*np.ones(len(ra_peaks)))
        n_obs_peak_array.append(n_obs_peaks)
        n_obs_half_peak_array.append(n_obs_half_peaks)
        n_model_peak_array.append(n_model_peaks)
        
    try:
        ra_peak_array = np.concatenate(ra_peak_array)
        dec_peak_array = np.concatenate(dec_peak_array)
        r_peak_array = np.concatenate(r_peak_array)
        sig_peak_array = np.concatenate(sig_peak_array)
        distance_modulus_array = np.concatenate(distance_modulus_array)
        n_obs_peak_array = np.concatenate(n_obs_peak_array)
        n_obs_half_peak_array = np.concatenate(n_obs_half_peak_array)
        n_model_peak_array = np.concatenate(n_model_peak_array)
    except ValueError:
        print('No arrays to concatenate')

    return ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, distance_modulus_array, n_obs_peak_array, n_obs_half_peak_array, n_model_peak_array

def search(mcid, position, survey, merged_data, sims_at_pos, iso_survey='lsst', outfile=None, save=False, verbose=True):
    """ Search for a satellite.

    Parameters
    ----------
    mcid : int
        satellite mc_source_id to search for
    position : tuple(float, float)
        ra [deg] and dec [deg] of the position on sky
    survey : string
        survey name
    merged_data : pd.DataFrame
        combined dc2 and simulated satellite data
    sims_at_pos : pd.DataFrame
        simulated satellites at the given position
    iso_survey : string
        survey isochrone to use, either 'lsst' or 'des'
    outfile : string
        name of the output file

    Returns
    -------
    iso_selection : tuple(list[bool], np.array[float], np.array[float], np.array[float])
        isochrone information. The first element is the actual isochrone cut used in the search, the rest are used for plotting. The second is the array of magnitude bin centers, the third is the upper bounds of the color, the fourth is the lower bounds of the color. 
    """
    ### KB the iso_survey argument is because I was using the wrong isochrone for so long, now it is needed for testing
    
    if verbose: print('Searching for MC_SOURCE_ID ', mcid)
    ra = position[0]
    dec = position[1]
    region = simple_adl.survey.Region(survey, ra, dec)
    region.data = merged_data
    distance_modulus = sims_at_pos.loc[sims_at_pos['MC_SOURCE_ID'] == mcid]['DISTANCE_MODULUS']
    distance_modulus = distance_modulus.values[0]
    # Create isochrone
    iso_search = simple_adl.isochrone.Isochrone(survey=iso_survey, # survey.isochrone['survey']
                                           band_1=survey.band_1.lower(),
                                           band_2=survey.band_2.lower(),
                                           age=survey.isochrone['age'], # 12 Gyr
                                           metallicity=survey.isochrone['metallicity'],  # 0.00010
                                           distance_modulus=distance_modulus)

    iso_selection = cut_isochrone_path(region.data[survey.mag_dered_1], 
                                          region.data[survey.mag_dered_2],
                                          region.data[survey.mag_err_1],
                                          region.data[survey.mag_err_2],
                                          iso_search,
                                          survey.catalog['mag_max'],
                                          radius=0.1,
                                          return_all=True)
    results = search_by_distance(survey, region, distance_modulus, iso_selection[0], verbose=verbose) 
    ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, distance_modulus_array, n_obs_peak_array, n_obs_half_peak_array, n_model_peak_array = np.asarray(results)
    if len(sig_peak_array) == 0:
        return
    best_ra_peak, best_dec_peak, best_r_peak, best_distance_modulus, n_obs_peak, n_obs_half_peak, n_model_peak, best_sig_peak = 0, 0, 0, 0, 0, 0, 0, 0

    if mcid: 
        mc_source_id_array = np.full_like(distance_modulus_array, mcid)
    else:
        mc_source_id_array = np.zeros(len(distance_modulus_array))

    # Sort peaks according to significance
    index_sort = np.argsort(sig_peak_array)[::-1]
    ra_peak_array = ra_peak_array[index_sort]
    dec_peak_array = dec_peak_array[index_sort]
    r_peak_array = r_peak_array[index_sort]
    sig_peak_array = sig_peak_array[index_sort]
    distance_modulus_array = distance_modulus_array[index_sort]
    n_obs_peak_array = n_obs_peak_array[index_sort]
    n_obs_half_peak_array = n_obs_half_peak_array[index_sort]
    n_model_peak_array = n_model_peak_array[index_sort]
    mc_source_id_array = mc_source_id_array[index_sort]

    # Collect overlapping peaks
    for ii in range(0, len(sig_peak_array)):
        if sig_peak_array[ii] < 0:
            continue
        sep = angsep(ra_peak_array[ii], dec_peak_array[ii], ra_peak_array, dec_peak_array)
        sig_peak_array[(sep < r_peak_array[ii]) & (np.arange(len(sig_peak_array)) > ii)] = -1.

    # Prune the list of peaks
    ra_peak_array = ra_peak_array[sig_peak_array > 0.]
    dec_peak_array = dec_peak_array[sig_peak_array > 0.]
    r_peak_array = r_peak_array[sig_peak_array > 0.]
    distance_modulus_array = distance_modulus_array[sig_peak_array > 0.]
    n_obs_peak_array = n_obs_peak_array[sig_peak_array > 0.]
    n_obs_half_peak_array = n_obs_half_peak_array[sig_peak_array > 0.]
    n_model_peak_array = n_model_peak_array[sig_peak_array > 0.]
    mc_source_id_array = mc_source_id_array[sig_peak_array > 0.]
    sig_peak_array = sig_peak_array[sig_peak_array > 0.] # Update the sig_peak_array last!

    if sig_peak_array[0] > best_sig_peak:
        best_sig_peak = sig_peak_array[0]
        best_ra_peak = ra_peak_array[0]
        best_dec_peak = dec_peak_array[0]
        best_r_peak = r_peak_array[0]
        best_distance_modulus = distance_modulus_array[0]
        n_obs_peak = n_obs_peak_array[0]
        n_obs_half_peak = n_obs_half_peak_array[0]
        n_model_peak = n_model_peak_array[0]
        mc_source_id = mc_source_id_array[0]
        
    if verbose:
        # show results from all peaks
        for ii in range(0, len(sig_peak_array)):
            print('{:0.2f} sigma; (RA, Dec) = ({:0.2f}, {:0.2f}); r = {:0.2f} deg; d = {:0.1f}, mu = {:0.2f} mag, mc_source_id: {:0.2f}'.format(sig_peak_array[ii], 
                     ra_peak_array[ii], 
                     dec_peak_array[ii], 
                     r_peak_array[ii],
                     distanceModulusToDistance(distance_modulus_array[ii]),
                     distance_modulus_array[ii],
                     mc_source_id_array[ii]))
    else:
        print('{:0.2f} sigma; (RA, Dec) = ({:0.2f}, {:0.2f}); r = {:0.2f} deg; d = {:0.1f}, mu = {:0.2f} mag, mc_source_id: {:0.2f}'.format(sig_peak_array[0], 
                 ra_peak_array[0], 
                 dec_peak_array[0], 
                 r_peak_array[0],
                 distanceModulusToDistance(distance_modulus_array[0]),
                 distance_modulus_array[0],
                 mc_source_id_array[0]))

    if best_sig_peak < 5.5:
        print(f'--> {mcid} NOT FOUND')
        print('---------------------------------')
    else:   
        print(f'--> {mcid} FOUND')
        print('---------------------------------')

    if save and outfile is not None:
        try:
            if (len(sig_peak_array) > 0):
                write_output(survey.output['results_dir'], survey.catalog['nside'], region.pix_center, best_ra_peak, best_dec_peak,
                                best_r_peak, best_distance_modulus, 
                                n_obs_peak, n_obs_half_peak, n_model_peak, 
                                best_sig_peak, mcid, 0, outfile)
            else:
                print('No significant hotspots found.')
                nan_array = [np.nan]
                write_output(survey.output['results_dir'], survey.catalog['nside'], region.pix_center,
                                 nan_array, nan_array, nan_array, nan_array, 
                                 nan_array, nan_array, nan_array, nan_array,
                                 [mc_source_id], 0, outfile)
        except Exception as e: 
            print(e)
            print('Data missing, cannot write to file')
    return iso_selection
