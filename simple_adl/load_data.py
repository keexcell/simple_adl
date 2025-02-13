#!usr/bin/env python
"""
Load sim data.
"""

__author__ = 'Kabelo Tsiane'

import os
import glob

import numpy as np
import fitsio as fits
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord

import simple_adl.survey
from simple_adl.query_TAP import query, query_truth
from lsst.rsp import get_tap_service
from simple_adl.search import cut_isochrone_path

service = get_tap_service()
assert service is not None
assert service.baseurl == "https://data.lsst.cloud/api/tap"

radius = 2.     # degrees

def get_catalog_file(catalog_dir, mc_source_id):
    """
    Parameters
    ----------
    catalog_dir: str
        String corresponding to directory containing the stellar catalog infiles
    mc_source_id: int
        Integer corresponding to the target MC_SOURCE_ID value

    Returns
    -------
    catalog_infile: str
        String corresponding to filename of stellar catalog containing mc_source_id
    """
    catalog_infiles = sorted(glob.glob(catalog_dir + '/*catalog*.fits'))
    mc_source_id_array = []
    catalog_infile_index_array = []
    for ii, catalog_infile in enumerate(catalog_infiles):
        mc_source_id_min = int(os.path.basename(catalog_infile).split('.')[0].split('mc_source_id_')[-1].split('-')[0])
        mc_source_id_max = int(os.path.basename(catalog_infile).split('.')[0].split('mc_source_id_')[-1].split('-')[1])
        assert (mc_source_id_max > mc_source_id_min) & (mc_source_id_min >= 1), 'Found invalue MC_SOURCE_ID values in filenames'
        mc_source_id_array.append(np.arange(mc_source_id_min, mc_source_id_max + 1))
        catalog_infile_index_array.append(np.tile(ii, 1 + (mc_source_id_max - mc_source_id_min)))

    mc_source_id_array = np.concatenate(mc_source_id_array)
    catalog_infile_index_array = np.concatenate(catalog_infile_index_array)

    assert len(mc_source_id_array) == len(np.unique(mc_source_id_array)), 'Found non-unique MC_SOURCE_ID values in filenames'
    assert np.in1d(mc_source_id, mc_source_id_array), 'Requested MC_SOURCE_ID value not among files'
    mc_source_id_index = np.nonzero(mc_source_id == mc_source_id_array)[0][0] # second [0] added by smau 7/23/18 to fix incompatiable type bug
    return catalog_infiles[catalog_infile_index_array[mc_source_id_index]]


def load_sim_data(sim_dir, mc_source_id):
    """ Load info for injecting satellite simulations

    Parameters
    ----------
    sim_dir: str
        Satellite simulation directory
    mc_source_id: int
        Satellite ID
    """
    cat_file = get_catalog_file(sim_dir, mc_source_id)
    cat_fits = fits.FITS(cat_file)
    w = cat_fits[1].where(f'MC_SOURCE_ID == {mc_source_id}')
    try:
        data = cat_fits[1][w]
        cat_fits.close()
        return data
    except IndexError:
        print(f'NO SIM DATA, {mc_source_id} NOT FOUND')
        return None


def load_simdf(catalog_dir: str, results_dir: str) -> pd.DataFrame:
    """ Combines information from the sim catalog files and the results directory to build the sims dataframe for results plots.
    
    Parameters
    ----------
    catalog_dir: str
        Directory with the sim catalogs
    results_dir: str
        Directory with sim search results 
        
    Returns
    -------
    sims: DataFrame
        Sims dataframe
    """
    results_df = combine_sim_results(results_dir)
    pop_df = combine_sim_pop(catalog_dir, results_dir)
    
    sims = pd.merge(pop_df,results_df[['MC_SOURCE_ID','SIG']],on='MC_SOURCE_ID',how='left')
    sims.set_index('MC_SOURCE_ID', inplace=True)
    sims = sims[sims['FRACDET_CORE'] == 1]
    sims = sims[sims['FRACDET_WIDE'] == 1]
    sims = sims[sims['FRACDET_HALF'] == 1]
    sims.loc[sims.DIFFICULTY == 2, 'SIG'] = 37.5
    
    return sims


def combine_sim_results(results_dir: str) -> pd.DataFrame:
    """ Take all the sim results dataframes from the results directory and make them one big dataframe.
    """
    files = glob.glob(os.path.join(results_dir, '*.csv'))
    first_file = files[0]
    results_df = pd.read_csv(first_file)
    for f in files:
        if f == first_file:
            continue
        df = pd.read_csv(f)
        results_df = pd.concat((results_df, df), ignore_index=True)

    return results_df


def combine_sim_pop(catalog_dir: str, results_dir: str) -> pd.DataFrame:
    """ Take all the sim catalog population files and combine them into one big sim population dataframe.
    """
    population_file = glob.glob(os.path.join(catalog_dir, '*population*'))
    sim_pop = fits.read(population_file[0])
    sim_pop = sim_pop.byteswap().newbyteorder() 
    pop_df = pd.DataFrame(sim_pop)
    file_ids = [files[8:23] for files in os.listdir(results_dir) if 'ipynb' not in files]
    for i,f in enumerate(population_file):
        if i == 0:
            continue
        for ids in file_ids:
            if ids in f:
                sim_pop = fits.read(f)
                sim_pop = sim_pop.byteswap().newbyteorder()
                df = pd.DataFrame(sim_pop)
                pop_df = pd.concat((pop_df, df), ignore_index=True)
                break
    return pop_df


def get_sim_file(sim_dir, sim_id = None, truth_matching=True):
    """ Given the sims directory and a sim_id return the corresponding sim catalog file.
    This needs a little work.
    """
    if truth_matching: 
        if sim_id is not None:
            sims_batch = get_sim_batch(sim_id)
            file = sim_dir + 'sim_population_lsst_dc2_v7_mc_source_id_' + sims_batch + '.fits'
            return(file)
        else: 
            file = glob.glob(os.path.join(sim_dir, '*population*'))
            return(file)
    else: 
        if sim_id is not None:
            sims_batch = get_sim_batch(sim_id)
            file = sim_dir + 'sim_population_lsst_dc2_v6_mc_source_id_' + sims_batch + '.fits'
            return(file)
        else: 
            file = glob.glob(os.path.join(sim_dir, '*population*'))
            return(file)


def get_sim_batch(pop_file=None, sim_id=None):
    """ Gets the batch number for the sims

    Given a population file it will extract the batch directly from the file name.
    
    Given a sim id it will find the corresponding sim population file batch 
    e.g sim_id 27355 has batch 0027301-0027400
    """
    if sim_id is not None and pop_file is None:
        file_start = str((sim_id//100)*100 + 1).zfill(7)
        file_end = str((sim_id//100)*100 + 100).zfill(7)
        sims_batch = file_start + '-' + file_end
    elif pop_file is not None and sim_id is None:
        start = pop_file.find('0')
        end = start + 15
        sims_batch = pop_file[start:end]
    return sims_batch


def clean_sim_pop(sim_population):
    """ Remove sims on the edge of the DC2 footprint to prevent detection anomalies.
    """
    sim_population  = sim_population.byteswap().newbyteorder()    # resetting byte order for compatibility
    sim_population = sim_population[sim_population['FRACDET_CORE'] == 1]
    sim_population = sim_population[sim_population['FRACDET_WIDE'] == 1]
    sim_population = sim_population[sim_population['FRACDET_HALF'] == 1]
    return sim_population
    

def get_merged_data(real_data, sim_data, survey):
    """ Merge the sim data and DC2 data into one dataframe.
    """
    # merging the sims and dc2 data
    frames = [real_data[real_data.columns[:-1]], sim_data[real_data.columns[:-1]]]
    merged_data = pd.concat(frames)  # pd.Dataframe
    # perform mag cut on the merged data
    good_snr = (merged_data['magerr_g'] < 0.2) & (merged_data['magerr_r'] < 0.2)
    merged_data = merged_data[good_snr]
    good_mag = (merged_data['mag_g'] < survey.catalog['mag_max']) & (merged_data['mag_r'] < survey.catalog['mag_max'])
    merged_data = merged_data[good_mag]
    return merged_data

    
def mask_region_sims(sim_data, position, radius):
    """ Mask to ensure we only use sims within the queried data's footprint
    """
    
    c2 = SkyCoord(sim_data['ra'], sim_data['dec'], unit='deg', frame='icrs')
    center = SkyCoord(position[0], position[1], unit='deg')
    d2d = center.separation(c2) 
    catalogmsk = d2d < radius*u.deg
    sim_data = sim_data[catalogmsk]
    return sim_data

def sim_pop(file):
    """ Load sim population and their positions
    """
    sim_population = fits.read(file)
    sim_positions = np.unique(sim_population[['RA', 'DEC']])
    sim_population = clean_sim_pop(sim_population)
    return sim_population, sim_positions

def load_field(sim_population, position, service=service, truth_match=True, verbose=True):
    """ Load DC2 data and sim data at given position
    """
    sim_population_at_position = sim_population[sim_population[['RA', 'DEC']] == position]
    sim_population_at_position = pd.DataFrame(sim_population_at_position)
    
    if len(sim_population_at_position['MC_SOURCE_ID'].values) == 0:
        if verbose: print(f'No sims at {position}')
        return
        
    if verbose: 
        print(f'Satellites at {position} :\n {sim_population_at_position['MC_SOURCE_ID'].values}')
        
    print(f'Querying region {position}')
        
    if truth_match:
        real_data = query_truth(service, position[0], position[1], radius)
    else:
        real_data = query(service, position[0], position[1], radius)
    
    return real_data, sim_population_at_position


def load_merge(mcid, real_data, position, survey, truth_match=True, verbose=True):
    if truth_match:
        sim_dir = '/project/shared/data/satsim/lsst_dc2_v7'
    else:
        sim_dir = '/project/shared/data/satsim/lsst_dc2_v6'
    sim_data = load_sim_data(sim_dir, mcid)
    if sim_data is not None:
        sim_data = sim_data.byteswap().newbyteorder()   # resetting byte order for compatibility
        sim_data = pd.DataFrame(sim_data)
        sim_data = mask_region_sims(sim_data, position, radius)
        if sim_data.empty:
            if verbose:
                print(f'No sim data to inject into region at ({position[0]},{position[1]}) after applying mask')
            return None
        merged_data = get_merged_data(real_data, sim_data, survey)
        return merged_data, sim_data
    else:
        return None
    

    
def load_and_merge(sim_id, truth_matching=True):
    """ Create merged dataframe and display CMDs using true information or observational information

    This is mainly used for the plot_cmd notebook but is now broken.

    Parameters
    ---------z
    sim_id: int
        The ID for the sim to plot
    truth_matching: bool
        Flag to use true or measured star-galaxy separation

    """
    if truth_matching: 
        sim_dir = '/project/shared/data/satsim/lsst_dc2_v7/'
    else:
        sim_dir = '/project/shared/data/satsim/lsst_dc2_v6/'
        
    with open('config.yaml') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        survey = simple_adl.survey.Survey(cfg)

    file = get_sim_file(sim_dir, sim_id, truth_matching) 
    sim_population = fits.read(file)
    sim_positions = np.unique(sim_population[['RA', 'DEC']])
    sim_population = clean_sim_population_data(sim_population)
    
    radius = 2
    for iii,position in enumerate(sim_positions):
        sim_population_at_position = sim_population[sim_population[['RA', 'DEC']] == position]
        patch_df = pd.DataFrame(sim_population_at_position)
        
        if len(patch_df['MC_SOURCE_ID'].values) == 0:
            print(f'No sims at {position}')
            continue
        elif sim_id not in patch_df['MC_SOURCE_ID'].values:
            continue
            
        print(f'Querying region {position}')
        if truth_matching:
            real_data = query_truth(service, position[0], position[1], radius)
        else:
            real_data = query(service, position[0], position[1], radius)    # takes significantly longer due to galaxy contamination
    
        print('Satellites at ', position, ':\n', patch_df['MC_SOURCE_ID'].values)
        for mcid in sim_population_at_position['MC_SOURCE_ID']:
            if mcid == sim_id:
                sim_data = load_sim_data(sim_dir, mcid)
                if sim_data is not None:
                    sim_data = sim_data.byteswap().newbyteorder()   # resetting byte order for compatibility
                    sim_data = pd.DataFrame(sim_data)
                    sim_data = mask_region_sims(sim_data, position, radius)
                    if sim_data.empty:
                        print(f'No sim data to inject into region at ({position[0]},{position[1]}) after applying mask')
                        continue
                    merged_data = get_merged_data(real_data, sim_data, survey)
                    ra = position[0]
                    dec = position[1]
                    region = simple_adl.survey.Region(survey, ra, dec)
                    region.data = merged_data
                    distance_modulus = patch_df.loc[patch_df['MC_SOURCE_ID'] == mcid]['DISTANCE_MODULUS']
                    distance_modulus = distance_modulus.values[0]
                    merged_data['DISTANCE_MODULUS'] = pd.Series(distance_modulus)
                    # Create isochrone
                    iso_search = simple_adl.isochrone.Isochrone(survey=survey.isochrone['survey'],
                                                           band_1=survey.band_1.lower(),
                                                           band_2=survey.band_2.lower(),
                                                           age=12.0, #survey.isochrone['age'],
                                                           metallicity=0.00010, #survey.isochrone['metallicity'],
                                                           distance_modulus=distance_modulus)
    
                    iso_selection = cut_isochrone_path(region.data[survey.mag_dered_1], 
                                                          region.data[survey.mag_dered_2],
                                                          region.data[survey.mag_err_1],
                                                          region.data[survey.mag_err_2],
                                                          iso_search,
                                                          survey.catalog['mag_max'],
                                                          radius=0.1, 
                                                          return_all=True)
                    return sim_data, real_data, merged_data, iso_search, iso_selection

    return

