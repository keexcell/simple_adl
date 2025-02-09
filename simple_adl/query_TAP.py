"""
Query Rubin Table Access Protocol for DC2 data.
"""
__author__ = "Sidney Mau and Kabelo Tsiane"

from astropy import units as u
from astropy.coordinates import SkyCoord

from dustmaps.sfd import SFDQuery
from dustmaps.config import config

# Redenning coefficients
# R_g = 3.185
# R_r = 2.140
# R_i = 1.571   # I can't find any reference for these old values 

R_g = 3.64
R_r = 2.70
R_i = 2.06

config['data_dir'] = '/home/kb/software/simple_adl/notebooks/dustmaps'     # dustmaps
sfd = SFDQuery()

def query(service, ra, dec, radius=1.0, gmax=23.5):
    """ Return data queried from Rubin TAP
    
    Parameters
    ----------
    service: str 
        TAP service
    ra: float
        Right Ascension [deg]
    dec: float 
        Declination [deg]
    radius: float
        Radius around (ra, dec) [deg]

    Returns
    -------
    good_results: DataFrame
    """

    # Redenning coefficients
    # R_g = 3.185
    # R_r = 2.140
    # R_i = 1.571   # I can't find any reference for these old values 

    R_g = 3.64
    R_r = 2.70
    R_i = 2.06

    # Define our reference position on the sky and E(B-V) at that position using SFD reddening maps
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    E_BV = sfd(coord)
    radius = radius * u.deg

    A_g = R_g*E_BV
    A_r = R_r*E_BV
    A_i = R_i*E_BV
    
    # Quality selection and star--galaxy separation adapted from
    # https://github.com/LSSTDESC/DC2-analysis/blob/master/tutorials/object_pandas_stellar_locus.ipynb

    snr_threshold = 5
    mag_err_threshold = 1/snr_threshold
    mag_threshold =26

    # assuming extendedness is the same in all bands we assume g_extendedness matches extendedness for dp0.2 tables
    
    safe_max_extended = 1.0
    
    query = f"""
        SELECT
            coord_ra AS ra, coord_dec AS dec,
            scisql_nanojanskyToAbMag(g_cModelFlux) AS mag_g,
            scisql_nanojanskyToAbMag(r_cModelFlux) AS mag_r,
            scisql_nanojanskyToAbMagSigma(g_cModelFlux, g_cModelFluxErr) AS magerr_g, 
            scisql_nanojanskyToAbMagSigma(r_cModelFlux, r_cModelFluxErr) AS magerr_r,
            scisql_nanojanskyToAbMag(g_cModelFlux) - {A_g} AS mag_corrected_g,
            scisql_nanojanskyToAbMag(r_cModelFlux) - {A_r} AS mag_corrected_r,
            g_extendedness
        FROM dp02_dc2_catalogs.Object
        WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {coord.ra.value}, {coord.dec.value}, {radius.value})) = 1
        AND g_extendedness < {str(safe_max_extended)}
    """
    
    job = service.submit_job(query)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    async_results = job.fetch_result()
    results = async_results.to_table().to_pandas()
    job.delete()

    good_snr = (results['magerr_g'] < mag_err_threshold) & (results['magerr_r'] < mag_err_threshold)
    good_results = results[good_snr]
    
    return good_results

    
def query_truth(service, ra, dec, radius=1):
    """ Return data queried from Rubin TAP using true star-galaxy separation
    Parameters
    ----------
    service: str 
        TAP service
    ra: float
        Right Ascension [deg]
    dec: float 
        Declination [deg]
    radius: float
        Radius around (ra, dec) [deg]

    Returns
    -------
    df: DataFrame
        Data from Rubin TAP
    """

    # Define our reference position on the sky and E(B-V) at that position using SFD reddening maps
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    E_BV = sfd(coord)
    radius = radius * u.deg

    A_g = R_g*E_BV
    A_r = R_r*E_BV
    A_i = R_i*E_BV
    
    query = f"""
        SELECT
            coord_ra AS ra, coord_dec AS dec,
            scisql_nanojanskyToAbMag(g_cModelFlux) AS mag_g,
            scisql_nanojanskyToAbMag(r_cModelFlux) AS mag_r,
            scisql_nanojanskyToAbMagSigma(g_cModelFlux, g_cModelFluxErr) AS magerr_g, 
            scisql_nanojanskyToAbMagSigma(r_cModelFlux, r_cModelFluxErr) AS magerr_r,
            scisql_nanojanskyToAbMag(g_cModelFlux) - {A_g} AS mag_corrected_g,
            scisql_nanojanskyToAbMag(r_cModelFlux) - {A_r} AS mag_corrected_r,
            g_extendedness AS extended_class
        FROM dp02_dc2_catalogs.Object as obj
        JOIN dp02_dc2_catalogs.MatchesTruth as truth
        ON truth.match_objectId = obj.objectId
        WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {coord.ra.value}, {coord.dec.value}, {radius.value})) = 1
        AND truth.match_objectId >= 0 
        AND truth.match_candidate = 1
        AND truth.truth_type = 2
    """
    df = service.search(query).to_table().to_pandas()
    df['MC_SOURCE_ID'] = 0
    
    return df