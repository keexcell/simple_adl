"""
Plotting the observational selection function
"""
__author__ = 'Alex Drlica-Wagner, Sidney Mau and Kabelo Tsiane'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

def gen_contours(mbins, rbins, dmin, dmax, pdet, survey):
    """ Generates the 50% detection efficiency contours used in the observational selection function.

    Parameters
    ----------
    mbins: ndarray
        Magnitude bins
    rbins (ndarray):
        Size bins
    dmin: float
        Minimum distance
    dmax: float
        Maximum distance
    pdet: ndarray
        Probability of detection
    survey: str
        Name of the survey the observational selection function is based on
    """
    # contour expects bin centers
    mcent = (mbins[:-1]+mbins[1:])/2.
    rcent = (rbins[:-1]+rbins[1:])/2.
    CS = plt.contour(mcent,rcent,pdet.T,levels=[0.50],
                     colors='r',linstyles='dashed',linewidths=0.5)

    data = CS.allsegs[0][0]
    outfile= '%s_p50_d%06.2f.npy'%(survey,np.sqrt(dmin*dmax))
    print("Writing %s..."%outfile)
    np.save(outfile,data)

    return


def setdefaults(kwargs,defaults):
    for k,v in defaults.items():
        kwargs.setdefault(k,v)
    return kwargs


def draw_survey(dist,survey,**kwargs):
    """ Draws the 50% detection efficiency contour for a survey at a specific distance

    Parameters
    ----------
    dist: float
        Geometric mean distance of the distance bin
    survey: str
        Name of the survey
    """
    defaults = dict(ls='--',color='k',zorder=1)
    kwargs = setdefaults(kwargs,defaults)
    abs_mag, r_physical,a,b,c = calc_survey(dist,survey)
    sel = abs_mag < b
    plt.plot(abs_mag[sel],r_physical[sel], **kwargs)

    return


def calc_survey(dist,survey='lsst'):
    """ Calculate the 50% detection efficiency contour for a survey at a specific distance

    Parameters
    ----------
    dist: float
        Geometric mean distance of the distance bin
    survey: str
        Name of the survey
    """
    mbins = np.arange(-11,2.5,0.75)

    if survey == 'des':
        #    Dist   A0     Mv0    logr0
        P0=[[11.3 ,  21.5,   7.8 ,   3.8],
            [22.6 ,  24.1,   8.3 ,   4.2],
            [45.2 ,  17.2,   5.2 ,   4.3], #tuned
            [90.5 ,  8.6 ,   1.2 ,   4.1],
            [181.0,  6.6 ,   -1.1,   4.1],
            [362.0,  6.3 ,   -2.3,   4.3]]

    elif survey == 'ps1':
        #    Dist      A0     Mv0    logr0
        P0=[[11.3 ,  21.7,   6.7 ,   3.9],
            [22.6 ,  16.4,   4.3 ,   4.0],
            [45.2 ,  11.5,   1.0 ,   4.0],
            [90.5 ,  8.6 ,   -1.0,   4.0],
            [181.0,  7.2 ,   -2.4,   4.2],
            [362.0,  4.2 ,   -4.8,   4.0]]
        P0=[[11.3 ,  22.8 ,  7.1  ,  4.0],
            [22.6 ,  19.0 ,  5.0  ,  4.1],
            [45.2 ,  14.1 ,  1.8  ,  4.2],
            [90.5 ,  11.0 ,  -0.3 ,  4.3],
            [181.0,  7.5  ,  -2.2 ,  4.2],
            [362.0,  6.8  ,  -4.0 ,  4.4],]

    elif survey=='lsst_true':
        #    Dist      A0     Mv0    logr0
        P0=[[11.3 ,  22.7 ,  10.0 ,  4.0  ],
            [22.6 ,  24.7 ,  10.0 ,  4.3  ],
            [45.2 ,  25.1 ,  8.7  ,  4.7  ],
            [90.5 ,  24.4 ,  7.6  ,  5.0  ],
            [181.0,  11.0 ,  3.2  ,  4.4  ],
            [362.0,  8.6  ,  0.2  ,  4.7  ]]

    elif survey=='lsst_measured':
        #    Dist      A0     Mv0    logr0
        P0=[[11.3 ,  22.4 ,  10.0 ,  3.9  ],
            [22.6 ,  25.3 ,  10.0 ,  4.3  ],
            [45.2 ,  31.6 ,  9.9  ,  4.9  ],
            [90.5 ,  21.5 ,  6.6  ,  4.8  ],
            [181.0,  9.0  ,  1.9  ,  4.3  ],
            [362.0,  4.1  ,  -1.2 ,  3.9  ]]

    elif survey=='lsst_corrected':
        #    Dist   A0     Mv0    logr0
        P0=[[11.3 ,  25.0 ,  10.0 ,  4.0  ],
            [22.6 ,  30.8 ,  10.0 ,  4.6  ],
            [45.2 ,  20.7 ,  6.9  ,  4.4  ],
            [90.5 ,  23.4 ,  5.8  ,  5.0  ],
            [181.0,  14.3 ,  2.1  ,  4.8  ],
            [362.0,  6.5  ,  -1.4 ,  4.3  ]]



    # A = normalization factor (how curved)
    # B = Mv0 -- the absolute magnitude cut
    # C = R0  -- the r_physical cut
    PARAMS = np.rec.fromrecords(P0,names=['D','A','B','C'])

    a = np.interp(dist,PARAMS['D'],PARAMS['A'])
    b = np.interp(dist,PARAMS['D'],PARAMS['B'])
    c  = np.interp(dist,PARAMS['D'],PARAMS['C'])

    abs_mag = np.linspace(mbins.min(),mbins.max(),1000)
    r_physical = a/(abs_mag - b) + c

    return abs_mag, r_physical, a, b, c


def plot_osf(sims: pd.DataFrame, title: str, save: bool = False, out_name: str = None,
             contours: bool = False, survey: str = None, threshold: float = 5.5, 
             cmap: str = 'viridis', **kwargs) -> None:
    """ Plot the observational selection function

    Parameters
    ----------
        sims: DataFrame
            Sims dataframe
        title: str
            Title for the plot
        save: bool
            Flag to save figure
        out_name: str
            Name of the output file if saved
        gen_contours: bool
            Flag to generate 50% detection efficiency contours
        survey: str
            Survey name for efficiency contours
    """
    # bins in distance
    dbins = 2**np.arange(3,10)
    # bins in absolute magnitude
    mbins = np.arange(-11,2.5,0.75)
    # bins in physical radius
    rbins = np.arange(0,3.75,0.3)

    fig,axes = plt.subplots(2,3,figsize=(11,7))
    plt.subplots_adjust(wspace=0, hspace=0)
    axes[0,0].axes.get_xaxis().set_visible(False)
    axes[0,1].axes.get_yaxis().set_visible(False)
    axes[0,2].axes.get_yaxis().set_visible(False)
    axes[1,2].axes.get_yaxis().set_visible(False)
    axes[1,1].axes.get_yaxis().set_visible(False)
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, ls='--', label='LSST Ideal'),
        Line2D([0], [0], color='red', lw=1, ls='-.', alpha=0.8, label='LSST Measured'),
        Line2D([0], [0], color='red', lw=1, ls=':', alpha=0.6, label='LSST Corrected'),
        Line2D([0], [0], color='black', lw=1, ls='--', label='DES')
    ]
    
    fig.suptitle(title, fontsize=15, x=0.51, y=0.97)
    for i,(dmin,dmax) in enumerate(zip(dbins[:-1],dbins[1:])):
        plt.sca(axes.flat[i])
        plt.xlabel("$M_V$", fontsize=15)
        plt.ylabel("log$_{10}$(r/pc)", fontsize=15)
        plt.text(-10, 0.5, "%i < D < %i kpc"%(dmin,dmax), fontsize=9, )
        plt.xlim(-10.25,1.75)
        plt.ylim(0, 3.3)
        s = sims[(sims['DISTANCE'] >= dmin)&(sims['DISTANCE'] < dmax)]

        det = s["SIG"] >= threshold

        total = np.histogram2d(s['ABS_MAG'],np.log10(s['R_PHYSICAL']*1e3),
                           bins=[mbins,rbins])[0]
        ndet = np.histogram2d(s['ABS_MAG'],np.log10(s['R_PHYSICAL']*1e3),
                           weights=det,bins=[mbins,rbins])[0]

        pdet = ndet.astype(float)/total
        im = plt.pcolormesh(mbins,rbins,pdet.T,rasterized=True, cmap=cmap)

        draw_survey(np.sqrt(dmin*dmax), survey='lsst_true', color='red', ls='--', lw=2, label='LSST Ideal')
        draw_survey(np.sqrt(dmin*dmax), survey='lsst_measured', color='red', ls='-.', alpha=0.8, label='LSST Measured')
        draw_survey(np.sqrt(dmin*dmax), survey='lsst_corrected', color='red', ls= ':', alpha=0.6, label='LSST Corrected')
        draw_survey(np.sqrt(dmin*dmax), survey='des', color='black', label='DES')
        
        if contours:
            gen_contours(mbins, rbins, dmin, dmax, pdet, survey=survey)
            
    cb_ax = fig.add_axes([0.92, 0.11, 0.025, 0.77])  # [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cb_ax, label='Detection Efficiency')
    # cb = fig.colorbar(im, ax=axes.ravel().tolist(), label='Detection Efficiency')

    fig.legend(legend_elements, [e.get_label() for e in legend_elements],
               loc='upper center',
               bbox_to_anchor=(0.51, 0.94),
               ncol=4,
               frameon=False,
               fontsize='medium')

    if save:
        plt.savefig(out_name, bbox_inches='tight')


    return
