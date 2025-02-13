"""
A bunch of plotting functionality.
"""

__author__ = 'Kabelo Tsiane and Sidney Mau'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

import simple_adl.isochrone

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

def gnomic(mu_ra, mu_dec, ra, dec):
    """
    https://mathworld.wolfram.com/GnomonicProjection.html
    The Gnomic projection is a conformal (angle-preserving) map of
    coordinates on a sphere to coordinates on a plane around some central
    coordinate.
    """
    mu_ra = np.deg2rad(mu_ra)
    mu_dec = np.deg2rad(mu_dec)
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    cos_c = np.sin(mu_dec) * np.sin(dec) + np.cos(mu_dec) * np.cos(dec) * np.sin(ra - mu_ra)
    x = np.cos(dec) * np.sin(ra - mu_ra) / cos_c
    y = (np.cos(mu_dec) * np.sin(dec) - np.sin(mu_dec) *np.cos(dec) * np.cos(ra - mu_ra)) / cos_c

    return x, y


def plot_cmd(data, axs):
    """ Plots a color magnitude diagram.

    Inputs:
        data (pd.DataFrame): DataFrame with photometry data
        axs: pyplot axes object
    """
    y = data['mag_g']
    x = data['mag_g'] - data['mag_r']

    xlims = [-1, 1.5]  #need to find better way to restrict axes
    ylims = [16,28]
    axs.set_xlim(xlims);
    axs.set_ylim(ylims);

    axs.set_ylabel('$Magnitude (g)$')
    axs.set_xlabel('$Color (g-r)$')

    axs.plot(x, y, 'ko', markersize=0.3, alpha=0.3)
    axs.invert_yaxis()


    plt.show()

    return


def plot_cmd_sep(rdata, sdata, axs, cbar=True, show_iso=False, iso_selection=None, fname=None, save=False):
    """ Plots a color magnitude diagram with simulated satellite objects highlighted.

    Inputs:
        rdata: DataFrame
            Background field photometry data
        sdata: DataFrame
            Simulated satellite photometry data
        axs: pyplot axes object
        show_iso: bool
            Whether to show the isochrone path
        iso_selection: Isochrone object
            The isochrone data to plot
        fname: str
            Filename to save the plot
        save: bool
            Whether to save the plot
    """
    y = rdata['mag_g']  
    x = rdata['mag_g'] - rdata['mag_r']
    
    xlims = [-0.5, 1]
    ylims = [17, 27]
    axs.set_xlim(xlims)
    axs.set_ylim(ylims)
    axs.locator_params(axis='x', nbins=4)
    axs.set_ylabel('$g$', fontsize=22)
    axs.set_xlabel('$g-r$', fontsize=22)
    
    n, x, y, p = axs.hist2d(x, y, cmap='Greys', bins=[np.linspace(-0.5, 1, 50), np.linspace(17, 27, 50)], label='DC2 object', norm=LogNorm(vmax=1000), rasterized=True)
    if cbar: 
        cbar = plt.colorbar(p, label='Number of objects')
        cbar.ax.tick_params(labelsize=15)
    
    y = sdata['mag_g']  
    x = sdata['mag_g'] - sdata['mag_r']
    axs.plot(x, y, 'o', color='red', markersize=8, alpha=1, label='satellite star', markeredgecolor='white', mew=0.5)

    if show_iso and iso_selection is not None:
        _, mag_centers, iso_upper, iso_lower = iso_selection  
    
        # Create a closed path: move along upper boundary, then lower boundary in reverse
        vertices = np.vstack([
            np.column_stack([iso_upper, mag_centers]),  # Upper boundary (left to right)
            np.column_stack([iso_lower[::-1], mag_centers[::-1]])  # Lower boundary (right to left)
        ])
    
        codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
        path = Path(vertices, codes)
    
        # Create and add a PathPatch to outline the region
        patch = PathPatch(path, facecolor='none', edgecolor='red', linewidth=2)
        axs.add_patch(patch)
        
    axs.invert_yaxis()
    
    if save and fname is not None:
        plt.savefig(fname)
    
    # plt.show()
    
    return p


def draw_mu_vs_ng24(mu,ng24,color,data=None,**kwargs):
    """ Draw one surface brightness (mu) vs the number of stars with g < 24 (N_g24)

    Parameters
    ----------
    mu    : recarray of surface brightness
    ng22  : recarray of N(g < 24)
    color : recarray of color values
    kwargs: passed to matplotlib.scatter

    Returns
    -------
    ax0, ax1 : the axes of the figure produced
    """

    defaults = dict(vmin=0., vmax=20, s=5, alpha=1.0)

    ax0 = plt.gca()

    ax0.set_xscale('log')
    c1 = ax0.scatter(ng24,mu,c=color,
                     rasterized=True,edgecolor='none')

    kwargs = dict(facecolor='none',alpha=1.0,hatch='x',edgecolor='0.5',lw=1)

    # DIFFICULTY = 1
    ax0.fill_between([5,8e5],[39,26],[48,48],**kwargs)
    ax0.text(1e3,40,'DIFFICULTY=1',fontsize=18)

    # DIFFICULTY = 2
    ax0.fill_between([6e1,8e5],[10,10],[23.5,23.5],**kwargs)
    ax0.text(2e3,20,'DIFFICULTY=2',fontsize=18)

    ax0.set_xticks([10,100,1000,1e4,1e5])
    ax0.set_xticklabels(['10^1','10^2','10^3','10^4','10^5'],
                        fontsize=16)
    ax0.set_yticks([15,20,25,30,35,40])
    ax0.set_yticklabels([15,20,25,30,35,40],fontsize=16)

    ax0.set_xlim(5.5,5e5)
    ax0.set_ylim(14,42)

    return ax0, c1

def plot_mu_vs_ng24(sims,data=None):
    """ Plot surface brightness (mu) vs the number of stars with g < 24 (N_g24)

    Parameters
    ----------
    sims : recarray of simulated satellite properties

    Returns
    -------
    axs : the axs of the figure produced
    """
    fig,axs = plt.subplots(figsize=(6,6))

    sel = (sims['N_G24'] > 5)

    mu = sims['SURFACE_BRIGHTNESS'][sel]
    ng24 = sims['N_G24'][sel]
    sig = sims['SIG'][sel]

    # simple
    plt.sca(axs)
    _,c1 = draw_mu_vs_ng24(mu,ng24,sig)
    # y-label
    axs.set_ylabel('$\mu [mag arcsec^{-2}]$',fontsize=18,labelpad=8)
    axs.set_xlabel('N(g < 24)',fontsize=18,labelpad=8)
    axs.set_xlim(5*1e0, 1e6)
    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.1075, 0.0175, 0.78])
    cbar2 = plt.colorbar(c1, cax=cbar_ax, ticks=[0,5,10,15,20,25,30,35])
    cbar2.ax.set_yticklabels([0,5,10,15,20,25,30,35], fontsize=12)
    cbar2.set_label('SIG',size=18,labelpad=6)
    cbar2.solids.set_rasterized(True)
    cbar2.solids.set_edgecolor("face")

    return axs


def plots(position, real_data, sim_data, merged_data, mcid, iso_selection, cmap):
    """ Plots the objects from DC2 data, the sim satellite data, and DC2+sim data
    on separate plots. Also plots a color magnitude diagram.

    Parameters
    ----------
        position: tuple(ra, dec)
            Position of the object
        real_data: DataFrame
            DC2 data at position
        sim_data: DataFrame
            Simulated satellite data
        merged_data: DataFrame
            DC2 data + sim data
        mcid: int
            Simulated satellite id 
        cmap: ListedColormap
            Color map to use for plots
    """

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

    hb = axs[0].hexbin(*gnomic(*position, real_data['ra'], real_data['dec']), mincnt=1, cmap=cmap, gridsize=50)
    cb = fig.colorbar(hb, ax=axs[0])
    axs[0].set_title('DC2 Data')
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$y$')

    hb = axs[1].hexbin(*gnomic(*position, sim_data['ra'], sim_data['dec']), mincnt=1, cmap=cmap, gridsize=50)
    cb = fig.colorbar(hb, ax=axs[1])
    axs[1].set_title(f'Simulated Satellite {mcid}')
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$y$')

    hb = axs[2].hexbin(*gnomic(*position, merged_data['ra'], merged_data['dec']), mincnt=1, cmap=cmap, gridsize=50)
    cb = fig.colorbar(hb, ax=axs[2])
    axs[2].set_title('Merged')
    axs[2].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$y$')

    # plot_cmd(merged_data, axs[3])
    p = plot_cmd_sep(real_data, sim_data, axs[3], cbar=False, show_iso=True, iso_selection=iso_selection, fname=None, save=False)
    cbar = plt.colorbar(p, ax=axs[3])
    cbar.set_label('Number of objects')
    plt.show()
