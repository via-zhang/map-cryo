from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import load
from pathlib import Path
import harmonica as hm
from scipy import interpolate
from sklearn.preprocessing import QuantileTransformer
import gstatsim
import skgstat as skg
from skgstat import models
import gstools as gs
import xarray as xr
import xrft
import verde as vd

import warnings
warnings.filterwarnings("ignore")

from prisms import PrismGen
from diagnostics import acceptance_rate
from utilities import xy_into_grid

def loss_fun(data, pred):
    res = data-pred
    return np.mean(res**2)+np.mean(res)**2

def sum_sq_err(data, pred):
    return np.sum(np.square(data-pred))
    
def mcmc_rbf(ds, x0, grav, grid, sigma, rfgen, pgen, save_path=None, adapt=False, density=False, iter_num=500, quiet=False, parallel=True, num_mp=1):
    """
    MCMC for bathymetry inference.
    """
    rng = np.random.default_rng(seed=num_mp)
    
    bed = x0
    y = np.unique(ds.y.data)
    x = np.unique(ds.x.data)
    te_dist = grav[grid].values

    shelf_msk = np.where((ds.mask==3) & (ds.dist_msk==True), True, False)
    
    # initialize caches
    bed_cache = np.zeros((iter_num, bed.shape[0], bed.shape[1]))
    loss_cache = np.zeros(iter_num)
    step_cache = np.zeros(iter_num)
    grav_cache = np.zeros((iter_num, grav.shape[0]))
    
    if density==True:
        # make cache of terrain effects
        est = load(Path('processed_data/gravity_te_density.joblib'))
        rng = np.random.default_rng()
        dens_cache = rng.normal(loc=2700, scale=80, size=iter_num)
        te_dist_cache = est.predict(dens_cache.reshape(-1,1))
    
    # initialize loss
    pred_coords = (grav.x, grav.y, grav.height)
    prisms_inv, densities_inv = pgen.make_prisms(bed, 'inv')
    prisms_dist, densities_dist = pgen.make_prisms(bed, 'dist_not_inv')
    prisms_no_ice, densities_no_ice = pgen.make_prisms(bed, 'inv', ice=False)
    g_z_inv = hm.prism_gravity(pred_coords, prisms_inv, densities_inv, 
                               field='g_z', parallel=parallel)
    g_z_dist = hm.prism_gravity(pred_coords, prisms_dist, densities_dist, 
                               field='g_z', parallel=parallel)
    g_z_no_ice = hm.prism_gravity(pred_coords, prisms_no_ice, densities_no_ice, 
                               field='g_z', parallel=parallel)
    g_z_ice = g_z_inv-g_z_no_ice
    loss_prev = sum_sq_err(te_dist, g_z_inv+g_z_dist)
    
    pbar = tqdm(range(iter_num), position=0, leave=True, disable=quiet)
    for i in pbar:
        # random gaussian field perturbation
        field_cond = rfgen.generate_field(condition=True)
        bad_i = 0
        while np.any(np.isnan(field_cond))==True:
            if bad_i > 100:
                print('cant generate good field')
                return
            else:
                field_cond = rfgen.generate_field(condition=True)
                bad_i += 1

        # add to previous bed
        bed_next = bed+field_cond

        # make sure bed below shelf bottom
        bed_next = np.where((shelf_msk==True) & (bed_next > (ds.surface-ds.thickness)), 
                            ds.surface-ds.thickness, bed_next)
        
        # get random terrain effect
        if density==True and i%1==0:
            pgen.rock_dens = dens_cache[i]
            te_dist = te_dist_cache[i]

        prisms, densities = pgen.make_prisms(bed_next, 'inv', ice=False)
        g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z', parallel=parallel)
        
        # compute loss
        loss_next = sum_sq_err(te_dist, g_z+g_z_dist+g_z_ice)
        
        #acceptance
        # alpha = min(1,np.exp((loss_prev**2-loss_next**2)/(2*sigma**2)))
        alpha = min(1,np.exp((loss_prev-loss_next)/(2*sigma**2)))
        
        # accept or not, save cachestep
        u = rng.uniform(size = 1)
        if (u <= alpha):
            bed = bed_next
            loss_cache[i] = loss_next
            step_cache[i] = True
            loss_prev = loss_next
        else:
            loss_cache[i] = loss_prev
            step_cache[i] = False
        
        bed_cache[i,:,:] = bed
        grav_cache[i,:] = g_z
        if density==True:
            dens_cache[i] = pgen.rock_dens
        
        if adapt==True:
            if (i%500==0) & (i < 10e3) & (i > 1):
                acc_rate = acceptance_rate(step_cache[i-500:i], 0)
                if acc_rate > 0.234:
                    rfgen.high_step += 5
                    print('variance raised')
                else:
                    rfgen.high_step -= 5
                    print('variance lowered')

        if (i>0) & (i%10_000==0) & (save_path is not None):
            np.savez(
                save_path, 
                bed_cache=bed_cache[:i,...], 
                loss_cache=loss_cache[:i], 
                step_cache=step_cache[:i], 
                grav_cache=grav_cache[:i,:]
            )
                
        pbar.set_description(f'#{num_mp} loss: {loss_cache[i]:.3f}')
        
    result = {
        'bed_cache' : bed_cache,
        'loss_cache' : loss_cache,
        'step_cache' : step_cache,
        'grav_cache' : grav_cache
    }
    if density==True:
        result['density_cache'] = dens_cache
    
    return result

#@threadpool_limits.wrap(limits=1)
def mp_mcmc_rbf(args):
    """
    Multiprocessing wrapper to unpack parameters
    """
    
    [ds, x0, grav, sigma, rfgen, pgen, adapt, density, iter_num, quiet, parallel, num_mp] = args
    
    return mcmc_rbf(ds, x0, grav, sigma, rfgen, pgen, adapt, density, iter_num, quiet, parallel, num_mp)

def nte_correction(ds, grav, density):
    density_dict = {
        'ice' : 917,
        'water' : 1027,
        'rock' : density
    }
    pgen = PrismGen(ds, density_dict)

    prisms, densities = pgen.make_prisms(ds.bed.values, msk='all')

    pred_coords = (grav.x, grav.y, grav.height)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')

    residual = grav.faa-g_z

    coords = (grav.x[grav.inv_msk==False], grav.y[grav.inv_msk==False], grav.height[grav.inv_msk==False])
    values = residual[grav.inv_msk==False]
    interp_lin = interpolate.LinearNDInterpolator(coords, values, rescale=True)
    interp_nn = interpolate.NearestNDInterpolator(coords, values, rescale=True)
    trend1 = interp_lin(grav.x, grav.y, grav.height)
    trend2 = interp_nn(grav.x, grav.y, grav.height)
    trend = np.where(np.isnan(trend1), trend2, trend1)

    return grav.faa - trend

def nte_correction_eq(ds, grav, density):
    density_dict = {
        'ice' : 917,
        'water' : 1027,
        'rock' : density
    }
    pgen = PrismGen(density_dict)

    prisms, densities = pgen.make_prisms(ds, ds.bed.values, msk='all')

    pred_coords = (grav.x, grav.y, grav.height)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')

    residual = grav.faa-g_z

    grav_int = grav[grav['inv_msk']==False][::100]
    res_int = residual[grav['inv_msk']==False][::100]
    coords_int = (grav_int.x, grav_int.y, grav_int.height)

    equivalent_sources = hm.EquivalentSources(depth=5e3, damping=1)
    equivalent_sources.fit(coords_int, res_int)
    nte = equivalent_sources.predict((grav.x, grav.y, grav.height))
    nte = np.where(grav['inv_msk']==True, nte, residual)

    return grav.faa - nte

