import numpy as np
import xarray as xr
import boule as bl
    
class PrismGen:
    def __init__(self, density_dict):
        self.ice_dens = density_dict['ice']
        self.water_dens = density_dict['water']
        self.rock_dens = density_dict['rock']

    def make_prisms(self, ds, bed, msk='all', ice=True):

        res = ds.res
        half_res = res/2
        
        xx, yy = np.meshgrid(ds.x.values, ds.y.values)
        
        surf = ds.surface.values
        thick = ds.thickness.values

        # make prisms for entire x/y domain
        if msk=='all':
            dist_msk = np.ones(ds.bed.values.shape)
        # make prisms within distance mask
        elif msk=='dist':
            dist_msk = ds.dist_msk.values
        # make prisms within inversion domain
        elif msk=='inv':
            dist_msk = ds.inv_msk.values
        # make prisms within distance mask outside of inversion mask
        elif msk=='dist_not_inv':
            dist_msk = np.where((ds.dist_msk==True) & (ds.inv_msk==False), True, False)

        # water prisms mask
        water_msk = np.where(((ds.mask==0) ^ (ds.mask==3)) & (dist_msk==True), True, False)
        
        # ice prisms mask
        ice_msk = np.where(((ds.mask==3) ^ (ds.mask==2)) & (dist_msk==True), True, False)
        
        # positive rock mask
        rock_msk = np.where((bed > 0) & (dist_msk==True), True, False)
        
        # negative rock mask
        negrock_msk = np.where((surf < 0) & (dist_msk==True), True, False)
        
        # calculate water column thickness
        water_thickness = surf - thick - bed
        
        # water prisms
        prisms_water = np.array([
            xx[water_msk]-half_res,
            xx[water_msk]+half_res,
            yy[water_msk]-half_res,
            yy[water_msk]+half_res,
            bed[water_msk],
            (bed+water_thickness)[water_msk]
        ]).T
        
        prisms_water, idx_water_pos = self.split_prisms(prisms_water)
        
        # rock prisms
        prisms_rock = np.array([
            xx[rock_msk]-half_res,
            xx[rock_msk]+half_res,
            yy[rock_msk]-half_res,
            yy[rock_msk]+half_res,
            np.zeros(xx.shape)[rock_msk],
            bed[rock_msk]
        ]).T
        
        # negative rock prisms
        prisms_negrock = np.array([
            xx[negrock_msk]-half_res,
            xx[negrock_msk]+half_res,
            yy[negrock_msk]-half_res,
            yy[negrock_msk]+half_res,
            surf[negrock_msk],
            np.zeros(xx.shape)[negrock_msk]
        ]).T

        if ice==True:
            # ice prisms
            prisms_ice = np.array([
                xx[ice_msk]-half_res,
                xx[ice_msk]+half_res,
                yy[ice_msk]-half_res,
                yy[ice_msk]+half_res,
                (surf-thick)[ice_msk],
                surf[ice_msk]
            ]).T
        
            prisms_ice, idx_ice_pos = self.split_prisms(prisms_ice)
            densities_ice = np.where(idx_ice_pos, self.ice_dens, self.ice_dens-self.rock_dens)
        

        densities_water = np.where(idx_water_pos, self.water_dens, self.water_dens-self.rock_dens)
        densities_rock = np.full(prisms_rock.shape[0], self.rock_dens)
        densities_negrock = np.full(prisms_negrock.shape[0], -self.rock_dens)

        if ice==True:
            prisms = np.vstack([
                prisms_ice,
                prisms_water,
                prisms_rock,
                prisms_negrock
            ])
            densities = np.concatenate([
                densities_ice,
                densities_water,
                densities_rock,
                densities_negrock
            ])
        else:
            prisms = np.vstack([
                prisms_water,
                prisms_rock,
                prisms_negrock
            ])
            densities = np.concatenate([
                densities_water,
                densities_rock,
                densities_negrock
            ])

        # remove bad prisms where bottom is above top
        bad_idx = np.nonzero(prisms[:,4] > prisms[:,5])[0]
        prisms = np.delete(prisms, bad_idx, axis=0)
        densities = np.delete(densities, bad_idx, axis=0)

        return prisms, densities

    def split_prisms(self, prisms):
        '''
        Function to split prisms above and below the ellipsoid.
        Rerturns combined prisms and an index of which ones are above the ellipsoid.
        '''
        prisms_pos = prisms[prisms[:,5] >= 0, :]
        prisms_neg = prisms[prisms[:,4] < 0, :]
        prisms_pos[prisms_pos[:,4] < 0, 4] = 0.0
        prisms_neg[prisms_neg[:,5] > 0, 5] = 0.0
        prisms = np.vstack([prisms_pos, prisms_neg])
        idx_pos = np.full(prisms.shape[0], False)
        idx_pos[:prisms_pos.shape[0]] = True
        return prisms, idx_pos