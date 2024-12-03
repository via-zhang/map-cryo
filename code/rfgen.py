import gstools as gs
import numpy as np
from scipy import interpolate

class RFGen:
    def __init__(self, ds, range_max, range_min, high_step, nug_max, eps, model='Gaussian', const_var=False, neighbors=None, seed=None, rbf=True):
        
        y = ds.y.data
        x = ds.x.data
        xx, yy = np.meshgrid(x, y)
        XX = np.stack([xx.flatten(), yy.flatten()]).T
        X = XX[ds.inv_msk.values.flatten()==False, :]
        
        self.x = x
        self.y = y
        self.xx = xx
        self.yy = yy
        self.X = X
        self.XX = XX
        self.cond_msk = ~ds.inv_msk.data
        
        self.mean = 0
        self.var = 1
        self.range_max = range_max
        self.range_min = range_min
        self.high_step = high_step
        self.nug_max = nug_max
        self.eps = eps
        self.neighbors = neighbors
        self.seed = seed
        self.cond_edges = None
        self.dist_weights = None
        self.rbf = rbf
        self.const_var = const_var
        
        if model=='Gaussian':
            self.model = gs.Gaussian
        elif model=='Exponential':
            self.model = gs.Exponential

        self.get_cond_edges()
        
    def get_cond_edges(self):
        a = self.cond_msk.copy()
        for i in range(1, self.cond_msk.shape[0]-1):
            for j in range(1, self.cond_msk.shape[1]-1):
                if self.cond_msk[i,j] == True:
                    if np.all([self.cond_msk[i+1,j],
                               self.cond_msk[i-1,j],
                               self.cond_msk[i,j+1],
                               self.cond_msk[i,j-1]]) == True:
                        a[i,j] = False
                        
        self.cond_edges = a
        self.X = self.XX[a.flatten()==True,:]
        
    def get_dist_weights(self, ds):
        dists = np.zeros(ds.x.shape)
        for i in range(ds.x.shape[0]):
            for j in range(ds.x.shape[1]):
                east_cond = np.where(ds.cond_msk, ds.x, np.nan)
                north_cond = np.where(ds.cond_msk, ds.y, np.nan)
                dist_mat = np.sqrt((ds.x[i,j].values-east_cond)**2+(ds.y[i,j].values-north_cond)**2)
                dists[i,j] = np.nanmin(dist_mat)
                
        x = np.log(dists)
        x = np.where(np.isinf(x), np.nanmin(x[~np.isinf(x)]), x)
        # rescale
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        rescale = (x-xmin)/(xmax-xmin)
                
        self.dist_weights = rescale
            
        
    def generate_field(self, condition=False):
        # random parameters
        rng = np.random.default_rng()
        if self.const_var==False:
            scale  = rng.uniform(low=1, high=self.high_step, size=1)[0]/3
        else:
            scale = self.high_step/3
        nug = rng.uniform(low=0.0, high=self.nug_max, size=1)[0]
        range1 = rng.uniform(low=self.range_min[0], high=self.range_max[0], size=1)[0]
        range2 = rng.uniform(low=self.range_min[1], high=self.range_max[1], size=1)[0]
        angle = rng.uniform(low=0, high=180, size=1)[0]
        model = self.model(dim=2,
                            var = self.var,
                            len_scale = [range1/np.sqrt(3),range2/np.sqrt(3)],
                            angles = angle*np.pi/180,
                            nugget = nug)
        if self.seed is not None:
            srf = gs.SRF(model,seed=self.seed)
        else: 
            srf = gs.SRF(model)

        field = srf.structured([self.x, self.y]).T*scale + self.mean
        
        if condition==True:
            return self.rbf_cond_field(field)
        else:
            return field
    
    def rbf_cond_field(self, field):
        if self.rbf==True:
            z_field = field.flatten()[self.cond_edges.flatten()==True]
            z_interp = interpolate.RBFInterpolator(self.X, z_field, kernel='inverse_multiquadric',
                                       epsilon=self.eps, neighbors=self.neighbors)(self.XX)
        
            field_cond = field - np.where(self.cond_msk==True, field, z_interp.reshape(field.shape))
        else:
            field_cond = field*self.dist_weights
        return field_cond
    
def generate_m_2D(theta, x, y, seed=None):
    model = gs.Gaussian(dim=2,
                var= theta[1],
                len_scale=[theta[2]/np.sqrt(3),theta[3]/np.sqrt(3)],
                angles=theta[4]*np.pi/180,
                nugget=theta[5])
    if seed is not None:
        srf = gs.SRF(model,seed=seed)
    else: 
        srf = gs.SRF(model)
        
    field = srf.structured([x, y], store=False, post_process=False).T*theta[6] + theta[0]
    return field