import os,sys,pickle,glob
import scipy.stats as stats
import mne
import numpy as np

from T_to_Z import T_to_Z

import do_stats
import pylab as plt

import do_stats
#import write_nifti
import qvalue

import nibabel
import copy
from nilearn import image, plotting





alph = 0.05
RESULTS_DIR = 'bf_results_TEST2_mxf'
# ~ RESULTS_DIR = 'bf_results_pchip'
# ~ RESULTS_DIR = 'bf_results_pchip_WNnone'



def get_individual_nai_stats(pid,protocol,fband_str):
    """Return the mean and variance of the Neural Activity Index at every beamformer grid point, and the number of epochs used."""
    
    fn = '/fred/oz266/%s/2003xMSNA_%s_PL_%s_5_%s.pkl'%(RESULTS_DIR,pid,protocol,fband_str)

    with open(fn,'rb') as f:
        nai,nai_mean,nai_sem,num_epochs,filters = pickle.load(f)
    
    nai_var = (nai_sem * nai_sem)* num_epochs       # Need variance, not standard error
    
    return nai_mean,nai_var,num_epochs

def write_interpolated_nifti(bf_vals, grid_spacing, src_fn, cached_fn, template_brain, out_filename):

    import nibabel
    from scipy import ndimage
    from scipy.interpolate import RegularGridInterpolator

    nr = nibabel.load(template_brain)
    nifti_data = nr.get_fdata()
    X,Y,Z = np.mgrid[0:nifti_data.shape[0]:grid_spacing,0:nifti_data.shape[1]:grid_spacing,0:nifti_data.shape[2]:grid_spacing]
    X = X.astype(int)
    Y = Y.astype(int)
    Z = Z.astype(int)
    gd = np.zeros(nifti_data.shape,nifti_data.dtype)
    gd[X,Y,Z] = 1


    grid_mri0 = np.transpose(np.nonzero(gd))
    #convert mne-python grid coordinates to mm
    s1 = mne.read_source_spaces(src_fn, verbose=False)
    common_vertno = s1[0]['vertno']
    s0 = mne.read_source_spaces(cached_fn, verbose=False)
    s2 = np.round(s0[0]['rr'][common_vertno,:]*1000).astype(np.int64)

    #generate a map from s2 to gd
    pd1 = {grid_mri0[i].tobytes():i for i in range(len(grid_mri0))}
    pd2 = {s2[i].tobytes():i for i in range(len(s2))}
    pd = {v:pd1[k] for k,v in pd2.items()}
    m = [pd[i] for i in range(len(pd))]

    grid_mri = grid_mri0[m]
    grid_vals = bf_vals

    mp = np.transpose(np.nonzero(nifti_data))
    Xc = np.zeros(X.shape)         # This is an arbitrary choice, X,Y and Z all have the same shape
    Xc[(grid_mri[:,0]/grid_spacing).astype(int),(grid_mri[:,1]/grid_spacing).astype(int),(grid_mri[:,2]/grid_spacing).astype(int)] = grid_vals

    dz = np.zeros(nifti_data.shape)

    #~ zi = ndimage.map_coordinates(Xc, mp.T*1.0/grid_spacing, order=2)
    #~ dz[mp[:,0],mp[:,1],mp[:,2]] = zi

    my_interpolating_function = RegularGridInterpolator((X[:,0,0],Y[0,:,0],Z[0,0,:]), Xc, method="nearest")
    dz[mp[:,0],mp[:,1],mp[:,2]] = my_interpolating_function(mp)

    #img3 = nibabel.Nifti1Image(dz, nr.get_affine(), header=nr.get_header())
    img3 = nibabel.Nifti1Image(dz, nr.affine, header=nr.header)
    img3.to_filename(out_filename)
    print ('Output volume written to:',out_filename)

  
if __name__ == "__main__":
        
    ############################################################
    ### For generating the brain atlas
    sys.path.append('/dagg/public/neuro/MEG_scripts/')
    from atlas_timeseries import getAALAtlasDct, getHOCombinedAtlasDct

    src_fn = '/fred/oz266/forward/2003xMSNA_002_5-src.fif'
    cached_fn = '/dagg/public/neuro/MEG_data/MNI152-src-cached/MNI152_5-src.fif'

   
    label_dct = getHOCombinedAtlasDct(src_fn, grid_spacing=5)
    label_names = list(label_dct.keys())
    for label in label_names:
        print(label)
    print(len(label_names))
    ###################################################################

    #pid_lst=["002", "004", "005", "007" "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "032", "033"]
    pid_lst=["001", "003", "002", "004", "005", "007" "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030","031" "032", "033", "034"]
    #pid_lst=["001", "002", "006", "008", "011", "014", "015", "018", "022", "023", "024", "027", "028", "029","031", "033", "034"]
    #pid_lst=["004","005","013","016","019", "020","021","025","026","030","032"]
                
    protocol1 = 'STRESS'
    protocol2 = 'REST'
    fband_str=str(sys.argv[1])
    print ('\n%s Hz\n'%fband_str,flush=True)
    
    fl = glob.glob('/fred/oz266/bf_results_TEST2_mxf/*_STRESS,REST_5_%s.pkl'%(fband_str))
    Z_lst = []
    for fn in fl:
        print (fn,flush=True)
        with open(fn,'rb') as f:
            nai, num_epochs, filters, filters_SAM = pickle.load(f)
        T = stats.ttest_ind(nai[:,:num_epochs[0]], nai[:,num_epochs[0]:],axis=1)[0]
        dof = sum(num_epochs) - 2
        Z = T_to_Z(T,dof)
        Z_lst.append(Z)
    Z_data = np.array(Z_lst)
    N = None
    ### Do permutation test on your roi data - change num_perms to 10,000 for reproducibility when you are happy with your data     
    mnth, mxth, null_mn, null_mx, ts0, pvals = do_stats.do_perms(Z_data, alph, num_perms=10000, return_null_dist=True)
    print ('\nalph=%0.2f\nMin thresh: %0.2f Max thresh: %0.2f'%(alph,mnth,mxth))
    print ('Min value: %0.2f Max value: %0.2f\n'%(ts0[0].min(),ts0[0].max()))
    
    ### Get the labels of regions which contain signficant grid points
    si = np.where(ts0[0]>mxth)[0]
    if len(si) > 0:
        print ('Max regions:',set([l for i in si for l in label_names if i in label_dct[l]]))
    si = np.where(ts0[0]<mnth)[0]
    if len(si) > 0:
        print ('Min regions:',set([l for i in si for l in label_names if i in label_dct[l]]))

    ### Use this sys exit the first time through to see if there is anything sig, then uncomment to plot
    #sys.exit(0)

    roi_name_str='ALL'
    grid_spacing = 5
    src_fn = '/fred/oz266/forward/2003xMSNA_002_5-src.fif'
    cached_fn = '/fred/oz266/NIFTI/MNI152_%d-src.fif'%grid_spacing
    template_brain = '/fred/oz266/NIFTI/MNI152_T1_1mm_brain_mask.nii.gz'

    out_filename = '/fred/oz266/NIFTI/%s_nifti/contrast/%s-%s_%s_%s.nii.gz' %(RESULTS_DIR,protocol1,protocol2,roi_name_str,fband_str)
    
    if not os.path.isdir(os.path.dirname(out_filename)):
        os.makedirs(os.path.dirname(out_filename))
    write_interpolated_nifti(ts0[0], grid_spacing, src_fn, cached_fn, template_brain, out_filename)


    ###Plot glassbrain
    ### If you have min regions, then the threshold will be mnth
    ### If you have max regions, then the threshold will be mxth
    
    threshold = -mnth
    threshold = mxth
    
    ### For your colour bar - you can change this, but this plots to the max value of the t-scores, and makes symmetrical (eg plots from -vmax to vmax)
    abs_T=np.abs(ts0[0]) ### Finds the max absolute value in t-scores (positive or negative)
    vmax = np.nanmax(abs_T)
    vmin=-vmax


    stat_map_img = out_filename
    display = plotting.plot_glass_brain(stat_map_img, display_mode='lyrz', colorbar=True, cmap="RdYlBu_r", threshold=threshold,black_bg=False, symmetric_cbar=True, plot_abs=False, vmax=vmax)
    
    plotting.show()
        
    
    
