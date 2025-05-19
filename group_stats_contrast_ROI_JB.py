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
        
    #pid_lst=["008","009", "011", "013", "014", "015", "016"]
    #pid_lst=["008", "011", "013", "014", "015", "016", "019", "020", "021", "022", "023"]
    #pid_lst=["005", "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023"]
    #pid_lst=["002", "004", "005", "007" "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "032", "033"]
    pid_lst=["001","002", "004", "005", "007" "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "032", "033", "034"]
    #pid_lst=["001", "002", "006", "008", "011", "014", "015", "018", "022", "023", "024", "027", "028", "029","032" "033", "034"]
    #pid_lst=["004","005","013","016","019", "020","021","025","026","030","032"]
    #pid_lst=["004"]
    #pid_lst=["003","004","005","009","012", "014","015","016","020","021","025","027"]


    protocol1 = 'STRESS'
    protocol2 = 'REST'

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
    #sys.exit(0)
    
    ### ROI - defining areas of interest - make lists of whichever areas you like, and give a name so you know which are which
    #pre_roi_list=('MAJOR',['Left Hippocampus','Right Hippocampus','Right Inferior Frontal Gyrus, pars opercularis','Left Inferior Frontal Gyrus, pars opercularis','Right Frontal Medial Cortex','Left Frontal Medial Cortex','Right Frontal Orbital Cortex','Left Frontal Orbital Cortex','Right Frontal Medial Cortex','Left Frontal Medial Cortex','Right Amygdala','Left Amygdala','Left Cingulate Gyrus, anterior division','Right Cingulate Gyrus, anterior division', 'Left Cingulate Gyrus, posterior division', 'Right Cingulate Gyrus, posterior division','Right Insular Cortex','Left Insular Cortex'])
    #pre_roi_list=('FRONTAL',['Left Frontal Medial Cortex','Right Superior Frontal Gyrus','Right Frontal Pole','Left Frontal Operculum Cortex','Left Frontal Orbital Cortex','Left Inferior Frontal Gyrus, pars opercularis','Right Frontal Operculum Cortex','Right Inferior Frontal Gyrus, pars opercularis','Right Frontal Medial Cortex','Left Inferior Frontal Gyrus, pars triangularis','Left Middle Frontal Gyrus','Right Middle Frontal Gyrus','Right Inferior Frontal Gyrus, pars triangularis','Left Superior Frontal Gyrus','Left Frontal Pole','Right Frontal Orbital Cortex'])         
    #pre_roi_list=('TEMPORAL',['Right Inferior Temporal Gyrus, posterior division','Right Inferior Temporal Gyrus, anterior division','Right Temporal Fusiform Cortex, posterior division','Left Inferior Temporal Gyrus, posterior division','Right Middle Temporal Gyrus, posterior division','Left Superior Temporal Gyrus, posterior division','Right Middle Temporal Gyrus, anterior division','Right Planum Temporale','Right Temporal Fusiform Cortex, anterior division','Left Middle Temporal Gyrus, posterior division','Left Middle Temporal Gyrus, temporooccipital part','Left Inferior Temporal Gyrus, temporooccipital part','Left Temporal Fusiform Cortex, posterior division','Right Superior Temporal Gyrus, anterior division','Left Inferior Temporal Gyrus, anterior division','Right Superior Temporal Gyrus, posterior division','Left Temporal Pole','Left Planum Temporale','Left Temporal Occipital Fusiform Cortex','Right Inferior Temporal Gyrus, temporooccipital part','Left Middle Temporal Gyrus, anterior division','Right Middle Temporal Gyrus, temporooccipital part','Left Temporal Fusiform Cortex, anterior division','Right Temporal Pole','Left Superior Temporal Gyrus, anterior division'])
    #pre_roi_list=('ORB',['Right Frontal Orbital Cortex','Left Frontal Orbital Cortex'])
    #pre_roi_list=('ACC',['Left Cingulate Gyrus, anterior division','Right Cingulate Gyrus, anterior division', 'Left Cingulate Gyrus, posterior division', 'Right Cingulate Gyrus, posterior division'])
    #pre_roi_list=('AMY',['Right Amygdala','Left Amygdala'])
    #pre_roi_list=('INS',['Right Insular Cortex','Left Insular Cortex'])
    #pre_roi_list=('THAL',['Left Thalamus', 'Rigth Thalamus'])
    #pre_roi_list=('BrainS',['Brain-Stem'])
    #pre_roi_list=('INS',['Left Precuneous Cortex', 'Right Precuneous Cortex'])
    #pre_roi_list=('WHOLEA',['Right Amygdala','Left Amygdala','Right Cingulate Gyrus, anterior division', 'Left Cingulate Gyrus, anterior division', 'Right Caudate', 'Right Cingulate Gyrus, posterior division', 'Right Central Opercular Cortex', 'Right Thalamus', 'Left Hippocampus', 'Right Hippocampus', 'Left Cingulate Gyrus, posterior division', 'Left Insular Cortex', 'Left Central Opercular Cortex', 'Left Caudate', 'Left Thalamus', 'Right Insular Cortex', 'Brain-Stem'])
    pre_roi_list=('WHOLEA2',['Left Precuneous Cortex', 'Right Precuneous Cortex', 'Right Frontal Medial Cortex', 'Left Frontal Medial Cortex', 'Right Frontal Orbital Cortex', 'Left Frontal Orbital Cortex', 'Left Thalamus', 'Right Thalamus', 'Right Amygdala', 'Left Amygdala', 'Right Cingulate Gyrus, anterior division', 'Left Cingulate Gyrus, anterior division', 'Right Caudate', 'Right Cingulate Gyrus, posterior division', 'Left Hippocampus', 'Right Hippocampus', 'Left Precuneous Cortex', 'Right Precuneous Cortex', 'Left Cingulate Gyrus, posterior division', 'Left Insular Cortex', 'Left Caudate', 'Right Insular Cortex', 'Brain-Stem'])
    #pre_roi_list=('INTEREST',['Left Precuneous Cortex', 'Right Precuneous Cortex', 'Left Parahippocampal Gyrus, anterior division','Right Parahippocampal Gyrus, anterior division','Right Parahippocampal Gyrus, posterior division','Left Parahippocampal Gyrus, posterior division','Right Amygdala','Left Amygdala','Right Cingulate Gyrus, anterior division', 'Left Cingulate Gyrus, anterior division', 'Right Caudate', 'Right Cingulate Gyrus, posterior division', 'Right Central Opercular Cortex', 'Right Thalamus', 'Left Hippocampus', 'Right Hippocampus', 'Left Cingulate Gyrus, posterior division', 'Left Insular Cortex', 'Left Central Opercular Cortex', 'Left Caudate', 'Left Thalamus', 'Right Insular Cortex', 'Brain-Stem'])
    #pre_roi_list=('TEMP',['Right Middle Temporal Gyrus, posterior division','Left Superior Temporal Gyrus, posterior division', 'Right Middle Temporal Gyrus, anterior division','Left Middle Temporal Gyrus, posterior division', 'Right Pallidum', 'Left Pallidum', 'Right Accumbens', 'Right Putamen', 'Right Superior Temporal Gyrus, anterior division', 'Left Accumbens', 'Right Superior Temporal Gyrus, posterior division', 'Left Temporal Pole', 'Left Middle Temporal Gyrus, anterior division', 'Left Putamen', 'Right Temporal Pole', 'Left Superior Temporal Gyrus, anterior division'])

    ### Print the list of areas you have selected and how many areas there are
    print(pre_roi_list)
    print("Number of regions in ROI:", len(pre_roi_list[1]))
    
    ### Get the gridpoints associated with these areas
    roi_idxs = [i for region in pre_roi_list[1] for l in label_names for i in label_dct[l] if region.lower() in l.lower()]
    ### For saving your images later
    roi_name_str = '_'+pre_roi_list[0]

    print(roi_idxs)
    print("Number of gridpoints in ROI:",len(roi_idxs))
    print(roi_name_str)
    #sys.exit(0)

    ### Do one frequency band at a time for now - change between ['1,4','4,8','8,13','13,30','30,80']
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
    
    group_T = stats.ttest_1samp(Z_data,0,axis=0)[0]

    ### ROI
    # Make a smaller array with only regions of interest in it (should be size [num_participants, num_gridpoints in ROI])
    Z_lst_roi=Z_data[:,roi_idxs]

    print(Z_lst_roi.shape)
    print(np.count_nonzero(Z_lst_roi))
    
    import random
    rnd_seed=10
    np.random.seed(seed=rnd_seed)
    
    ### Do permutation test on your roi data - change num_perms to 10,000 for reproducibility when you are happy with your data     
    mnth, mxth, null_mn, null_mx, ts0, pvals = do_stats.do_perms(Z_lst_roi, alph, num_perms=10000, return_null_dist=True)
    
    #### Print min max thresholds and values
    print ('\nalph=%0.2f\nMin thresh: %0.2f Max thresh: %0.2f'%(alph,mnth,mxth))
    print ('Min value: %0.2f Max value: %0.2f\n'%(ts0[0].min(),ts0[0].max()))

    
    ### Rebuild large array with everything set to 0 apart from at ROI
    
    T0=np.zeros_like(group_T)
    T0[roi_idxs]=ts0[0]
    
    ###Get the labels of regions which contain signficant grid points
    
    si = np.where(T0>mxth)[0]
    if len(si) > 0:
        print ('Max regions:',set([l for i in si for l in label_names if i in label_dct[l]]))
    else:
        print('No significant max regions')

    si = np.where(T0<mnth)[0]
    if len(si) > 0:
        print ('Min regions:',set([l for i in si for l in label_names if i in label_dct[l]]))
    else:
        print('No significant min regions')
    #### First time you run through, stop here and see what you are getting as significant
    #sys.exit(0)

    import sys
    sys.path.append('/dagg/public/neuro/MEG_scripts')
    from mne_source_space_warp_MNI import indices_to_MNI,MNI2indices
    src_fn = '/fred/oz266/forward/2003xMSNA_001_5-src.fif'  # any one will do, the result will be the same
    vox_loc,mni_coords = indices_to_MNI(src_fn, grid_spacing=5)
    print (mni_coords[2086])  # get the MNI coordinates for beamformer result index 3088
    for idx in [180,181,182]: # list of indices you want the coords for
        print (','.join(['%0.2f'%v for v in mni_coords[idx]]))


    
    #### For generating your nifi image for plotting
    grid_spacing = 5
    src_fn = '/fred/oz266/forward/2003xMSNA_002_5-src.fif'
    cached_fn = '/fred/oz266/NIFTI/MNI152_%d-src.fif'%grid_spacing
    template_brain = '/fred/oz266/NIFTI/MNI152_T1_1mm_brain_mask.nii.gz'

    out_filename = '/fred/oz266/NIFTI/%s_nifti/contrast/%s-%s_%s_%s.nii.gz' %(RESULTS_DIR,protocol1,protocol2,roi_name_str,fband_str)

    if not os.path.isdir(os.path.dirname(out_filename)):
        os.makedirs(os.path.dirname(out_filename))
    write_interpolated_nifti(T0, grid_spacing, src_fn, cached_fn, template_brain, out_filename)


    ###Make ROI Overlay - this just contains all areas from your ROI, useful for visualisation
    roi_arr = np.zeros_like(group_T)
    print(roi_arr.shape)
    roi_arr[roi_idxs] = 1
    print(roi_arr.shape)
    print(np.count_nonzero(roi_arr))
    
    out_filename_overlay = '/fred/oz266/NIFTI/%s_nifti/contrast/Overlay_%s-%s_%s_%s.nii.gz' %(RESULTS_DIR,protocol1,protocol2,roi_name_str,fband_str)
    write_interpolated_nifti(roi_arr, grid_spacing, src_fn, cached_fn, template_brain, out_filename_overlay)

    overlay = out_filename_overlay

    ### Plot the glassbrain

    ### If you have max regions, then the threshold will be mxth 
    ### If you have min regions, then the threshold will be mnth 
    
    threshold = mxth
    
    #threshold = -mnth
    
    ### For your colour bar - you can change this, but this plots to the max value of the t-scores, and makes symmetrical (eg plots from -vmax to vmax)
    abs_T=np.abs(T0) ### Finds the max absolute value in t-scores (positive or negative)
    vmax = np.nanmax(abs_T)
    vmin=-vmax
    
    stat_map_img = out_filename
    ### You can change the parameters here however you like, see https://nilearn.github.io/dev/auto_examples/01_plotting/plot_demo_glass_brain_extensive.html
    

    display = plotting.plot_glass_brain(stat_map_img, display_mode='lyrz', colorbar=True, cmap="RdYlBu_r", threshold=threshold,black_bg=False, symmetric_cbar=True, plot_abs=False, vmax=vmax,vmin=-vmax)
    #display = plotting.plot_glass_brain(stat_img, title='plot_glass_brain with display_mode="lzr"', black_bg=True, display_mode="lzr", threshold=threshold,black_bg=False, symmetric_cbar=True, plot_abs=False, vmax=vmax,vmin=-vmax)

    ### This adds your ROI overlay of areas included, can change colour or comment if not needed
    display.add_contours(overlay, filled=True, colors='grey', alpha=0.4)

    plotting.show()
    

