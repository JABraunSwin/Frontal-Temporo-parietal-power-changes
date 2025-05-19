import os,sys,pickle,glob
import scipy.stats
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

def get_individual_nai_stats(pid,protocol,fband_str):
    """Return the mean and variance of the Neural Activity Index at every beamformer grid point, and the number of epochs used."""
    
    fn = '/fred/oz266/%s/2003xMSNA_%s_PL_%s_5_%s.pkl'%(RESULTS_DIR,pid,protocol,fband_str)

    with open(fn,'rb') as f:
        #nai,nai_mean,nai_sem,num_epochs,filters = pickle.load(f)
        nai, num_epochs, filters, filters_SAM = pickle.load(f)

    print("Nai shape:",nai.shape)
    ### For group level statistics, we just want mean power across epochs and standard error of the mean for this condition
    nai_mean = nai.mean(axis=1)
    from scipy import stats
    nai_sem = stats.sem(nai, axis=1)
    print("Nai mean shape:",nai_mean.shape)
    nai_var = (nai_sem * nai_sem)* num_epochs       # Need variance, not standard error
    
    return nai_mean,nai_var,num_epochs

if __name__ == "__main__":
        
    ### Put all participants in one group - change this to reflect all participants
    pid_lst_total=["001", "002", "003", "004", "005", "007", "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "031", "032", "033", "034"]
    #pid_lst_total=["001", "002", "003", "004", "005", "007", "008", "011", "013", "014", "015", "016", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "031", "032", "033", "034"]
    
    protocol1 = 'REST'
    #protocol1 = 'STRESS'

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

    ### Choose one area at the time (you can combine left and right hem or do separately, just make sure you define the name of the list for saving later)
    
    #pre_roi_list=('FrontML',['Left Frontal Medial Cortex'])
    #pre_roi_list=('FrontMR',['Right Frontal Medial Cortex'])
    #pre_roi_list=('MidFL',['Left Middle Frontal Gyrus'])
    #pre_roi_list=('MidFR',['Right Middle Frontal Gyrus'])
    #pre_roi_list=('FrontPL',['Right Frontal Pole'])
    #pre_roi_list=('FrontPR',['Left Frontal Pole'])
    #pre_roi_list=('CingAL',['Left Cingulate Gyrus, anterior Division'])
    #pre_roi_list=('CingAR',['Right Cingulate Gyrus, anterior division'])
    #pre_roi_list=('SupFL',['Left Superior Frontal Gyrus'])
    #pre_roi_list=('SupFR',['Right Superior Frontal Gyrus'])
    #pre_roi_list=('InsL',['Left Insular Cortex'])
    #pre_roi_list=('InsR',['Right Insular Cortex'])
    #pre_roi_list=('CINGPL',['Left Cingulate Gyrus, posterior division'])
    #pre_roi_list=('CingPR',['Right Cingulate Gyrus, posterior division'])
    #pre_roi_list=('PrecL',['Left Precuneous Cortex'])
    #pre_roi_list=('PrecR',['Right Precuneous Cortex'])
    #pre_roi_list=('HippL',['Left Hippocampus']) 
    #pre_roi_list=('HippR',['Right Hippocampus'])
    #pre_roi_list=('ThalL',['Left Thalamus'])
    #pre_roi_list=('ThalR',['Right Thalamus'])
    #pre_roi_list=('SupPL',['Left Superior Parietal Lobule']) 
    #pre_roi_list=('SupPR',['Right Superior Parietal Lobule'])
    #pre_roi_list=('PostcL',['Left Postcentral Gyrus'])
    #pre_roi_list=('PostcR',['Right Postcentral Gyrus'])
    #pre_roi_list=('AmyL',['Left amygdala'])
    #pre_roi_list=('AmyR',['Right amygdala'])
    pre_roi_list=('Brainstem',['Brain-Stem'])

    print("Number of regions in ROI:", len(pre_roi_list[1]))

    ### Get the gridpoints associated with these areas
    roi_idxs = [i for region in pre_roi_list[1] for l in label_names for i in label_dct[l] if region.lower() in l.lower()]
    
    roi_name_str = pre_roi_list[0]
    print(len(roi_idxs))
    print(roi_idxs)
    

    fband_str=str(sys.argv[1])
    print ('\n%s Hz\n'%fband_str,flush=True)
    

    nai_dict={}
    nai_dict_norm={}
    for pid in pid_lst_total:
        ### Get beamformer data for each person
            mn1,v1,n1 = get_individual_nai_stats(pid, protocol1, fband_str)
            
            ### For normalisation
            mean_head=np.mean(mn1)
            
            std_head=np.std(mn1)
            
            norm_mn1=[]
            for vox in mn1:
                norm=(vox-mean_head)/std_head
                norm_mn1.append(norm)
            norm_mn1=np.array(norm_mn1)

            ### Find nai at each gridpoint
            reg_idx=[]
            reg_idx_norm=[]
            for id in roi_idxs:
                reg_idx.append(mn1[id])
                reg_idx_norm.append(norm_mn1[id])
            #print(len(reg_idx))

            ### Take the mean over all gridpoints within a region
            nai_reg=np.mean(reg_idx)
            nai_reg_norm=np.mean(reg_idx_norm)
           
            ### Will print out this number for every person, will also save to a text file below
            #print("Participant:",pid)
            print("NAI:",nai_reg)
            print("NAI norm:",nai_reg_norm)

            nai_dict[pid]=nai_reg
            nai_dict_norm[pid]=nai_reg_norm
            #sys.exit(0)
    
    ### Prints the whole group together
    print(roi_name_str)
    print(fband_str)
    print("Un-normalised:")
    print(nai_dict)
    print("Normalised:")
    print(nai_dict_norm)

    ### Saves this output to a text file you can open in /fred/oz266/nai_regions
    filename='/fred/oz266/nai_regions/JB_NAI_REST/'+roi_name_str+'_'+fband_str+'.txt'

    with open(filename,'w') as f:
        print(nai_dict,file=f)

    filename_norm='/fred/oz266/nai_regions/JB_NAI_NORM_REST/'+roi_name_str+'_'+fband_str+'.txt'

    with open(filename_norm,'w') as f:
        print(nai_dict_norm,file=f)
   
    