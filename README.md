# Frontal-Temporo-parietal-power-changes
Repository for code from the paper "Frontal and temporo-parietal changes in delta and alpha band accompany stress-induced vasoconstriction and blood pressure response".

Scripts import several external standardized tools openly available at https://mne.tools/stable/index.html

The models used in analysis are generated via LCMV weighted source power contrast profiles warped to an MNE source space based on each indiviuals T1 weighted MRI. Analysis assessed two conditions - rest and stress 

01_group_stats_contrast_JB.py - runs whole head power contrast

02_group_stats_contrast_ROI_JB.py - runs region of interest power contrast

03_nai_regions2_JB_TEXT.py - runs brain region of interest neural activity indices power output for relative 

04_T_to_Z.py - converts T score to Z score

Frequency band on interest-sys.arguments broken into 
delta " 1,4"; theta " 4,8"; alpha " 8,13"; beta " 13,30"; low gamma " 30,80"; high gamma " 80,120"


