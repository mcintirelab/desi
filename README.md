# Spatial Metabolomics Analyses McIntire Lab

This github repository contains all the code used for the McIntire lab's paper titled "Spatial lipidomics reveals accretion of docosahexaenoic and eicosapentaenoic acids in globus pallidus of Alzheimer’s disease mouse model".

## Folders

1. Averaging and Filtering Ion Scripts - Contains the python notebooks used to average the left and right regions of interest and shows the filtering process for each region of interest (*Note that we used the filtered ions from the whole brain for our analyses instead of filtering ions from each region). Results from these files are used for downstream analyses in the SLAT pipeline.

2. Batch Correction and Statistical Testing - Contains the RMarkdown files used to implement batch correction and statistical testing for all regions of interest in this work. Results from these files are found in the boxplots in Figure 2 and Figure 3. 

3. Ion Pattern Analysis - Contains the python notebooks used for unsupervised pattern identification within all brains. Results from these files are found in the boxplots in Supplementary Figure S4A and Supplementary Figure S4B.

4. Ion Trajectory Analysis - Contains the python notebooks used for unsupervised clustering of trajectories for significant ions in the globus pallidus and whole brain for negative and positive modes. Results from these files are found in Figure 6.

5. Pie Charts - Contains the python notebook used for creation of the pie charts showing the distribution of significant ions within each group subjected to statistical testing using the SLAT pipeline. Results from these files are found in in Figure 1.

6. Segmentation Scripts - Contains the python notebooks used for the manual segmentation of positive and negative mode brain regions of interest. Results from these files are used for downstream analyses in the SLAT pipeline.
