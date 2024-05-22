# Spatial Metabolomics Analyses McIntire Lab

This github repository contains all the code used for the McIntire lab's paper titled "Spatial lipidomics reveals accretion of docosahexaenoic and eicosapentaenoic acids in globus pallidus of Alzheimer’s disease mouse model".

Please contact Dr. Laura Beth McIntire at lbm7002@med.cornell.edu for further information about this research.

## Folders

1. Averaging and Filtering Ion Scripts - Contains the python notebooks used to average the left and right regions of interest and shows the filtering process for each region of interest (*Note that we used the filtered ions from the whole brain for our analyses instead of filtering ions from each region). Results from these files are used for downstream analyses in the SLAT pipeline.

2. Batch Correction and Statistical Testing - Contains the RMarkdown files used to implement batch correction and statistical testing for all regions of interest in this work. Results from these files are found in the boxplots in Figure 2 and Figure 3.

3. Demo - Contains a subsample of data for one of the brains used in our analysis. There are two .ipynb files that help the user work through the preprocessing steps used for our analysis. 

4. Ion Pattern Analysis - Contains the python notebooks used for unsupervised pattern identification within all brains. Results from these files are found in the boxplots in Supplementary Figure S4A and Supplementary Figure S4B.

5. Ion Trajectory Analysis - Contains the python notebooks used for unsupervised clustering of trajectories for significant ions in the globus pallidus and whole brain for negative and positive modes. Results from these files are found in Figure 6.

6. Pie Charts - Contains the python notebook used for creation of the pie charts showing the distribution of significant ions within each group subjected to statistical testing using the SLAT pipeline. Results from these files are found in in Figure 1.

7. Segmentation Scripts - Contains the python notebooks used for the manual segmentation of positive and negative mode brain regions of interest. Results from these files are used for downstream analyses in the SLAT pipeline.


## System Requirements
This code has been tested on Windows 11 OS.

## Code Dependencies
The SLAT pipeline requires the following:
* python (> = 3.10.9)
* seaborn (> = 0.12.2)
* scipy (> = 1.10.0)
* ipympl (> = 0.9.3)
* ipywidgets (> = 0.8.6)
* mpl-interactions (> = 0.23.0)
* pillow (> = 9.4.0)
* R (> = 4.3.1)
* maplet (> = 1.1.2)
* magrittr (> = 2.0.3)
* dply (> = 1.1.2)
* tidyverse (> = 2.0.0)
* car (> = 3.1.2)
* data.table (> = 1.14.8)
* plyr (> = 1.8.9)
* ggpubr (> = 0.6.0)

## Installation
There are no special requirements for installation. All files can be inspected and directly downloaded from github. 

## Demo 
There are two .ipynb files in the Demo folder. A subset of the data from one of the brains used in our analysis is included as "225_a1wt_subsampled.txt", and all output files are included as well from the preprocessing pipeline. The expected output of the "Data Preprocessing Demo" is a final list of percentages of ions within the globus pallidus. The expected output of the "Data Filtering Demo" is a list of ions with abundance greater than two standard deviations than the mean abundance of ions within the whole brain. The run time will vary depending on the user because it requires manual segmentation in the "Data Preprocessing Demo" file. Overall, it will take 10-15 minutes to run both files. 
