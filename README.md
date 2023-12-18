# Likelihood ratio (LR) map

This directory contains the codes of likelihood ratio (LR) map for exoplanet detection by Angular Differential Imaging. Moreover, it contains an alternative ROC curve generation for direct exoplanet detection.

### CONTENTS:

* README: this file
* likelihoodmap_generator.py: the main code for likelihood ratio map with using class structure (new).
* likelihood_source.py: the main code for likelihood ratio map
* util.py: the utilized functions for likelihood ratio map
* hci_postproc_with_likelihood.py: the modified version of hci_postproc.py file of VIP-HCI package to add likelihood ratio map as an alternative to SNR map.
* roc_updated_vip_roc.py: the modified version of roc.py file of VIP-HCI package to suggest an alternative ROC curve generation.
* test_LRmap.ipynb: test of likelihood ratio map as a detection map using class structure (new).
* test_likelihoodmap.ipynb: test of likelihood ratio map as a detection map.
* test_roc_curve.ipynb: test ROC curve suggestion using likelihood ratio map.

### CITE:
Please cite "Likelihood ratio map for direct exoplanet detection" (https://doi.org/10.48550/arXiv.2210.10609) (https://doi.org/10.1109/IPAS55744.2022.10052997)

Please also provide a link to this webpage in your paper (https://github.com/hazandaglayan/likelihoodratiomap.git)

### Dependencies:
You need to install VIP_HCI, numpy, multiprocessing, and tqdm for the likelihood ratio. 

