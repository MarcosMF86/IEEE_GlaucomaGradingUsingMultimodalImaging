# Code to create and optimize CNN models for glaucoma stage classification using two imaging modalities, fundus images and 3D OCT volumes.



This repository contains source codes used for grading glaucoma stage using fundus images and OCT volume scans, relesead by GAMMA challenge

#Problem

Automate glaucoma grading stage using two images modalities, fundus images and 3D OCT volumes scan.

#Proposed Method

We proposed a glaucoma grading method usind fundus images and 3D OCT volumes through deep learning. In first step, a hyperparameter optimization was carried out, evaluating different CNN models, handling each imaging modality as input, whole fundus images, optic disc region image cropped from fundus images and 3D OCT. In second step, ensemble and feature combination strategies were evaluated to grade glaucoma stage.

#Results

The Table below presents the results obtained during the model optimization step, taking each of the imaging modalities as input. In some cases, the models were not evaluated due to the limitation of submitting results on the challenge organizers' website. The models were chosen to be evaluated according to the validation loss obtained during the training step. The best results are shown in bold.
![alt text](https://github.com/MarcosMF86/Glaucoma-Grading/blob/main/Results.PNG?raw=true)

The Table below presents the results obtained using ensemble and feature combination.
![alt text](https://github.com/MarcosMF86/Glaucoma-Grading/blob/main/results_ensemble.PNG?raw=true)
![alt text](https://github.com/MarcosMF86/Glaucoma-Grading/blob/main/results_ensemblePNG?raw=true)
In the official challenge website: <a href="https://aistudio.baidu.com/aistudio/competition/detail/807/0/introduction" target="_blank">GAMMA</a>, the dataset can be found, and classifications results can be submitted. 

