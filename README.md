# Multimodal Fusion Strategies for Outcome Prediction in Stroke

This repository contains the implementation and evaluation of a clinical predictors based MLP model, a neuroimaging based CNN model, as well as two multimodal fusion strategies based on neural networks for combining clinical and neuroimaging data for predicting stroke outcome in acute-ischemic patients.

## Publication


## Data
313 patients with acute ischemic stroke from the 1000plus study were included. Currently the data cannot be made publicly accessible due to data protection.

__Inputs:__ 

    1. CLINICAL DATA (7 predictors: 2 continuous, 5 categorical)
    AD_NIH: NIHSS at admission
    AT_LY: thrombolysis
    CH: history of cardiac diseases
    DG_SEX: gender
    RF_DM: diabetes
    RF_HC: hypercholesterolemia
    DG_AG: age
    
    2. NEUROIMAGING DATA (156x192x64 voxels)
    3D Raw TOF-MRA scans


__Output:__ 3 months post-stroke mRS score (good (mrs<=2) : 226 patients , bad (mrs>=3) : 87 patients)

__Exclusion Criteria:__

* Patients with no mRS information were excluded.
* Patients with missing or distorted acute TOF-MRA scans excluded.
* Clinical predictors with more than 5% missing values were excluded.
* Categorical clinical predictors with a yes/no ratio larger than 1/4 were excluded, in order to prevent category imbalance.                               

The table below presents the clinical characteristics of the 1000plus dataset.

| Clinical Information             | Value       | Clinical Information           | Value    | 
|----------------------------------|-------------|--------------------------------|----------|
| Median age (IQR)                 | 72.0 (16.0) | Cardiac history (yes/ no)      | 87/ 226  | 
| Median Initial NIHSS (IQR)       | 3 (4)       | Diabetes (yes/ no)             | 84/ 229  | 
| Thrombolysis (yes/ no)           | 58/ 255     | Hypercholesterolemia (yes/ no) | 187/ 126 |  
| Sex (Females/ males)             | 197/ 116    | 								                |          |

The figure below shows the middle slices of a TOF-MRA image taken from the sagittal, coronal and horizontal planes.


## Methods:

### Data Preprocessing
1. CLINICAL DATA
* The continuous predictors were centered using zero-mean unit-variance normalization.
* Missing values were imputed using mean imputation

2. IMAGING DATA
* Images were resized from 312x384x127 to 156x192x64 voxels due to memory constraints.  
* The voxel intensity values were centered using zero-mean unit-variance normalization.
* Images were loaded into a numpy array and saved as a compressed numpy file (.npz) to be used as input to train the model.
The array has dimensions (313x156x192x64x1) where the final dimension indicates the number of channels (1). The size of the compressed file was around 4,5GB.

### Clinical Data Framework: Multilayer Perceptron (MLP)
The clinical data was modeled using an MLP with a single  fully  connected  (FC) hidden layer.
* The hidden layer  neurons  were  rectified linear units (ReLUs).  
* L2 norm regularization was introduced to penalize weights in the hidden and output layer neurons.
* Dropout was used on the hidden layer neurons.

### Neuroimaging Data Framework: Convolutional Neural Network (CNN)
The 3D imaging data was modeled using a 3D convolutional neural network (CNN) consisting of three convolutional blocks followed by a single FC layer.
