# Documentation: PSMA-FDG-PET-CT-Lesions

The data inside /data/PSMA-FDG-PET-CT-Lesions

is retreived from https://doi.org/10.57754/FDAT.6gjsg-zcg93, and is used by https://autopet-iv.grand-challenge.org/dataset/

## Info

### FDG-PET-CT-Lesions | A whole-body FDG-PET/CT dataset with manually annotated tumor lesions

Description
Introduction
A publicly available dataset of annotated Positron Emission Tomography/Computed Tomography (PET/CT) studies. 1014 whole body Fluorodeoxyglucose (FDG)-PET/CT studies (900 patients) and 597 prostate-specific membrane antigen (PSMA)-PET/CT studies (378 patients) acquired between 2014 and 2022 were included. The FDG cohort comprises 501 patients diagnosed with histologically proven malignant melanoma, lymphoma, or lung cancer, along with 513 negative control patients. The PSMA cohort includes pre- and/or post-therapeutic PET/CT images of male individuals with prostate carcinoma, encompassing images with (537) and without PSMA-avid tumor lesions (60). Notably, the training datasets exhibit distinct age distributions: the FDG UKT cohort spans 570 male patients (mean age: 60; std: 16) and 444 female patients (mean age: 58; std: 16), whereas the PSMA MUC cohort tends to be older, with 378 male patients (mean age: 71; std: 8). Additionally, there are variations in imaging conditions between the FDG Tübingen and PSMA Munich cohorts, particularly regarding the types and number of PET/CT scanners utilized for acquisition. The PSMA Munich dataset was acquired using three different scanner types (Siemens Biograph 64-4R TruePoint, Siemens Biograph mCT Flow 20, and GE Discovery 690), whereas the FDG Tübingen dataset was acquired using a single scanner (Siemens Biograph mCT).

Structure and usage
The data is organized in the nnUNet structure:

|--- imagesTr\
|--- tracer_patient1_study1_0000.nii.gz (CT image resampled to PET)\
|--- tracer_patient1_study1_0001.nii.gz (PET image in SUV)\
|--- ...\
|--- labelsTr\
|--- tracer_patient1_study1.nii.gz (manual annotations of tumor lesions)

|--- dataset.json (nnUNet dataset description)\
|--- dataset_fingerprint.json (nnUNet dataset fingerprint)

|--- splits_final.json (reference 5fold split)

|--- psma_metadata.csv (metadata csv for psma)\
|--- fdg_metadata.csv (original metadata csv for fdg)

We demonstrate how this dataset can be used for deep learning-based automated analysis of PET/CT data and provide the trained deep learning model: www.autopet.org

PET/CT acquisition protocol
FDG dataset: Patients fasted at least 6 h prior to the injection of approximately 350 MBq 18F-FDG. Whole-body PET/CT images were acquired using a Biograph mCT PET/CT scanner (Siemens, Healthcare GmbH, Erlangen, Germany) and were initiated approximately 60 min after intravenous tracer administration. Diagnostic CT scans of the neck, thorax, abdomen, and pelvis (200 reference mAs; 120 kV) were acquired 90 sec after intravenous injection of a contrast agent (90-120 ml Ultravist 370, Bayer AG) or without contrast agent (in case of existing contraindications). PET Images were reconstructed iteratively (three iterations, 21 subsets) with Gaussian post-reconstruction smoothing (2 mm full width at half-maximum). Slice thickness on contrast-enhanced CT was 2 or 3 mm.

PSMA dataset: Examinations were acquired on different PET/CT scanners (Siemens Biograph 64-4R TruePoint, Siemens Biograph mCT Flow 20, and GE Discovery 690). The imaging protocol mainly consisted of a diagnostic CT scan from the skull base to the mid-thigh using the following scan parameters: reference tube current exposure time product of 143 mAs (mean); tube voltage of 100kV or 120 kV for most cases, slice thickness of 3 mm for Biograph 64 and Biograph mCT, and 2.5 mm for GE Discovery 690 (except for 3 cases with 5 mm). Intravenous contrast enhancement was used in most studies (571), except for patients with contraindications (26).

The whole-body PSMA-PET scan was acquired on average around 74 minutes after intravenous injection of 246 MBq 18F-PSMA (mean, 369 studies) or 214 MBq 68Ga-PSMA (mean, 228 studies), respectively. The PET data was reconstructed with attenuation correction derived from corresponding CT data. For GE Discovery 690 the reconstruction process employed a VPFX algorithm with voxel size 2.73 mm × 2.73 mm × 3.27 mm, for Siemens Biograph mCT Flow 20 a PSF+TOF algorithm (2 iterations, 21 subsets) with voxel size 4.07 mm × 4.07 mm × 3.00 mm, and for Siemens Biograph 64-4R TruePoint a PSF algorithm (3 iterations, 21 subsets) with voxel size 4.07 mm × 4.07 mm × 5.00 mm.

Annotation
FDG PET/CT training and test data from UKT was annotated by a Radiologist with 10 years of experience in Hybrid Imaging and experience in machine learning research. FDG PET/CT test data from LMU was annotated by a radiologist with 8 years of experience in hybrid imaging. PSMA PET/CT training and test data from LMU as well as PSMA PET/CT test data from UKT was annotated by a single reader and reviewed by a radiologist with 5 years of experience in hybrid imaging.

The following annotation protocol was defined:
Step 1: Identification of tracer-avid tumor lesions by visual assessment of PET and CT information together with the clinical examination reports.
Step 2: Manual free-hand segmentation of identified lesions in axial slices.

## Citation

Gatidis, S., Küstner, T., Ingrisch, M., Hepp, T., Früh, M., Nikolaou, K., La Fougère, C., Pfannenberg, C., Fabritius, M., Jeblick, K., Schachtner, B., Wesp, P., Mittermeier, A., Unterrainer, L., Sheikh, G., Böning, G., Brendel, M., Ricke, J., Gu, S., … Cyran, C. (2024). PSMA-FDG-PET-CT-Lesions. University of Tübingen, Ludwig-Maximilians-University Munich. https://doi.org/10.57754/FDAT.6gjsg-zcg93
