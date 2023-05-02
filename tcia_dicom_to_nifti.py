## based on python 3.8.12

import os
import SimpleITK as sitk


def get_all_dicom_files(root_folder):
    dicom_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files


def convert_dicom_folder_to_nifti(input_folder, output_folder):
    # list all dicom files in the folders (sub-folders)
    for subdir in sorted(os.listdir(input_folder)):
        subdir_path = os.path.join(input_folder, subdir)

        if os.path.isdir(subdir_path):
            # create output folder, and get series of dicom files
            os.makedirs(output_folder, exist_ok=True)
            dicom_files = get_all_dicom_files(subdir_path)

            if not dicom_files:
                print(f"No DICOM files found in {subdir_path}. Skipping...")
                continue

            # hundreds of dicom files to single nifti file
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)
            image = reader.Execute()

            # save_nifti_files
            nifti_file = os.path.join(output_folder, f"{subdir}.nii.gz")
            sitk.WriteImage(image, nifti_file)


'''
# SPIE-AAPM Lung CT Challenge
input_folder = r"F:\sarcopenia-unet\data\TCIA\dicom\manifest-1682752878304\SPIE-AAPM Lung CT Challenge"
output_folder = r"F:\sarcopenia-unet\data\TCIA\nifti\SPIE-AAPM Lung CT Challenge"

convert_dicom_folder_to_nifti(input_folder, output_folder)

# LungCT-Diagnosis: unclear
# Qin Lung CT: not cover T12
'''

# Lung CT Segmentation Challenge 2017 (LCTSC)
input_folder = r"F:\sarcopenia-unet\data\TCIA\dicom\manifest-1683006587994\LCTSC"
output_folder = r"F:\sarcopenia-unet\data\TCIA\nifti\LCTSC"

convert_dicom_folder_to_nifti(input_folder, output_folder)
