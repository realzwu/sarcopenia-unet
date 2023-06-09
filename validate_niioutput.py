import os
import numpy as np
import SimpleITK as sitk
import sys
from log import Logger
import torch
import torchvision
from model import UNET
from utils import load_checkpoint
import albumentations as A
import nibabel as nib
from matplotlib import pyplot as plt

def running_slices(ct_dir, slice_dir, output_ct):
    # we have a series of CT volumes and T4 segmentations
    images = sorted(os.listdir(ct_dir))
    slices = sorted(os.listdir(slice_dir))

    for i in range(len(images)):
        caseid = images[i].replace('.gz', '').replace('.nii', '')
        print(caseid)

        # Read ct and mask files using SimpleITK
        nii = sitk.ReadImage(os.path.join(slice_dir, slices[i]))
        arr = sitk.GetArrayFromImage(nii)
        
        nii_ct = sitk.ReadImage(os.path.join(ct_dir, images[i]))
        arr_ct = sitk.GetArrayFromImage(nii_ct)

        # Are CTs and masks are in the same order?
        if np.shape(arr) != np.shape(arr_ct):
            print("error:" + images[i] +"is not same as"+ slices[i])
            continue

        # find the mask slice
        vector = np.sum(np.sum(arr, axis=1), axis=1)
        seg_level = int(np.where(vector > 10)[0])
        mask = arr[seg_level, :, :].astype(np.float32)

        # get ct slice according to mask's location
        ct_2d= arr_ct[seg_level, :, :].astype(np.float32)

        mask = mask.astype(np.int16)
        if mask.dtype != "int16" or ct_2d.dtype != "float32":
            print(f"{caseid} data type: {mask.dtype}, {ct_2d.dtype}")

        np.save(os.path.join(output_ct, f"{caseid}.npy"), ct_2d)

        ct_2d = np.clip(ct_2d, -100, None)

        transform = A.Normalize(mean=[0.0], std=[1.0])
        transformed = transform(image=ct_2d)
        normalized_ct_2d = transformed["image"]
        x = torch.from_numpy(normalized_ct_2d[np.newaxis, np.newaxis, :, :]).to(device="cuda")

        model = UNET(in_channels=1, out_channels=1).to("cuda")

        load_checkpoint(torch.load(os.path.join(ROOT_DIR, f"model\\t12_checkpoint.pth.tar")), model)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, os.path.join(ROOT_DIR, f"data\\png\\pred_{caseid}.png"))

        np_arr = preds.detach().cpu().numpy()

        case_mask = os.path.join(ROOT_DIR, f"data\\mask\\92.nii.gz")
        mask_nii = nib.load(case_mask)
        mask = mask_nii.get_fdata()

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                for z in range (mask.shape[2]):
                    if (mask[x,y,z] != 0):
                        T12_slice = z
                        break

        mask_output = mask
        mask_output[:, :, T12_slice] = np.rot90(np_arr, 3, (2,3))

        #plt.imshow(mask_output[:, :, T12_slice], interpolation='nearest')
        #plt.show()

        '''
        cos_gamma = np.cos(4.71239)
        sin_gamma = np.sin(4.71239)
        #rotation_affine = np.array([[1, 0, 0, 0],
        #                            [0, cos_gamma, -sin_gamma, 0],
        #                            [0, sin_gamma, cos_gamma, 0],
        #                            [0, 0, 0, 1]])

        rotation_affine = np.array([[cos_gamma, -sin_gamma, 0, 0],
                                    [sin_gamma, cos_gamma, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        affine_rot = rotation_affine.dot(mask_nii.affine)
        '''

        output_image = nib.Nifti1Image(mask_output, affine=mask_nii.affine, header=mask_nii.header, dtype='mask')

        ornt = np.array([[0, 1],
                         [1, -1],
                         [2, 1]])

        output_image = output_image.as_reoriented(ornt)

        nib.save(output_image, os.path.join(ROOT_DIR, f"data\\output\\92_output.nii.gz"))



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

ct_dir = os.path.join(ROOT_DIR, 'data/image/')

'''
def testing_t4():
    t4_dir = os.path.join(ROOT_DIR, 'data/nifti/test_t4')
    output_ct_t4 = os.path.join(ROOT_DIR, 'data/nifti/npy_t4')
    sys.stdout = Logger(os.path.join(ROOT_DIR, 't4.txt'))
    running_slices(ct_dir, t4_dir, output_ct_t4)
'''

def testing_t12():
    t12_dir = os.path.join(ROOT_DIR, 'data/mask/')
    output_ct_t12 = os.path.join(ROOT_DIR, 'data/npy')
    sys.stdout = Logger(os.path.join(ROOT_DIR, 't12.txt'))
    running_slices(ct_dir, t12_dir, output_ct_t12)


testing_t12()