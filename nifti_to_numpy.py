## based on python 3.8.12

import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage

def save_t4_slices(ct_dir, t4_dir, output_t4, output_ct_t4):
    # we have a series of CT volumes and T4 segmentations
    images = sorted(os.listdir(ct_dir))
    t4s = sorted(os.listdir(t4_dir))

    for i in range(149):
        # Read mask files using SimpleITK
        nii_t4 = sitk.ReadImage(os.path.join(t4_dir, t4s[i]))
        arr_t4 = sitk.GetArrayFromImage(nii_t4)

        # find the mask slice and combine pectoralis major (=1) and minor (=2)
        vector = np.sum(np.sum(arr_t4, axis=1), axis=1)
        t4_seg_level = int(np.where(vector > 100)[0])
        mask = arr_t4[t4_seg_level, :, :]
        mask[mask == 2.] = 1.

        # Resize to actual image (may help? or may not)
        space = nii_t4.GetSpacing()
        scale_factor = (0.38 / space[0], 0.38 / space[1])
        mask_resized = scipy.ndimage.zoom(mask, scale_factor, order=3)

        # Read ct volumes using SimpleITK
        nii_ct = sitk.ReadImage(os.path.join(ct_dir, images[i]))
        arr_ct = sitk.GetArrayFromImage(nii_ct)

        # Are cts and masks are in the same order?
        if np.shape(arr_t4) != np.shape(arr_ct):
            print(images[i])
            print(t4s[i])
            continue

        # get ct slice according to mask's location
        caseid = images[i].replace('.gz', '').replace('.nii', '')
        ct_2d= arr_ct[t4_seg_level, :, :]

        # We wish to centralize the image. So, we normalize the arrays and find the centroid
        ct_2d = np.clip(ct_2d, -100, None)
        ct_resized = scipy.ndimage.zoom(ct_2d, scale_factor, order=3)

        # Normalize intensity
        norm_ct = (ct_resized + 100) / 100
        norm_ct[norm_ct > 1] = 1

        # Compute centroid
        y, x = np.where(norm_ct > 0)
        centroid = (int(np.mean(y)), int(np.mean(x)))
        print(centroid)

        # Pad the original image with background intensity, then cropped
        pad_size = ((128, 128), (128, 128))
        mask_padded = np.pad(mask_resized, pad_size, mode='constant', constant_values=0)
        ct_padded = np.pad(ct_resized, pad_size, mode='constant', constant_values=-100)

        y_min = centroid[0] - 128 + 128
        y_max = centroid[0] + 128 + 128
        x_min = centroid[1] - 128 + 128
        x_max = centroid[1] + 128 + 128

        mask_cropped = mask_padded[y_min:y_max, x_min:x_max]
        ct_cropped = ct_padded[y_min:y_max, x_min:x_max]

        # To expand dataset, we perform horizontal & vertical flip, to there will be four folds images for input
        mask_h_flip = np.fliplr(mask_cropped)
        mask_v_flip = np.flipud(mask_cropped)
        mask_both_flip = np.flipud(mask_h_flip)

        ct_h_flip = np.fliplr(ct_cropped)
        ct_v_flip = np.flipud(ct_cropped)
        ct_both_flip = np.flipud(ct_h_flip)

        np.save(os.path.join(output_t4, f"{caseid}_hflip.npy"), mask_h_flip)
        np.save(os.path.join(output_t4, f"{caseid}_vflip.npy"), mask_v_flip)
        np.save(os.path.join(output_t4, f"{caseid}_bflip.npy"), mask_both_flip)
        np.save(os.path.join(output_t4, f"{caseid}.npy"), mask_cropped)

        np.save(os.path.join(output_ct_t4, f"{caseid}_hflip.npy"), ct_h_flip.astype(np.float32))
        np.save(os.path.join(output_ct_t4, f"{caseid}_vflip.npy"), ct_v_flip.astype(np.float32))
        np.save(os.path.join(output_ct_t4, f"{caseid}_bflip.npy"), ct_both_flip.astype(np.float32))
        np.save(os.path.join(output_ct_t4, f"{caseid}.npy"), ct_cropped.astype(np.float32))

# The resize of T12 slice is according to its final size on numpy arrays.
def save_t12_slices(ct_dir, t12_dir, output_t12, output_ct_t12):
    images = sorted(os.listdir(ct_dir))
    t12s = sorted(os.listdir(t12_dir))

    for i in range(149):
        nii_t12 = sitk.ReadImage(os.path.join(t12_dir, t12s[i]))
        arr_t12 = sitk.GetArrayFromImage(nii_t12)
        # space = nii_t12.GetSpacing()
        # scale_factor = (1 / space[0], 1 / space[1])

        vector = np.sum(np.sum(arr_t12, axis=1),axis=1)
        t12_seg_level = int(np.where(vector > 500)[0])
        mask = arr_t12[t12_seg_level,:,:]

        nii_ct = sitk.ReadImage(os.path.join(ct_dir, images[i]))
        arr_ct = sitk.GetArrayFromImage(nii_ct)

        if np.shape(arr_t12) != np.shape(arr_ct):
            print(images[i])
            print(t12s[i])
            continue

        caseid = images[i].replace('.gz','').replace('.nii','')
        ct_2d = arr_ct[t12_seg_level,:,:]
        ct_2d = np.clip(ct_2d, -100, None)

        # Normalize intensity
        norm_ct = (ct_2d + 100) / 100
        norm_ct[norm_ct > 1] = 1

        # Compute centroid and standard deviation. Noted that standard deviation is for calculation and resizing the T12 slice.
        y, x = np.where(norm_ct > 0)
        centroid = (int(np.mean(y)), int(np.mean(x)))
        std_dev = (int(np.std(y)), int(np.std(x)))
        print(centroid)
        print(std_dev)

        # Pad the original image
        pad_size = ((128, 128), (128, 128))
        mask_padded = np.pad(mask, pad_size, mode='constant', constant_values= 0)
        ct_padded = np.pad(ct_2d, pad_size, mode='constant', constant_values= -100)

        # calculate resize factor. "50" is manually adjusted for the biggest muscle area on numpy arrays.
        scale_factor = (50 / std_dev[0], 50 / std_dev[1])
        print(scale_factor)
        mask_resized = scipy.ndimage.zoom(mask_padded, scale_factor, order=3)
        ct_resized = scipy.ndimage.zoom(ct_padded, scale_factor, order=3)

        y_min = - 128 + int((centroid[0] + 128) * scale_factor[0])
        y_max = + 128 + int((centroid[0] + 128) * scale_factor[0])
        x_min = - 128 + int((centroid[1] + 128) * scale_factor[1])
        x_max = + 128 + int((centroid[1] + 128) * scale_factor[1])

        mask_cropped = mask_resized[y_min:y_max, x_min:x_max]
        print(np.shape(mask_cropped))
        ct_cropped = ct_resized[y_min:y_max, x_min:x_max]
        print(np.shape(ct_cropped))

        # Perform horizontal & vertical flip
        mask_h_flip = np.fliplr(mask_cropped)
        mask_v_flip = np.flipud(mask_cropped)
        mask_both_flip = np.flipud(mask_h_flip)

        ct_h_flip = np.fliplr(ct_cropped)
        ct_v_flip = np.flipud(ct_cropped)
        ct_both_flip = np.flipud(ct_h_flip)

        np.save(os.path.join(output_t12, f"{caseid}_hflip.npy"), mask_h_flip)
        np.save(os.path.join(output_t12, f"{caseid}_vflip.npy"), mask_v_flip)
        np.save(os.path.join(output_t12, f"{caseid}_bflip.npy"), mask_both_flip)
        np.save(os.path.join(output_t12, f"{caseid}.npy"), mask_cropped)

        np.save(os.path.join(output_ct_t12, f"{caseid}_hflip.npy"), ct_h_flip.astype(np.float32))
        np.save(os.path.join(output_ct_t12, f"{caseid}_vflip.npy"), ct_v_flip.astype(np.float32))
        np.save(os.path.join(output_ct_t12, f"{caseid}_bflip.npy"), ct_both_flip.astype(np.float32))
        np.save(os.path.join(output_ct_t12, f"{caseid}.npy"), ct_cropped.astype(np.float32))

# Not finished
def save_l3_slices(ct_dir, l3_dir, output_l3, output_raw_t12):
    images = sorted(os.listdir(ct_dir))
    l3s = sorted(os.listdir(l3_dir))

    for i in range(149):
        nii_l3 = sitk.ReadImage(os.path.join(l3_dir, l3s[i]))
        arr_l3 = sitk.GetArrayFromImage(nii_l3)

        vector = np.sum(np.sum(arr_l3, axis=1),axis=1)
        l3_seg_level = int(np.where(vector > 500)[0])
        caseid = l3s[i].replace('.nii.gz','').replace('pp','')
        slice2d = arr_l3[l3_seg_level,:,:]
        np.save(os.path.join(output_l3, f"{caseid}.npy"), slice2d)

        nii_ct = sitk.ReadImage(os.path.join(ct_dir, images[i]))
        arr_ct = sitk.GetArrayFromImage(nii_ct)
        print(np.shape(arr_l3))
        print(np.shape(arr_ct))
        if np.shape(arr_l3) != np.shape(arr_ct):
            print(images[i])
            print(l3s[i])
            continue
        caseid = images[i].replace('.gz','').replace('.nii','')
        slice2d = arr_ct[l3_seg_level,:,:]
        slice2d = np.clip(slice2d, -100, None)

        np.save(os.path.join(output_raw_t12, f"{caseid}.npy"), slice2d.astype(np.float32))


ct_dir = r'F:\sarcopenia-unet\data\nifti\ct'

t4_dir = r'F:\sarcopenia-unet\data\nifti\t4'
output_t4 = r'F:\sarcopenia-unet\data\npy\t4'
output_ct_t4 = r'F:\sarcopenia-unet\data\npy\ct_t4'

t12_dir = r'F:\sarcopenia-unet\data\nifti\t12'
output_t12 = r'F:\sarcopenia-unet\data\npy\t12'
output_ct_t12 = r'F:\sarcopenia-unet\data\npy\ct_t12'

l3_dir = r'F:\sarcopenia-unet\data\nifti\l3'
output_l3 = r'F:\sarcopenia-unet\data\npy\l3'
output_ct_l3 = r'F:\sarcopenia-unet\data\npy\ct_l3'


save_t4_slices(ct_dir, t4_dir, output_t4, output_ct_t4)
save_t12_slices(ct_dir, t12_dir, output_t12, output_ct_t12)
# save_l3_slices(ct_dir, l3_dir, output_l3, output_ct_l3)


'''
# images can be visualized here:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mask_dir = r'F:\sarcopenia-unet\data\npy\t4'
    image_dir = r'F:\sarcopenia-unet\data\npy\ct_t4'

    npy_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.npy')]
    npy_files.sort()

    for i in range(144):
        image = np.load(npy_files[i])
        plt.subplot(12, 12, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.show()
'''