import numpy as np
from scipy.io import loadmat
# import matplotlib
# matplotlib.use('TkAgg', force=True)
# print("Switched to:",matplotlib.get_backend())
import torchvision.transforms as T
import torch
import torch.utils.data as data


def descale_params(paramnorm, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return ((((paramnorm - a1) / (b1 - a1)) ) * (Upper_bound - Lower_bound)) + Lower_bound

def scaling(param, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return (b1 - a1) * ((param - Lower_bound) / (Upper_bound - Lower_bound)) + a1

def descaling_params(d, f, dp, bounds):

    D_descaled = descale_params(d, bounds[0, 1], bounds[1, 1])
    F_descaled = descale_params(f, bounds[0, 2], bounds[1, 2])
    Dp_descaled = descale_params(dp, bounds[0, 3], bounds[1, 3])
    return D_descaled, F_descaled, Dp_descaled

def scale_params(d, f, dp, bounds):
    D_scaled = scaling(d,bounds[0, 1], bounds[1, 1])
    F_scaled = scaling(f,bounds[0, 2], bounds[1, 2])
    Dp_scaled = scaling(dp,bounds[0, 3], bounds[1, 3])
    return D_scaled, F_scaled, Dp_scaled

def descale_sigma(d, f, dp, bounds):
    scale_factor_d = bounds[1, 1] - bounds[0, 1]
    scale_factor_f = bounds[1, 2] - bounds[0, 2]
    scale_factor_dp = bounds[1, 3] - bounds[0, 3]
    return d * scale_factor_d, f * scale_factor_f, dp * scale_factor_dp

def ivim_data_load_vivo(path, norm_type='mean'):
    # load simulations

    Dataset = loadmat(path)

    # Prepare Training data
    X_train_2D = Dataset['images']
    X_train_2D = np.transpose(X_train_2D, (0, 3, 1, 2))

   # X_train_2D = X_train_2D[:, 1:, :, :]

    # Normalize with respect to b0 (first b-value)
    mask = Dataset['mask']
    batch_size, bvals, width, height = X_train_2D.shape
    X_train_2D_scaled = np.zeros_like(X_train_2D)

    if norm_type == 's0':

        S0 = np.expand_dims(X_train_2D[:, 0, :, :], axis=1)
        X_train_2D = X_train_2D/ S0  # normalize
        X_train_2D = X_train_2D[:, 1:, :, :]
        X_train_2D_scaled = np.clip(X_train_2D, a_min=None, a_max=1)
        for i in range(bvals-1):
            test_matrix = X_train_2D_scaled[:, i, :, :]
            test_matrix = np.where(mask > 0, test_matrix, 0)
            X_train_2D_scaled[:, i, :, :] = test_matrix  # masking signal background at 0

    elif norm_type == 'mean':

        # Iterate over each sample in the batch
        for i in range(batch_size):
            # Extract bval=0 slice for the current sample
            b0 = X_train_2D[i, 0, :, :]  # This is the b0 image (bval=0)

            roi_mask = mask[i, :, :] > 0  # Mask where D_true > 0
            # Find the min and max values of b0 (over width and height)

            min_b0 = np.min(b0[roi_mask])
            max_b0 = np.max(b0[roi_mask])


            # Perform Min-Max scaling for the entire sample (using the min/max of b0)
            for j in range(bvals):  # Iterate over all bvals
                # Extract the current b-value slice (across the width and height)
                bval_slice = X_train_2D[i, j, :, :]

                # Min-Max scaling with respect to b0 min and max
                X_train_2D_scaled[i, j, :, :] = (bval_slice - min_b0) / (max_b0 - min_b0)


        for i in range(bvals):
            test_matrix = X_train_2D_scaled[:, i, :, :]
            test_matrix = np.where(mask > 0, test_matrix, 0)
            X_train_2D_scaled[:, i, :, :] = test_matrix  # masking signal background at 0

    return X_train_2D_scaled, mask


def ivim_data_load_voxelwise(path):
    # load simulations
    Dataset = loadmat(path)

    # define b values
    b_values = Dataset['b_values'].astype(np.float32)
    b_values = np.transpose(np.squeeze(b_values))

    # load true parameters
    D_true = Dataset['D_true']
    f_true = Dataset['f_true']
    Dstar_true = Dataset['Dstar_true']
    X_train_2D = Dataset['matrix']

    X_train_2D_scaled = np.zeros_like(X_train_2D)

    S0 = np.expand_dims(X_train_2D[:, 0], axis=1)
    X_train_2D = X_train_2D / S0  # normalize
    X_train_2D = X_train_2D[:, 1:]
    X_train_2D_scaled = np.clip(X_train_2D, a_min=None, a_max=1)
    b_values = b_values[1:]

    return X_train_2D_scaled, b_values, D_true[0,:], f_true[0,:], Dstar_true[0,:]

