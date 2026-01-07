import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg', force=True)
# from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())
from scipy.stats import median_abs_deviation
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import matplotlib.pyplot as plt


def metric_bias(Dt, D_true, Fp, f_true, Dp, Dstar_true, return_raw=False):

    mask_Dt = D_true != 0
    mask_Fp = f_true != 0
    mask_Dp = Dstar_true != 0

    err_D = (D_true[mask_Dt] - Dt[mask_Dt]) / D_true[mask_Dt]
    err_Dstar = (Dstar_true[mask_Dp] - Dp[mask_Dp]) / Dstar_true[mask_Dp]
    err_f = (f_true[mask_Fp] - Fp[mask_Fp]) / f_true[mask_Fp]

    # Calculate metrics
    max_err_D = np.max(err_D)
    max_err_f = np.max(err_f)
    max_err_Dstar = np.max(err_Dstar)

    min_err_D = np.min(err_D)
    min_err_f = np.min(err_f)
    min_err_Dstar = np.min(err_Dstar)

    median_err_D = np.median(err_D)
    median_err_f = np.median(err_f)
    median_err_Dstar = np.median(err_Dstar)

    mad_err_D = np.median(np.abs(err_D - np.median(err_D)))
    mad_err_f = np.median(np.abs(err_f - np.median(err_f)))
    mad_err_Dstar = np.median(np.abs(err_Dstar - np.median(err_Dstar)))

    mean_err_D = np.mean(err_D)
    mean_err_f = np.mean(err_f)
    mean_err_Dstar = np.mean(err_Dstar)

    std_err_D = np.std(err_D)
    std_err_f = np.std(err_f)
    std_err_Dstar = np.std(err_Dstar)

    # Create a DataFrame
    D_metrics = [median_err_D, mad_err_D, mean_err_D, std_err_D, max_err_D, min_err_D]
    f_metrics = [median_err_f, mad_err_f, mean_err_f, std_err_f, max_err_f, min_err_f]
    Dstar_metrics = [median_err_Dstar, mad_err_Dstar, mean_err_Dstar, std_err_Dstar, max_err_Dstar, min_err_Dstar]
    Metrics = ['Median', 'Mad', 'Mean', 'Std', 'Max', 'Min']

    data = {'Metrics': Metrics, 'D_metrics': D_metrics, 'f_metrics': f_metrics, 'Dstar_metrics': Dstar_metrics}
    df = pd.DataFrame(data)

    if return_raw:
        return df, err_D, err_f, err_Dstar

    return df

def metric_absolute_error(Dt, D_true, Fp, f_true, Dp, Dstar_true, return_raw=False):
    """
    Calculate the RMSE (Root Mean Squared Error) between the predicted and true parameters.

    Args:
        Dt (np.ndarray): Predicted pseudo-diffusion coefficient (shape: bsize, 76, 76).
        D_true (np.ndarray): True pseudo-diffusion coefficient (shape: bsize, 76, 76).
        Fp (np.ndarray): Predicted perfusion fraction (shape: bsize, 76, 76).
        f_true (np.ndarray): True perfusion fraction (shape: bsize, 76, 76).
        Dp (np.ndarray): Predicted diffusion coefficient (shape: bsize, 76, 76).
        Dstar_true (np.ndarray): True diffusion coefficient (shape: bsize, 76, 76).

    Returns:
        rmse_D: RMSE for the pseudo-diffusion coefficient.
        rmse_Dstar: RMSE for the diffusion coefficient.
        rmse_f: RMSE for the perfusion fraction.
    """

    mask_Dt = D_true != 0
    mask_Fp = f_true != 0
    mask_Dp = Dstar_true != 0

    err_D = np.absolute(D_true[mask_Dt] - Dt[mask_Dt]) / D_true[mask_Dt]
    err_Dstar = np.absolute(Dstar_true[mask_Dp] - Dp[mask_Dp]) / Dstar_true[mask_Dp]
    err_f = np.absolute(f_true[mask_Fp] - Fp[mask_Fp]) / f_true[mask_Fp]

    # Calculate metrics
    max_err_D = np.max(err_D)
    max_err_f = np.max(err_f)
    max_err_Dstar = np.max(err_Dstar)

    min_err_D = np.min(err_D)
    min_err_f = np.min(err_f)
    min_err_Dstar = np.min(err_Dstar)

    median_err_D = np.median(err_D)
    median_err_f = np.median(err_f)
    median_err_Dstar = np.median(err_Dstar)

    mad_err_D = np.median(np.abs(err_D - np.median(err_D)))
    mad_err_f = np.median(np.abs(err_f - np.median(err_f)))
    mad_err_Dstar = np.median(np.abs(err_Dstar - np.median(err_Dstar)))

    mean_err_D = np.mean(err_D)
    mean_err_f = np.mean(err_f)
    mean_err_Dstar = np.mean(err_Dstar)

    std_err_D = np.std(err_D)
    std_err_f = np.std(err_f)
    std_err_Dstar = np.std(err_Dstar)

    # Create a DataFrame
    D_metrics = [median_err_D, mad_err_D, mean_err_D, std_err_D, max_err_D, min_err_D]
    f_metrics = [median_err_f, mad_err_f, mean_err_f, std_err_f, max_err_f, min_err_f]
    Dstar_metrics = [median_err_Dstar, mad_err_Dstar, mean_err_Dstar, std_err_Dstar, max_err_Dstar, min_err_Dstar]
    Metrics = ['Median', 'Mad', 'Mean', 'Std', 'Max', 'Min']

    data = {'Metrics': Metrics, 'D_metrics': D_metrics, 'f_metrics': f_metrics, 'Dstar_metrics': Dstar_metrics}
    df = pd.DataFrame(data)

    if return_raw:
        return df, err_D, err_f, err_Dstar

    # Save the DataFrame to a CSV file
    #df.to_csv(f'abserr_SNR_{SNR}.csv', index=False)

    return df

def RCV(D, f, Dstar, return_raw=False):
    # Function to create a modified Shepp-Logan phantom
    def create_brain_mask():

        # Generate phantom
        phantom = shepp_logan_phantom()
        phantom = resize(phantom, (76, 76), anti_aliasing=False)

        # Quantize phantom to specific values
        phantom[phantom < 0] = 0.5
        brain_mask = np.round(phantom * 10) / 10

        return brain_mask

    # Function to create binary ROIs
    def create_rois(brain_mask, values):
        rois = []
        for value in values:
            roi = (brain_mask == value).astype(np.uint8)
            rois.append(roi)
        return rois

    # Robust Coefficient of Variation calculation
    def calculate_rcv(parameter, rois):
        rcv = []
        for roi in rois:
            # Extract ROI values
            roi_values = parameter[roi == 1]
            if len(roi_values) > 0:
                # Compute RCV
                mad = median_abs_deviation(roi_values)  # Scale for normal distribution
                median = np.median(roi_values)
                rcv.append(1.4826*mad / median if median != 0 else np.nan)
            else:
                rcv.append(np.nan)
        return np.array(rcv)

    # Main processing script
    brain_mask = create_brain_mask()

    # 2. Define ROIs based on intensity levels
    roi_values = [1, 0.5, 0.4, 0.3, 0.2, 0.1]
    rois = create_rois(brain_mask, roi_values)

    # 3. Calculate RCV for each parameter
    rcv_D = np.array([calculate_rcv(D[i, :, :], rois) for i in range(D.shape[0])])
    rcv_f = np.array([calculate_rcv(f[i, :, :], rois) for i in range(f.shape[0])])
    rcv_Dstar = np.array([calculate_rcv(Dstar[i, :, :], rois) for i in range(Dstar.shape[0])])

    # 4. Compute Median and MAD of RCVs
    rcv_D_median, rcv_D_mad = np.nanmedian(rcv_D), median_abs_deviation(rcv_D.flatten(), nan_policy='omit')
    rcv_f_median, rcv_f_mad = np.nanmedian(rcv_f), median_abs_deviation(rcv_f.flatten(), nan_policy='omit')
    rcv_Dstar_median, rcv_Dstar_mad = np.nanmedian(rcv_Dstar), median_abs_deviation(rcv_Dstar.flatten(),
                                                                                    nan_policy='omit')

    # 5. Save Results
    metrics = {
        "Parameter": ["D", "f", "Dstar"],
        "Median_RCV": [rcv_D_median, rcv_f_median, rcv_Dstar_median],
        "MAD_RCV": [rcv_D_mad, rcv_f_mad, rcv_Dstar_mad],
    }
    df = pd.DataFrame(metrics)
    if return_raw:
        return df, rcv_D, rcv_f, rcv_Dstar


    return df


def in_vivo_metrics(Dp, Dt, Fp, mask, output_path, multi_sample=True):
    """
    Compute per-sample statistics (median, MAD-based std, CV) for Dp, Dt, Fp maps,
    filtered using a binary mask, and save results to a CSV.
    """

    def calc_metrics(arr):
        if arr.size == 0:
            return None
        med = np.median(arr)
        mad = median_abs_deviation(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        rcv = 1.4826 * mad / med if med != 0 else np.nan
        return [med, mad, mean, std, rcv]

    Metrics = ['Median', 'MAD', 'Mean', 'STD', 'RCV']

    if multi_sample:
        all_dt, all_fp, all_dp = [], [], []

        # Squeeze singleton dims
        Dp = np.squeeze(Dp)
        Dt = np.squeeze(Dt)
        Fp = np.squeeze(Fp)
        mask = np.squeeze(mask)

        for i in range(Dp.shape[0]):
            if np.sum(mask[i]) == 0:
                # print(f"Skipping slice {i}: empty mask.")
                continue

            dp_masked = Dp[i][mask[i] == 1]
            dt_masked = Dt[i][mask[i] == 1]
            fp_masked = Fp[i][mask[i] == 1]


            dp_metrics = calc_metrics(dp_masked) #calcolo metrica per la slice
            dt_metrics = calc_metrics(dt_masked)
            fp_metrics = calc_metrics(fp_masked)

            if dp_metrics and dt_metrics and fp_metrics:
                all_dp.append(dp_metrics)
                all_dt.append(dt_metrics)
                all_fp.append(fp_metrics)

        dpm = np.array(all_dp)
        dtm = np.array(all_dt)
        fm = np.array(all_fp)

        # Compute aggregate stats across valid samples only
        def reduce_stats(metric_array):
            return [
                np.nanmedian(metric_array[:, 0]),
                median_abs_deviation(metric_array[:, 0]),
                np.nanmedian(metric_array[:, -1]),
                median_abs_deviation(metric_array[:, -1])
            ]

        final_dt = reduce_stats(dtm)
        final_f = reduce_stats(fm)
        final_dp = reduce_stats(dpm)

        df = pd.DataFrame({
            'Metric': ['Median', 'MAD', 'Median RCV', 'MAD RCV'],
            'Dt': final_dt,
            'f': final_f,
            'Dstar': final_dp
        })

    else:

        Dp = np.squeeze(Dp)
        Dt = np.squeeze(Dt)
        Fp = np.squeeze(Fp)

        mask = np.squeeze(mask)

        rows = []

        for i in range(Dp.shape[0]):

            if np.sum(mask[i]) == 0:
                print(f"Skipping slice {i}: empty mask.")

                continue

            dp_masked = Dp[i][mask[i] == 1]
            dt_masked = Dt[i][mask[i] == 1]
            fp_masked = Fp[i][mask[i] == 1]
            dp_metrics = calc_metrics(dp_masked)
            dt_metrics = calc_metrics(dt_masked)
            fp_metrics = calc_metrics(fp_masked)

            for metric_name, dt, f, dp in zip(Metrics, dt_metrics, fp_metrics, dp_metrics):
                rows.append({'Slice': i, 'Metric': metric_name, 'Dt': dt, 'f': f, 'Dstar': dp})

        df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

def save_metric_arrays_to_excel(filename, D_vals, f_vals, Dstar_vals):
    """
    Save metric arrays (e.g., bias, abs error, RCV) to an Excel file.

    Args:
        filename (str): Path to save the Excel file.
        D_vals (np.ndarray): Array of D metric values.
        f_vals (np.ndarray): Array of f metric values.
        Dstar_vals (np.ndarray): Array of D* metric values.
    """

    # Flatten arrays if needed (e.g., if shape is (N, H, W) or (N, R) for ROIs)
    D_vals = D_vals.flatten()
    f_vals = f_vals.flatten()
    Dstar_vals = Dstar_vals.flatten()

    # Optional: Remove NaNs if needed
    df = pd.DataFrame({
        'D': D_vals,
        'f': f_vals,
        'Dstar': Dstar_vals
    })

    df.to_excel(filename, index=False)