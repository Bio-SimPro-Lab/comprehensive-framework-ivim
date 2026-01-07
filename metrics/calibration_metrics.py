import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from properscoring import crps_ensemble

def batch_crps(obs: np.ndarray, forecast: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    """
    Compute CRPS in batches for memory efficiency.
    """
    n = obs.shape[0]
    crps_list = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_obs = obs[i:end]
        batch_forecast = forecast[i:end]
        batch_crps = crps_ensemble(batch_obs, batch_forecast)
        crps_list.append(batch_crps.astype('float32'))
        #print('iteration done!')
    return np.concatenate(crps_list)


def compute_and_save_metrics(
    samples_dict: dict,
    ground_truth_dict: dict,
    name: str,
    interval_levels: list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
    output_dir: str = 'metrics_output'
) -> None:
    """
    Compute CRPS, PICP, MPIW for ensemble forecasts of multiple parameters and save results to CSV and plots.
    """
    output_dir = os.path.join(output_dir, 'simulations', 'calibration')
    os.makedirs(output_dir, exist_ok=True)

    interval_levels = np.array(interval_levels)
    alpha_levels = 1 - interval_levels
    lower_q = alpha_levels / 2
    upper_q = 1 - lower_q

    for param in samples_dict:
        print(f'Now calculating calibration metrics for {param}')

        samples = samples_dict[param]  # shape: (n_samples, n_ensembles)
        ground_truth = ground_truth_dict[param].reshape(-1)  # shape: (n_samples,)

        mask = ground_truth != 0
        ground_truth = ground_truth[mask]
        #samples = samples[mask]  # IMPORTANT: filter samples as well!

        metrics = {}

        # CRPS computation in batches
        print(" - Computing CRPS in batches...")
        crps_scores = batch_crps(ground_truth, samples, batch_size=10000)
        metrics['CRPS_Mean'] = float(np.mean(crps_scores))
        metrics['CRPS_Std'] = float(np.std(crps_scores))
        metrics['CRPS_Median'] = float(np.median(crps_scores))
        metrics['CRPS_MAD'] = float(np.median(np.abs(crps_scores - np.median(crps_scores))))

        del crps_scores

        y_range = np.ptp(ground_truth)  # max - min
        if y_range == 0:
            y_range = 1e-8

        # Vectorized quantile calculation for all interval levels at once
        lower_bounds = np.quantile(samples, lower_q, axis=1).T  # shape: (n_samples, n_levels)
        upper_bounds = np.quantile(samples, upper_q, axis=1).T

        within_interval = (ground_truth[:, None] >= lower_bounds) & (ground_truth[:, None] <= upper_bounds)
        picp = np.mean(within_interval, axis=0)  # shape: (n_levels,)
        mpiw = np.mean(upper_bounds - lower_bounds, axis=0)
        pinaw = mpiw / y_range

        temp_pinaw = (upper_bounds - lower_bounds)/y_range
        metrics['PINAW_90_MEDIAN'] = np.median(temp_pinaw[:,-3])
        metrics['PINAW_90_MAD'] = np.median(abs(temp_pinaw[:,-3]-np.median(temp_pinaw[:,-3])))

        for i, level in enumerate(interval_levels):
            metrics[f'PICP_{int(level * 100)}'] = picp[i]
            metrics[f'MPIW_{int(level * 100)}'] = mpiw[i]
            metrics[f'PINAW_{int(level * 100)}'] = pinaw[i]

        # Miscalibration Area
        miscalibration_area = np.trapz(np.abs(picp - interval_levels), interval_levels)
        metrics['Miscalibration_Area'] = miscalibration_area

        # Save metrics CSV
        df_metrics = pd.DataFrame([metrics])
        output_file = os.path.join(output_dir, f'_SNR_{name[:-4]}_{param}_metrics.csv')
        df_metrics.to_csv(output_file, index=False)
        print(f"Metrics for parameter '{param}' saved to '{output_file}'")

        # Plot calibration curve
        plt.figure(figsize=(6, 6))
        plt.plot(interval_levels, picp, marker='o', label='Observed')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Predicted Coverage (α)')
        plt.ylabel('Observed Coverage')
        if param=='Dp':
            param_title='D*'
        else:
            param_title=param

        plt.title(f'MDN Calibration Plot for {param_title}', fontsize=20)
        plt.grid(True)
        plt.legend()
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.text(0.05, 0.9, f'Miscalibration Area: {miscalibration_area*100:.2f}%', fontsize=15,
                 bbox=dict(facecolor='white', alpha=0.8))

        plot_file = os.path.join(output_dir, f'_SNR_{name[:-4]}_{param}_calibration_plot.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Calibration plot for parameter '{param}' saved to '{plot_file}'")
