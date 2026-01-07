import numpy as np
import os
from scipy.io import loadmat
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg', force=True)
from scipy.stats import median_abs_deviation
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
from statistics import mean
import pickle
from torchsummary import summary
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import tools.utils_model as um
from tools.utils_model import InferenceMode
import metrics.ivim_metrics as immet
import tools.dataloader as dl
import metrics.calibration_metrics as cal

import re

class IvimModelEstimator:
    def __init__(self, dataset, norm, model_class, num_neurons, mix_components, run_id, mode, test_name):

        self.dataset = dataset
        self.model_class = model_class
        self.num_neurons = num_neurons
        self.mix_components = mix_components
        self.run_id = run_id
        self.mode = mode
        self.test_name = test_name
        self.norm = norm
        self.__load_configs__()

    def __load_configs__(self):
        self.dataset_path = os.path.join(os.getcwd(), 'dataset', self.dataset)
        self.exper_path = os.path.join(os.getcwd(), 'experiments', self.dataset.split('.')[0], self.model_class,
                                      'neurons_'+str(self.num_neurons), 'mixture_'+str(self.mix_components), 'mode_'+self.mode, 'norm_'+self.norm, 'run_' + self.run_id)
        os.makedirs(self.exper_path, exist_ok=True)
        self.config_path = os.path.join(os.getcwd(), 'experiments', self.dataset.split('.')[0], self.model_class,
                                       'neurons_' + str(self.num_neurons), 'mixture_' + str(self.mix_components),
                                       'mode_' + self.mode, 'norm_' + self.norm)

        with open(os.path.join(self.config_path, 'exper_configs.json')) as f:
            self.exper_confs = json.load(f)

        if self.model_class == 'MLP':
            self.Model = um.MLP
        elif self.model_class == 'MDN':
            self.Model = um.IVIMMDN
        else:
            exit('Unknown model!')

        self.bounds = np.array(([0, 0, 0, 3e-3], [1, 3e-3, 0.4, 200e-3])) #bounds of ivim parameters

    def set_seed(self):
        random.seed(42)  # Python random module
        np.random.seed(42)  # NumPy
        torch.manual_seed(42)  # PyTorch (CPU)
        torch.cuda.manual_seed(42)  # PyTorch (current CUDA device)
        torch.cuda.manual_seed_all(42)  # All CUDA devices (if using multi-GPU)

    @staticmethod
    def mdn_loss(alpha, mu, sigma, target, eps=1e-8):
        """
        alpha: [B, K]
        sigma: [B, K]
        mu:    [B, K]
        target: [B]
        """
        target = target.unsqueeze(1).expand_as(mu)  # [B] -> [B, K]
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = m.log_prob(target)  # [B, K]
        log_alpha = torch.log(alpha + eps)  # [B, K]
        log_sum = torch.logsumexp(log_alpha + log_prob, dim=1)  # [B]

        return -log_sum.mean()

    def train_model(self, use_gpu=True):
        print('************************************************************************************')
        print('   Running experiment: ' + self.dataset + ' - run_id: ' + self.run_id)
        print('************************************************************************************')

        lr = self.exper_confs['learning_config']['lr']
        batch_size = self.exper_confs['learning_config']['batch_size']
        max_epochs = self.exper_confs['learning_config']['max_epochs']
        patience = self.exper_confs['learning_config']['patience']
        early_stop_on_train_loss = self.exper_confs['learning_config']['early_stop_on_train_loss']

        if self.model_class == 'MLP':
            criterion = nn.MSELoss()
        elif self.model_class == 'MDN':
            criterion = self.mdn_loss
        else:
            exit('Unknown loss!')

        #self.set_seed()

        # Setup device
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = 'cpu'
        print(device)

        # Import data
        X_train, b_values, D_true, f_true, Dstar_true = dl.ivim_data_load_voxelwise(os.path.join(self.dataset_path, 'training.mat'))
        X_val, b_values, D_true_val, f_true_val, Dstar_true_val = dl.ivim_data_load_voxelwise(os.path.join(self.dataset_path, 'validation.mat'))
        b_values_no0 = torch.FloatTensor(b_values)

        # Load Model
        model = self.Model(b_values_no0.to(device), num_neurons=self.num_neurons, mix_components=self.mix_components, mode=self.mode)
        input_shape = (len(b_values_no0),)
        # Generate the model summary
        summary(model.to(device), input_shape)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Print statistics
        print('train shape is: ' +str(X_train.shape))
        print('val shape is: ' +str(X_val.shape))
        print('max_'+str(X_train.max()))
        print('mean'+str(X_train.mean()))
        print('std' +str(X_train.std()))
        print('min'+str(X_train.min()))

        # Scale parameters
        D_train_scaled, f_train_scaled, Dstar_train_scaled = dl.scale_params(D_true, f_true, Dstar_true, self.bounds)
        D_val_scaled, f_val_scaled, Dstar_val_scaled = dl.scale_params(D_true_val, f_true_val, Dstar_true_val, self.bounds)

        # Convert to tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        D_train_tensor = torch.from_numpy(D_train_scaled).float()
        f_train_tensor = torch.from_numpy(f_train_scaled).float()
        Dstar_train_tensor = torch.from_numpy(Dstar_train_scaled).float()

        X_val_tensor = torch.from_numpy(X_val).float()
        D_val_tensor = torch.from_numpy(D_val_scaled).float()
        f_val_tensor = torch.from_numpy(f_val_scaled).float()
        Dstar_val_tensor = torch.from_numpy(Dstar_val_scaled).float()

        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor.to(device), D_train_tensor.to(device), f_train_tensor.to(device), Dstar_train_tensor.to(device))
        val_dataset = TensorDataset(X_val_tensor.to(device), D_val_tensor.to(device), f_val_tensor.to(device), Dstar_val_tensor.to(device))

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        signal_criterion = nn.MSELoss()
        # Train
        history = {'loss_tr': [],
                   'loss_val': [],
                   'signal_MSE_train': [],
                   'signal_MSE_val': []}

        best_loss = 1e16
        num_bad_epochs = 0

        for epoch in range(max_epochs):
            print("-----------------------------------------------------------------")
            print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
            model.train()
            running_loss = 0.
            loss_tr_ep = []
            signal_mse_train = []
            for i, (X_batch, D_true_batch, f_true_batch, Dp_true_batch) in enumerate(tqdm(trainloader), 0):

                optimizer.zero_grad()

                if self.model_class == 'MDN':

                    output = model(X_batch)

                    # Compute loss for each parameter
                    loss_d = criterion(*output['dt'], D_true_batch)
                    loss_f = criterion(*output['fp'], f_true_batch)
                    loss_dstar = criterion(*output['dp'], Dp_true_batch)

                    # Compute overall loss as the mean of the three
                    loss = (loss_d + loss_f + loss_dstar)/3

                    #mu_dt, mu_fp, mu_dp = model.inference(X_batch, mode=InferenceMode.MODE)
                    #Dt_pred, Fp_pred, Dp_pred = dl.descaling_params(mu_dt, mu_fp, mu_dp, self.bounds)

                else: #MLP training
                    Dt_pred, Fp_pred, Dp_pred = model(X_batch)

                    # rescaling to 0-1 values
                    # Compute MSE loss for each parameter
                    loss_d = criterion(Dt_pred, D_true_batch)
                    loss_f = criterion(Fp_pred, f_true_batch)
                    loss_dstar = criterion(Dp_pred, Dp_true_batch)

                    # Compute overall loss as the mean of the three
                    loss = (loss_d + loss_f + loss_dstar)/3

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loss_tr_ep.append(loss.item())

            print("Loss: {}".format(mean(loss_tr_ep)))

            if early_stop_on_train_loss:
                # early stop is performed on the running loss
                history['loss_tr'].append(running_loss)
                history['loss_val'].append(running_loss)
                es_loss = running_loss

            else:
                # Evaluation phase (validation set)
                val_loss = 0.0
                num_batches = len(valloader)  # The number of batches in the validation set
                model.eval()
                with torch.no_grad():  # Disable gradient computation
                    for i, (X_batch, D_true_batch, f_true_batch, Dp_true_batch) in enumerate(tqdm(valloader), 0):

                        if self.model_class == 'MDN':

                            output = model(X_batch)
                            # rescaling to 0-1 values
                            # Compute MSE loss for each parameter
                            loss_d = criterion(*output['dt'], D_true_batch)
                            loss_f = criterion(*output['fp'], f_true_batch)
                            loss_dstar = criterion(*output['dp'], Dp_true_batch)

                            # Compute overall loss as the mean of the three
                            loss = (loss_d + loss_f + loss_dstar)/3
                            print(
                                f"D loss: {loss_d.item():.4f}, F loss: {loss_f.item():.4f}, D* loss: {loss_dstar.item():.4f}")


                        else:
                            Dt_pred, Fp_pred, Dp_pred = model(X_batch)

                            # Compute MSE loss for each parameter
                            loss_d = criterion(Dt_pred, D_true_batch)
                            loss_f = criterion(Fp_pred, f_true_batch)
                            loss_dstar = criterion(Dp_pred, Dp_true_batch)

                            # Compute overall loss as the mean of the three
                            loss = (loss_d + loss_f + loss_dstar)/3
                            print(
                                f"D loss: {loss_d.item():.4f}, F loss: {loss_f.item():.4f}, D* loss: {loss_dstar.item():.4f}")

                        val_loss += loss.item()  # Sum the loss across all batches


                # Compute the mean validation loss
                mean_val_loss = val_loss / num_batches
                print("Mean Validation Loss: {}".format(mean_val_loss))

                # Store the mean validation loss to track performance
                history['loss_tr'].append(mean(loss_tr_ep))
                history['loss_val'].append(mean_val_loss)

                es_loss = mean_val_loss

            # early stopping
            if es_loss < best_loss:
                print("############### Saving good model ###############################")
                final_model = model.state_dict()
                best_loss = es_loss
                num_bad_epochs = 0
                model_state_file = 'model_state'
                torch.save(final_model, os.path.join(self.exper_path, model_state_file))
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == patience:
                    print("Done, best loss: {}".format(best_loss))
                    break

            print(f"Epoch {epoch}")

        # save model
        model_state_file = 'model_state'
        torch.save(final_model, os.path.join(self.exper_path, model_state_file))
        # save history
        with open(os.path.join(self.exper_path, 'train_history.p'), 'wb') as f:
            pickle.dump(history, f)

    def test_model(self, use_ensemble=True, use_gpu=True):
        """
        Run model testing and evaluation (simulation or in-vivo data).
        Handles ensemble inference, uncertainty descaling, metric computation,
        and result visualization.
        """

        # -------------------------------------------------------------------------
        # Device setup
        # -------------------------------------------------------------------------
        device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
        print(f"Using device: {device}")

        # -------------------------------------------------------------------------
        # Select test data files
        # -------------------------------------------------------------------------
        if self.test_name in ['simulation_dataset_brain', 'simulation_dataset_v2']:
            test_files = ['test_data_uniform_GM_25.mat', 'test_data_uniform_GM_50.mat', 'test_data_uniform_GM_100.mat']  # Could add multiple test files
        else:
            test_files = ['test_data.mat']

        # -------------------------------------------------------------------------
        # Iterate over test datasets
        # -------------------------------------------------------------------------

        for name in test_files:
            print(f"\n--- Testing on dataset: {name} ---")

            # ---------------------------------------------------------------------
            # Load test data
            # ---------------------------------------------------------------------
            if self.test_name in ['simulation_dataset_v2', 'simulation_dataset_brain']:
                X_test, b_values_no0, D_true, f_true, Dstar_true = dl.ivim_data_load_cnn(
                    os.path.join(self.dataset_path, name), norm_type='s0'
                )
                shape_or = D_true.shape
                b_values_no0 = torch.tensor(b_values_no0)

            else:
                invivo_path = re.sub(
                    r'(?:simulations_dataset_muscle|simulation_dataset_brain|simulation_dataset_v2|simulation_dataset_v2_muscle)',
                    self.test_name,
                    self.dataset_path
                )
                X_test, mask = dl.ivim_data_load_vivo(
                    os.path.join(invivo_path, name), norm_type='s0'
                )
                shape_or = (X_test.shape[0], X_test.shape[2], X_test.shape[3])
                b_values_no0 = torch.arange(X_test.shape[1])  # placeholder if not provided

            # ---------------------------------------------------------------------
            # Preprocess test data
            # ---------------------------------------------------------------------
            X_test = torch.from_numpy(X_test.astype(np.float32))
            X_test = X_test.permute(0, 2, 3, 1).reshape(-1, len(b_values_no0))

            # if self.test_name in ['simulations_dataset_v2', 'simulations_dataset_muscle_v2']:
            #     mask = (D_true.reshape(-1) != 0)
            #     X_test = X_test[mask]

            # ---------------------------------------------------------------------
            # Ensemble inference
            # ---------------------------------------------------------------------
            with torch.no_grad():
                results = self.ensemble_inference(X_test, device, b_values_no0)
                (
                    Dt, Fp, Dp,
                    alea_dt, alea_fp, alea_dp,
                    epi_dt, epi_fp, epi_dp,
                    uq_dt, uq_fp, uq_dp
                ) = results

            # ---------------------------------------------------------------------
            # Descale parameters and uncertainties
            # ---------------------------------------------------------------------
            Dt, Fp, Dp = dl.descaling_params(Dt, Fp, Dp, self.bounds)
            epi_dt, epi_fp, epi_dp = dl.descale_sigma(epi_dt.cpu(), epi_fp.cpu(), epi_dp.cpu(), self.bounds)

            if alea_dt is not None:
                alea_dt, alea_fp, alea_dp = dl.descale_sigma(alea_dt.cpu(), alea_fp.cpu(), alea_dp.cpu(), self.bounds)

            Dp = Dp.cpu().numpy()
            Dt = Dt.cpu().numpy()
            Fp = Fp.cpu().numpy()

            # ---------------------------------------------------------------------
            # Calibration metrics (only on raw, unreshaped MDN samples)
            # ---------------------------------------------------------------------
            if self.model_class == 'MDN' and self.test_name in ['simulation_dataset_v2',
                                                                'simulations_dataset_brain'] and name=='test_data.mat':
                # Eliminate background before calibration — without reshaping
                mask = (D_true.reshape(-1) != 0)
                uq_dt, uq_fp, uq_dp = dl.descaling_params(uq_dt[:, mask], uq_fp[:, mask], uq_dp[:, mask], self.bounds)

                self._evaluate_calibration(
                    D_true, f_true, Dstar_true,
                    name, uq_dt, uq_fp, uq_dp
                )

            # Reshape to image volume
            Dt, Fp, Dp = [x.reshape(shape_or) for x in (Dt, Fp, Dp)]

            # Properly reshape uncertainty tensors if they exist
            if alea_dt is not None:
                alea_dt = alea_dt.reshape(shape_or)
                alea_fp = alea_fp.reshape(shape_or)
                alea_dp = alea_dp.reshape(shape_or)

            if epi_dt is not None:
                epi_dt = epi_dt.reshape(shape_or)
                epi_fp = epi_fp.reshape(shape_or)
                epi_dp = epi_dp.reshape(shape_or)

            # ---------------------------------------------------------------------
            # Swap Dp and Dt if needed
            # ---------------------------------------------------------------------
            if np.mean(Dp) < np.mean(Dt):
                Dp, Dt = Dt, Dp
                Fp = 1 - Fp

            # ---------------------------------------------------------------------
            # Run evaluation (simulation vs. in-vivo)
            # ---------------------------------------------------------------------
            if self.test_name in ['simulation_dataset_v2', 'simulation_dataset_brain']: #testing simulations or in vivo data
                self._evaluate_simulation(
                    Dt, Fp, Dp,
                    D_true, f_true, Dstar_true,
                    name, alea_dt, alea_fp, alea_dp,
                    epi_dt, epi_fp, epi_dp,
                    use_ensemble
                )
            else:
                self._evaluate_invivo(
                    Dt, Fp, Dp,
                    alea_dt, alea_fp, alea_dp,
                    epi_dt, epi_fp, epi_dp,
                    use_ensemble
                )

    # -----------------------------------------------------------------------------
    # Helper: Simulation dataset evaluation
    # -----------------------------------------------------------------------------
    def _normalize_and_save_uncertainties(self, alea_dt, alea_fp, alea_dp,
                                          epi_dt, epi_fp, epi_dp,
                                          mask=None, output_path=None):
        """
        Normalize and summarize uncertainty maps (aleatoric + epistemic).
        Optionally applies a mask (e.g. in-vivo ROI) and saves results to CSV.
        """

        def normalize(array, low, high):
            return array / (high - low)


        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            alea_dt = alea_dt[mask_tensor]
            alea_fp = alea_fp[mask_tensor]
            alea_dp = alea_dp[mask_tensor]
            epi_dt = epi_dt[mask_tensor]
            epi_fp = epi_fp[mask_tensor]
            epi_dp = epi_dp[mask_tensor]

        # Extract & normalize
        if self.test_name=='simulation_dataset_v2':

            alea_dt_n = normalize(alea_dt.cpu().numpy(), self.bounds[0, 1], self.bounds[1, 1])
            alea_fp_n = normalize(alea_fp.cpu().numpy(), self.bounds[0, 2], self.bounds[1, 2])
            alea_dp_n = normalize(alea_dp.cpu().numpy(), self.bounds[0, 3], self.bounds[1, 3])
            epi_dt_n = normalize(epi_dt.cpu().numpy(), self.bounds[0, 1], self.bounds[1, 1])
            epi_fp_n = normalize(epi_fp.cpu().numpy(), self.bounds[0, 2], self.bounds[1, 2])
            epi_dp_n = normalize(epi_dp.cpu().numpy(), self.bounds[0, 3], self.bounds[1, 3])

        else:
            alea_dt_n = normalize(alea_dt.cpu().numpy(), 0.84e-3-1.96*0.05e-3, 0.84e-3+1.96*0.05e-3)
            alea_fp_n = normalize(alea_fp.cpu().numpy(),  0.14-1.96*0.02, 0.14+1.96*0.02)
            alea_dp_n = normalize(alea_dp.cpu().numpy(), 8.2e-3-1.96*0.9e-3, 8.2e-3+1.96*0.9e-3)
            epi_dt_n = normalize(epi_dt.cpu().numpy(), 0.84e-3-1.96*0.05e-3, 0.84e-3+1.96*0.05e-3)
            epi_fp_n = normalize(epi_fp.cpu().numpy(), 0.14-1.96*0.02, 0.14+1.96*0.02)
            epi_dp_n = normalize(epi_dp.cpu().numpy(),  8.2e-3-1.96*0.9e-3, 8.2e-3+1.96*0.9e-3)

        # 🔹 Reshape to (n_sim, -1) if not already
        n_sim = 198
        alea_dt_n = alea_dt_n.reshape(n_sim, -1)
        alea_fp_n = alea_fp_n.reshape(n_sim, -1)
        alea_dp_n = alea_dp_n.reshape(n_sim, -1)
        epi_dt_n = epi_dt_n.reshape(n_sim, -1)
        epi_fp_n = epi_fp_n.reshape(n_sim, -1)
        epi_dp_n = epi_dp_n.reshape(n_sim, -1)

        # 🔹 Compute per-simulation median and MAD
        data = {
            'alea_dt_median': np.median(alea_dt_n, axis=1),
            'alea_dt_mad': median_abs_deviation(alea_dt_n, axis=1),
            'alea_fp_median': np.median(alea_fp_n, axis=1),
            'alea_fp_mad': median_abs_deviation(alea_fp_n, axis=1),
            'alea_dp_median': np.median(alea_dp_n, axis=1),
            'alea_dp_mad': median_abs_deviation(alea_dp_n, axis=1),
            'epi_dt_median': np.median(epi_dt_n, axis=1),
            'epi_dt_mad': median_abs_deviation(epi_dt_n, axis=1),
            'epi_fp_median': np.median(epi_fp_n, axis=1),
            'epi_fp_mad': median_abs_deviation(epi_fp_n, axis=1),
            'epi_dp_median': np.median(epi_dp_n, axis=1),
            'epi_dp_mad': median_abs_deviation(epi_dp_n, axis=1),
        }

        df = pd.DataFrame(data)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Uncertainty summary saved to {output_path}")

        return df

    def _evaluate_calibration(self, D_true, f_true, Dstar_true,
                              name,
                              uq_dt, uq_fp, uq_dp):
        """
        Evaluate calibration metrics using MDN samples.
        Runs only for simulation datasets (test_data.mat) and removes background pixels.
        """
        if name == 'test_data.mat':
            ground_truth_dict = {'D': D_true, 'f': f_true, 'Dp': Dstar_true}
            samples_dict = {'D': uq_dt.astype('float32'),
                            'f': uq_fp.astype('float32'),
                            'Dp': uq_dp.astype('float32')}
            cal.compute_and_save_metrics(samples_dict, ground_truth_dict, name, output_dir=self.exper_path)


    def _evaluate_simulation(self, Dt, Fp, Dp, D_true, f_true, Dstar_true,
                             name, alea_dt, alea_fp, alea_dp, epi_dt, epi_fp, epi_dp,
                             use_ensemble):
        base_path = os.path.join(self.exper_path, 'simulations')
        os.makedirs(base_path, exist_ok=True)

        # --- Metrics ---
        Bias = immet.metric_bias(Dt, D_true, Fp, f_true, Dp, Dstar_true)
        AbsErr = immet.metric_absolute_error(Dt, D_true, Fp, f_true, Dp, Dstar_true)
        CV = immet.RCV(Dt, Fp, Dp)

        Bias.to_csv(os.path.join(base_path, f"{name[:-4]}_bias.csv"), index=False)
        AbsErr.to_csv(os.path.join(base_path, f"{name[:-4]}_absolute_error.csv"), index=False)
        CV.to_csv(os.path.join(base_path, f"{name[:-4]}_rcv.csv"), index=False)

        print("Test set Bias and Absolute Error results saved.")

        # --- Plots ---
        plots_dir = os.path.join(base_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for idx in [20, 50, 90]:
            save_path = os.path.join(plots_dir, f"parameter_maps_{name[:-4]}_idx_{idx}.png")
            if use_ensemble:
                self.plot_all_maps(D_true, Dt, Dstar_true, Dp, f_true, Fp,
                                   alea_dt, alea_dp, alea_fp, epi_dt, epi_dp, epi_fp,
                                   random_idx=idx, save_path=save_path)
            else:
                self.plot_all_maps(D_true, Dt, Dstar_true, Dp, f_true, Fp,
                                   random_idx=idx, save_path=save_path)
        mask_uq = ((D_true) != 0)

        if use_ensemble:
            self._normalize_and_save_uncertainties(
                alea_dt, alea_fp, alea_dp,
                epi_dt, epi_fp, epi_dp, mask = mask_uq,
                output_path=os.path.join(base_path, f"{name[:-4]}_uncertainty_values_normalized.csv")
            )

        print("Simulation maps plotted and saved.")

        # -----------------------------------------------------------------------------
        # Helper: In-vivo dataset evaluation
        # -----------------------------------------------------------------------------

    def ensemble_inference(self, X_test, device, b_values_no0):
        """
        Perform ensemble inference across multiple trained models.
        Supports MDN and MLP modes. Returns predictive means, aleatoric
        and epistemic uncertainties, and sample distributions.
        """

        # -------------------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------------------
        model_files = [f"run_{i + 1}/model_state" for i in range(5)]
        parent_path = os.path.dirname(self.exper_path)

        # Containers for predictions and uncertainties
        all_mu_dt, all_mu_fp, all_mu_dp = [], [], []
        all_std_dt, all_std_fp, all_std_dp = [], [], []
        sample_dt_list, sample_fp_list, sample_dp_list = [], [], []

        val_likelihood = []

        # -------------------------------------------------------------------------
        # Helper: Load a model and set to eval mode
        # -------------------------------------------------------------------------
        def load_model(file_path):
            model = self.Model(b_values_no0, self.num_neurons, self.mix_components, mode=self.mode)
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)
            model.eval()
            return model

        # -------------------------------------------------------------------------
        # Ensemble inference loop
        # -------------------------------------------------------------------------
        for file in model_files:
            print(f"Inference using {file}")
            model = load_model(os.path.join(parent_path, file))
            model_dir = file.split('/')[0]

            with torch.no_grad():
                if self.model_class == 'MDN':
                    # ------------------ MDN MODE ------------------
                    mean = model.inference(X_test.to(device), mode=InferenceMode.UNCERTAINTY)

                    # Unpack means and standard deviations
                    mu_dt_i, std_dt_i = mean[0]
                    mu_fp_i, std_fp_i = mean[1]
                    mu_dp_i, std_dp_i = mean[2]

                    # Store results
                    all_mu_dt.append(mu_dt_i)
                    all_mu_fp.append(mu_fp_i)
                    all_mu_dp.append(mu_dp_i)

                    all_std_dt.append(std_dt_i)
                    all_std_fp.append(std_fp_i)
                    all_std_dp.append(std_dp_i)

                    # Generate samples for uncertainty metrics
                    model_cpu = model.to('cpu')
                    sample_dt, sample_fp, sample_dp = model_cpu.inference(
                        X_test.to('cpu'), mode=InferenceMode.SAMPLE, n_samples=100
                    )

                    sample_dt_list.append(sample_dt.numpy())
                    sample_fp_list.append(sample_fp.numpy())
                    sample_dp_list.append(sample_dp.numpy())

                else:
                    # ------------------ MLP MODE ------------------
                    mu_dt_i, mu_fp_i, mu_dp_i = model(X_test.to(device))
                    all_mu_dt.append(mu_dt_i)
                    all_mu_fp.append(mu_fp_i)
                    all_mu_dp.append(mu_dp_i)

            # ---------------------------------------------------------------------
            # Record validation likelihood from training history
            # ---------------------------------------------------------------------
            with open(os.path.join(parent_path, model_dir, 'train_history.p'), 'rb') as fp:
                history = pickle.load(fp)
                val_likelihood.append(np.min(history['loss_val']))

        # -------------------------------------------------------------------------
        # Compute ensemble statistics
        # -------------------------------------------------------------------------
        mu_dt = torch.stack(all_mu_dt).mean(dim=0)
        mu_fp = torch.stack(all_mu_fp).mean(dim=0)
        mu_dp = torch.stack(all_mu_dp).mean(dim=0)

        epistemic_dt = torch.stack(all_mu_dt).std(dim=0)
        epistemic_fp = torch.stack(all_mu_fp).std(dim=0)
        epistemic_dp = torch.stack(all_mu_dp).std(dim=0)

        if self.model_class == 'MDN':
            # Aleatoric uncertainty (mean predicted variance)
            var_dt = torch.stack(all_std_dt)
            var_fp = torch.stack(all_std_fp)
            var_dp = torch.stack(all_std_dp)

            alea_std_dt = torch.sqrt((var_dt ** 2).mean(dim=0).clamp(min=0))
            alea_std_fp = torch.sqrt((var_fp ** 2).mean(dim=0).clamp(min=0))
            alea_std_dp = torch.sqrt((var_dp ** 2).mean(dim=0).clamp(min=0))

            # Combine samples from all ensemble models
            sample_dt_list = np.concatenate(sample_dt_list, axis=1)
            sample_fp_list = np.concatenate(sample_fp_list, axis=1)
            sample_dp_list = np.concatenate(sample_dp_list, axis=1)
        else:
            alea_std_dt = alea_std_fp = alea_std_dp = None

        # -------------------------------------------------------------------------
        # Save validation likelihood summary
        # -------------------------------------------------------------------------
        df = pd.DataFrame(
            [val_likelihood + [np.mean(val_likelihood)]],
            columns=[f"val_likelihood_{i + 1}" for i in range(5)] + ["mean"]
        )
        df.to_csv(os.path.join(parent_path, 'val_likelihoods.csv'), index=False)

        # -------------------------------------------------------------------------
        # Return results
        # -------------------------------------------------------------------------
        return (
            mu_dt, mu_fp, mu_dp,
            alea_std_dt, alea_std_fp, alea_std_dp,
            epistemic_dt, epistemic_fp, epistemic_dp,
            sample_dt_list, sample_fp_list, sample_dp_list
        )


    def plot_train_history(self):
        with open(os.path.join(self.exper_path, 'train_history.p'), 'rb') as fp:
            history = pickle.load(fp)
        plt.plot(history['loss_tr'], label='loss_tr')
        plt.plot(history['loss_val'], label='loss_val')
        plt.ylim(-4,-2)
        plt.legend()
        plt.grid()
        plt.savefig((os.path.join(self.exper_path, 'loss.png')))
        plt.close('all')


    def plot_all_maps(self, D_true, Dt, Dstar_true, Dp, f_true, Fp,
                              alea_dt=None, alea_dstar=None, alea_fp=None, epi_dt=None, epi_dp=None, epi_fp=None,
                              random_idx=0, save_path="output.png"):
        """Plots true and predicted (and optionally uncertainty) parameter maps."""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        has_uncertainty = alea_dt is not None and alea_dstar is not None and alea_fp is not None


        # Prepare parameter maps
        if self.test_name=='simulation_dataset_brain':

          d_low, d_up, f_low, f_up, ds_low, ds_up = 0.84e-3-1.96*0.05e-3, 0.84e-3+1.96*0.05e-3, 0.14-1.96*0.02, 0.14+1.96*0.02, 8.2e-3-1.96*0.9e-3, 8.2e-3+1.96*0.9e-3


        else:

          d_low, d_up, f_low, f_up, ds_low, ds_up = self.bounds[0, 1], self.bounds[1, 1], self.bounds[0, 2], self.bounds[1, 2],self.bounds[0, 3], self.bounds[1, 3]

        param_maps = [
            (D_true[random_idx], Dt[random_idx],  d_low, d_up, alea_dt[random_idx] if has_uncertainty else None, epi_dt[random_idx] if has_uncertainty else None, "D","mm²/s",0.0015),
            (f_true[random_idx]*100, Fp[random_idx]*100, f_low, f_up, alea_fp[random_idx] if has_uncertainty else None, epi_fp[random_idx] if has_uncertainty else None, "f",'%',20),
            (Dstar_true[random_idx], Dp[random_idx], ds_low, ds_up, alea_dstar[random_idx] if has_uncertainty else None, epi_dp[random_idx] if has_uncertainty else None, "D*","mm²/s",0.03)
            ,
        ]

        ncols = 4 if has_uncertainty else 2
        fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(5 * ncols, 12))

        for i, (true_map, pred_map, min_bound, max_bound, alea_map, epi_map, title, unit, vmax) in enumerate(param_maps):
            mask = (1 - (true_map == 0)).astype(int)
            vmin, vmax = min(true_map.min(), pred_map.min()), max(true_map.max(), pred_map.max())

            # True map
            img = axes[i, 0].imshow(true_map*mask, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"True {title}", fontsize=20)
            axes[i, 0].axis("off")
            cbar = fig.colorbar(img, ax=axes[i, 0])
            cbar.set_label(unit, fontsize=21)
            cbar.ax.tick_params(labelsize=18)
            # Predicted map
            img = axes[i, 1].imshow(pred_map*mask, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"Predicted {title}", fontsize=20)
            axes[i, 1].axis("off")
            cbar = fig.colorbar(img, ax=axes[i, 1])
            cbar.set_label(unit, fontsize=21)
            cbar.ax.tick_params(labelsize=18)
            # Uncertainty map if available
            if has_uncertainty:
                norm_alea_map = (alea_map*100)/(max_bound-min_bound)
                img = axes[i, 2].imshow(norm_alea_map * mask, cmap='magma', vmin=vmin, vmax=50)
                axes[i, 2].set_title(f"AU {title}", fontsize=20)
                axes[i, 2].axis("off")
                cbar = fig.colorbar(img, ax=axes[i, 2])
                cbar.set_label("%", fontsize=21)
                cbar.ax.tick_params(labelsize=18)

                norm_epi_map = (epi_map*100) / (max_bound - min_bound)
                img = axes[i, 3].imshow(norm_epi_map * mask, cmap='magma', vmin=vmin, vmax=30)
                axes[i, 3].set_title(f"EU {title}", fontsize=20)
                axes[i, 3].axis("off")
                cbar = fig.colorbar(img, ax=axes[i, 3])
                cbar.set_label("%", fontsize=20)
                cbar.ax.tick_params(labelsize=15)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for title
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

