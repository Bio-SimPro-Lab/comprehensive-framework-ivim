from tools.ivim_tools import IvimModelEstimator
import argparse

def parse_args():
    def exper_arg(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument("-datasets", type=exper_arg,
                        help="Setups to experiment: comma separated List (e.g., 'SNR10.mat','SNR15.mat')")
    parser.add_argument("-run_ids", type=exper_arg,
                        help="Id of the experiment run: comma separated List (e.g., '1','2')")
    parser.add_argument("-model_class", choices=['MLP', 'UNetIVIM'],
                        help="Select model class between 'MLP', UNetIVIM'",
                        default='MLP')
    parser.add_argument("-num_neurons", type=exper_arg,
                        help="Select number neurons",
                        default=32)
    parser.add_argument("-num_mixture", type=exper_arg,
                        help="Select number mixture components",
                        default=1)
    # parser.add_argument("-loss_class", choices=['MSE', 'Rician', 'MAE'],
    #                     help="Select loss class between 'MSE', 'Rician', 'MAE'",
    #                     default='MSE')

    return parser.parse_args()

def main(args):
    debug = True
    if debug:
        datasets = ['simulation_dataset_v2']
        model_class = 'MDN' # MDN or MLP
        num_neurons = [64] #
        mixture_components = [10]
        run_ids = ['1', '2', '3', '4', '5']
        norm_type = 's0' #normalizing by the signal at b=0

    else:
        datasets = args.datasets
        model_class = args.model_class
        num_neurons = args.num_neurons
        mixture_components = args.mixture_components
        run_ids = args.run_ids

    for dataset in datasets:
        for num_neuron in num_neurons:
            for mix_comp in mixture_components:
                for run_id in run_ids:

                    IvimModelEstimator(dataset=dataset,
                                       norm=norm_type,
                                       model_class=model_class,
                                       num_neurons=num_neuron,
                                       mix_components=mix_comp,
                                       run_id=run_id, mode='supervised', test_name="simulation_dataset_v2").train_model()
                    #
                    # IvimModelEstimator(dataset=dataset,
                    #                    norm=norm_type,
                    #                    model_class=model_class,
                    #                    num_neurons=num_neuron,
                    #                    mix_components=mix_comp,
                    #                    run_id=run_id, mode='supervised', test_name="simulation_dataset_v2").test_model()

                run_id='ensemble' # after the five repetitions

                IvimModelEstimator(dataset=dataset,
                                   norm=norm_type,
                                   model_class=model_class,
                                   num_neurons=num_neuron,
                                   mix_components=mix_comp,
                                   run_id=run_id, mode='supervised', test_name="simulation_dataset_v2").test_model(use_ensemble=True, use_gpu=True)

    print('Done!')

if __name__ == "__main__":
    args = parse_args()
    main(args)


















