import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from baselines import train_oxon, train_exp_grad, train_fairret, train_mifpo
from utils import soft_estimator, test_monotonicity, return_tradeoff

import argparse
from tqdm import tqdm
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='ICML 2026 Fair Soft')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='gaussian',
                        help='Dataset name')
    # Params for gaussian dataset
    parser.add_argument('--num_train_samples', type=int, default=10000,
                        help='Number of training samples for gaussian dataset')
    parser.add_argument('--num_test_samples', type=int, default=5000,
                        help='Number of test samples for gaussian dataset')
    parser.add_argument('--input_dim', type=int, default=2,
                        help='Input dimension for gaussian dataset')
    parser.add_argument('--var', type=float, default=2,
                        help='Variance for gaussian dataset')
    # Params for adult dataset
    parser.add_argument('--adult_soft_model', type=str, default='LR',
                        help='Model to generate soft labels for adult dataset (lr/rf)')
    # Params for cifar-10h dataset
    parser.add_argument('--cifar_class_setup', type=str, default='animals',
                        help='Class setup for CIFAR-10H dataset')
    parser.add_argument('--cifar_attr_setup', type=str, default='animals',
                        help='Attribute setup for CIFAR-10H dataset')
    parser.add_argument('--cifar_model_choice', type=str, default='resnet18',
                        help='Model choice for CIFAR-10H embeddings')
    
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--metric', type=str, default='DP',
                        help='Fairness metrics to evaluate (DP, EOP)')
    parser.add_argument('--test_monotonicity', action='store_true',
                        help='Whether to test monotonicity of the estimator')
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(f"Arguments: {args}")

    if args.dataset == 'gaussian':
        from dataset_utils import gaussian_data
        priors = {
            'a_1': 0.7,
            'a_0': 0.3,
            'y_1_a_1': 0.7,
            'y_0_a_1': 0.3,
            'y_1_a_0': 0.4,
            'y_0_a_0': 0.6
        }
        train_X, train_A, train_Y, train_soft, test_X, test_A, test_Y, test_soft = gaussian_data(
            priors=priors,
            dim=args.input_dim,
            var=args.var,
            train_samples=args.num_train_samples,
            val_samples=args.num_test_samples,
            seed=args.seed
        )
    elif args.dataset == 'adult':
        from dataset_utils import load_adult_data
        train_X, train_Y, train_A, train_soft, test_X, test_Y, test_A, test_soft = load_adult_data(
            soft_model=args.adult_soft_model,
            seed=args.seed
        )
    elif args.dataset == 'compas':
        from dataset_utils import load_compas_data
        train_X, train_Y, train_A, train_soft, test_X, test_Y, test_A, test_soft = load_compas_data(
            seed=args.seed
        )
    elif args.dataset == 'cifar10h':
        from dataset_utils import embed_points
        # train_X, train_Y, train_A, train_soft, test_X, test_Y, test_A, test_soft, all_X, all_Y, all_A, all_soft = load_cifar10h(
        #     class_setup=args.cifar_class_setup,
        #     attr_setup=args.cifar_attr_setup,
        #     model_choice=args.cifar_model_choice,
        #     seed=args.seed
        # )
        train_X, train_Y, train_A, train_soft, test_X, test_Y, test_A, test_soft, all_X, all_Y, all_A, all_soft = embed_points(
            model_choice=args.cifar_model_choice,
            metric=args.metric,
            plot_soft=False,
            seed=args.seed
        )
    
    if args.metric == 'DP':
        mifpo_pareto = train_mifpo(train_X, train_Y, train_A, train_soft)
    else:
        mifpo_pareto = None
    exp_unfs, exp_errs, exp_eps = train_exp_grad(train_X, train_Y, train_A, test_X, test_Y, test_A, metric=args.metric)
    oxon_errs, oxon_unfs = train_oxon(train_X, train_Y, train_A, test_X, test_Y, test_A, metric=args.metric)
    strength_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    #strength_values = [0, 0.01] # only two values for testing
    fairret_points = {}
    for val in tqdm(strength_values):
        fairret_points[val] = train_fairret(train_X, train_Y, train_A, test_X, test_Y, test_A, metric=args.metric, fairness_strength=val)
    # Our method
    # A real authentic way would be to split train set further for estimation of pi and then alpha
    estimated_priors = soft_estimator(train_soft, train_A)
    # Used for estimation of alpha
    D_alpha = {
        'C': np.asarray(train_soft),
        'A': train_A
    }
    # Used for estimation of beta
    D_beta = {
        'C': np.asarray(test_soft),
        'A': test_A
    }
    
    if args.test_monotonicity:
        test_monotonicity(estimated_priors, D_alpha, metric=args.metric)
    
    # No point of true vs estimated prior beyond gaussian data
    est_tradeoff, est_bayes, est_unf = return_tradeoff(D_beta, estimated_priors, constraint=args.metric, delta=0.0001, eps=0.0001)

    # Store all results
    results = {
        'mifpo_pareto': mifpo_pareto,
        'exp_unfs': exp_unfs,
        'exp_errs': exp_errs,
        'exp_eps': exp_eps,
        'oxon_errs': oxon_errs,
        'oxon_unfs': oxon_unfs,
        'fairret_points': fairret_points,
        'est_tradeoff': est_tradeoff,
        'est_bayes': est_bayes,
        'est_unf': est_unf
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.metric}_{args.seed}.pickle")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_path}")