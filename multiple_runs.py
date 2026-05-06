# Script to call main.py with multiple random seeds
import subprocess
import argparse

# Basic argparse for experiment name
parser = argparse.ArgumentParser(description='Run multiple experiments with different seeds')
parser.add_argument('--dataset', type=str, default='gaussian',
                    help='Dataset name')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory to save outputs (and will also act as experiment name)')

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
parser.add_argument('--test_monotonicity', action='store_true',
                    help='Whether to test monotonicity of the estimator')

seeds = [42, 100, 2021, 7, 1234]
metrics = ['DP', 'EOP']

args = parser.parse_args()
for seed in seeds:
    for metric in metrics:
        args.metric = metric
        cmd = [
            'python', 'main.py',
            '--dataset', args.dataset,
            '--output_dir', args.output_dir,
            '--num_train_samples', str(args.num_train_samples),
            '--num_test_samples', str(args.num_test_samples),
            '--input_dim', str(args.input_dim),
            '--var', str(args.var),
            '--adult_soft_model', args.adult_soft_model,
            '--cifar_class_setup', args.cifar_class_setup,
            '--cifar_attr_setup', args.cifar_attr_setup,
            '--cifar_model_choice', args.cifar_model_choice,
            '--metric', args.metric,
            '--seed', str(seed)
        ]
        if args.test_monotonicity:
            cmd.append('--test_monotonicity')
    
        print(f"Running command: {' '.join(cmd)}")
        # Run command and pipe output to console
        subprocess.run(cmd)