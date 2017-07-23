import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--change_label_prob', type=float, default=0.0)
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data')
parser.add_argument('--experiment_type', type=int, default=0)
parser.add_argument('--other_training_label', type=int, default=0)

def parse_args():
    args = parser.parse_args()
    return args
