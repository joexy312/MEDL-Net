import argparse
from argparse import RawTextHelpFormatter
import os


def parse_opts():
    """
    Set the network parameters here.
    """
    parser = argparse.ArgumentParser(description="DeepLearning", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--num_worker', default=0, type=int)
    parser.add_argument('--data_dir', default=os.path.expanduser('~') + '/codes/data/knee/', type=str) # data directory
    parser.add_argument('--acceleration', default=4, type=int)
    parser.add_argument('--center_fraction', default=0.08, type=float)
    parser.add_argument('--sequence', default='coronal_pd', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--seed', default=1000, type=int)
    args = parser.parse_args()

    return args
