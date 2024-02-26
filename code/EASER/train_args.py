import argparse

def parse_args():
    parser = argparse.ArgumentParser("arguments for EASER")
    parser.add_argument('--K', default=10, help="decision top-K", type=int)
    parser.add_argument('--output_dir', default='./submission/')
    parser.add_argument('--data_path', default="../../data/train/", type=str)
    
    ###################### EASER ##########################
    parser.add_argument('--threshold', default=3500, type=float)
    parser.add_argument('--lambdaBB', default=500, type=float)
    parser.add_argument('--lambdaCC', default=10000, type=float)
    parser.add_argument('--rho', default=50000, type=float)
    parser.add_argument('--epochs', default=100, type=int)    
    parser.add_argument('--seed', default=99, type=float)
    parser.add_argument('--valid_samples', default=10, type=float)


    
    args = parser.parse_args()

    return args