import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    #ensemble configs
    parser.add_argument("--outputs", default=',', type=str, help='output files that will be ensembled')
    
    args = parser.parse_args()
    return args