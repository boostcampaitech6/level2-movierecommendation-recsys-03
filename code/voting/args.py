import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    #ensemble configs
    parser.add_argument("--outputs", default=',', type=str, help='output files that will be ensembled')
    parser.add_argument("--weights", default='0.5,0.5', type=str, help='output files weight')
    
    args = parser.parse_args()
    return args