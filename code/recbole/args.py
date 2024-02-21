import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default='seq', type=str, help='model name in recbole library')
    parser.add_argument("--model_path", default='saved/GRU4Rec-Feb-21-2024_12-52-42.pth', type=str, help='saved model path to predict')
    parser.add_argument("--sequence", default=False, type=str, help='sequence model have to delete history during inference')
    
    args = parser.parse_args()
    return args