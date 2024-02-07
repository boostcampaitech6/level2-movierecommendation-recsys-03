import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default='recvae', type=str, help='model name in recbole library')
    parser.add_argument("--model_path", default='saved/RecVAE-Feb-06-2024_16-27-17.pth', type=str, help='saved model path to predict')
    
    args = parser.parse_args()
    return args