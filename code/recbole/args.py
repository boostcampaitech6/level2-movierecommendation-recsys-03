import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default='recvae', type=str, help='model name in recbole library')
    parser.add_argument("--model_path", default='saved/RecVAE-Feb-06-2024_16-27-17.pth', type=str, help='saved model path to predict')
    
    #ensemble configs
    parser.add_argument("--models", default=',', type=str, help='path of saved models that will be ensembled')
    parser.add_argument("--voting", default='soft', type=str, help='choose soft or hard')
    parser.add_argument("--weights", default='0.5,0.5', type=str, help='weights of models when soft voting')
    
    args = parser.parse_args()
    return args