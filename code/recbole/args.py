import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default='seq', type=str, help='model name in recbole library')
    parser.add_argument("--model_path", default='saved/GRU4Rec-Feb-21-2024_12-52-42.pth', type=str, help='saved model path to predict')
    parser.add_argument("--sequence", default=False, type=str, help='sequence model have to delete history during inference')
    
    #ensemble configs
    parser.add_argument("--models", default=',', type=str, help='path of saved models that will be ensembled')
    parser.add_argument("--voting", default='soft', type=str, help='choose soft or hard')
    parser.add_argument("--weights", default='0.5,0.5', type=str, help='weights of models when soft voting')
    
    args = parser.parse_args()
    return args