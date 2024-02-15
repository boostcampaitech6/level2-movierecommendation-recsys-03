import torch
import numpy as np
from trainer import naive_sparse2tensor


def predict(args, model, data_tr):
    model.eval()
    
    top_items = []
    
    with torch.no_grad():
        data_tensor = naive_sparse2tensor(data_tr).to('cuda')
        predicts = model(data_tensor)

        if args.model == 'Multi-VAE':
            predicts = predicts[0].cpu().numpy()    
        elif args.model == 'Multi-DAE':
            predicts = predicts.cpu().numpy()
            
        predicts[data_tr.nonzero()] = -np.inf
        
        top_scores, top_ids = torch.topk(torch.from_numpy(predicts).float().to('cuda'), k=10, dim=1)
        for user_id, item_ids in enumerate(top_ids):
            for item_id in item_ids:
                top_items.append((user_id, item_id.item()))

    return top_items