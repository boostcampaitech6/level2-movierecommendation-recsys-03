import torch
import time
import bottleneck as bn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import wandb


def run(args, model, loader, logger, setting):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    model_path = os.path.join(args.save, setting.get_submit_filename(args)[9:-4]+'.pt')
    

    train_data = loader.load_data('train')
    N = train_data.shape[0]
    idxlist = list(range(N))
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')

    if args.model=='Multi-DAE':
        is_VAE=False
        criterion = loss_function_dae
    else:
        is_VAE=True
        criterion = loss_function_vae
        
    best_n100 = -np.inf
    global update_count
    update_count = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(args, model, criterion, optimizer, epoch, train_data, idxlist, N, is_VAE)
        val_loss, n100, r10, r20 = evaluate(args, model, criterion, vad_data_tr, vad_data_te, N, is_VAE)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n100, r10, r20))
        print('-' * 89)
        wandb.log(dict(epoch=epoch,
                       train_loss=train_loss,
                       valid_loss=val_loss,
                       n100=n100,
                       r10=r10,
                       r20=r20))

        n_iter = epoch * len(range(0, N, args.batch_size))


        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(model_path, 'wb') as f:
                torch.save(model, f)
            best_n100 = n100
    
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, n100, r10, r20 = evaluate(args, model, criterion, test_data_tr, test_data_te, N, is_VAE)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | '
            'r20 {:4.2f}'.format(test_loss, n100, r10, r20))
    print('=' * 89)
    return model


def train(args, model, criterion, optimizer, epoch, train_data, idxlist, N, is_VAE):

    global update_count

    total_tr_loss_list = []

    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()

    np.random.shuffle(idxlist)

    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(args.device)
        optimizer.zero_grad()

        if is_VAE:
          if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                            1. * update_count / args.total_anneal_steps)
          else:
              anneal = args.anneal_cap

          optimizer.zero_grad()
          recon_batch, mu, logvar = model(data)

          loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
          recon_batch = model(data)
          loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        total_tr_loss_list.append(loss.item())
    
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))


            start_time = time.time()
            train_loss = 0.0
    return np.nanmean(total_tr_loss_list)

def evaluate(args, model, criterion, data_tr, data_te, N, is_VAE):


    global update_count

    # Turn on evaluation mode
    model.eval()
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    total_val_loss_list = []
    n100_list = []
    r10_list = []
    r20_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(args.device)
            if is_VAE :

              if args.total_anneal_steps > 0:
                  anneal = min(args.anneal_cap,
                                1. * update_count / args.total_anneal_steps)
              else:
                  anneal = args.anneal_cap

              recon_batch, mu, logvar = model(data_tensor)

              loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)

            total_val_loss_list.append(loss.item())

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)

            n100_list.append(n100)
            r10_list.append(r10)
            r20_list.append(r20)

    n100_list = np.concatenate(n100_list)
    r10_list = np.concatenate(r10_list)
    r20_list = np.concatenate(r20_list)

    return np.nanmean(total_val_loss_list), np.nanmean(n100_list), np.nanmean(r10_list), np.nanmean(r20_list)

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE