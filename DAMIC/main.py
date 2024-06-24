import torch

import tqdm
from torch.optim import Adam
from time import time

import opt
from utils import *
from encoder import *
from DAMIC import DAMIC
from data_loader import load_data
import math
def calc_loss(x, x_aug, temperature=0.2, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        #    print(pos_sim,sim_matrix.sum(dim=0))
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

    return loss


def pretrain_ae(model, x):
    print("Pretraining AE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z = model.encoder(x)
        x_hat = model.decoder(z)
        loss = F.mse_loss(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pretrain_gae(model, x, adj):
    print("Pretraining GAE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z, a = model.encoder(x, adj)
        z_hat, z_adj_hat = model.decoder(z, adj)
        a_hat = a + z_adj_hat
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(a_hat, adj.to_dense())
        loss = loss_w + opt.args.alpha_value * loss_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def pre_train(model, X1, A1, X2, A2):
    print("Pretraining fusion model...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    for epoch in tqdm.tqdm(range(opt.args.fus_epoch)):

        # input & output
        hat1, z_fused_hat1, hat2, z_fused_hat2, z1, z2, z_fused, q_z1, q_z2, q_z_fused, p_z_fused = model(X1, A1, X2, A2, pretrain=True)
        L_DRR = 0
        L_REC1 = reconstruction_loss(X1, A1, hat1)
        L_REC2 = reconstruction_loss(X2, A2, hat2)
        L_REC3 = reconstruction_loss(X1, A1, z_fused_hat1)
        L_REC4 = reconstruction_loss(X2, A2, z_fused_hat2)

        loss = L_REC1 + L_REC2 + L_REC3 + L_REC4 + L_DRR
 
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), './model_pretrained/{}_pretrain.pkl'.format(opt.args.name))
    

def train(model, X1, A1, X2, A2, y):
    if not opt.args.pretrain:
        # loading pretrained model
        model.load_state_dict(torch.load('./model_pretrained/{}_pretrain.pkl'.format(opt.args.name), map_location='cpu'))

        with torch.no_grad():
            hat1, z_fused_hat1, hat2, z_fused_hat2, z1, z2, z_fused, q_z1, q_z2, q_z_fused, p_z_fused = model(X1, A1, X2, A2)
        
        _, _, _, _, centers = clustering(z_fused, y)

        # initialize cluster centers
        model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)
    
    print("Training...")

    optimizer = Adam(model.parameters(), lr=(opt.args.lr))
    loss_device = torch.device("cuda")
    pbar = tqdm.tqdm(range(opt.args.epoch), ncols=200)
    for epoch in pbar:

        # input & output
        hat1, z_fused_hat1, hat2, z_fused_hat2, z1, z2, z_fused, q_z1, q_z2, q_z_fused, p_z_fused = model(X1, A1, X2, A2)

        L_DRR = calc_loss(q_z1.T, q_z_fused.T) + calc_loss(q_z2.T, q_z_fused.T)

        L_REC1 = reconstruction_loss(X1, A1, hat1)
        L_REC2 = reconstruction_loss(X2, A2, hat2)
        L_REC3 = reconstruction_loss(X1, A1, z_fused_hat1)
        L_REC4 = reconstruction_loss(X2, A2, z_fused_hat2)

        loss = L_REC1 + L_REC2 + L_REC3 + L_REC4 + L_DRR


        p_z1 = target_distribution(q_z1)
        p_z2 = target_distribution(q_z2)
        L_KL = F.kl_div((q_z1.log() + q_z2.log()) / 2, p_z_fused, reduction='batchmean')


        loss = loss +L_KL

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clustering & evaluation
        ari, nmi, ami, acc, y_pred = assignment((p_z1 + p_z2).data, y)

        pbar.set_postfix({'loss':'{0:1.4f}'.format(loss), 'ARI':'{0:1.4f}'.format(ari),'NMI':'{0:1.4f}'.format(nmi),
                          'AMI':'{0:1.4f}'.format(ami),'ACC':'{0:1.4f}'.format(acc)})
    
        if ari > opt.args.ari:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.ami = ami
            best_epoch = epoch

    pbar.close()
    
    print("Best_epoch: {},".format(best_epoch),"ARI: {:.4f},".format(opt.args.ari), "NMI: {:.4f},".format(opt.args.nmi), 
            "AMI: {:.4f}".format(opt.args.ami), "ACC: {:.4f}".format(opt.args.acc))
    
    print("Final_epoch: {},".format(epoch),"ARI: {:.4f},".format(ari), "NMI: {:.4f},".format(nmi), 
            "AMI: {:.4f}".format(ami), "ACC: {:.4f}".format(acc))

    torch.save(model.state_dict(),'./output/{}model.pt'.format(opt.args.name))

if __name__ == '__main__':
    # setup
    print("setting:")

    setup_seed(opt.args.seed)

    opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("alpha value   : {:.0e}".format(opt.args.alpha_value))
    print("k value       : {}".format(opt.args.k))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")

    # load data
    Xr, y, Ar = load_data(opt.args.name, 'RNA', opt.args.method, opt.args.k, show_details=False)
    Xa, y, Aa = load_data(opt.args.name, 'ATAC', opt.args.method, opt.args.k, show_details=False)
    opt.args.n_clusters = int(max(y) - min(y) + 1)

    Xr = numpy_to_torch(Xr).to(opt.args.device)
    Ar = numpy_to_torch(Ar, sparse=True).to(opt.args.device)

    Xa = numpy_to_torch(Xa).to(opt.args.device)
    Aa = numpy_to_torch(Aa, sparse=True).to(opt.args.device)

    ae1 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z).to(opt.args.device)

    ae2 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z).to(opt.args.device)
    
    if opt.args.pretrain:
        opt.args.dropout = 0.4
    gae1 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    gae2 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    if opt.args.pretrain:
        t0 = time()
        pretrain_ae(ae1, Xr)
        pretrain_ae(ae2, Xa)

        pretrain_gae(gae1, Xr, Ar)
        pretrain_gae(gae2, Xa, Aa)

        model = DAMIC(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)

        pre_train(model, Xr, Ar, Xa, Aa)
        t1 = time()
        print("Time_cost: {}".format(t1-t0))
    else:
        t0 = time()
        model = DAMIC(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)

        train(model, Xr, Ar, Xa, Aa, y)
        t1 = time()
        print("Time_cost: {}".format(t1-t0))
