# -*- coding: utf-8 -*-
"""
Original EpiDeep implementation (backup)
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from rnnAttention import RNN
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from utils import EarlyStopping
dtype = torch.float
STOP_INIT_EPIDEEP=350

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def mse_loss(inp, target):
    return torch.sum((inp - target)**2) / inp.data.nelement()

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="leakyReLU":
            net.append(nn.LeakyReLU())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class EpiDeep(nn.Module):
    '''
    '''
    def __init__(self, input1_dim, embed1_dim, input2_dim, embed2_dim, n_centroids, encode_layers=[500, 200], decode_layers=[ 200, 500], mapping_layers=[100,200, 100],\
                    device=torch.device("cpu"),encod_out_dim=20):
        super(EpiDeep, self).__init__()
        self.device = device
        self.input1_dim = input1_dim
        self.embed1_dim = embed1_dim
        self.n_centroids = n_centroids
        self.input2_dim = input2_dim
        self.embed2_dim = embed2_dim
        
        self.first_encoder = buildNetwork([input1_dim]+encode_layers+[embed1_dim]).to(self.device)
        self.first_decoder = buildNetwork([embed1_dim]+encode_layers+[input1_dim]).to(self.device)
        
        self.first_cluster_layer = Parameter(torch.Tensor(n_centroids, embed1_dim)).to(self.device)
        torch.nn.init.xavier_normal_(self.first_cluster_layer.data)
        
        self.second_encoder = buildNetwork([input2_dim]+encode_layers+[embed2_dim]).to(self.device)
        self.second_decoder = buildNetwork([embed2_dim]+encode_layers+[input2_dim]).to(self.device)
        
        self.second_cluster_layer = Parameter(torch.Tensor(n_centroids, embed2_dim)).to(self.device)
        torch.nn.init.xavier_normal_(self.second_cluster_layer.data)
        
        self.mapper = buildNetwork([embed1_dim] + mapping_layers+[embed2_dim], activation="LeakyReLu").to(self.device)
        self.encoder = RNN(1, encod_out_dim, 2, device=device).to(self.device)

        deco_layers = [self.encoder.hidden_size + embed2_dim, 20, 20]
        self.decoder = buildNetwork(deco_layers, activation="LeakyReLu").to(self.device)
        self.regressor_layers = [20, 20, 20, 1]
        self.regressor = buildNetwork(self.regressor_layers, activation="LeakyReLu").to(self.device)
        self.alpha = 1


    def pre_train(self, qdata, fdata, feature_data, pre_train_epochs=1000):
        x1 = qdata
        x2 = fdata
        
        optimizer = torch.optim.Adam(self.parameters())
        for _ in range(pre_train_epochs):        
            z1 = self.first_encoder(x1)
            x1_bar = self.first_decoder(z1)
    
            optimizer.zero_grad()
            loss = F.mse_loss(x1_bar, x1)
            loss.backward()
            optimizer.step()
        for _ in range(pre_train_epochs):  
            z2 = self.second_encoder(x2)
            x2_bar = self.second_decoder(z2)
            optimizer.zero_grad()
            loss = F.mse_loss(x2_bar, x2)
            loss.backward()
            optimizer.step()
    
    
    def forward_clustering_first(self, x1):
        z1 = self.first_encoder(x1)
        x1_bar = self.first_decoder(z1)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z1.unsqueeze(1) - self.first_cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x1_bar, q ,z1
    

    def forward_clustering_second(self, x2):
        z2 = self.second_encoder(x2)
        x2_bar = self.second_decoder(z2)
        
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z2.unsqueeze(1) - self.second_cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x2_bar, q , z2
        
        
        
    def loss_function(self, x1, recon_x1, x2, recon_x2, rnn_data, rnn_labels, q1, q2, emb1, emb2, k, epoch,init_epideep, feature_data, cost_vectors):

        p1 = target_distribution(q1)
        loss_val1 = F.kl_div(q1.log(), p1.detach())
        p2 = target_distribution(q2)
        loss_val2 = F.kl_div(q2.log(), p2.detach())

        translated_emb = self.mapper(emb1)
        rnn_out = self.encoder(rnn_data)
        out = self.decoder(torch.cat((rnn_out, translated_emb),1))

        # NOTE: the following will be skipped if there are no added modules
        # forward pass on each module and concat with previous embedding
        feat_mod_rloss = torch.tensor([0.], dtype=dtype, device=self.device)
        cost_vector = np.ones((rnn_out.shape[0], 1))  # default, i.e. all have the same cost
        pred = self.regressor(out)
        # apply weighted cost to get prediction loss
        pred_loss = torch.tensor(cost_vector, dtype=torch.float, device=self.device) * F.mse_loss(pred, rnn_labels, reduction='none')
        pred_loss = pred_loss.mean()  # now reduce
        loss = 0.1*(F.mse_loss(x1, recon_x1)+F.mse_loss(x2, recon_x2)+ F.mse_loss(translated_emb, emb2) + loss_val1 + loss_val2) + 10*pred_loss +  feat_mod_rloss
        return loss

    
    def fit(self, qdata, fdata, rnn_data, rnn_labels, feature_data=None, cost_vectors=None, lr = 0.001, num_epoch = 10, train_seasons=None, \
        first_year=None, pre_train_epochs=1000, init_epideep=False,model_path=None,epiweek=None):
        """
        """
        self.train() 
        in_q_data = Variable(torch.Tensor(qdata).to(self.device), requires_grad= True)
        in_f_data = Variable(torch.Tensor(fdata).to(self.device), requires_grad= True)
        rnn_data = Variable(torch.Tensor(rnn_data).to(self.device), requires_grad= True)
        rnn_labels = Variable(torch.Tensor(rnn_labels).to(self.device))

        k=None  # default value - will not be used when enters to if 
        self.pre_train(in_q_data, in_f_data, feature_data, pre_train_epochs)
        
        kmeans = KMeans(n_clusters=self.n_centroids, n_init=10)
        z1 = self.first_encoder(in_q_data)
        kmeans.fit_predict(z1.cpu().detach().numpy())
        self.first_cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=self.device)
        
        z2 = self.second_encoder(in_f_data)
        kmeans.fit_predict(z2.cpu().detach().numpy())
        self.second_cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=self.device)
    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        import time
        start=time.time()
        for epoch in range(num_epoch):
            
            x1_bar, q1, z1 = self.forward_clustering_first(in_q_data)
            x2_bar, q2, z2 = self.forward_clustering_second(in_f_data)
            
            loss = self.loss_function(in_q_data,x1_bar, in_f_data, x2_bar, rnn_data, rnn_labels, q1, q2, z1, z2, k,epoch, init_epideep, feature_data, cost_vectors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.item())
            if np.isnan(loss.item()) and epoch>STOP_INIT_EPIDEEP:  # break only after normal epideep
                break
        end=time.time()
        print('train time: ',end-start)
        # deactivate dropout (if it was set)
        self.eval()     
        
       

    def predict(self, data, rnn_data, feature_data=None, in_f_data=None):
        '''
            Right now guidance is just the full season data, we have to make it be anything
        '''
        in_data = Variable(torch.Tensor(data).to(self.device))
        in_rnn_data = Variable(torch.Tensor(rnn_data).to(self.device))
        emd = self.mapper(self.first_encoder(in_data))
        rnn_out = self.encoder(in_rnn_data)
        out = self.decoder(torch.cat((rnn_out, emd),1))
        pred = self.regressor(out)

        return pred
        


class EpiDeepCN(EpiDeep):
    '''
    '''
        

    def fit_with_dataloader(self, data_loader_hist_train, lr = 0.001, num_epoch = 10, train_seasons=None, \
            model_path=None,epiweek=None):
        self.train() 
        
        # get data from one batch
        in_q_data, in_f_data, _, _ = next(iter(data_loader_hist_train))
        
        self.pre_train(in_q_data, in_f_data, [], 10)
        
        kmeans = KMeans(n_clusters=self.n_centroids, n_init=10)
        z1 = self.first_encoder(in_q_data)
        kmeans.fit_predict(z1.cpu().detach().numpy())
        self.first_cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=self.device)
        
        z2 = self.second_encoder(in_f_data)
        kmeans.fit_predict(z2.cpu().detach().numpy())
        self.second_cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=self.device)
    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        es = EarlyStopping(patience=10, min_delta=0.1)
        import time
        start=time.time()
        feature_data=[]
        cost_vectors=[]
        
        for epoch in range(num_epoch):
            for bath_query_length,bath_full_length,bath_rnn_data,bath_rnn_label in data_loader_hist_train: 
                optimizer.zero_grad()
                k=None  # default value - will not be used when enters to if use_seldonian
                x1_bar, q1, z1 = self.forward_clustering_first(bath_query_length)
                x2_bar, q2, z2 = self.forward_clustering_second(bath_full_length)
                loss = self.loss_function(bath_query_length,x1_bar, bath_full_length, x2_bar, bath_rnn_data, bath_rnn_label, q1, q2, z1, z2, k,epoch, False, feature_data, cost_vectors)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # stopping criteria
                if es.step(loss):
                    break  # early stop criterion is met, we can stop now
            print(epoch, loss.item())
        end=time.time()
        print('train time: ',end-start)
        # deactivate dropout (if it was set)
        self.eval()



if __name__ == "__main__":
    pass
