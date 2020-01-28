import gzip
import itertools
import string
import numpy as np
import pandas as pd
import datetime as dt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix
import torch
from torch import nn, optim
from torch.distributions import gamma, Normal, kl
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import sys


alpha = float(sys.argv[1])
beta = float(sys.argv[2])
latent_dim = int(sys.argv[3])
tensor_num = int(sys.argv[4])
core_num = int(sys.argv[5])
lr = float(sys.argv[6])

X_train = pd.read_csv('X_train_sub.csv',names=['user', 'asin',  'time'])
X_test = pd.read_csv('X_test_sub.csv',names=['user', 'asin',  'time'])
y_train = pd.read_csv('y_train_sub.csv', header=None)
y_test = pd.read_csv('y_test_sub.csv', header=None)

y = pd.concat([y_train,y_test]).reset_index(drop=True)
y=(y-0)/(y.max()-0)
y_train1 = y.iloc[0:len(y_train)]
y_test1 = y.iloc[len(y_train):len(y)].reset_index(drop=True)

user_ids = X_train['user'].unique()
print(max(user_ids),len(user_ids))
times_ids = X_train['time'].unique()
print(max(times_ids),len(times_ids))
asin_ids = X_train['asin'].unique()
print(max(asin_ids),len(asin_ids))

factors = dict()
factors['user'] = dict()
factors['time'] = dict()
factors['asin'] = dict()

gamma_dist = gamma.Gamma(torch.tensor([alpha]), torch.tensor([beta]))
for user_id in user_ids:
    factors['user'][user_id] = gamma_dist.sample(torch.Size([latent_dim]))

for time_id in times_ids:
    factors['time'][time_id] = gamma_dist.sample(torch.Size([latent_dim]))

for asin_id in asin_ids:
    factors['asin'][asin_id] = gamma_dist.sample(torch.Size([latent_dim]))

weights = {}
weights['user'] = []
weights['time'] = []
weights['asin'] = []
normal_dist = Normal(torch.Tensor([30]), torch.Tensor([10]))
normal_dist1 = Normal(torch.Tensor([50]), torch.Tensor([10]))


for i in range(latent_dim):
    weights['user'].append(normal_dist.sample(torch.Size([tensor_num])).requires_grad_())
    weights['time'].append(normal_dist.sample(torch.Size([tensor_num])).requires_grad_())
    weights['asin'].append(normal_dist.sample(torch.Size([tensor_num])).requires_grad_())

for i in range(latent_dim):
    weights['user'].append(normal_dist1.sample(torch.Size([tensor_num])).requires_grad_())
    weights['time'].append(normal_dist1.sample(torch.Size([tensor_num])).requires_grad_())
    weights['asin'].append(normal_dist1.sample(torch.Size([tensor_num])).requires_grad_())

    
for i in range(latent_dim):
    weights['user'].append(normal_dist.sample(torch.Size([1])).requires_grad_())
    weights['time'].append(normal_dist.sample(torch.Size([1])).requires_grad_())
    weights['asin'].append(normal_dist.sample(torch.Size([1])).requires_grad_())

for i in range(latent_dim):
    weights['user'].append(normal_dist1.sample(torch.Size([1])).requires_grad_())
    weights['time'].append(normal_dist1.sample(torch.Size([1])).requires_grad_())
    weights['asin'].append(normal_dist1.sample(torch.Size([1])).requires_grad_())




def test_vae():
    pred_data = []
    softplus = nn.Softplus()
    for rind, row in X_test.iterrows():
        z_prod = []
        uid = row['user']
        wid = row['time']
        vid = row['asin']
        for i in range(latent_dim):
            z_prod.append(factors['user'][uid][i] * factors['time'][wid][i] * factors['asin'][vid][i])
        pred_data.append(torch.sum(torch.cat(z_prod)))
    pred_data_final = torch.stack(pred_data)
    print(max(pred_data_final))
    return pred_data_final


X_train_groups={}
X_train_groups['user']=X_train.groupby(by='user')
X_train_groups['time']=X_train.groupby(by='time')
X_train_groups['asin']=X_train.groupby(by='asin')

def vae_target_multi(target_name, sf1_name, sf2_name, target_id):    
    kl_div = torch.zeros(1)
    target_pred_train = []
    target_data_train = []
    z_alpha_prod = [[] for i in range(latent_dim)]
    z_beta_prod = [[] for i in range(latent_dim)]
    X = [None] * latent_dim 
    X_train_target = X_train_groups[target_name].get_group(target_id)
    for rind, data_row in X_train_target.iterrows():
        sf1_id = data_row[sf1_name]
        sf2_id = data_row[sf2_name]
        target_data_train.append(y_train.loc[rind,0])
        for i in range(latent_dim):
            X[i] = torch.stack([torch.tensor([y_train1.iloc[rind,0]]), factors[sf1_name][sf1_id][i], factors[sf2_name][sf2_id][i]],1)                       
            z_alpha_prod[i].append(torch.nn.Softplus()(torch.matmul(X[i], weights[target_name][i]) + weights[target_name][2 * latent_dim + i]))
            z_beta_prod[i].append(torch.nn.Softplus()(torch.matmul(X[i], weights[target_name][latent_dim + i]) + weights[target_name][3 * latent_dim + i]) )        
    target = [None] * latent_dim
    for i in range(latent_dim):        
        gamma_dist1 = gamma.Gamma(torch.sum(torch.cat(z_alpha_prod[i])), torch.sum(torch.cat(z_beta_prod[i])) )
        target[i] = gamma_dist1.rsample(torch.Size([1]))
        kl_div = torch.add(kl_div, kl.kl_divergence(gamma_dist1, gamma_dist))
    #factors[target_name][target_id] = Variable(torch.stack(target), requires_grad=False)
    j = 0
    for rind, data_row in X_train_target.iterrows():
        sf1_id = data_row[sf1_name]
        sf2_id = data_row[sf2_name]
        target_pred_train.append(torch.sum(torch.cat([target[i] * factors[sf1_name][sf1_id][i] * factors[sf2_name][sf2_id][i] for i in range(latent_dim)])))
        j += 1
    target_pred_train_final = torch.stack(target_pred_train)
    return target_pred_train_final, kl_div, target_data_train


def optimize_vae_multi(target_name, sf1_name, sf2_name, target_id):
    #print(weights['user'][0][0].grad) 
    target_pred_data_train, kl_div, target_data_train = vae_target_multi(target_name, sf1_name, sf2_name, target_id)
    target_data_train = torch.from_numpy(np.array(target_data_train)).float()
    optimizer = optim.Adam(weights[target_name], lr=lr)
    optimizer.zero_grad()
    #criterion = torch.nn.MSELoss(log_input=False, reduction='sum')
    criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction='sum')
    loss = criterion(target_pred_data_train, target_data_train) + kl_div
    loss.backward()
    return [weights[target_name][i].grad.numpy() for i in range(len(weights[target_name]))]

def update_factors(target_name, target_id, sf1_name, sf2_name):
    z_alpha_prod = [[] for i in range(latent_dim)]
    z_beta_prod = [[] for i in range(latent_dim)]
    soft_plus = torch.nn.Softplus()    
    X = [None] * latent_dim
    X_train_target = X_train_groups[target_name].get_group(target_id)
    for rind, data_row in X_train_target.iterrows():
        sf1_id = data_row[sf1_name]
        sf2_id = data_row[sf2_name]
        for i in range(latent_dim):            
            X[i] = torch.stack([torch.tensor([y_train1.iloc[rind,0]]), factors[sf1_name][sf1_id][i], factors[sf2_name][sf2_id][i]],1)
            z_alpha_prod[i].append(soft_plus(torch.matmul(X[i], weights[target_name][i]) + weights[target_name][2 * latent_dim + i]))
            z_beta_prod[i].append(soft_plus(torch.matmul(X[i], weights[target_name][latent_dim + i]) + weights[target_name][3 * latent_dim + i]))   
    target = [None] * latent_dim
    for i in range(latent_dim):
        gamma_dist1 = gamma.Gamma(torch.sum(torch.cat(z_alpha_prod[i])), torch.sum(torch.cat(z_beta_prod[i])))
        target[i] = gamma_dist1.rsample(torch.Size([1]))
        #target[i] = torch.tensor([torch.sum(torch.cat(z_alpha_prod[i])).data.numpy() / torch.sum(torch.cat(z_beta_prod[i])).data.numpy()])
    return (target_id ,torch.stack(target).data)


def poisson_log_likelihood(data_test1, pred_data_test1):
    return np.sum(data_test1 * np.log(pred_data_test1) -  pred_data_test1)



import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
err_train_last = 1.0

for iter in range(5000):
    weights_grads = {}
    weights_grads['user'] = []
    weights_grads['time'] = []
    weights_grads['asin'] = []
    for i in range(2 * latent_dim):
        weights_grads['user'].append(torch.zeros(tensor_num,1))
        weights_grads['time'].append(torch.zeros(tensor_num,1))
        weights_grads['asin'].append(torch.zeros(tensor_num,1))
        
    for i in range(2 * latent_dim):
        weights_grads['user'].append(torch.zeros(1,1))
        weights_grads['time'].append(torch.zeros(1,1))
        weights_grads['asin'].append(torch.zeros(1,1))
    params = []
    pool = mp.Pool(processes=core_num)
    target_keys = user_ids
    optimizer1 = optim.Adam(weights['user'], lr=lr)
    for target_key in target_keys:
        params.append(('user', 'time', 'asin', target_key))

    results = pool.starmap(optimize_vae_multi, params)
    pool.close()
    pool.join()
    for result in results:
        for sub in range(len(result)):
            weights_grads['user'][sub] += torch.tensor(result[sub])
    for sub in range(len(weights_grads['user'])):
        weights_grads['user'][sub] /= torch.tensor([float(len(X_train))])
        weights['user'][sub].grad = weights_grads['user'][sub]        
    optimizer1.step()
    params = []
    target_keys = user_ids
    for target_key in target_keys:
        params.append(('user', target_key, 'time','asin'))

    pool = mp.Pool(processes=core_num)
    results=pool.starmap(update_factors, params)
    pool.close()
    pool.join()
    for result in results:
        factors['user'][result[0]] = result[1]
    print('---------------user factors-----------------------')

    params = []
    pool = mp.Pool(processes=core_num)
    target_keys = times_ids
    optimizer4 = optim.Adam(weights['time'], lr=lr)
    for target_key in target_keys:
        params.append(('time', 'user','asin', target_key))

    results = pool.starmap(optimize_vae_multi, params)
    pool.close()
    pool.join()

    for result in results:
        for sub in range(len(result)):
            weights_grads['time'][sub] +=  torch.tensor(result[sub])  

    for sub in range(len(weights_grads['time'])):
        weights_grads['time'][sub] /= torch.tensor([float(len(X_train))])
        weights['time'][sub].grad = weights_grads['time'][sub]        
    optimizer4.step()


    params = []
    target_keys = times_ids
    for target_key in target_keys:
        params.append(('time', target_key, 'user','asin'))

    pool = mp.Pool(processes=core_num)
    results=pool.starmap(update_factors, params)
    pool.close()
    pool.join()
    for result in results:
        factors['time'][result[0]] = result[1]

    print('---------------time factors-----------------------')
    params = []
    pool = mp.Pool(processes=core_num)
    target_keys = asin_ids
    optimizer4 = optim.Adam(weights['asin'], lr=lr)
    for target_key in target_keys:
        params.append(('asin', 'user','time', target_key))

    results = pool.starmap(optimize_vae_multi, params)
    pool.close()
    pool.join()

    for result in results:
        for sub in range(len(result)):
            weights_grads['asin'][sub] +=  torch.tensor(result[sub])  

    for sub in range(len(weights_grads['asin'])):
        weights_grads['asin'][sub] /= torch.tensor([float(len(X_train))])
        weights['asin'][sub].grad = weights_grads['asin'][sub]        
    optimizer4.step()


    params = []
    target_keys = asin_ids
    for target_key in target_keys:
        params.append(('asin', target_key, 'user','time'))

    pool = mp.Pool(processes=core_num)
    results=pool.starmap(update_factors, params)
    pool.close()
    pool.join()
    for result in results:
        factors['asin'][result[0]] = result[1]

    print('---------------asin factors-----------------------')

    pred_data_test = test_vae()
    print(y_test.values.reshape(1,-1)[0],pred_data_test.data.numpy())
    print(mean_squared_error(y_test.values.reshape(1,-1)[0] , pred_data_test.data.numpy()), mean_absolute_error(y_test.values.reshape(1,-1)[0] , pred_data_test.data.numpy()), poisson_log_likelihood(y_test.values.reshape(1,-1)[0], pred_data_test.data.numpy())) 

    if iter % 1 == 0:
        factors1 = dict()
        factors1['user'] = []
        factors1['time'] = []
        factors1['asin'] = []
        for time_id in times_ids:
            if len(factors1['time']) == 0:
                factors1['time'] = factors['time'][time_id].data.numpy().reshape(1,-1)
            else:
                factors1['time'] = np.concatenate((factors1['time'], factors['time'][time_id].data.numpy().reshape(1,-1)))
        np.savetxt("time_factors.csv", factors1['time'], delimiter=",")
        for user_id in user_ids:
            if len(factors1['user']) == 0:
                factors1['user'] = factors['user'][user_id].data.numpy().reshape(1,-1)
            else:
                factors1['user'] = np.concatenate((factors1['user'], factors['user'][user_id].data.numpy().reshape(1,-1)))
        np.savetxt("user_factors.csv", factors1['user'], delimiter=",") 
        for asin_id in asin_ids:
            if len(factors1['asin']) == 0:
                factors1['asin'] = factors['asin'][asin_id].data.numpy().reshape(1,-1)
            else:
                factors1['asin'] = np.concatenate((factors1['asin'], factors['asin'][asin_id].data.numpy().reshape(1,-1)))
        np.savetxt("asin_factors.csv", factors1['asin'], delimiter=",") 


