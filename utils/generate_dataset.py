#!/usr/bin/env python
# coding: utf-8

# In[84]:


import json
import numpy as np
import torch
import os


# In[92]:


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


# In[93]:


def get_data(num_user, dataset_item=('synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1', 'mnist', 'femnist')):
    
    train_path = os.path.join('data', dataset_item, 'data', 'train')
    test_path = os.path.join('data', dataset_item, 'data', 'test')
        
    clients, groups, train_data_orig, test_data_orig = read_data(train_path, test_path)
    train_data = { 'x': {}, 'y': {},  'num_samples': []}
    test_data = { 'x': {}, 'y': {},  'num_samples': []}
    for i in range(num_user):
        client = clients[i]
        client_train_data = train_data_orig[client]
        train_data['x'][i] = client_train_data['x']
        train_data['y'][i] = client_train_data['y']
        train_data['num_samples'].append(len(client_train_data['y']))
    
        client_test_data = test_data_orig[client]
        test_data['x'][i] = client_test_data['x']
        test_data['y'][i] = client_test_data['y']
        test_data['num_samples'].append(len(client_test_data['y']))

    # 生成global_train 和global_test
    global_train_x = []
    global_train_y = []
    global_test_x = []
    global_test_y = []
    global_train_data = {'x': {}, 'y': {}}
    global_test_data = {'x': {}, 'y': {}}
    for i in range(num_user):
        train_x = torch.tensor(train_data['x'][i])
        train_y = torch.tensor(train_data['y'][i])
        global_train_x.append(train_x)
        global_train_y.append(train_y)
        test_x = torch.tensor(test_data['x'][i])
        test_y = torch.tensor(test_data['y'][i])
        global_test_x.append(test_x)
        global_test_y.append(test_y)
    
    global_train_x = torch.vstack(global_train_x)
    global_train_y = torch.hstack(global_train_y)
    global_train_data['x'], global_train_data['y'] = global_train_x, global_train_y

    global_test_x = torch.vstack(global_test_x)
    global_test_y = torch.hstack(global_test_y)
    global_test_data['x'], global_test_data['y'] = global_test_x, global_test_y
    
    return train_data, global_train_data, global_test_data


# In[ ]:




