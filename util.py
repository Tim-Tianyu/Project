import torch
import torch.nn.functional as F
import numpy as np

from scipy.spatial import distance

def CoSenLogSoftmaxLoss(out, y, cost_matrix):
    # tensor object
    cost_matrix = torch.tensor(cost_matrix).float()
    assert(not cost_matrix.requires_grad)
    assert(not y.requires_grad)
    assert(out.requires_grad)
    costs_used = cost_matrix[y,:]
    weighted_exp_output = costs_used * torch.exp(out)
    weighted_softmax = weighted_exp_output / torch.sum(weighted_exp_output, axis=1).view(-1,1)
    loss = F.nll_loss(torch.log(weighted_softmax), y)
    return loss

def class_separability(p, q, intra_p):
    # p: N_1 * k
    # q: N_2 * k
    # intra_p: N_1 * 1
    inter_p_q = min_inter_class_dist(p, q)
    return np.mean(intra_p / inter_p_q)

def class_separability_matrix(data, targets):
    # data: N * k matrix
    # targets: N * 1 array
    # return: len(classes) * len(classes) 
    classes = np.unique(targets)
    intra_dists = []
    samples = []
    matrix = np.zeros((len(classes), len(classes)))
    
    for i in classes:
        samples.append(data[targets == i])
        intra_dists.append(min_intra_class_dist(samples[-1]))
    
    for i in range(0,len(classes)):
        for j in range(0,len(classes)):
            if (i == j):
                matrix[classes[i],classes[j]] = 1
            else:
                matrix[classes[i],classes[j]] = class_separability(samples[i], samples[j], intra_dists[i])
    return matrix

def min_intra_class_dist(p):
    # p: N * k
    # return: N * 1
    intra_dist = distance.cdist(p, p)
    return np.amin(intra_dist+np.eye(p.shape[0])*np.amax(intra_dist), axis=1)

def min_inter_class_dist(p, q):
    # p: N_1 * k
    # q: N_2 * k
    # return: N_1 * 1
    return np.amin(distance.cdist(p, q), axis=1)

def matrix_H(distribution):
    vector = histogram_vector(distribution)
    matrix = np.zeros((len(vector), len(vector)))
    for i in range(0, len(vector)):
        for j in range(0,len(vector)):
            if (i==j):
                matrix[i,j] = vector[i]
            else:
                matrix[i,j] = max(vector[i], vector[j])
    return matrix
    
def histogram_vector(distribution):
    return distribution / np.sum(distribution)

def matrix_T(H, S, R):
    return H * np.exp(-sudo_normalize(S)) * np.exp(-sudo_normalize(R))
    
def sudo_normalize(M):
    return (M - np.mean(M)) ** 2 / (2 * np.std(M) ** 2)

def cost_matrix_gradient(cost_matrix, confusion_matrix, data, targets, distribution, lr=0.1):
    S = class_separability_matrix(data,targets)
    H = matrix_H(distribution)
    T = matrix_T(H, S, confusion_matrix)
    return lr * (cost_matrix - T)
    
