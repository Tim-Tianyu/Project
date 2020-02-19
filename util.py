import numpy as np

from scipy.spatial import distance

def CoSenLogSoftmaxLoss(x, y, cost_matrix):
    # tensor object
    cots_used = self.cost_matrix[y,:]
    weighted_exp_output = cots_used * torch.exp(x)
    weighted_softmax = weighted_exp_output / torch.sum(weighted_exp_output, axis=0)
    loss = torch.mean(-torch.log(weighted_softmax[:,y]))
    return loss

def class_separability(p, q, intra_p):
    # p: N_1 * k
    # q: N_2 * k
    # intra_p: N_1 * 1
    inter_p_q = min_inter_class_dist(p, q)
    print(intra_p/inter_p_q)
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

def matrix_H(histogram_vector):
    matrix = np.zeros((len(histogram_vector), len(histogram_vector)))
    for i in range(0, len(histogram_vector)):
        for j in range(0,len(histogram_vector)):
            if (i==j):
                matrix[i,j] = histogram_vector[i]
            else:
                matrix[i,j] = max(histogram_vector[i], histogram_vector[j])
    return matrix
    
def histogram_vector(distribution):
    return distribution / np.sum(distribution)

def matrix_T(H, S, R):
    return H * np.exp(-sudo_normalize(S)) * np.exp(-sudo_normalize(R))
    
def sudo_normalize(M):
    return (M - np.mean(M)) ** 2 / (2 * np.std(M) ** 2)

def cost_matrix_gradient(cost_matrix, confusion_matrix, data, targets, distribution, lr=0.1):
    S = class_separability_matrix(data,targets)
    H = histogram_vector(v)
    T = matrix_T(H, S, confusion_matrix)
    return lr * (cost_matrix - T)
    
