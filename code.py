# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load Data
print("Loading Raw Data")
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")

# Helper Functions 
def get_features(raw_data):
    cols = []
    # Get data of each row from pixel0 to pixel783 
    for px in range(784):
        cols.append("pixel"+str(px))   
    #return (raw_data.as_matrix(cols) /255) - 0.5
    return (raw_data.as_matrix(cols) > 0 ) * 1

def cross_validated(X, n_samples):
    kf = KFold(n_samples, shuffle = True)
    result = [group for group in kf.split(X)]
    return result        
    
# Deep Neural Net
# Initialize Parameters 
def init_dnn_parameters(n, activations, epsilons, filter1=None):
    L = len(n)
    params = {}
    vgrad = {}
    d_rms = {}
    for l in range(1,L):
        W = np.random.randn(n[l],n[l-1]) * epsilons[l] 
        # Experiment, multiply filter in case of input layer weights 
        if filter1 is not None and l == 1:
            W = np.dot(W, filter1) 
        b = np.zeros((n[l],1))
        params["W"+str(l)] = W
        params["b"+str(l)] = b
        # Normalization Parameters
        params["mu"+str(l)] = 0
        params["sig"+str(l)] = 1
        
        vgrad["W"+str(l)] = W * 0
        vgrad["b"+str(l)] = b * 0
        d_rms["W"+str(l)] = W * 0
        d_rms["b"+str(l)] = b * 0

        params["act"+str(l)] = activations[l]
    params["n"] = n
    return params, vgrad, d_rms

# Activation Functions 
def gdnn(X, activation_function):
    leak_factor = 1/100000
    if activation_function == 'tanh':
        return np.tanh(X)
    if activation_function == 'lReLU':
        return ((X > 0) * X) + ((X <= 0)* X * leak_factor)
    if activation_function == 'linear':
        return X
    if activation_function == 'softmax':
        t = np.exp(X - np.max(X, axis = 0))
        t_sum = np.reshape(np.sum(t, axis = 0),(1,-1))
        return t/t_sum
    else: 
        return 1 / (1 +np.exp(-X))

def gdnn_prime(X, activation_function):
    leak_factor = 1/100000
    if activation_function == 'tanh':
        return 1-np.power(X,2)
    if activation_function == 'lReLU':
        return ((X > 0) * 1) + ((X <= 0)* leak_factor)
    if activation_function == 'linear':
        return X**0
    else: 
        return (1 / (1 +np.exp(-X)))*(1-(1 / (1 +np.exp(-X))))

# Cost 
def get_dnn_cost(Y_hat, Y):
    #print(Y.shape)
    m = Y.shape[1]
    # in case of softmax, we do not include (1-Y) term 
    logprobs = np.multiply(np.log(Y_hat),Y) # + np.multiply(np.log(1-Y_hat),1-Y)
    cost = - np.sum(logprobs) /m
    return cost
    
# Forward Propagation 
def forward_dnn_propagation(X, params):
    # Get Network Parameters 
    n = params["n"]
    L = len(n)
    
    A_prev = X
    cache = {}
    cache["A"+str(0)] = X
    for l in range(1,L):
        W = params["W"+str(l)]
        b = params["b"+str(l)]
        #print("DEBUG FF l[{: <2}] - Mu {:.2E}, Sig {:.2E}".format(l, params["mu"+str(l)],params["sig"+str(l)] ))
        Z = (np.dot(W,A_prev)+b - params["mu"+str(l)]) / (params["sig"+str(l)] + 1e-8)
        A = gdnn(Z,params['act'+str(l)])
        cache["Z"+str(l)] = Z
        cache["A"+str(l)] = A
        
        A_prev = A
    return A, cache, params 

# Backward Propagation
def back_dnn_propagation(X, Y, params, cache, alpha = 0.01, _lambda=0, keep_prob=1):
    n = params["n"]
    L = len(n) -1
    
    m = X.shape[1]
    W_limit = 5
    A = cache["A"+str(L)]
    A1 = cache["A"+str(L-1)]
    grads = {}
    
    # Outer Layer 
    dZ = A - Y#gdnn_prime(A - Y, params["act"+str(L)])
    dW = 1/m * np.dot(dZ, A1.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    grads["dZ"+str(L)] = dZ
    grads["dW"+str(L)] = dW + _lambda/m * params["W"+str(L)]
    grads["db"+str(L)] = db
    
    # Update Outer Layer
    params["W"+str(L)] -= alpha * dW
    params["b"+str(L)] -= alpha * db
    for l in reversed(range(1,L)):
        params["mu"+str(l)] = np.mean(cache["Z"+str(l)])
        params["sig"+str(l)] = np.std(cache["Z"+str(l)])
        dZ2 = dZ
        W2 = params["W"+str(l+1)]
        b = params["b"+str(l)]
        A2 = cache["A"+str(l)]
        A1 = cache["A"+str(l-1)]
        d = np.random.randn(A1.shape[0],A1.shape[1]) > keep_prob
        A1 = A1 * d/keep_prob
        dZ = np.dot(W2.T, dZ2)*gdnn_prime(A2, params["act"+str(l)])
        dW = 1/m * np.dot(dZ, A1.T) + _lambda/m * params["W"+str(l)]
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        grads["dZ"+str(l)] = dZ
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
        params["W"+str(l)] -= alpha *dW
        params["b"+str(l)] -= alpha *db
    
    return grads, params    

# Momentum Gradient Descent 
def back_dnn_propagation_with_momentum(X, Y, params, cache, alpha = 0.01, _lambda=0, 
                    keep_prob=1, beta=0.9, 
                    vgrad = {}, d_rms={}, t=0):
    n = params["n"]
    L = len(n) -1
    
    beta2 = 0.999
    
    m = X.shape[1]
    W_limit = 5
    A = cache["A"+str(L)]
    A1 = cache["A"+str(L-1)]
    grads = {}

    v_corr = {}
    s_corr = {}
    # Outer Layer 
    dZ = A - Y#gdnn_prime(A - Y, params["act"+str(L)])
    dW = 1/m * np.dot(dZ, A1.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    grads["dZ"+str(L)] = dZ
    grads["dW"+str(L)] = dW + _lambda/m * params["W"+str(L)]
    grads["db"+str(L)] = db
    
    vgrad["W"+str(L)] = beta * vgrad["W"+str(L)] + (1 - beta) * grads["dW"+str(L)] 
    vgrad["b"+str(L)] = beta * vgrad["b"+str(L)] + (1 - beta) * grads["db"+str(L)]

    v_corr["W"+str(L)] = vgrad["W"+str(L)] / (1 - beta ** t)  
    v_corr["b"+str(L)] = vgrad["b"+str(L)] / (1 - beta ** t) 

    # RMS
    d_rms["W"+str(L)] = beta2 * d_rms["W"+str(L)] + (1 - beta2) * grads["dW"+str(L)] ** 2 
    d_rms["b"+str(L)] = beta2 * d_rms["b"+str(L)] + (1 - beta2) * grads["db"+str(L)] ** 2

    s_corr["W"+str(L)] = d_rms["W"+str(L)] / (1 - beta2 ** t) 
    s_corr["b"+str(L)] = d_rms["b"+str(L)] / (1 - beta2 ** t)

    #print("Debug - ADAM (L)")
    #print( v_corr["W"+str(L)] / np.sqrt((s_corr["W"+str(L)]) + 1e-8))
    # Update Outer Layer
    params["W"+str(L)] -= alpha * v_corr["W"+str(L)] / (np.sqrt(s_corr["W"+str(L)]) + 1e-8)
    params["b"+str(L)] -= alpha * v_corr["b"+str(L)] / (np.sqrt(s_corr["b"+str(L)]) + 1e-8)
    
    for l in reversed(range(1,L)):
        if l < L:
            params["mu"+str(l)] = (1 - alpha) * params["mu"+str(l)] + alpha * np.reshape(np.nanmean(cache["Z"+str(l)], axis=1),(-1,1))
            params["sig"+str(l)] = (1 - alpha) * params["sig"+str(l)] + alpha * np.reshape(np.nanstd(cache["Z"+str(l)], axis = 1),(-1,1))

        dZ2 = dZ
        W2 = params["W"+str(l+1)]
        b = params["b"+str(l)]
        A2 = cache["A"+str(l)]
        A1 = cache["A"+str(l-1)]
        d = np.random.randn(A1.shape[0],A1.shape[1]) > keep_prob
        A1 = A1 * d/keep_prob
        dZ = np.dot(W2.T, dZ2)*gdnn_prime(A2, params["act"+str(l)])
        dW = 1/m * np.dot(dZ, A1.T) + _lambda/m * params["W"+str(l)]
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        grads["dZ"+str(l)] = dZ
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
        vgrad["W"+str(l)] = beta * vgrad["W"+str(l)] + (1 - beta) * grads["dW"+str(l)] 
        vgrad["b"+str(l)] = beta * vgrad["b"+str(l)] + (1 - beta) * grads["db"+str(l)]
        v_corr["W"+str(l)] = vgrad["W"+str(l)] / (1 - beta ** t)  
        v_corr["b"+str(l)] = vgrad["b"+str(l)] / (1 - beta ** t) 
        
        d_rms["W"+str(l)] = beta2 * d_rms["W"+str(l)] + (1 - beta2) * grads["dW"+str(l)] ** 2 
        d_rms["b"+str(l)] = beta2 * d_rms["b"+str(l)] + (1 - beta2) * grads["db"+str(l)] ** 2
        s_corr["W"+str(l)] = d_rms["W"+str(l)] / (1 - beta2 ** t) 
        s_corr["b"+str(l)] = d_rms["b"+str(l)] / (1 - beta2 ** t)

        #print("Debug - ADAM ({})".format(l))
        #print( v_corr["W"+str(l)] / np.sqrt((s_corr["W"+str(l)]) + 1e-8))
        
        params["W"+str(l)] -= alpha * v_corr["W"+str(l)] / (np.sqrt(s_corr["W"+str(l)]) + 1e-8)
        params["b"+str(l)] -= alpha * v_corr["b"+str(l)] / (np.sqrt(s_corr["b"+str(l)]) + 1e-8)
    
    return grads, params, vgrad, d_rms    

def batch_back_propagation(X, Y, params, cache, alpha = 0.01, 
            _lambda=0, keep_prob=1,chunk_size=128, beta=0.9, 
            vgrad={}, d_rms={}):
    # slice input and output data into smaller chunks 
    m = X.shape[1]
    include_probability = keep_prob
    idx_from = 0
    batch_size = chunk_size 
    idx_to = min(batch_size, m)
    print("Mini-Batch - Shuffling Training Data")
    shuffled_idx = list(np.random.permutation(m))
    X_shuffle = X[:,shuffled_idx]
    y_shuffle = Y[:,shuffled_idx]
    counter = 0
    while idx_to < m:
        counter += 1
        if idx_from < idx_to:
            #print(" [{: >3d}], Size [{}] End @ {:5.2f}%, Alph {:.2E}".format(counter,
            #                                                            batch_size, 
            #                                                            100*idx_to/m,
            #                                                            alpha * (0.9 ** (counter-1))),end="")
            X_train = X_shuffle[:,idx_from:idx_to]
            y_train = y_shuffle[:,idx_from:idx_to]
    
            A, cache, params = forward_dnn_propagation(X_train, params)
            #grads, params= back_dnn_propagation(X_train, y_train, params, cache, alpha ,_lambda, keep_prob)
            grads, params, vgrad, d_rms= back_dnn_propagation_with_momentum( X_train, 
                                                                    y_train, 
                                                                    params, 
                                                                    cache, 
                                                                    alpha, # * ((1-alpha) ** (counter-1)),
                                                                    _lambda, 
                                                                    keep_prob,
                                                                    beta,
                                                                    vgrad,
                                                                    d_rms,
                                                                    counter)
            #print(" Tr. Score {:.2E}".format(np.mean(get_dnn_cost(A, y_train))))
        idx_from += batch_size
        idx_from = min(m, idx_from)
        idx_to += batch_size
        idx_to = min(m, idx_to)
    return grads, params, vgrad, d_rms
    
# Train Model 
print("Loading Training and Dev Data ")
X2 = get_features(train_raw)

labels = np.array(train_raw['label'])
m = labels.shape[0]
y = np.zeros((m,10))
for j in range(10):
    y[:,j]=(labels==j)*1
# TODO: implement softmax as output layer 
k = 38
folds = 5
oinst = 1
h_layers = 4
beta = 0.9
np.random.seed(1)
print("Cross Validation using {} folds".format(folds))
print("Building Deep Network of {} Hidden Layer Groups".format(h_layers))
print("Cross Validation ..")
cv_groups = cross_validated(X2, folds)
print("Done")
alphas = np.linspace(0.00125, 0.00125, oinst)
epsilons = np.linspace(0.76,0.78,oinst)
gammas =  np.linspace(0.01,0.01,oinst)
lambdas=  np.linspace(1.0,1.0,oinst)
keep_probs=  np.linspace(0.99,0.99,oinst)
alph_decays = np.linspace(0.9,0.9,oinst) 
iterations = 100
n_1 = []
break_tol = 0.00001
etscost = []
etrcost= []
seeds = []
layers = []
for j in range(oinst):
    batch_processing = True
    base_batch_size = 1024 # min size

    print("Building Network")
    X = X2 # Direct Map
    n = [X.shape[1]]
    acts = ['input']
    gamma = [0]
    for layer in range(h_layers):
        n.append((17)**2) #((28-layer*3))**2)
        acts.append('lReLU') #tanh')
        gamma.append(np.sqrt(2/n[layer-1]))
        print("Hidden Layer[{: ^3d}] n = {: >4}, Activation Fn [{: >8}], Weight init Factor = {:.2E}".format(
            len(n)-1, n[-1], acts[-1], gamma[-1]))
    #for layer in range(h_layers):
    #    n.append((28)**2) #((28-layer*3))**2)
    #    acts.append('lReLU') #tanh')
    #    gamma.append(np.sqrt(2/n[layer-1]))
    #    print("Hidden Layer[{:03d}] n = {}, Activation Fn [{}], Weight init Factor = {:3.2f}".format(
    #        len(n)-1, n[-1], acts[-1], gamma[-1]))
    layers.append(j+1)    
    n.append(y.shape[1])
    acts.append('softmax')
    gamma.append(np.sqrt(1/n[layer-1]))
    print("Output Layer n = {}, Activation Function [{}], Weight init Factor = {:3.2f}".format(
            n[-1], acts[-1], gamma[-1]))
    n_1.append(j+4)
    np.random.seed(1)
   
    alpha = alphas[j]#0.166# 
    _lambda = lambdas[j] # 0.5#
    keep_prob = keep_probs[j]
    epsilon = 0.76#epsilons[j] #0.02 
    print("Hyper-parameters")
    print("alpha = {:.2E}, # Epochs = {}, lambda = {:3.2f}, keep probability = {:3.2f} % ".format(
        alpha, iterations, _lambda, keep_prob*100))
    print("Momentum (Beta) = {:3.2f}".format(beta))

    L = len(n) - 1

    # Prepare Training and testing sets 
    X_train = X[cv_groups[0][0],:].T 
    y_train = y[cv_groups[0][0],:].T 
    labels_train = labels[cv_groups[0][0]]
    # Experiment - Filter based on linear correlation
    
    depth = 1024
    print("Building Input Layer Initialization Filter, Depth = {}".format(depth))
    filter1 = np.zeros((n[0],n[0]))
    for dim in range(10):
        for monomial in range(1,min(2, h_layers)):
            X_sample = X_train[:,:depth].T**monomial
            X_mean = np.reshape(np.mean(X_sample,axis=0),(1,-1))
            y_sample = np.reshape(y_train[dim, :depth],(-1,1))

            y_mean = np.mean(y_sample)
            y_var = (y_sample - y_mean)*X_sample**0
            numer = (np.dot((X_sample-X_mean).T,y_var))
            denom = np.sqrt(np.sum(np.dot((X_sample-X_mean).T,(X_sample-X_mean))))*np.sqrt(np.dot((y_sample - y_mean).T,(y_sample - y_mean)))
            filter1 += np.abs(np.diag((numer/denom)[:,0]))
    filter1 /= np.linalg.norm(filter1)
    filter2 = 1*(np.abs(filter1) > 0.0001 )
    params, vgrad, d_rms = init_dnn_parameters(n, acts,gamma) #,np.abs(filter1))
    #alpha /= np.linalg.norm(np.abs(filter1)) # Normalize alpha to match weight adjustment 
    # Experiment 
    
    X_test = X[cv_groups[0][1],:].T 
    y_test = y[cv_groups[0][1],:].T
    print("Experiment [{}] - Eps = {}, Alph = {:3.2f}, Decay = {:3.2f}, lambda={:3.2f}".format(j, epsilon, alpha,alph_decays[j], _lambda))
    print("k = {}, |X| = {}, max(i) = {}".format( k, X_test.shape[0], iterations))
    #print("Keep Prob = {}%, gamma = {}".format(keep_prob*100, gamma))
    print("Network Size {}".format(n))
    print("Network Activation{}".format(acts))
    cost = []
    tcost=[]
    print("Mini-Batch : [{}], Mini-Batch Size [{}]".format(batch_processing, base_batch_size))
    print("Measuring Cost for [Training Set]",end="")
    A, cache, params = forward_dnn_propagation(X_train, params)
    cost.append(np.mean(get_dnn_cost(A, y_train)))
    print(",[Dev. Set]")
    A2, vectors, _ = forward_dnn_propagation(X_test, params)
    tcost.append(get_dnn_cost(A2, y_test))
    print("Pre-Training Cost")
    print("i = {:3d}, trc = {:3.2f}, tsc={:3.2f}".format(-1,cost[-1],tcost[-1]))
    print(" active alpha = {:.2E}".format(alpha))        
    
    for i in range(iterations):
        if batch_processing:
            #batch_power = np.random.randint(0,int(np.log2(2048/base_batch_size)))
            batch_size = base_batch_size #* 2 ** batch_power
            #print("Epoch [{}], batch size [{}] [pwr {}], Training".format(i, batch_size, batch_power))
            grads, params, vgrad, d_rms = batch_back_propagation(X_train, 
                                                   y_train, 
                                                   params, 
                                                   cache, 
                                                   alpha * ( batch_size/2048),
                                                   _lambda, 
                                                   keep_prob,                                                  
                                                   batch_size,
                                                   beta **( batch_size/2048),
                                                   vgrad,
                                                   d_rms)
            print("Epoch [{}], Evaluating, [Training] ".format(i),end="")
            A, cache, params = forward_dnn_propagation(X_train, params)
            cost.append(np.mean(get_dnn_cost(A, y_train)))
            print(" Evaluating, [Dev] ")
            A2, vectors, _ = forward_dnn_propagation(X_test, params)
            tcost.append(get_dnn_cost(A2, y_test))
            #batch_size *= 2
        else:
            A, cache = forward_dnn_propagation(X_train, params)
            cost.append(get_dnn_cost(A, y_train))
            grads, params= back_dnn_propagation(X_train, 
                                                y_train, 
                                                params, 
                                                cache, 
                                                alpha,
                                                _lambda, 
                                                keep_prob)
            A2, vectors = forward_dnn_propagation(X_test, params)
            tcost.append(get_dnn_cost(A2, y_test))
        
        if alpha*np.abs(np.linalg.norm(grads["dW"+str(L)])) < break_tol:
            print("Reached Change Tolerance")
            break
        if i % 1 == 0:
            alpha *= (1-alpha) #alph_decays[j]
            print("---------------------------------------------------------------")
            print("i = {:3d}, trc = {:3.2f}, tsc={:3.2f}, |dWL|_L = {:.2E}".format(i,cost[-1],
                                                                   tcost[-1], 
                                                                   alpha*np.abs(np.linalg.norm(grads["dW"+str(L)]))))
            print(" active alpha = {:.2E}".format(alpha))
            if 1==1:
                print("Number Matching (Dev. Set)")
                for num in range(10):
                    y_hat = A2[num,:] > 0.5
                    y_star = y_test[num,:]
                    matched = np.sum((1-np.abs(y_star-y_hat))*y_star)
                    tp = np.sum((y_hat == y_star) * y_star * 1)
                    tn = np.sum((y_hat == y_star)* (1-y_star) * 1)
                    fp = np.sum((y_hat == (1-y_star))*(1-y_star)*1)
                    fn = np.sum((y_hat == (1-y_star))*y_star*1)
                    distance = np.linalg.norm((y_star - A2[num,:])*y_star)
                    m_test = sum(y_test[num,:]==1)
                    y_size = y_test.shape[1]
                    pct = matched/m_test
                    print("[{}] Dst {:5.2f}".format(num, distance ), end='')
                    print(" T+ve {: >6d}/{: <6d}, T-ve {: >6d}/{: <6d}, F+ve {: >6d}, F-ve {: >6d}".format(int(tp), 
                                                                                int(np.sum(y_star)),
                                                                                int(tn),
                                                                                int(np.sum((1-y_star))),
                                                                                int(fp),
                                                                                int(fn)))
                print("---------------------------------------------------------------")
    etscost.append(tcost[-1])
    etrcost.append(cost[-1])

    
    # Prepare Data For submission
print("Preparing Data for submission")
X_test = get_features(test_raw)
print("Running Test Data On Model")
A2, vectors, _ = forward_dnn_propagation(X_test.T, params)
print("Output Vector Shape {}".format(A2.shape))

data = np.clip(A2.T, 0,1)
data = data.argmax(axis=1)
#data = np.reshape(data,(-1,1))
print(data.shape)
#data[len(data),0] = 0 # Add missing entry 
data = np.reshape(data,(-1,1))

print("Prepared Output Vector Shape {}".format(data.shape))
index = np.reshape(np.arange(1, data.shape[0]+1),(-1,1))
s1 = pd.Series(data[:,0], index=index[:,0])
s0 = pd.Series(index[:,0])
df = pd.DataFrame(data = s1, index=index[:,0])
df.index.name = 'ImageId'
df.columns = ['Label']
df.replace([np.inf, -np.inf, np.nan], 0)
df = df.astype(int)
file_name = "deep_nn.csv"
print("Saving Data to [{}]".format(file_name))
df.to_csv(file_name, sep=',')
print("========= End ===========")
