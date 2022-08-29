"""Based on dsnmf(2017) Only Main function, and it's related methods are developed, the original dsnmf code is available " \
"at https://github.com/trigeorgis/Deep-Semi-NMF"" """

from __future__ import print_function
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from scipy.sparse.linalg import svds
import statistics

from Utils.metrics_evaluation import evaluate_nmi, accuracy
from Utils.utils import init_kmeans

relu = lambda x: 0.5 * (x + abs(x))


def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)


def appr_seminmf(M, r):
    """
        Approximate Semi-NMF factorisation. 
        
        Parameters
        ----------
        M: array-like, shape=(n_features, n_samples)
        r: number of components to keep during factorisation
    """

    if r < 2:
        raise ValueError("The number of components (r) has to be >=2.")

    A, S, B = svds(M, r - 1)
    S = np.diag(S)
    A = np.dot(A, S)

    m, n = M.shape

    for i in range(r - 1):
        if B[i, :].min() < (-B[i, :]).min():
            B[i, :] = -B[i, :]
            A[:, i] = -A[:, i]

    if r == 2:
        U = np.concatenate([A, -A], axis=1)
    else:
        An = -np.sum(A, 1).reshape(A.shape[0], 1)
        U = np.concatenate([A, An], 1)

    V = np.concatenate([B, np.zeros((1, n))], 0)

    if r >= 3:
        V -= np.minimum(0, B.min(0))
    else:
        V -= np.minimum(0, B)

    return U, V


def adam(loss, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """Adam updates

    Adam updates implemented as in [1]_.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float
        Learning rate
    beta_1 : float
        Exponential decay rate for the first moment estimates.
    beta_2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """

    all_grads = theano.grad(loss, params)
    t_prev = theano.shared(floatX(0.))
    updates = OrderedDict()

    for param, g_t in zip(params, all_grads):
        m_prev = theano.shared(param.get_value() * 0.)
        v_prev = theano.shared(param.get_value() * 0.)
        t = t_prev + 1
        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * g_t ** 2
        a_t = learning_rate * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates


def init_weights(X, num_components, svd_init=True):
    if svd_init:
        return appr_seminmf(X, num_components)

    Z = 0.08 * np.random.rand(X.shape[0], num_components)
    H = 0.08 * np.random.rand(num_components, X.shape[1])

    return Z, H


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

rng = RandomStreams()


def dropout(x, p=0):
    if p == 0:
        return x
    else:
        p = 1 - p
        x /= p

        return x * rng.binomial(x.shape, p=p, dtype=theano.config.floatX)


class DSNMF(object):

    def __init__(self, data, layers, verbose=False, l1_norms=[], pretrain=True, learning_rate=1e-3):
        """
        Parameters
        ----------
        :param data: array-like, shape=(n_samples, n_features)
        :param layers: list, shape=(n_layers) containing the size of each of the layers
        :param verbose: boolean
        :param l1_norms: list, shape=(n_layers) the l1-weighting of each of the layers
        :param pretrain: pretrain layers using svd
        """
        H = data.T

        assert len(layers) > 0, "You have to provide a positive number of layers."

        params = []
        for i, l in enumerate(layers, start=1):

            Z, H = init_weights(H, l, svd_init=pretrain)
            params.append(theano.shared(floatX(Z), name='Z_%d' % (i)))

        params.append(theano.shared(floatX(H), name='H_%d' % len(layers)))

        self.params = params
        self.layers = layers

        cost = ((data.T - self.get_h(-1)) ** 2).sum()

        for norm, param in zip(l1_norms, params):
            cost += ((abs(param)) * norm).sum()

        H = relu(self.params[-1])

        updates = adam(cost, params, learning_rate=learning_rate)

        self.cost = cost
        self.train_fun = theano.function([], cost, updates=updates)
        self.get_features = theano.function([], H)

        self.get_reconstruction = theano.function([], self.get_h(-1))

        ## Add on
        self.recon_mat = np.identity(theano.function([], self.params[0])().shape[0])
        for i in range(len(self.params)):
            self.recon_mat = self.recon_mat @ theano.function([], self.params[i])()
        self.z1 = theano.function([], self.params[0])
        self.z2 = theano.function([], self.params[1])

    def finetune_features(self):

        updates = adam(self.cost, self.params[-1:])
        self.train_fun = theano.function([], self.cost, updates=updates)

    def get_param_values(self):
        return [p.get_value() for p in self.params]

    def set_param_values(self, values):
        params = self.params

        if len(params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                             (len(values), len(params)))

        for p, v in zip(params, values):
            if p.get_value().shape[0] != v.shape[0]:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    def get_h(self, layer_num, have_dropout=False):
        h = relu(self.params[-1])

        if have_dropout:
            h = dropout(h, p=.1)

        for z in reversed(self.params[1:-1][:]):
            h = relu(z.dot(h))

        if layer_num == -1:
            h = self.params[0].dot(h)

        return h


def run_model(matImg, y, k1_list, k2_list, maxiter_kmeans):
    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    data = matImg / norma
    # data = matImg / 255.0

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    for k1 in k1_list:

        for k2 in k2_list:

            dsnmf = DSNMF(data, layers=(k1, k2))

            # Train model
            n = data.shape[0]
            recon_matrix = dsnmf.z1() @ dsnmf.z2() @ dsnmf.get_features()
            E = [((np.linalg.norm(data.T - recon_matrix, 'fro')) ** 2) / n]
            epoch = 0
            err = 1
            for epoch in range(1000):
                # while(epoch <= 300 and err >= 1e-06):
                residual = float(dsnmf.train_fun())

                # calculate convergence criteria
                epoch += 1
                recon_matrix = dsnmf.z1() @ dsnmf.z2() @ dsnmf.get_features()
                a = np.linalg.norm(data.T - recon_matrix, 'fro')
                recon_reeor = (a ** 2) / n
                E.append(recon_reeor)
                err = (E[epoch - 1] - E[epoch]) / max(1, E[epoch - 1])

            fea = dsnmf.get_features().T  # this is the last layers features i.e. h_2

            # Kmeans task
            lst_acc = []
            lst_nmi = []

            for i in range(1, maxiter_kmeans):
                pred = kmeans.fit_predict(fea)

                ## NMI
                nmi = 100 * evaluate_nmi(y, pred)

                ## ACC
                acc = 100 * accuracy(y, pred)

                lst_acc.append(acc)
                lst_nmi.append(nmi)
                ## End for

            print(
                "**********************************************************************************************************")
            print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs\n")
            print(f"k1 = {k1} : k2 = {k2} : best max_acc = {max(lst_acc)} ")
            print(f"k1 = {k1} : k2 = {k2} : best max_nmi = {max(lst_nmi)} ")
            print("\n\nThe results of running the Kmeans method 20 times and the report of average of 20 runs\n")
            print(f"k1 = {k1} :  k2 = {k2} : avg_acc = {statistics.mean(lst_acc)} ")
            print(f"k1 = {k1} :  k2 = {k2} : avg_nmi = {statistics.mean(lst_nmi)}  ")
            print(
                "**********************************************************************************************************")

    print(f"done!")
