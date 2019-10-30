
import numpy as np
import numpy

import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
import math

##############################################################################
################implemenations of the basic functionality###################
##############################################################################

def error(y,pred):
    return torch.mean(torch.ne(y,pred))


def accuracy(y,pred):
    return torch.mean(torch.eq(y,pred))

def clip(x,min,max):
    return torch.clamp(x,min,max)


def floor(x):
    return torch.floor(x).int()


def ceil(x):
    return torch.ceil(x).int()

def sigmoid(x):
    return F.sigmoid(x)


def relu(x):
    return F.relu(x)

def leaky_relu(x,negative_slope):
    return F.leaky_relu(x,negative_slope=negative_slope)


def softplus(x):
    return F.softplus(x)

def softmax(x):
    return F.softmax(x)


def tanh(x):
    return F.tanh(x)

def l2_norm(x,epsilon = 0.00001):
    square_sum = torch.sum(torch.pow(x,exponent=2))
    norm = torch.sqrt(torch.add(square_sum,epsilon))
    return norm

def l2_norm_2d(x, epsilon = 0.00001):
    square_sum = torch.sum(torch.pow(x,exponent=2))
    norm = torch.mean(torch.sqrt(torch.add(square_sum,epsilon)))

    return norm

# we assume afa=beta
def neg_likelihood_gamma(x, afa ,epsilon = 0.00001):
    #norm = T.maximum(x, epsilon)
    norm = torch.add(x,epsilon)
    neg_likelihood = -(afa-1)*torch.log(norm)+afa*norm
    return  torch.mean(neg_likelihood)

# KL(lambda_t||lambda=1)
def kl_exponential(x, epsilon = 0.00001):
    norm = torch.add(x,epsilon)
    kl = -torch.log(norm)+norm
    return  torch.mean(kl)
 
def likelihood(x,y, epsilon = 0.00001):
    norm = torch.add(x,epsilon)
    kl = -torch.log(norm)+norm*y
    return  0.25*torch.mean(kl)



def shape(x):

    return x.shape

def reshape(x, shape):
    y = torch.reshape(x, shape).float()
    return y




def Linear_Function(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input,weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret



##############################################################################
################implemenations of the Neuro Networks##########################
##############################################################################



class Dense(Module):

    __constants__ = ['bias', 'features', 'features']
    def __init__(self, in_features, out_features, bias = True):
        super(Dense,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fam_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fam_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        return Linear_Function(input, self.weight,self.bias)
    '''def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features,self.out_features, self.bias is not None)'''




###   Note: this class can work, but when the number of layers > 1, it cannot work well,
###   I'm still working on imporve it.
###   If you need to use RNN, please choose nn.LSTM, if you have some ideas about how to improve this class,
###   please connect me! Thanks!

class NaiveLSTM(Module):

    """Naive LSTM like nn.LSTM"""

    def __init__(self, input_size, hidden_size):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # forget gate
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # output gate
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        # cell
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, state = None):
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """
        #inputs = inputs.cuda()
        batch_size, seq_size, _ = inputs.size()
        #print(inputs)
        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t().cuda()
            c_t = torch.zeros(1, self.hidden_size).t().cuda()
        else:
            #print("===============+++++++++++++++++++++=================")
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        #seq_size = 1
        for t in range(seq_size):
            x = inputs[:, t, :].t()
            # input gate

            #print(self.w_ii)
            i = torch.sigmoid(torch.matmul(self.w_ii,x) + self.b_ii + torch.matmul(self.w_hi , h_t) +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(torch.matmul(self.w_if , x) + self.b_if + torch.matmul(self.w_hf, h_t) +
                              self.b_hf)
            # cell
            g = torch.tanh(torch.matmul(self.w_ig , x) + self.b_ig + torch.matmul(self.w_hg , h_t)
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(torch.matmul(self.w_io , x) + self.b_io + torch.matmul(self.w_ho , h_t) +
                              self.b_ho)
            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_next_t, c_next_t)













