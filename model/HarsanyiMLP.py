import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

EPS = 1e-30

# functions of initializing layers
def init_layer(layer,weight_init):
    '''Initialize a Linear or Convolutional layer.'''
    if weight_init == 'xavier':
        nn.init.xavier_uniform_(layer.weight)
    elif weight_init == 'uniform':
        nn.init.uniform_(layer.weight)
    if hasattr(layer,'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class STEFunction(torch.autograd.Function):
    '''backward function for F'''
    @staticmethod
    def forward(ctx, input_, beta=1, slope=1):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.beta = beta
        out = (input_>0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = ctx.beta * grad_input * ctx.slope * torch.exp(-ctx.slope * input_)/((torch.exp(-ctx.slope * input_)+1)**2 + EPS )
        return grad, None

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator,self).__init__()

    def forward(self, x, beta=1):
        output = STEFunction.apply(x, beta)
        return output

class HarsanyiBlock(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 device:str='cuda:0',
                 beta: int=10,
                 gamma: int = 100, 
                 initial_V:float = 1.,
                 act_ratio: float = 0.1,
                 comparable_DNN: bool=False,
                 weight_init: str='xavier',
                 )->None:

        super(HarsanyiBlock,self).__init__()
        
        self.beta = beta
        self.gamma = gamma
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.comparable_DNN = comparable_DNN
        
        # set parameters used to select children nodes
        self.initial_V = initial_V
        self.act_ratio = act_ratio
        self.v = nn.Linear(input_dim, output_dim, bias=False)  # self.v.weight ->paras. tau in the paper
        self.init_v()
        
        if not comparable_DNN:
            self.fc = nn.Linear(input_dim, output_dim, bias=False)    
            self.ste = StraightThroughEstimator()
            self.activation = nn.functional.relu
            self.init_weights(weight_init)
            
        if comparable_DNN:
            self.conv_comparable_DNN = nn.Linear(input_dim,output_dim)
            self.activation = nn.functional.relu
            init_layer(self.conv_comparable_DNN, weight_init)

    def init_v(self):
        self.v.weight.data[:,:] = -self.initial_V
        # randomly choose certain number of input variables as initial children nodes
        # if the value >0, it means the corresponding node is the children node;
        # otherwise, if the value <=0, it means the corresoinding node is not the children node.
        for i in range(self.output_dim):
            act_index = np.random.choice(self.input_dim, int(self.act_ratio*self.input_dim), replace=False)
            self.v.weight.data[i][act_index] = self.initial_V

    def init_weights(self, weight_init:str) -> None:
        init_layer(self.fc, weight_init)
    
    def forward(self, x:Tensor) -> Tensor:
        if self.comparable_DNN:
            x = self.conv_comparable_DNN(x)
            delta = torch.zeros(x.shape[0], self.output_dim).to(self.device)
        else:
            x, delta = self._layer(x) 
        if self.activation !=None:
            out = self.activation(x)
        return out, delta

    def _layer(self, x: Tensor) -> Tensor:
        
        x_enlarge = self._extend_layer(x)
        x_enlarge = x_enlarge * self.ste(self.v.weight, beta=self.beta)
        
        weight = self.fc.weight * self.ste(self.v.weight, beta=self.beta)
        x = torch.matmul(x, weight.transpose(0,1))     
        delta = self._get_trigger_value(x_enlarge)
        x = x * delta    # AND operation
        return x, delta

    def _extend_layer(self, x:Tensor) -> Tensor:
        x = x.unsqueeze(1).repeat(1,self.output_dim, 1)
        return x

    def _get_trigger_value(self, input:Tensor) -> Tensor:
        '''
            Each element of delta is in [0,1), which refers to a unit is activated (>0) or not (=0).
            If all children nodes are activated, then the activation score of the unit can pass through the AND operation,
            otherwise, if any children node is not activated, then the activation score of the unit is strictly equal to 0.
            Children nodes can be selected by parameter v.
        '''
        v = self.v.weight
        device = input.device
        v_is_child = self.ste(v, beta=self.beta)
        ones = torch.ones(v_is_child.shape)
        ones = ones.to(device)
        v_is_not_child =ones - v_is_child
        delta_en = torch.tanh(self.gamma* torch.abs(input))
        delta_en = delta_en*v_is_child + v_is_not_child
        delta_prod = torch.prod(delta_en,dim=-1)
        
        return delta_prod
    


class HarsanyiNet(nn.Module):

    def __init__(self, 
                 input_dim:int=12,
                 num_classes:int=2,
                 num_layers:int=1,
                 hidden_dim:int=100,
                 beta:int =10,
                 gamma:int=100,
                 initial_V: float = 1.,
                 act_ratio:float = 0.1,
                 device:str = 'cuda:0',
                 comparable_DNN: bool=False,
                 weight_init: str='xavier'
                 )->None:
        super(HarsanyiNet,self).__init__()
        
        self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.relu = nn.ReLU()
        assert num_layers >0, "should have at least one Harsanyi Layer"
        
        self.HarsanyiBlocks = nn.ModuleList()
        self.fc = nn.ModuleList()
        # to ensure the the firt hidden dim to have at least one children node
        first_layer_act_ratio = max((1+EPS)/self.input_dim, act_ratio) 

        for layer_index in range(num_layers):
            if layer_index == 0:
                harsanyiblock = HarsanyiBlock(input_dim, 
                                               hidden_dim,
                                               device = device,
                                               beta = beta,
                                               gamma = gamma,
                                               initial_V=initial_V, 
                                               act_ratio=first_layer_act_ratio, 
                                               comparable_DNN=comparable_DNN,
                                               weight_init=weight_init)
                self.fc.append(nn.Linear(hidden_dim, num_classes, bias=False))
                
            else:
                harsanyiblock = HarsanyiBlock(hidden_dim, 
                                               hidden_dim, 
                                               device=device,
                                               beta = beta,
                                               gamma = gamma,
                                               initial_V=initial_V, 
                                               act_ratio=act_ratio, 
                                               comparable_DNN=comparable_DNN,
                                               weight_init=weight_init)
                self.fc.append(nn.Linear(hidden_dim, num_classes, bias=False))
            self.HarsanyiBlocks.append(harsanyiblock)
        
        self.init_weights()

    def init_weights(self):
        for layer in self.fc:
            init_layer(layer,'xavier')

    def forward(self,x:Tensor) -> Tensor:
        if len(x.shape)>2:
            x = x.squeeze(1)
        
        hidden_y = None
            
        for layer_index in range(self.num_layers):
            layer= self.HarsanyiBlocks[layer_index]
            x,_= layer(x)
            y_ = self.fc[layer_index](x)
            if hidden_y == None:
                hidden_y = y_
            else:
                hidden_y += y_
        output = hidden_y
        return output
    
    def _get_value(self,x:Tensor)->Tensor:
        if len(x.shape)>2:
            x = x.squeeze(1)
        deltas = []
        zs, ys = [], []
        hidden_y = None
        for layer_index in range(self.num_layers):
            layer = self.HarsanyiBlocks[layer_index]
            x, delta = layer(x)
            y_ = self.fc[layer_index](x)
            zs.append(x)
            ys.append(y_)
            deltas.append(delta)
            if hidden_y == None:
                hidden_y = y_
            else:
                hidden_y += y_
        
        output = hidden_y
        return output, zs, ys, deltas

