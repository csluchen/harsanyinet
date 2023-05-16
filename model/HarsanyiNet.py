import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-30   # pow(EPS, 1/(kernel_size ** 2)) < THRESHOLD_EPS (By default, kernel_size=3).
THRESHOLD_EPS = 1e-3

class STEFunction(torch.autograd.Function):
    '''
       implement Straight-Through Estimator (STE) to train the parameter \tau_u^{(l)} in paper, where
       forward:  1(input>0)
       backward: beta * exp(-input) / (1 + exp(-input))^2
    '''
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
        grad = ctx.beta * grad_input * ctx.slope * torch.exp(-ctx.slope * input_)/((torch.exp(-ctx.slope * input_)+1)**2 + EPS)
        return grad,None

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x, beta=1):
        output = STEFunction.apply(x, beta)
        return output

def conv3x3(in_channels:int, out_channels:int, 
            stride:int=1, padding:int=1)-> nn.Conv2d:
    return nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size = 3,
            stride=stride,
            padding=padding,
            bias=False)


class HarsanyiBlock(nn.Module):
    def __init__(
            self,
            conv_size: int,
            in_channels: int,
            out_channels: int,
            beta: int = 1000,
            gamma: float = 1.,
            threshold_t: float = 0.,
            device: str = 'cuda:0',
            comparable_DNN: bool = False,
    )->None:
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels, stride=3, padding=0)
        self.v = nn.Linear(conv_size*3, conv_size*3,bias=False)  # parameter \tau in paper, which used to select children nodes
        self.conv_size = conv_size
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.threshold_t = threshold_t
        self.comparable_DNN = comparable_DNN
        
        self.relu = nn.ReLU(inplace=False)
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), stride=3)
        self.fold = torch.nn.Fold(output_size=(conv_size, conv_size), kernel_size=(conv_size, conv_size), stride=conv_size)
        self.sigmoid = torch.sigmoid
        self.ste = StraightThroughEstimator()

        self._init_weights()

        # whether to use a traditional DNN with comparable size
        if self.comparable_DNN:
            self.conv_comparable_DNN = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=3, stride=1, padding=1, bias=False)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, x:Tensor) -> Tensor:
        if self.comparable_DNN: # Traditional DNN with comparable size
            x = self.conv_comparable_DNN(x)
            delta = torch.zeros(x.shape[0], self.conv_size, self.conv_size).to(self.device)
        else:   # HarsanyiNet
            x, delta = self._layer(x)

        output = self.relu(x)

        return output, delta

    def _layer(self, x: Tensor) -> Tensor:
        
        x_enlarge = self._extend_layer(x)
        x_enlarge = x_enlarge * self.ste(self.v.weight, beta=self.beta)
        x = self.conv(x_enlarge)  # Linear operation on children nodes

        delta = self._get_trigger_value(x_enlarge)
        delta = delta.unsqueeze(dim=1)
        x = x * delta    # AND operation
        
        return x, delta

    def _extend_layer(self, x: Tensor) -> Tensor:
        # extend x from (batch_size, channel, conv_size, conv_size) to (batch_size, channel, conv_size*3, conv_size*3)
        # first, padding x with 0
        batch_size, channel_num, conv_size = x.shape[0], x.shape[1], x.shape[2]
        p1d = (1, 1, 1, 1)
        x = F.pad(x, p1d)

        # second, gather x twice
        indice = torch.tensor([int(i/3) + i % 3 for i in range(0, conv_size*3)]).to(self.device)  # generate [0,1,2,1,2,3,2,3,4,...]
        row_indice = indice.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, channel_num, conv_size+2, 1)
        col_indice = indice.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(batch_size, channel_num, 1, conv_size*3)
        x = torch.gather(x, 3, row_indice)
        x = torch.gather(x, 2, col_indice)
        return x

    def _get_trigger_value(self, input_en: Tensor) -> Tensor:
        '''
            Each element of delta is in [0,1), which refers to a unit is activated (>0) or not (=0).
            If all children nodes are activated, then the activation score of the unit can pass through the AND operation,
            otherwise, if any children node is not activated, then the activation score of the unit is strictly equal to 0.
            Children nodes can be selected by parameter v.
        '''
        threshold_t = self.threshold_t
        gamma = self.gamma
        v = self.v.weight
        
        input_norm = torch.norm(input_en, p=1, dim=1)  # l1 norm of all channels
        delta_en = torch.tanh(gamma * (input_norm - threshold_t))
        delta_en = delta_en.unsqueeze(dim=1)
        delta_unfold = self.unfold(delta_en)

        v_is_child = self.ste(v, beta=1)  # change backward to sigmoid
        v_is_child_unfold = self.unfold(v_is_child.unsqueeze(0).unsqueeze(0))

        # calculate the geometric mean of children nodes
        delta_prod = torch.exp( (torch.sum(torch.log(delta_unfold+EPS) * v_is_child_unfold, 1)) / (torch.sum(v_is_child_unfold,1) + EPS) )
        # force EPS to 0, if any children node is not activated, then the activation score is strictly equal to 0.
        ZEROS = torch.zeros_like(delta_prod)
        delta_prod = torch.where(delta_prod > THRESHOLD_EPS, delta_prod, ZEROS)
        # find the position where all children nodes are zeros in parameter v
        zero_position = torch.sum(v_is_child_unfold, 1)/(torch.sum(v_is_child_unfold,1) + EPS)
        delta_prod = delta_prod * zero_position

        delta_prod = delta_prod.unsqueeze(dim=2)
        delta_fold = self.fold(delta_prod)
        delta = delta_fold.reshape(-1, self.conv_size, self.conv_size)

        return delta



class HarsanyiNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_layers: int = 1,
        channel_extend: int = 512,
        beta: int = 1000,
        gamma: int = 1,
        conv_size: int = 16,
        fc_size: int = 16,
        threshold_t: float = 0.,
        device: str = 'cuda:0',
        in_channels: int = 3,
        comparable_DNN: bool = False
    ) -> None:

        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.channels = channel_extend
        self.device = device
        self.conv_size = conv_size

        self.conv1 = conv3x3(in_channels=in_channels, out_channels=channel_extend)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu =nn.ReLU(inplace=False)

        self.HarsanyiBlocks = []
        self.fc = []

        for i in range(num_layers):
            self.HarsanyiBlocks.append(HarsanyiBlock(conv_size=conv_size,
                                                     in_channels=channel_extend,
                                                     out_channels=channel_extend,
                                                     beta=beta,
                                                     gamma=gamma,
                                                     threshold_t=threshold_t,
                                                     device=device,
                                                     comparable_DNN=comparable_DNN))
            self.fc.append(nn.Linear(channel_extend*conv_size*conv_size, fc_size, bias=False))

        self.HarsanyiBlocks = nn.ModuleList(self.HarsanyiBlocks)
        self.fc = nn.ModuleList(self.fc)
        self.fc_final = nn.Linear(fc_size, num_classes, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x:Tensor) -> Tensor:
        x = self._get_z0(x)

        hidden_y = None 
        for layer in range(self.num_layers):
            x, _ = self.HarsanyiBlocks[layer](x)
            y = self.fc[layer](torch.flatten(x,1))
            if hidden_y == None:
                hidden_y = y
            else:
                hidden_y += y

        output = self.fc_final(hidden_y)

        return output

    def _get_z0(self, x:Tensor):
        # If the input is the original image x, not z0
        if x.shape[1] != self.channels:
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.relu(x)
        return x

    # Used to calculate Shapley value
    def _get_value(self, x:Tensor):
        x = self._get_z0(x)

        hidden_y = None
        deltas, zs, ys = [], [], []
        for layer in range(self.num_layers):
            x, delta = self.HarsanyiBlocks[layer](x)
            y = self.fc[layer](torch.flatten(x,1))
            deltas.append(delta)
            zs.append(x)
            ys.append(y)
            if hidden_y == None:
                hidden_y = y
            else:
                hidden_y += y

        output = self.fc_final(hidden_y)

        return output, ys, zs, deltas

