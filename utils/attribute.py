import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from model.HarsanyiNet import HarsanyiNet
from model.HarsanyiMLP import HarsanyiNet as HarsanyiMLP

class HarsanyiNetAttribute(HarsanyiNet):
    '''
        Given a pre-trained HarsanyiNet to compute Shapley values of all input variables in a sample
    '''
    def __init__(
            self,
            model,
            device
    ) -> None:
        super().__init__()
        assert isinstance(model, HarsanyiNet)

        self.device = device
        self.num_layers = model.num_layers
        self.conv_size = model.conv_size
        self.all_players = np.arange(1, self.conv_size*self.conv_size+1).tolist()

        # get weights of FC layers that connected to the output
        self.w = torch.zeros(model.num_layers, model.num_classes, model.fc[0].weight.shape[1]).to(self.device)
        for layer in range(model.num_layers):
            self.w[layer] = torch.matmul(model.fc_final.weight, model.fc[layer].weight)

        # get children nodes from parameter tau (V matrix), where each element of tau refers to a child node(=1) or not(=0)
        self.v = torch.zeros(model.num_layers, model.conv_size*3, model.conv_size*3).to(self.device)
        for layer in range(self.num_layers):
            self.v[layer] = (model.HarsanyiBlocks[layer].v.weight.data > 0).float()

        # get the receptive field R of each Harsanyi unit, and all affected Harsanyi units(R=S) given a coalition S
        self.V_to_coalition, self.coalition_pos = self.get_all_coalitions(model)


    def get_all_coalitions(self, model):
        '''function to get coalition S from parameter tau, given a pre-trained HarsanyiNet
        :return V_to_coalition: the receptive field R of each Harsanyi unit from V;
                               [num_layers, conv_size, conv_size], each element is the corresponding receptive field(set)
                coalition_pos: all affected Harsanyi units(R=S) given a coalition S;
                               {coalition: positions of all affected Harsanyi units(R=S)}
        '''
        V_to_coalition = [[[set() for _ in range(self.conv_size)] for _ in range(self.conv_size)] for _ in range(self.num_layers)]
        coalition_pos = {}
        all_players = torch.from_numpy(np.array(self.all_players)).reshape(self.conv_size, self.conv_size).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
        index = model.HarsanyiBlocks[0]._extend_layer(all_players).squeeze(dim=0).squeeze(dim=0)

        for layer in range(self.num_layers):
            if layer == 0:   # children nodes of z(1) are the z(0) units
                tmp = [[set([int(element)]) if element != 0 else set() for element in index_rows] for index_rows in index]

            else:   # children nodes of z(k) are the combination of z(k-1) units, and get the combination of z(0) units
                for j in range(3*self.conv_size):
                    for k in range(3*self.conv_size):
                        if index[j,k] == 0:  # if padding
                            tmp[j][k] = set()
                        else:    # if not padding
                            tmp[j][k] = V_to_coalition[layer-1][ (index[j,k]-1)//self.conv_size ][ (index[j,k]-1)%self.conv_size ]

            # if not a child node (V[j,k] = 0), then the coalition of [j,k] position is not considered
            for j in range(3*self.conv_size):
                for k in range(3*self.conv_size):
                    if self.v[layer][j,k] == 0:
                        tmp[j][k] = set()

            # union 3*3 children coalitions to get each Harsanyi unit's receptive field
            for j in range(self.conv_size):
                for k in range(self.conv_size):

                    V_to_coalition[layer][j][k] = tmp[3*j+1][3*k+1].union(tmp[3*j][3*k], tmp[3*j][3*k+1], tmp[3*j][3*k+2], \
                                            tmp[3*j+1][3*k], tmp[3*j+1][3*k+2],\
                                            tmp[3*j+2][3*k], tmp[3*j+2][3*k+1], tmp[3*j+2][3*k+2])

                    # update coalition_pos: {coalition: positions of all affected Harsanyi units(R=S)}
                    coalition = tuple(V_to_coalition[layer][j][k])
                    if coalition not in coalition_pos.keys():
                        coalition_pos[coalition] = [[layer, j, k]]
                    else:
                        coalition_pos[coalition].append([layer, j, k])

        return V_to_coalition, coalition_pos


    def attribute(self, model, image, target_label):
        '''function to get Harsanyi interaction I(S) from coalition S, given a pre-trained HarsanyiNet and an image
        :return harsanyi: the harsanyi intearctions for all Harsanyi units
                          {S: I(S)}'''
        n, channel, height, weight = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
        assert channel == model.channels  # attribution on z(0), not image x
        model = model.double()
        image = image.double()

        # get z(l)
        y = torch.zeros(self.num_layers, channel*height*weight)
        _, _, z, _ = model._get_value(image)

        # update y(l) = z(l) * w(l)
        for layer in range(self.num_layers):
            y[layer] = torch.flatten(z[layer],1) * self.w[layer][target_label]

        # add up all channels of y(l)
        y = y.reshape(self.num_layers, channel, self.conv_size, self.conv_size).sum(dim = 1)

        # update {S: I(S)}
        coalition_pos = self.coalition_pos
        harsanyi = {coalition: 0 for coalition in coalition_pos.keys()}
        for coalition, position in coalition_pos.items():
            for pos in position:
                harsanyi[coalition] += float(y[pos[0]][pos[1]][pos[2]])

        return harsanyi


    def get_shapley(self, harsanyi=None):
        '''function to get Shapley values from Harsanyi interactions I(S), given {S: I(S)}'''
        assert harsanyi

        shapley = np.zeros(len(self.all_players))
        for coalition, value in harsanyi.items():
            if coalition != ():
                for element in coalition:
                    shapley[element-1] += value / len(coalition)

        return shapley.reshape(self.conv_size, self.conv_size)




class HarsanyiMLPAttribute(HarsanyiMLP):
    '''
        Given a pre-trained HarsanyiNet to compute Shapley values
    '''
    def __init__(
            self,
            model,
            device
    ) -> None:
        super().__init__()
        assert isinstance(model, HarsanyiMLP)

        self.device = device
        self.num_layers = model.num_layers
        self.input_dim = model.input_dim

        # get children nodes from parameters
        self.tau = []
        self.coalitions = []
        for i in range(self.num_layers):
            F = model.HarsanyiBlocks[i].v.weight
            self.tau.append((F>0.).float())
        
            if i == 0:
                self.coalitions.append(self.get_coalition(i))
            else:
                self.coalitions.append(self.get_receptive_field(self.coalitions[i-1], self.get_coalition(i)))


    def compute_harsanyi(self,model, layer, gt_label,z):
        '''function to compute the Harsanyi interaction value on each Harsanyi unit (intermediate-layer neuron)
        :param model: the trained model
        :param layer: the index of the hiddden layer
        :param gt_label: the ground truth lable, the explaination is on the classification score on the gt_label
        :param z: the neural activation score of each Harsanyi unit
        :return harsanyi: the harsanyi intearctions for all Harsanyi units'''

        weight = model.fc[layer].weight
        weight_sum=weight[gt_label]
        weight_sum = weight_sum.reshape(z[layer].size())
        harsanyi = weight_sum*z[layer]
    
        return harsanyi.squeeze(0)

    def get_coalition(self,layer_index):
        '''funtion to get coalition of children nodes from previous layer
        :param v:tensor
        :return children_set: dictionary,
                          keys: index of the harsanyi unit,
                          values: list of index of the children nodes from previous layer
        '''
        tau = self.tau[layer_index]
        size_cur = tau.shape[0]
        size_pre = tau.shape[1]

        children_set = {}
        for i in range(size_cur):
            children_set[i] = []
            for j in range(size_pre):
                if tau[i][j] == 1:
                    children_set[i].append(j)
        return children_set

    def get_receptive_field(self, RF_prev, children_set):
        '''function to get receptive field on input sample of each Harsanyi unit
        :param RF_prev: dictionary, the previous layer's receptive filed
        :param children_set: dictionary, the current layer's children set
        :return RF_cur: dictionary, the receptive field of current layer's Harsanyi unit
        '''
        RF_cur = {}
        for key in children_set.keys():
            RF_cur[key] = []
            sub_keys = children_set[key]
            for sub_key in sub_keys:
                for value in RF_prev[sub_key]:
                    if value not in RF_cur[key]:
                        RF_cur[key].append(value)
        return RF_cur
 

    def attribute(self,model,x_te, target_label):
        '''function to calculate the Harsanyi interactions for each unique input variable set'''
        model = model.double()
        x_te = x_te.double()
        y_pred, zs, ys, deltas = model._get_value(x_te)
        HarsanyiValues =  []
    
        for ind in range(self.num_layers):
            HarsanyiValues.append(self.compute_harsanyi(model,ind,target_label,zs).detach().cpu().numpy())
        return HarsanyiValues

    def get_shapley(self, harsanyi=None):
        assert harsanyi

        shapley = np.zeros(self.input_dim)
        for ind in range(self.num_layers):
            for key, value in self.coalitions[ind].items():
                harsanyi_value = harsanyi[ind][key]
                player_num = len(value)
                if player_num > 0:
                    single_value = harsanyi_value/player_num # equally distribute the HarsanyiValue to all variables in the receptive field
                for i in range(player_num):
                    shapley[value[i]] += single_value
        return shapley


