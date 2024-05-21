

import torch
from torch import nn
import numpy as np

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from .maml import MAMLLayer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class LearningRateLearner(nn.Module):
    def __init__(
        self,
        input_size = 2,
        hidden_size = 20,
        num_layers = 1
    ):
        super(LearningRateLearner, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers
        )
    
    def forward(self, x):
        out = self.lstm(*x)
        return out

class MetaAdam(MetaModel):
    def __init__(self, inner_param,outer_param,feat_dim, **kwargs):
        super(MetaAdam, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.inner_param = inner_param
        self.outer_param = outer_param
        self.LearningRateLearner = LearningRateLearner()
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        # Outer loop, return output acc loss
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            fast_parameters = self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        
        # update lstm
        lr_lstm = self.outer_param["lstm_lr"]
        lstm_param = self.lstm.parameters()
        lstm_grad = torch.autograd.grad(loss, lstm_param, create_graph=True)
        for k,weight in enumerate(lstm_param):
            weight.data = weight.data - lr_lstm * lstm_grad[k].data
        
        acc = accuracy(output, query_target.contiguous().view(-1))
        
        
        return output, acc, loss
    
    def set_forward_adaptation(self, support_set, support_target):
        # Inner loop
        # "MetaMomentumInner" in paper
        # TODO:replace for with np matrix
        lr = self.inner_param["inner_lr"]
        fast_parameters = list(self.parameters())
        m = [0 for _ in fast_parameters]
        for parameter in self.parameters():
            parameter.fast = parameter
        
        self.emb_func.train()
        self.classifier.train()
        for t in range(
            self.inner_param["train_iter"]
            if self.training 
            else self.inner_param["test_iter"]
        ):
            output = self.forward_output(support_set)
            loss_fast = self.loss_func(output,support_target)
            grad = torch.autograd.grad(loss_fast, fast_parameters, create_graph=True)
            #fast_parameters = []
            theta_m = []
            theta_g = []
            
            
            for k, weight in enumerate(self.parameters()):
                theta_m.append(weight.fast - lr * m[k])
                theta_g.append(weight.fast - lr * grad[k])
            
            for k, weight in enumerate(self.parameters()):
                weight.fast = theta_m[k]
            output_m = self.forward_output(support_set)
            loss_m = self.loss_func(output_m, support_target)
            
            for k, weight in enumerate(self.parameters()):
                weight.fast = theta_g[k]
            output_g = self.forward_output(support_set)
            loss_g = self.loss_func(output_g, support_target)
            
            delta_loss_m = loss_m - loss_fast
            delta_loss_g = loss_g - loss_fast
            
            tmp = softmax([delta_loss_g,delta_loss_m])
            
            eta = []
            for k, _ in enumerate(theta_m):
                eta.append(self.lstm(tmp[0] * grad[k], tmp[1] * m[k]))
                m[k] = tmp[0] * grad[k] + tmp[1] * m[k]
            
            for k,weight in enumerate(self.parameters()):
                weight.fast = fast_parameters[k] - eta[k] * m[k]
                fast_parameters[k] = weight.fast
        
        return fast_parameters

    