

import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from .maml import MAMLLayer

class LearningRateLearner(nn.Module):
    def __init__(
        self,
        input_size = 2,
        hidden_size = 20,
        num_layers = 1
    ):
        super(LearningRateLearner, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers
        )
    
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.lstm.weight.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.lstm.weight.device)
        return (h_0, c_0)

    def forward(self, x):
        batch_size = x.size(1)  # Assuming x is of shape (seq_len, batch, input_size)
        hidden = self.init_hidden(batch_size)
        out, _ = self.lstm(x, hidden)
        return out[-1]  # Return the last time step's output

class MetaAdam(MetaModel):
    def __init__(self, inner_param,outer_param,feat_dim, **kwargs):
        super(MetaAdam, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.inner_param = inner_param
        self.outer_param = outer_param
        self.lstm = LearningRateLearner() # cause None grad
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def freeze_lstm(self):
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_lstm(self):
        for param in self.lstm.parameters():
            param.requires_grad = True

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
        lr_lstm = torch.tensor(self.outer_param["lstm_lr"], device=self.device)
        lstm_param = self.lstm.parameters()
        lstm_grad = torch.autograd.grad(loss, lstm_param, create_graph=True)
        for k,weight in enumerate(lstm_param):
            weight.data = weight.data - lr_lstm * lstm_grad[k].data
        
        acc = accuracy(output, query_target.contiguous().view(-1))
        
        
        return output, acc, loss
    
    def set_forward_adaptation(self, support_set, support_target):
        # Inner loop
        # "MetaMomentumInner" in paper
        # TODO:replace for with torch matrix
        self.freeze_lstm()
        
        lr = torch.tensor(self.inner_param["inner_lr"], device = self.device)
        fast_parameters = [p for p in self.parameters() if p.requires_grad]
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.fast = parameter
        m = [torch.zeros_like(p) for p in fast_parameters]
        
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
            
            # for debug
            '''
            model_state_dict = self.state_dict()
            for i,grad in enumerate(grad):
                if grad is None:
                    param_name = list(model_state_dict.keys())[i]
                    print(f"grad for param '{param_name}' is None ")
            '''
            k = 0
            for weight in self.parameters():
                if weight.requires_grad:
                    theta_m.append(weight.fast - lr * m[k])
                    theta_g.append(weight.fast - lr * grad[k])
                    k += 1
            
            k = 0
            for weight in self.parameters():
                if weight.requires_grad:
                    weight.fast = theta_m[k]
                    k += 1
            output_m = self.forward_output(support_set)
            loss_m = self.loss_func(output_m, support_target)
            
            
            k = 0
            for weight in self.parameters():
                if weight.requires_grad:
                    weight.fast = theta_g[k]
                    k += 1
            output_g = self.forward_output(support_set)
            loss_g = self.loss_func(output_g, support_target)
            
            delta_loss_m = loss_m - loss_fast
            delta_loss_g = loss_g - loss_fast
            
            tmp = F.softmax(torch.stack([delta_loss_g, delta_loss_m]), dim=0)
            
            eta = []
            for g, m_val in zip(grad, m):
                input_tensor = torch.stack([tmp[0] * g, tmp[1] * m_val], dim=0).unsqueeze(1)  # Shape (seq_len=2, batch=1, input_size)
                eta.append(self.lstm(input_tensor))

            m = [tmp[0] * g + tmp[1] * m_val for g, m_val in zip(grad, m)]
            
            #fast_parameters -= eta*m
            
            k = 0
            for weight in self.parameters():
                if weight.requires_grad:
                    weight.fast = fast_parameters[k] - eta[k] * m[k]
                    k += 1
            
            fast_parameters = []
            for weight in self.parameters():
                if weight.requires_grad:
                    fast_parameters.append(weight.fast)

            k = 0
        
        self.unfreeze_lstm()
        
        return fast_parameters

    