import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from .maml import MAMLLayer

# maml + dynamic weighting schema + momentum
class LearningRateLearner_MLP(nn.Module):
    def __init__(
        self,
        input_size = 2,
        output_size = 1
    ):
        super(LearningRateLearner_MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size 
        
        self.weight1 = nn.Parameter(torch.Tensor(10, self.input_size))
        self.bias1 = nn.Parameter(torch.zeros(10, requires_grad=True))
        self.weight2 = nn.Parameter(torch.Tensor(self.output_size, 10))
        self.bias2 = nn.Parameter(torch.zeros(self.output_size, requires_grad=True))
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.weight1, gain=0.02)  # Small positive values
        nn.init.uniform_(self.bias1, 0, 0)  # Small positive values
        nn.init.xavier_uniform_(self.weight2, gain=0.02)  # Small positive values
        nn.init.uniform_(self.bias2, 0, 0)  # Small positive values
        
        

    def forward(self, momentum, new_gradients):
        inputs = torch.stack((momentum, new_gradients), dim = 1)
        x = inputs.unsqueeze(1)
        
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.linear(x, self.weight1, self.bias1)
        x = F.relu(x)
        x = F.linear(x, self.weight2, self.bias2)
        # x = F.relu(x)
        
        return x

class MetaAdam3(MetaModel):
    def __init__(self, inner_param, outer_param, feat_dim, **kwargs):
        super(MetaAdam3, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.inner_param = inner_param
        self.outer_param = outer_param
        self.loss_func = nn.CrossEntropyLoss()
        self.mlp = LearningRateLearner_MLP().to(self.device)
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param
        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2
    
    def freeze_mlp(self):
        for param in self.mlp.parameters():
            param.requires_grad = False
    
    def unfreeze_mlp(self):
        for param in self.mlp.parameters():
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
        
        # self.unfreeze_mlp()
        
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        
        temperature = torch.tensor(self.outer_param["temperature"], device = self.device)
        query_target = query_target.view(-1)
        class_losses = []
        for class_idx in range(self.way_num):
            class_mask = query_target == class_idx
            if class_mask.any():
                class_output = output[class_mask] 
                class_target = query_target[class_mask]
                class_loss = self.loss_func(class_output, class_target)
                class_losses.append(class_loss)
            else:
                class_losses.append(torch.tensor(0.0, device=self.device))
        # Compute softmax weights for the class losses
        class_losses_tensor = torch.stack(class_losses)
        weights = F.softmax(class_losses_tensor / temperature, dim=0)
        # Compute final loss as weighted sum of class losses
        final_loss = torch.sum(weights * class_losses_tensor)
        '''
        lr_mlp = torch.tensor(self.outer_param["lstm_lr"], device=self.device)
        mlp_param = list(self.mlp.parameters())
        mlp_grad = torch.autograd.grad(final_loss, mlp_param)
        with torch.no_grad():  # Disable gradient tracking for parameter updates
            for k, weight in enumerate(mlp_param):
                weight.data -= lr_mlp * mlp_grad[k]
        
        self.freeze_mlp()'''
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, final_loss

    def set_forward_adaptation(self, support_set, support_target):
        alpha = torch.tensor(0.9, device = self.device)
        beta = torch.tensor(0.1, device = self.device)
        backbone = [self.emb_func, self.classifier]
        
        lr = torch.tensor(self.inner_param["inner_lr"], device = self.device)
        
        fast_parameters = [p for module in backbone for p in module.parameters()]
        for module in backbone:
            for parameter in module.parameters():
                parameter.fast = None
            
        m = [torch.zeros_like(p) for p in fast_parameters]

        w = self.mlp.weight1
        w2 = self.mlp.weight2
        self.emb_func.train()
        self.classifier.train()
        self.mlp.train()
        for i in range(
            self.inner_param["train_iter"]
            if self.training
            else self.inner_param["test_iter"]
        ):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            
            eta = []
            for g, m_val in zip(grad, m):
                #input_tensor = torch.stack([tmp[0] * g, tmp[1] * m_val], dim=0).unsqueeze(1)  # Shape (seq_len=2, batch=1, input_size)
                g_flat = g.view(-1)
                m_val_flat = m_val.view(-1)
                eta_flat = self.mlp(beta * m_val_flat, alpha * g_flat)
                eta.append(eta_flat.view_as(g))
                
            m = [alpha * g + beta * m_val for g, m_val in zip(grad, m)]
            
            k = 0
            fast_parameters = []
            for module in backbone:
                for weight in module.parameters():
                    if weight.fast is None:
                        weight.fast = weight - (eta[k] + lr) * m[k]
                    else:
                        weight.fast = weight.fast - (eta[k] + lr) * m[k]
                    k += 1
                    fast_parameters.append(weight.fast)
            
