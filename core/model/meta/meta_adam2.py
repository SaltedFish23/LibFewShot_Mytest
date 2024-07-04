import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from .maml import MAMLLayer

# maml + dynamic weighting schema + momentum


class MetaAdam2(MetaModel):
    def __init__(self, inner_param, outer_param, feat_dim, **kwargs):
        super(MetaAdam2, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.inner_param = inner_param
        self.outer_param = outer_param
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param
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
        
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, final_loss

    def set_forward_adaptation(self, support_set, support_target):
        alpha = torch.tensor(0.9, device = self.device)
        beta = torch.tensor(0.1, device = self.device)
        
        
        lr = self.inner_param["inner_lr"]
        fast_parameters = list(self.parameters())
        for parameter in self.parameters():
            parameter.fast = None
            
        m = [torch.zeros_like(p) for p in fast_parameters]

        self.emb_func.train()
        self.classifier.train()
        for i in range(
            self.inner_param["train_iter"]
            if self.training
            else self.inner_param["test_iter"]
        ):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            
            m = [alpha * g + beta * m_val for g, m_val in zip(grad, m)]

            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * m[k]
                else:
                    weight.fast = weight.fast - lr * m[k]
                fast_parameters.append(weight.fast)
            
