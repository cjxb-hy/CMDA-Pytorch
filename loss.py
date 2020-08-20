import torch
from torch import nn


class MyLoss(nn.Module):
    def __init__(self, train_config, cost_kwargs, net1, net2, net3, net4, net5):
        super(MyLoss, self).__init__()

        self.train_config = train_config

        self.cost_kwargs = cost_kwargs
        # coefficient for discriminator loss
        self.miu_dis = torch.Tensor([self.cost_kwargs["miu_dis"]])
        self.miu_gen = torch.Tensor([self.cost_kwargs["miu_gen"]])
        # used to be 0.5 0.5 1
        # weighting of mask critic score
        self.lambda_mask_loss = self.cost_kwargs["lambda_mask_loss"]
        self.reg_coeff = self.cost_kwargs["regularizer"]
        self.gan_reg_coeff = self.cost_kwargs["gan_regularizer"]
        self.joint_weights = []

        self.net_mr = net1
        self.net_ct = net2
        self.net_joint = net3
        self.net_cls = net4
        self.net_mask = net5

    def _get_cost(self, ct_logits, mr_logits,  ct_cls_logits, mr_cls_logits, ct_mask_logits, mr_mask_logits):

        dis_loss = -1 * self.miu_dis * \
            torch.mean(mr_cls_logits - ct_cls_logits)
        gen_loss = -1 * self.miu_gen * torch.mean(ct_cls_logits)

        m_dis_loss = -1 * self.miu_dis * \
            torch.mean(mr_mask_logits-ct_mask_logits)
        m_gen_loss = -1 * self.miu_gen * torch.mean(ct_mask_logits)

        total_weights = {**dict(self.net_mr.state_dict().items()),
                         **dict(self.net_joint.state_dict().items())}

        mr_front_reg = sum([torch.norm(parameters) for name, parameters in total_weights.items(
        ) if 'weight' in name and '.2.' not in name and 'extra.1' not in name and 'g_out' not in name])

        joint_reg = sum(torch.norm(variable)
                        for variable in self.joint_weights)
        fixed_coeff_reg = self.reg_coeff * (mr_front_reg + joint_reg)

        gen_reg = self.gan_reg_coeff * self.miu_gen * sum([torch.norm(parameters) for name, parameters in self.net_ct.state_dict().items(
        ) if 'weight' in name and '.2.' not in name and 'extra.1' not in name])

        dis_reg = self.gan_reg_coeff * self.miu_dis * sum([torch.norm(parameters) for name, parameters in self.net_cls.state_dict().items(
        ) if 'weight' in name and '.2.' not in name and 'extra.1' not in name and 'c_6.model2' not in name])

        m_dis_reg = self.gan_reg_coeff * self.miu_dis * sum([torch.norm(parameters) for name, parameters in self.net_mask.state_dict(
        ).items() if 'weight' in name and '.2.' not in name and 'extra.1' not in name])

        dis_loss += self.lambda_mask_loss * m_dis_loss
        gen_loss += self.lambda_mask_loss * m_gen_loss
        dis_reg += self.lambda_mask_loss * m_dis_reg

        return dis_loss + 1.0 / self.train_config['dis_sub_iter'] * dis_reg, fixed_coeff_reg, gen_loss + 1.0 / self.train_config['gen_sub_iter'] * gen_reg

    def forward(self, ct_logits, mr_logits,  ct_cls_logits, mr_cls_logits, ct_mask_logits, mr_mask_logits):
        return self._get_cost(ct_logits, mr_logits,  ct_cls_logits, mr_cls_logits, ct_mask_logits, mr_mask_logits)


if __name__ == "__main__":
    x = 0.005
    print(type(x))
    x = torch.Tensor([x])
    print(x.shape)
    # joint_weights = []
    # joint_reg = sum(torch.norm(variable) for variable in joint_weights)
    # print(joint_reg)
    pass
