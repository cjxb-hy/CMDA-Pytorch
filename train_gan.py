import os
import random
import argparse
from train_segmenter import train

import torch
from torch import optim
from torch.utils.data import DataLoader

from loss import MyLoss
from lib import load_model, _dice_eval
from datasetnpy import Data_Loader
from adversarial import Create_Mr_Network, Create_Ct_Network, Create_Second_Half, Create_Classifier, Create_Mask_Critic

random.seed(456)

cost_kwargs = {
    "regularizer": 1.0e-4,  # L2 norm regularizer segmentation model
    "gan_regularizer": 1.0e-4,  # L2 norm regularizer for WGAN variables
    "miu_gen": 0.002,  # weighing of generator loss
    "miu_dis": 0.002,  # weighing of discriminator loss
    # the trade-off parameter for mask discriminator, set it as 0.3
    "lambda_mask_loss": 1.0,
}

train_config = {
    # restore from the source segmenter and manually initialize DAM layers with learned early layers
    "dis_interval": 1,  # frequency of updating discriminator, normally, just set it to 1
    # frequency of updating generator (CT adaptation layers), normally, just set it to 1
    "gen_interval": 1,
    "dis_per_gen": 3,
    "train_per_eval": 3
}


def main(phase):

    channels = 3
    num_cls = 5
    batch_size = 2
    epochs = 6
    lr = 3e-4
    device = torch.device('cpu')

    baseline_model_path = 'models/mr_baseline/model2.pth'
    # ct_model_path = 'models/ct_model/model1.pth'
    cls_model_path = 'models/cls_model/model4.pth'

    # try:
    #     os.makedirs(output_path)
    # except:
    #     print("folder exist!")

    mr_train = DataLoader(dataset=Data_Loader('data/mr_train'),
                          batch_size=batch_size, shuffle=True)
    mr_val = DataLoader(dataset=Data_Loader('data/mr_val'),
                        batch_size=batch_size, shuffle=True)
    ct_train = DataLoader(dataset=Data_Loader('data/ct_train'),
                          batch_size=batch_size, shuffle=True)
    ct_val = DataLoader(dataset=Data_Loader('data/ct_val'),
                        batch_size=batch_size, shuffle=True)

    if phase == 'pre-train':  # pre-train the discriminator for CNN feature, before update the DAM and segmentation mask discriminator together
        print('pre-train-------------------------')
        train_config["gen_interval"] = 0
        train_config["dis_per_gen"] = 10000
        cost_kwargs["lambda_mask_loss"] = 0

    elif phase == 'train-gan':  # After warming-up, train the DAM and DCM together

        print('train-gan-------------------------')
        cost_kwargs["lambda_mask_loss"] = 0.3

    mr_network = Create_Mr_Network(channels=channels).to(device)
    ct_network = Create_Ct_Network(channels=channels).to(device)
    second_half = Create_Second_Half(
        n_class=num_cls, batch_size=batch_size).to(device)
    classifier = Create_Classifier(batch_size=batch_size).to(device)
    mask_critic = Create_Mask_Critic(n_class=num_cls).to(device)

    dis_criterion = MyLoss(cost_kwargs, mr_network,
                           ct_network, second_half, classifier, mask_critic).to(device)
    gen_criterion = MyLoss(cost_kwargs, mr_network,
                           ct_network, second_half, classifier, mask_critic).to(device)
    dis_optimizer = optim.RMSprop(classifier.parameters(), lr=lr)
    gen_optimizer = optim.RMSprop(ct_network.parameters(), lr=lr)

    mr_model_dict = load_model(mr_network, model_path=baseline_model_path)
    mr_network.load_state_dict(mr_model_dict)

    ct_model_dict = load_model(ct_network, model_path=baseline_model_path)
    ct_network.load_state_dict(ct_model_dict)

    second_half_dict = load_model(second_half, model_path=baseline_model_path)
    second_half.load_state_dict(second_half_dict)

    if phase == 'traing_gan':
        classifier.load_state_dict(torch.load(cls_model_path))

    dis_iter = 1
    for epoch in range(1, epochs+1):
        mr_network.eval()
        # ct_network.train()
        second_half.eval()
        # classifier.train()
        mask_critic.train()

        # print('start epoch:{}'.format(epoch))
        # if epoch % 3 != 0:
        print('Training at epoch: {},'.format(epoch))
        for [mr_image, mr_label], [ct_image, ct_label] in zip(mr_train, ct_train):
            if dis_iter % train_config["dis_per_gen"] != 0:
                print('Dis_iter:{}'.format(dis_iter))
                ct_network.eval()
                classifier.train()
                mr_4, mr_6 = mr_network(mr_image)
                ct_4, ct_6 = ct_network(ct_image)
                mr_9, mr_8, mr_7, mr_logits = second_half(mr_6)
                ct_9, ct_8, ct_7, ct_logits = second_half(ct_6)
                mr_class_logits = classifier(
                    mr_4, mr_6, mr_7, mr_9, mr_logits)
                ct_class_logits = classifier(
                    ct_4, ct_6, ct_7, ct_9, ct_logits)
                mr_mask_logits = mask_critic(mr_logits)
                ct_mask_logits = mask_critic(ct_logits)

                print('start get dis_loss! iter:{}'.format(dis_iter))
                dis_loss, _, _, dis_reg, _ = dis_criterion(
                    ct_logits, mr_logits, ct_class_logits, mr_class_logits, ct_mask_logits, mr_mask_logits)

                dis_loss_reg = torch.add(dis_loss, dis_reg)
                dis_optimizer.zero_grad()
                dis_loss_reg.backward()
                dis_optimizer.step()

                print('Training at epoch:{}, Dis Loss:{}'.format(
                    epoch, dis_loss.item()))
                # 权重限制在-0.03~0.03

            if train_config['gen_interval'] == 0:
                pass
            elif dis_iter % train_config["dis_per_gen"] == 0:
                print('Gen_iter:{}'.format(dis_iter))
                ct_network.train()
                classifier.eval()
                mr_4, mr_6 = mr_network(mr_image)
                ct_4, ct_6 = ct_network(ct_image)
                mr_9, mr_8, mr_7, mr_logits = second_half(mr_6)
                ct_9, ct_8, ct_7, ct_logits = second_half(ct_6)
                mr_class_logits = classifier(
                    mr_4, mr_6, mr_7, mr_9, mr_logits)
                ct_class_logits = classifier(
                    ct_4, ct_6, ct_7, ct_9, ct_logits)
                mr_mask_logits = mask_critic(mr_logits)
                ct_mask_logits = mask_critic(ct_logits)

                print('start get gen_loss! iter:{}'.format(dis_iter))
                _, gen_loss, _, _, gen_reg = gen_criterion(
                    ct_logits, mr_logits, ct_class_logits, mr_class_logits, ct_mask_logits, mr_mask_logits)

                gen_loss_reg = torch.add(gen_loss, gen_reg)
                gen_optimizer.zero_grad()
                gen_loss_reg.backward()
                gen_optimizer.step()

                print('Training at epoch:{}, Gen Loss:{}'.format(
                    epoch, gen_loss.item()))

            dis_iter += 1

        for param_group in dis_optimizer.param_groups:
            param_group["lr"] *= 0.98

        for param_group in gen_optimizer.param_groups:
            param_group["lr"] *= 0.98

        path = os.path.join('models/cls_model/', 'model{}.pth'.format(epoch))
        torch.save(classifier.state_dict(), path)
        print('Epoch: {}, save classifier model:{}'.format(epoch, path))
        path = os.path.join('models/ct_model/', 'model{}.pth'.format(epoch))
        torch.save(ct_network.state_dict(), path)
        print('Epoch: {}, save ct_network model:{}'.format(epoch, path))

        # else:  # eval
        if epoch % train_config["train_per_eval"] == 0:
            print('Eval at epoch: {},'.format(epoch))
            ct_network.eval()
            classifier.eval()
            m_dice_total = 0
            c_dice_total = 0
            # m_dice_arr_total = [0, 0, 0, 0, 0]
            c_dice_arr_total = [0, 0, 0, 0, 0]
            n_image = 0
            for [mr_val_image, mr_val_label], [ct_val_image, ct_val_label] in zip(mr_val, ct_val):
                mr_v4, mr_v6 = mr_network(mr_val_image)
                ct_v4, ct_v6 = ct_network(ct_val_image)
                mr_v9, mr_v8, mr_v7, mr_vlogits = second_half(mr_v6)
                ct_v9, ct_v8, ct_v7, ct_vlogits = second_half(ct_v6)

                mr_dice_eval, mr_dice_eval_arr = _dice_eval(
                    mr_vlogits, mr_val_label, num_cls)
                ct_dice_eval, ct_dice_eval_arr = _dice_eval(
                    ct_vlogits, ct_val_label, num_cls)

                m_dice_total += mr_dice_eval
                c_dice_total += ct_dice_eval

                # m_dice_arr_total = [m_dice_arr_total[i] + mr_dice_eval_arr[i]
                #                     for i in range(len(m_dice_arr_total))]
                c_dice_arr_total = [c_dice_arr_total[i] + ct_dice_eval_arr[i]
                                    for i in range(len(c_dice_arr_total))]
                n_image += 1

            # m_dice_arr_total = [v / n_image for v in m_dice_arr_total]
            c_dice_arr_total = [v / n_image for v in c_dice_arr_total]
            print("Eval at epoch: {}, mri eval dice: {}, ct eval dice: {}".format(
                epoch, m_dice_total / n_image, c_dice_total / n_image))
            print("ct_dice_eval_c1_lv_myo:{}".format(c_dice_arr_total[1]))
            print("ct_dice_eval_c2_la_blood:{}".format(c_dice_arr_total[2]))
            print("ct_dice_eval_c3_lv_blood:{}".format(c_dice_arr_total[3]))
            print("ct_dice_eval_c4_aa:{}".format(c_dice_arr_total[4]))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--phase", type=str, default=None)
    # args = parser.parse_args()
    # phase = args.phase
    main(phase='train-gan')
    # main(phase='pre-train')
