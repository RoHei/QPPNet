import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

import functools, os
import numpy as np
import json

from dataset.constants import terrier_dimensions, tpch_dimensions, tpcc_dimensions
from metric import Metric
from typing import Tuple

basic = 3

def squared_diff(output, target):
    return torch.sum((output - target) ** 2)


###############################################################################
#                        Operator Neural Unit Architecture                    #
###############################################################################
# Neural Unit that covers all operators
class NeuralUnit(nn.Module):
    """Define a Resnet block"""

    def __init__(self, node_type, dim_dict, num_layers=5, hidden_size=128,
                 output_size=32, norm_enabled=False):
        """
        Initialize the InternalUnit
        """
        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers, hidden_size, output_size,
                                            input_dim=dim_dict[node_type])

    def build_block(self, num_layers, hidden_size, output_size, input_dim):
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        assert (num_layers >= 2)
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for i in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        dense_block += [nn.Linear(hidden_size, output_size), nn.ReLU()]

        for layer in dense_block:
            try:
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
        return nn.Sequential(*dense_block)

    def forward(self, x):
        """ Forward function """
        out = self.dense_block(x)
        return out


###############################################################################
#                               QPP Net Architecture                          #
###############################################################################

class QPPNet():
    def __init__(self, opt):
        self.device = torch.device('cpu:0')
        self.save_dir: str = opt.save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.test: bool = False
        self.test_time: bool = opt.test_time
        self.batch_size: int = opt.batch_size
        self.dataset: str = opt.dataset

        if opt.dataset == "PSQLTPCH":
            self.dimensions = tpch_dimensions
        elif opt.dataset == "TerrierTPCH":
            self.dimensions = terrier_dimensions
        else:
            self.dimensions = tpcc_dimensions

        self.last_total_loss = None
        self.last_pred_err = None
        self.pred_err = None
        self.rq = 0
        self.last_rq = 0

        # Initialize the neural units
        self.units: dict = {}
        self.optimizers, self.schedulers = {}, {}
        self.best = 100000

        for operator in self.dimensions:
            self.units[operator] = NeuralUnit(operator, self.dimensions).to(self.device)
            if opt.SGD:
                optimizer = torch.optim.SGD(self.units[operator].parameters(),
                                            lr=opt.lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(self.units[operator].parameters(),
                                             opt.lr)  # opt.lr

            if opt.scheduler:
                sc = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                         gamma=opt.gamma)
                self.schedulers[operator] = sc

            self.optimizers[operator] = optimizer

        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.acc_loss = {operator: [self.dummy] for operator in self.dimensions}
        self.curr_losses = {operator: 0 for operator in self.dimensions}
        self.total_loss = None
        self._test_losses = dict()

        if opt.start_epoch > 0 or opt.test_time:
            self.load_model(opt.start_epoch)

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def _forward_oneQ_batch(self, sample_batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for a batch of queries from one query template.
        compute a dictionary of losses for each operator
        return output_vec, where 1st col is predicted time
        """
        feat_vec = sample_batch['feat_vec']

        # print(samp_batch['node_type'])
        # print(feat_vec.shape, print(samp_batch['children_plan']))
        input_vec = torch.from_numpy(feat_vec).to(self.device)
        # print(samp_batch['node_type'], input_vec)

        subplans_time = []
        # Recursively iterate over subplans and append to input_vector.
        # ToDo: How is this working if children have different operator type?

        for child_plan_dict in sample_batch['children_plan']:
            child_output_vec, _ = self._forward_oneQ_batch(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output_vec), axis=1)
                # first dim is subbatch_size
            else:
                subplans_time.append(torch.index_select(child_output_vec, 1, torch.zeros(1, dtype=torch.long).to(self.device)))

        # Exception handling if missing data
        expected_len = self.dimensions[sample_batch['node_type']]
        if expected_len > input_vec.size()[1]:
            add_on = torch.zeros(input_vec.size()[0], expected_len - input_vec.size()[1])
            print(sample_batch['real_node_type'], input_vec.shape, expected_len)
            input_vec = torch.cat((input_vec, add_on), axis=1)

        # Do actual forward pass according to the corresponding neural unit
        output_vec = self.units[sample_batch['node_type']](input_vec)

        # Predicted time is assumed to be the first column
        pred_time = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long).to(self.device))
        cat_res = torch.cat([pred_time] + subplans_time, axis=1)
        pred_time = torch.sum(cat_res, 1)

        # Compute and collect loss for current node type
        loss = (pred_time - torch.from_numpy(sample_batch['total_time']).to(self.device)) ** 2
        self.acc_loss[sample_batch['node_type']].append(loss)

        # added to deal with NaN
        try:
            assert (not (torch.isnan(output_vec).any()))
        except:
            print("feat_vec", feat_vec, "input_vec", input_vec)
            # if torch.cuda.is_available():
            #     print(samp_batch['node_type'], "output_vec: ", output_vec,
            #           self.units[samp_batch['node_type']].module.cpu().state_dict())
            # else:
            print(sample_batch['node_type'], "output_vec: ", output_vec,
                  self.units[sample_batch['node_type']].cpu().state_dict())
            exit(-1)
        return output_vec, pred_time

    def _forward(self, epoch):
        # self.input is a list of preprocessed plan_vec_dict
        total_loss = torch.zeros(1).to(self.device)
        total_losses = {operator: [torch.zeros(1).to(self.device)] for operator in self.dimensions}

        if self.test:
            test_loss = []
            pred_err = []

        all_tt, all_pred_time = None, None

        data_size = 0
        total_mean_mae = torch.zeros(1).to(self.device)
        for operator_index, sample_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in self.dimensions}
            _, pred_time = self._forward_oneQ_batch(sample_dict)

            if self.dataset == "PSQLTPCH":
                epsilon = torch.finfo(pred_time.dtype).eps
            else:
                epsilon = 0.001

            data_size += len(sample_dict['total_time'])

            if self.test:
                total_time = torch.from_numpy(sample_dict['total_time']).to(self.device)
                test_loss.append(torch.abs(total_time - pred_time))
                curr_pred_err = Metric.pred_err(total_time, pred_time, epsilon)
                pred_err.append(curr_pred_err)
                # if idx == 6 or \

                # print(samp_dict['feat_vec'])
                if np.isnan(curr_pred_err.detach()).any() or \
                        np.isinf(curr_pred_err.detach()).any():
                    print("feat_vec", sample_dict['feat_vec'])
                    print("pred_time", pred_time)
                    print("total_time", total_time)

                all_tt = total_time if all_tt is None else torch.cat([total_time, all_tt])
                all_pred_time = pred_time if all_pred_time is None \
                    else torch.cat([pred_time, all_pred_time])

                # if idx in self._test_losses and self._test_losses[idx] == curr_rq:
                #     print(f"^^^^^^^^^^^^^^^^^^{samp_dict['node_type']} ^^^^^^^^^^^^^^^\n",
                #           pred_time, '\n', tt, '\n')
                #           # samp_dict['feat_vec'], '\n')
                #     layer = self.units[samp_dict['node_type']].dense_block[0]
                #     print(type(layer), layer.weight.grad)
                #     for layer in self.units[samp_dict['node_type']].dense_block:
                #         try:
                #             print(type(layer), layer.weight.grad)
                #         except:
                #             assert(isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh))

                # self._test_losses[idx] = curr_rq
                curr_rq = Metric.r_q(total_time, pred_time, epsilon)

                curr_mean_mae = Metric.mean_mae(total_time, pred_time, epsilon)
                total_mean_mae += curr_mean_mae * len(total_time)

                if epoch % 50 == 0:
                    print("####### eval by temp: idx {}, test_loss {}, pred_err {}, " \
                          "rq {}, weighted mae {}, accumulate_err {} " \
                          .format(operator_index, torch.mean(torch.abs(total_time - pred_time)).item(),
                                  torch.mean(curr_pred_err).item(),
                                  curr_rq, curr_mean_mae,
                                  Metric.accumulate_err(total_time, pred_time, epsilon)))

            D_size = 0
            subbatch_loss = torch.zeros(1).to(self.device)
            for operator in self.acc_loss:
                # print(operator, self.acc_loss[operator])
                all_loss = torch.cat(self.acc_loss[operator])
                D_size += all_loss.shape[0]
                # print("all_loss.shape",all_loss.shape)
                subbatch_loss += torch.sum(all_loss)

                total_losses[operator].append(all_loss)

            subbatch_loss = torch.mean(torch.sqrt(subbatch_loss / D_size))
            # print("subbatch_loss.shape",subbatch_loss.shape)
            total_loss += subbatch_loss * sample_dict['subbatch_size']

        if self.test:
            all_test_loss = torch.cat(test_loss)
            # print(test_loss[0].shape, test_loss[1].shape, all_test_loss.shape)
            all_test_loss = torch.mean(all_test_loss)
            self.test_loss = all_test_loss

            all_pred_err = torch.cat(pred_err)
            self.pred_err = torch.mean(all_pred_err)

            self.rq = Metric.r_q(all_tt, all_pred_time, epsilon)
            self.accumulate_err = Metric.accumulate_err(all_tt, all_pred_time,
                                                        epsilon)
            self.weighted_mae = total_mean_mae / data_size

            if epoch % 50 == 0:
                print("test batch Pred Err: {}, R(q): {}, Accumulated Error: " \
                      "{}, Weighted MAE: {}".format(self.pred_err,
                                                    self.rq,
                                                    self.accumulate_err,
                                                    self.weighted_mae))

        else:
            self.curr_losses = {operator: torch.mean(torch.cat(total_losses[operator])).item() for operator in
                                self.dimensions}
            self.total_loss = torch.mean(total_loss / self.batch_size)
        # print("self.total_loss.shape", self.total_loss.shape)

    def backward(self):
        self.last_total_loss = self.total_loss.item()
        if self.best > self.total_loss.item():
            self.best = self.total_loss.item()
            self.save_model('best')
        self.total_loss.backward()
        self.total_loss = None

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.test = False
        self._forward(epoch)
        # clear prev grad first
        for operator in self.optimizers:
            self.optimizers[operator].zero_grad()

        self.backward()

        for operator in self.optimizers:
            self.optimizers[operator].step()
            if len(self.schedulers) > 0:
                self.schedulers[operator].step()

        self.input = self.test_dataset
        self.test = True
        self._forward(epoch)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.last_rq = self.rq
        self.test_loss, self.pred_err = None, None
        self.rq = 0

    def evaluate(self, eval_dataset):
        self.test = True
        self.set_input(eval_dataset)
        self._forward(0)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.last_rq = self.rq
        self.test_loss, self.pred_err = None, None
        self.rq = 0

    def get_current_losses(self):
        return self.curr_losses

    def save_model(self, epoch: int):
        for name, unit in self.units.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            # lql: disable cuda temporarily
            # if torch.cuda.is_available():
            #     torch.save(unit.module.cpu().state_dict(), save_path)
            #     unit.to(self.device)
            # else:
            torch.save(unit.cpu().state_dict(), save_path)

    def load_model(self, epoch):
        for name in self.units:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            if not os.path.exists(save_path):
                raise ValueError("model {} doesn't exist".format(save_path))
            self.units[name].load_state_dict(torch.load(save_path))
