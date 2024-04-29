import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import os
import numpy as np

from dataset.constants import terrier_dimensions, tpch_dimensions, tpcc_dimensions
from metric import Metric
from typing import Tuple


def squared_diff(output, target):
    return torch.sum((output - target) ** 2)


class NeuralUnit(nn.Module):
    """ Neural Unit that covers all operators """

    def __init__(self, node_type,
                 dimension: int,
                 num_layers: int = 5,
                 hidden_size: int = 128,
                 output_size: int = 32):

        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers=num_layers,
                                            hidden_size=hidden_size,
                                            output_size=output_size,
                                            input_dim=dimension)

    @staticmethod
    def build_block(num_layers: int, hidden_size: int, output_size: int, input_dim: int) -> nn.Sequential:
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        """
        assert num_layers >= 2, "Num of layers need to be greater than 1"
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
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


class QPPNet:
    """ QPPNet Architecture"""

    def __init__(self, params):
        self.device = torch.device('cpu:0')
        self.save_dir: str = params.save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.test: bool = False
        self.test_time: bool = params.test_time
        self.batch_size: int = params.batch_size
        self.dataset_name: str = params.dataset_name

        if params.dataset_name == "PSQLTPCH":
            self.dimensions_by_operator = tpch_dimensions

        elif params.dataset_name == "TerrierTPCH":
            self.dimensions_by_operator = terrier_dimensions

        else:
            self.dimensions_by_operator = tpcc_dimensions

        self.last_total_loss = None
        self.last_pred_err = None
        self.pred_err = None
        self.q_error = 0
        self.last_q_error = 0

        # Initialize the neural units
        self.units, self.optimizers, self.schedulers = {}, {}, {}
        self.best = 100000

        for operator_type, dimension in self.dimensions_by_operator.items():
            self.units[operator_type] = NeuralUnit(operator_type, dimension).to(self.device)
            self.optimizers[operator_type] = torch.optim.Adam(self.units[operator_type].parameters(), params.lr)
            if params.scheduler:
                self.schedulers[operator_type] = lr_scheduler.StepLR(self.optimizers[operator_type], step_size=params.step_size, gamma=params.gamma)

        self.loss_function = squared_diff

        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.accumulated_loss = {operator: [self.dummy] for operator in self.dimensions_by_operator}
        self.curr_losses = {operator: 0 for operator in self.dimensions_by_operator}

        self.total_loss = None
        self._test_losses = dict()

        if params.start_epoch > 0 or params.test_time:
            self.load_model(params.start_epoch)

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def forward_query_batch(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for a batch of queries from one query template.
        compute a dictionary of losses for each operator
        return output_vec, where 1st col is predicted time
        """
        feature_vector = batch['feat_vec']
        node_type = batch['node_type']
        children = batch['children_plan']
        features = torch.from_numpy(feature_vector).to(self.device)

        sub_plans_times = []
        # Recursively iterate over sub plans and append to input_vector.
        for children_batch in children:
            child_predictions, _ = self.forward_query_batch(children_batch)
            # In case, the operator is a leaf, add the current features to the input vector (?)
            if not children_batch['is_subplan']:
                features = torch.cat((features, child_predictions), axis=1)
                # first dim is subbatch_size
            else:
                sub_plan_time = torch.index_select(child_predictions, 1, torch.zeros(1, dtype=torch.long).to(self.device))
                sub_plans_times.append(sub_plan_time)

        # Exception handling if missing data
        expected_len = self.dimensions_by_operator[node_type]
        if expected_len > features.size()[1]:
            add_on = torch.zeros(features.size()[0], expected_len - features.size()[1])
            # print(batch['real_node_type'], input_vector.shape, expected_len)
            features = torch.cat((features, add_on), axis=1)

        # Do actual forward pass according to the corresponding neural unit
        predictions = self.units[node_type](features)

        # Predicted time is assumed to be the first column
        predicted_operator_times = torch.index_select(predictions, 1, torch.zeros(1, dtype=torch.long).to(self.device))

        # Adding predicted time to times of previous sub-plans.
        cat_res = torch.cat([predicted_operator_times] + sub_plans_times, axis=1)

        # ?
        predicted_operator_times = torch.sum(cat_res, 1)

        # Compute and collect loss for current node type
        real_time = torch.from_numpy(batch['total_time']).to(self.device)
        loss = (predicted_operator_times - real_time) ** 2

        # Gather loss in accumulated loss
        self.accumulated_loss[node_type].append(loss)

        if torch.isnan(predictions).any():
            print(f"NaN found in output_vector. "
                  f"feat_vec: {feature_vector} "
                  f"input_vec: {features} "
                  f"node_type: {node_type} "
                  f"output_vec: {predictions} "
                  f"state_dict: {self.units[node_type].cpu().state_dict()}")
            exit(-1)

        return predictions, predicted_operator_times

    def forward(self, epoch: int):
        # self.input is a list of preprocessed plan_vec_dict
        total_loss = torch.zeros(1).to(self.device)
        total_losses = {operator: [torch.zeros(1).to(self.device)] for operator in self.dimensions_by_operator}

        if self.test:
            test_loss = []
            pred_err = []

        all_tt, all_pred_time = None, None

        data_size = 0
        total_mean_mae = torch.zeros(1).to(self.device)

        for query_index, query_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.accumulated_loss
            self.accumulated_loss = {operator: [self.dummy] for operator in self.dimensions_by_operator}
            _, predicted_runtimes = self.forward_query_batch(query_dict)

            if self.dataset_name == "PSQLTPCH":
                epsilon = torch.finfo(predicted_runtimes.dtype).eps
            else:
                epsilon = 0.001

            data_size += len(query_dict['total_time'])

            if self.test:
                real_time = torch.from_numpy(query_dict['total_time']).to(self.device)
                test_loss.append(torch.abs(real_time - predicted_runtimes))
                curr_pred_err = Metric.pred_err(real_time, predicted_runtimes, epsilon)
                pred_err.append(curr_pred_err)

                if np.isnan(curr_pred_err.detach()).any() or np.isinf(curr_pred_err.detach()).any():
                    print("feat_vec", query_dict['feat_vec'])
                    print("pred_time", predicted_runtimes)
                    print("total_time", real_time)

                all_tt = real_time if all_tt is None else torch.cat([real_time, all_tt])
                all_pred_time = predicted_runtimes if all_pred_time is None else torch.cat([predicted_runtimes, all_pred_time])

                curr_q_error = Metric.q_error(real_time, predicted_runtimes, epsilon)
                curr_mean_mae = Metric.mean_mae(real_time, predicted_runtimes)
                total_mean_mae += curr_mean_mae * len(real_time)
                accumulate_err = Metric.accumulate_err(real_time, predicted_runtimes, epsilon)

                if epoch % 50 == 0:
                    print(f"eval by temp: "
                          f"idx {query_index}, "
                          f"test_loss: {torch.mean(torch.abs(real_time - predicted_runtimes)).item()}, "
                          f"pred_err: {torch.mean(curr_pred_err).item()}, "
                          f"q-error: {curr_q_error}, "
                          f"weighted mae: {curr_mean_mae}, "
                          f"accumulate_err: {accumulate_err}")

            D_size = 0
            subbatch_loss = torch.zeros(1).to(self.device)
            for operator_type, loss in self.accumulated_loss.items():
                all_loss = torch.cat(loss)
                D_size += all_loss.shape[0]
                subbatch_loss += torch.sum(all_loss)
                total_losses[operator_type].append(all_loss)

            subbatch_loss = torch.mean(torch.sqrt(subbatch_loss / D_size))
            total_loss += subbatch_loss * query_dict['subbatch_size']

        if self.test:
            all_test_loss = torch.cat(test_loss)
            self.test_loss = torch.mean(all_test_loss)

            # Compute metrics
            all_pred_err = torch.cat(pred_err)
            self.pred_err = torch.mean(all_pred_err)
            self.weighted_mae = total_mean_mae / data_size
            self.q_error = Metric.q_error(all_tt, all_pred_time, epsilon)
            self.accumulate_err = Metric.accumulate_err(all_tt, all_pred_time, epsilon)
            self.weighted_mae = total_mean_mae / data_size

            # Report metrics
            if epoch % 50 == 0:
                print(f"Test batch: "
                      f"Pred Err: {self.pred_err}, "
                      f"Q-Error: {self.q_error}, "
                      f"Accumulated Error: {self.accumulate_err}, "
                      f"Weighted MAE: {self.weighted_mae}")

        else:
            self.curr_losses = {operator: torch.mean(torch.cat(total_losses[operator])).item() for operator in self.dimensions_by_operator}
            self.total_loss = torch.mean(total_loss / self.batch_size)

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
        self.forward(epoch)

        # Clear previous gradients of the neural units
        for operator in self.optimizers:
            self.optimizers[operator].zero_grad()

        # Call backward function
        self.backward()

        # Do step on all operator-level optimizers
        for operator in self.optimizers:
            self.optimizers[operator].step()
            if len(self.schedulers) > 0:
                self.schedulers[operator].step()

        # Do validation
        self.input = self.test_dataset
        self.test = True
        self.forward(epoch)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.last_q_error = self.q_error
        self.test_loss, self.pred_err = None, None
        self.q_error = 0

    def evaluate(self, eval_dataset):
        self.test = True
        self.set_input(eval_dataset)
        self.forward(0)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.last_q_error = self.q_error
        self.test_loss, self.pred_err = None, None
        self.q_error = 0

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
