import time, torch
from typing import List, TextIO

from model_arch import QPPNet
from dataset.terrier_tpch_dataset.terrier_utils import TerrierTPCHDataSet
from dataset.postgres_tpch_dataset.tpch_utils import PSQLTPCHDataSet
from dataset.oltp_dataset.oltp_utils import OLTPDataSet
import argparse

parser = argparse.ArgumentParser(description='QPPNet Arg Parser')

parser.add_argument('--data_dir',
                    type=str,
                    default='./res_by_temp/',
                    help='Dir containing train data')

parser.add_argument('--dataset_name',
                    type=str,
                    default='PSQLTPCH',
                    help='Select dataset [PSQLTPCH | TerrierTPCH | OLTP]')

parser.add_argument('--test_time',
                    action='store_true',
                    help='if in testing mode')

parser.add_argument('-dir', '--save_dir',
                    type=str,
                    default='./saved_model',
                    help='Dir to save model weights (default: ./saved_model)')

parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='Learning rate (default: 1e-3)')

parser.add_argument('--scheduler',
                    action='store_true')

parser.add_argument('--step_size',
                    type=int,
                    default=1000,
                    help='step_size for StepLR scheduler (default: 1000)')

parser.add_argument('--gamma',
                    type=float,
                    default=0.95,
                    help='gamma in Adam (default: 0.95)')

parser.add_argument('--SGD',
                    action='store_true',
                    help='Use SGD as optimizer with momentum 0.9')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size used in training (default: 32)')

parser.add_argument('-s', '--start_epoch',
                    type=int,
                    default=0,
                    help='Epoch to start training with (default: 0)')

parser.add_argument('-t', '--end_epoch',
                    type=int,
                    default=200,
                    help='Epoch to end training (default: 200)')

parser.add_argument('-epoch_freq', '--save_latest_epoch_freq',
                    type=int,
                    default=100)

parser.add_argument('-logf', '--logfile',
                    type=str,
                    default='train_loss.txt')

parser.add_argument('--mean_range_dict',
                    type=str)


def save_parameters(opt, logf):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    logf.write(message)
    logf.write('\n')


def report_losses(losses: List, logfile: TextIO, epoch: int):
    loss_str = "losses: "
    for operator in losses:
        loss_str += str(operator) + " [" + str(losses[operator]) + "]; "

    if epoch % 50 == 0:
        print(f"epoch: {epoch}; "
              f"iter_num: {total_iterations}; "
              f"total_loss: {qpp.last_total_loss}; "
              f"test_loss: {qpp.last_test_loss}; "
              f"pred_err: {qpp.last_pred_err}; "
              f"Q-Error: {qpp.last_q_error}")
        print(loss_str)
    logfile.write(loss_str + '\n')


def report_scores(epoch: int, total_iterations: int, qpp: QPPNet, logfile: TextIO, ):
    logfile.write(f"epoch: {epoch}; "
                  f"iter_num: {total_iterations}; "
                  f"total_loss: {qpp.last_total_loss}; "
                  f"test_loss: {qpp.last_test_loss}; "
                  f"pred_err: {qpp.last_pred_err}; "
                  f"Q-Error: {qpp.last_q_error}")


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)

    # Initialize dataset
    parameters = parser.parse_args()
    if parameters.dataset_name == "PSQLTPCH":
        dataset = PSQLTPCHDataSet(parameters)
    elif parameters.dataset_name == "TerrierTPCH":
        dataset = TerrierTPCHDataSet(parameters)
    else:
        dataset = OLTPDataSet(parameters)

    # Initialize model
    qpp = QPPNet(parameters)

    total_iterations = 0
    if parameters.test_time:
        qpp.evaluate(dataset.train_dataset)
        print(f'total_loss: {qpp.last_total_loss}; '
              f'test_loss: {qpp.last_test_loss}; '
              f'pred_err: {qpp.last_pred_err}; '
              f'Q Error: {qpp.last_q_error}')

    else:
        logfile = open(parameters.logfile, 'w+')
        save_parameters(parameters, logfile)
        qpp.test_dataset = dataset.test_dataset

        for epoch in range(parameters.start_epoch, parameters.end_epoch):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration

            batch = dataset.sample_new_batch()
            total_iterations += parameters.batch_size

            # Learning iteration and obtaining losses
            qpp.set_input(batch)
            qpp.optimize_parameters(epoch)
            losses = qpp.get_current_losses()

            # Report metrics
            report_scores(epoch, total_iterations, qpp, logfile)
            report_losses(losses, logfile, epoch)

            # Regular model caching
            if (epoch + 1) % parameters.save_latest_epoch_freq == 0:
                print(f'Saving the latest model (epoch {epoch + 1}, total_iters {total_iterations})')
                qpp.save_model(epoch + 1)
        logfile.close()
