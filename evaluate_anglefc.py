import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from scipy import stats
from pycls.models.nas.nas import Cell
from utils import add_dropout, get_logger
from datasets.get_dataset_with_transform import get_datasets
from procedures import get_ntk_n, Linear_Region_Collector
from models import get_cell_based_tiny_net
import torch.nn as nn
from config_utils import load_config
from utils import get_angle_by_layer, sample_subset_by_class
import time
import json
import matplotlib.pyplot as plt
from mydataset import repeatDataset

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../data/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='/home/chenchen/tao/data/NASBench201/NAS-Bench-201-v1_1-096897.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()


def short_train(loader, loader_iter, model, criterion, lr=0.001, num_iter=10):
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    model.train()
    for batch_idx in range(num_iter):
        try:
            input, target = loader_iter.next()
        except Exception:
            del loader_iter
            loader_iter = iter(loader)
            input, target = loader_iter.next()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits, _ = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
    indices = torch.max(logits, dim=1)[1]
    acc = (indices == target).sum().cpu().numpy()
    cnt = target.size()[0]
    print('Short training ({} / {}) finished. Final loss: {}, acc: {}'.format(num_iter, len(loader), loss, acc/cnt))
    return loss.item()


def xavier_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_normal_fanout'):
    model.apply(kaiming_normal_fanin_init)


def get_weights(model):
    weights = []
    params = []
    for name, W in model.named_parameters():
        W_t = W.view(-1).detach()
        params.append(W_t)
        if 'weight' in name:
            W_t = W.view(-1).detach()
            weights.append(W_t)
    weights = torch.cat(weights, -1)
    params = torch.cat(params, -1)
    return weights, params


def get_weights_by_layer(model):
    weights = []
    params = []
    isnan = False
    params_t = []
    weights_t = []
    for name, W in model.stem.named_parameters():
        W_t = W.view(-1).detach()
        params_t.append(W_t)
        if 'weight' in name:
            W_t = W.view(-1).detach()
            weights_t.append(W_t)
    weights.append(torch.cat(weights_t, -1))
    params.append(torch.cat(params_t, -1))
    for i in range(17):  # 17 cells = 3*5 + 2
        params_t = []
        weights_t = []
        # for name, W in model.cells[i].named_parameters():
        #     W_t = W.view(-1).detach()
        #     params_t.append(W_t)
        #     if 'weight' in name:
        #         W_t = W.view(-1).detach()
        #         weights_t.append(W_t)
        for name, m in model.cells[i].named_modules():
            m_name = str(type(m))
            if 'AvgPool' in m_name:
                c_in, c_out = model.cells[i].in_dim, model.cells[i].out_dim
                W_t = (torch.ones(c_in, c_out) / 9).view(-1).detach().cuda()
                weights_t.append(W_t)
            elif 'Conv2d' in m_name or 'BatchNorm2d' in m_name:
                for name, W in m.named_parameters():
                    W_t = W.view(-1).detach()
                    params_t.append(W_t)
                    if 'weight' in name:
                        W_t = W.view(-1).detach()
                        weights_t.append(W_t)

        if len(weights_t) > 0:
            weights.append(torch.cat(weights_t, -1))
            # params.append(torch.cat(params_t, -1))
        else:
            weights.append([])
            params.append([])
            isnan = True
    params_t = []
    weights_t = []
    for name, W in model.classifier.named_parameters():
        W_t = W.view(-1).detach()
        params_t.append(W_t)
        if 'weight' in name:
            W_t = W.view(-1).detach()
            weights_t.append(W_t)
    weights.append(torch.cat(weights_t, -1))
    params.append(torch.cat(params_t, -1))
    return weights, isnan


def random_sample(data, train_split, num_class=10, ipc=10, dataname='cifar'):
    targets = data.targets.copy()
    if dataname == 'ImageNet16':
        targets = [t - 1 for t in targets]
    flag = np.zeros(num_class)
    # cifar_split = load_config('config_utils/cifar-split.txt', None, None)
    # train_split, valid_split = cifar_split.train, cifar_split.valid
    # train_split = list(range(50000))  # for Cifar-100
    random.shuffle(train_split)
    sample_list = []
    for i in train_split:
        c = targets[i]
        if flag[c] < ipc:
            flag[c] += 1
            sample_list.append(i)
        if sum(flag) == num_class * ipc:
            return sample_list


saved_path = os.path.join("logs2", 'cifar10_test', 'anglefcloss_param_100sample')
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, 'seed_{:02d}_kaimingnormal2.log'.format(args.seed)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N = 100  # number of samples
repeat = 1  # number of repeat for ensemble
repeat_seed = [(i+1) * 100 + args.seed for i in range(repeat)]
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
# if args.dataset == 'cifar10':
#     args.dataset = args.dataset + '-valid'
searchspace = nasspace.get_search_space(args)
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
# train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat,
#                                  args)
# dataloader for short training
train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)  #
cifar_split = load_config('config_utils/cifar-split.txt', None, None)
train_split, valid_split = cifar_split.train, cifar_split.valid
### sample train_data by 10ipc
# train_split = list(range(50000))  # cifar100: 50000, ImageNet16-120: 151700
class_idx = list(range(100))
random.shuffle(class_idx)
subset_indices = class_idx[:10]
loader_list = []
iter_list = []
for i in range(repeat):
    set_seed(repeat_seed[i])
    ### cifar10
    random_train_split = random_sample(train_data, train_split=train_split, num_class=10, ipc=10, dataname='cifar')  # cifar10
    sub_train_data = torch.utils.data.Subset(train_data, random_train_split)  # cifar10
    # sub_train_data, _ = sample_subset_by_class(train_data, None, subset_indices, ipc=10, dataname='cifar')
    # sub_train_data_repeat = repeatDataset(sub_train_data)
    shorttrain_loader = torch.utils.data.DataLoader(sub_train_data, batch_size=100, num_workers=0, shuffle=True, pin_memory=True)
    ###
    loader_iter = iter(shorttrain_loader)
    loader_list.append(shorttrain_loader)
    iter_list.append(loader_iter)
set_seed(args.seed)
criterion = torch.nn.CrossEntropyLoss().cuda()

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

# acc_exps = np.zeros(exps)
accs = np.zeros(N)
params_split = (json.load(open('params_split.txt', 'r')))  # 8 groups of networks where networks in the same group have the same number of parameters
params_split.append(list(range(len(searchspace))))  # all networks

scores = np.zeros(N)
scores_cells = np.zeros(N)
tmp_scores = np.zeros(repeat)
tmp_scores_cells = np.zeros(repeat)
params = np.zeros(N)
seed_idx = 0

for param_idx, uid_list in enumerate(params_split):
    if param_idx != 8:  # This is to search on the search space of all networks. This can be adjusted to search on different network groups
        continue
    set_seed(args.seed)
    end = time.time()
    uid_list_s = random.sample(uid_list, N)
    for i, uid in enumerate(uid_list_s):
        if i % 100 == 0:
            print(i, ' / ', N)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        params[i] = searchspace.api.get_cost_info(uid, args.dataset)['params']
        for run_idx in range(repeat):
            run_seed = repeat_seed[run_idx]
            set_seed(run_seed)  # to get different network initialization
            network = searchspace.get_network(uid)
            init_model(network)
            network = network.to(device)
            # Before short training
            weights_list1, isnan = get_weights_by_layer(network)
            if isnan:
                tmp_scores[run_idx] = 0
                tmp_scores_cells[run_idx] = -100
                continue
            # Do short training
            score_loss2 = - short_train(loader_list[run_idx], iter_list[run_idx], network, criterion, lr=0.2, num_iter=50)
            weights_list2, _ = get_weights_by_layer(network)
            fc1, fc2 = weights_list1[-1], weights_list2[-1]
            score = nn.functional.cosine_similarity(fc1, fc2, dim=-1)
            print('score: ', score)
            tmp_scores[run_idx] = score
            if score < 0.95:  # trivial networks will have score close to 1
                tmp_scores[run_idx] = score
                tmp_scores_cells[run_idx] = score_loss2
            else:
                tmp_scores[run_idx] = 0
                tmp_scores_cells[run_idx] = -100
        scores[i] = np.mean(tmp_scores)
        scores_cells[i] = np.mean(tmp_scores_cells)

    # idx = scores.argmax()
    rank_score = np.argsort(np.argsort(scores))
    rank_loss = np.argsort(np.argsort(scores_cells))
    rank_sl = rank_score + rank_loss
    rank_ens1 = np.argsort(np.argsort(rank_sl))
    rank_param = np.argsort(np.argsort(params))
    rank_ens = rank_param + rank_ens1
    msg = searchspace.api.get_net_config(uid, args.dataset)
    logger.info(msg)
    acc_anglelossparam = accs[rank_ens.argmax()]
    logger.info('Anglefc, ParamSplit: {} ({}), Time (s): {:.3f}, #Samples: {}, model acc: {}'.format(
        param_idx, len(uid_list), (time.time() - end), N, acc_anglelossparam))
