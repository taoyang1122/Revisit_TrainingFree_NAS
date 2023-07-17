import torch
from pycls.models.nas.nas import Cell
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import logging


class DropChannel(torch.nn.Module):
    def __init__(self, p, mod):
        super(DropChannel, self).__init__()
        self.mod = mod
        self.p = p
    def forward(self, s0, s1, droppath):
        ret = self.mod(s0, s1, droppath)
        return ret


class DropConnect(torch.nn.Module):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        dim1 = inputs.shape[2]
        dim2 = inputs.shape[3]
        channel_size = inputs.shape[1]
        keep_prob = 1 - self.p
        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, channel_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output    

def add_dropout(network, p, prefix=''):
    #p = 0.5
    for attr_str in dir(network):
        target_attr = getattr(network, attr_str)
        if isinstance(target_attr, torch.nn.Conv2d):
            setattr(network, attr_str, torch.nn.Sequential(target_attr, DropConnect(p)))
        elif isinstance(target_attr, Cell):
            setattr(network, attr_str, DropChannel(p, target_attr))
    for n, ch in list(network.named_children()):
        #print(f'{prefix}add_dropout {n}')
        if isinstance(ch, torch.nn.Conv2d):
            setattr(network, n, torch.nn.Sequential(ch, DropConnect(p)))
        elif isinstance(ch, Cell):
            setattr(network, n, DropChannel(p, ch))
        else:
            add_dropout(ch, p, prefix + '\t')
             



def orth_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.orthogonal_(m.weight)

def uni_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight)

def uni2_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight, -1., 1.)

def uni3_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight, -.5, .5)

def norm_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.norm_(m.weight)

def eye_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.eye_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.dirac_(m.weight)

def xavier_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)


def fixup_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.zero_(m.weight)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.zero_(m.weight)
        torch.nn.init.zero_(m.bias)


def init_network(network, init):
    if init == 'orthogonal':
        network.apply(orth_init)
    elif init == 'uniform':
        print('uniform')
        network.apply(uni_init)
    elif init == 'uniform2':
        network.apply(uni2_init)
    elif init == 'uniform3':
        network.apply(uni3_init)
    elif init == 'normal':
        network.apply(norm_init)
    elif init == 'identity':
        network.apply(eye_init)
    elif init == 'xavier':
        network.apply(xavier_init)


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def get_angle_by_layer(model):
    weights = []
    weights_t = []
    isnan = False
    fixed_cell_index_list = [5, 11]
    path_index = [[3], [1, 5], [0, 4], [0, 2, 5]]
    for name, W in model.stem.named_parameters():
        if 'weight' in name:
            W_t = W.view(-1).detach()
            weights_t.append(W_t)
    weights.append(torch.cat(weights_t, -1))
    for i in range(17):  # 17 cells = 3*5 + 2
        if i in fixed_cell_index_list:
            weights_t = []
            for name, m in model.cells[i].named_modules():
                m_name = str(type(m))
                if 'AvgPool' in m_name:
                    c_in, c_out = model.cells[i].in_dim, model.cells[i].out_dim
                    W_t = (torch.ones(c_in, c_out) / 9).view(-1).detach().cuda()
                    weights_t.append(W_t)
                elif 'Conv2d' in m_name or 'BatchNorm2d' in m_name:
                    for name, W in m.named_parameters():
                        W_t = W.view(-1).detach()
                        if 'weight' in name:
                            W_t = W.view(-1).detach()
                            weights_t.append(W_t)
        else:
            weights_t = []
            for path in path_index:
                weights_tt = []
                none_flag = False
                for layer_index in path:
                    for name, m in model.cells[i].layers[layer_index].named_modules():
                        m_name = str(type(m))
                        if 'AvgPool' in m_name:
                            c_in, c_out = model.cells[i].in_dim, model.cells[i].out_dim
                            W_t = (torch.ones(c_in, c_out) / 9).view(-1).detach().cuda()
                            weights_tt.append(W_t)
                        elif 'Conv2d' in m_name or 'BatchNorm2d' in m_name:
                            for name, W in m.named_parameters():
                                if 'weight' in name:
                                    W_t = W.view(-1).detach()
                                    weights_tt.append(W_t)
                        elif 'Identity' in m_name:
                            pass
                        elif 'Zero' in m_name:
                            none_flag = True
                    if none_flag:
                        weights_tt = []
                        break
                if len(weights_tt) > 0:
                    weights_t.append(torch.cat(weights_tt, -1))

        if len(weights_t) > 0:
            weights.append(torch.cat(weights_t, -1))
            # params.append(torch.cat(params_t, -1))
        else:
            weights.append([])
            isnan = True
    weights_t = []
    for name, W in model.classifier.named_parameters():
        if 'weight' in name:
            W_t = W.view(-1).detach()
            weights_t.append(W_t)
    weights.append(torch.cat(weights_t, -1))
    return weights, isnan


def get_features(model, dataloader):
    model.train()
    data_iter = iter(dataloader)
    input, target = data_iter.next()
    input = input.cuda(non_blocking=True)
    logits, features = model(input)
    return features, target


def sample_subset_by_class(full_data, data_split, class_indices, ipc=10, dataname='cifar'):
    subset_indices = []
    visit = [0 for i in range(len(class_indices))]
    for i, label in enumerate(full_data.targets):
        # if i not in data_split:
        #     continue
        if dataname == 'ImageNet16':
            label -= 1
        if label in class_indices:
            idx = class_indices.index(label)
            if visit[idx] < ipc:
                subset_indices.append(i)
                full_data.targets[i] = idx + 1 if dataname == 'ImageNet16' else idx  # set new target from 0 to C
                visit[idx] += 1
    sub_data = torch.utils.data.Subset(full_data, subset_indices)
    return sub_data, subset_indices


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('NASWOT')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


if __name__ == '__main__':
    syn_data_ckpt = torch.load('res_CIFAR10_ConvNet_10ipc.pt')
    syn_data, syn_label = syn_data_ckpt['data'][0][0], syn_data_ckpt['data'][0][1]
    syn_dataset = TensorDataset(syn_data, syn_label)
    shorttrain_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=256, num_workers=0, shuffle=True, pin_memory=True)
    loader_iter = iter(shorttrain_loader)
    img, label = next(loader_iter)