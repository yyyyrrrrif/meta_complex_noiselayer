
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
# import wideresnet as wrn
import torchvision.transforms as transforms

#hammer-spammer
def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
    (1 - mixing_ratio) * np.eye(num_classes)


def flip_smi(mixing_ratio, num_classes=10):
    # for CIFAR10
    r = mixing_ratio
    conf_matrix = np.eye(num_classes)
    conf_matrix[9][1] = r
    conf_matrix[9][9] = 1 - r
    conf_matrix[2][0] = r
    conf_matrix[2][2] = 1 - r
    conf_matrix[4][7] = r
    conf_matrix[4][4] = 1 - r
    conf_matrix[3][5] = r
    conf_matrix[3][3] = 1 - r
    return conf_matrix


def flip_nei(corruption_prob, num_classes):
    r = corruption_prob
    conf_matrix = np.eye(num_classes)
    for i in range(int(num_classes)):
        for j in range(int(num_classes)):
            if j == i:
                conf_matrix[i][j] = 1 - r
            elif j == (i + 1) % num_classes:
                conf_matrix[i][j] = r/2
            elif j == (i + 2) % num_classes:
                conf_matrix[i][j] = r/4
            elif j == (i + 3) % num_classes:
                conf_matrix[i][j] = r/4
    return conf_matrix
    #反对角阵
def flip_adver(corruption_prob, num_classes):
    conf_matrix = np.eye(num_classes)[::-1]
    return conf_matrix


def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C


###专门生成cvpr的四个矩阵
def special_nei(corruption_prob, num_classes):
    r = corruption_prob
    conf_matrix = np.eye(num_classes)
    for i in range(int(num_classes)):
        for j in range(int(num_classes)):
            if j == i:
                conf_matrix[i][j] = 0.5
            elif j == (i + 1) % 10:
                conf_matrix[i][j] = 0.3
            elif j == (i + 2) % 10:
                conf_matrix[i][j] = 0.1
            elif j == (i + 3) % 10:
                conf_matrix[i][j] = 0.1
    return conf_matrix


def special_smi(corruption_prob, num_classes):
    # for CIFAR10
    conf_matrix = np.eye(num_classes)
    conf_matrix[0][0] = 0.7
    conf_matrix[0][2] = 0.3
    conf_matrix[1][1] = 0.6
    conf_matrix[1][9] = 0.4
    conf_matrix[2][0] = 0.2
    conf_matrix[2][2] = 0.8
    conf_matrix[3][3] = 0.3
    conf_matrix[3][5] = 0.7
    conf_matrix[4][4] = 0.3
    conf_matrix[4][5] = 0.7
    conf_matrix[5][3] = 0.3
    conf_matrix[5][4] = 0.2
    conf_matrix[5][5] = 0.5
    conf_matrix[7][4] = 0.4
    conf_matrix[7][7] = 0.6
    conf_matrix[8][0] = 0.3
    conf_matrix[8][8] = 0.7
    conf_matrix[9][1] = 0.5
    conf_matrix[9][9] = 0.5
    return conf_matrix






class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]


    test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]


    def __init__(self, root='', train=True, meta=True, num_meta=1000,
    corruption_prob=0, corruption_type='unif', transform=None, target_transform=None,
    download=False, seed=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set
        self.meta = meta
        self.corruption_prob = corruption_prob
        self.corruption_type = corruption_type
        self.num_meta = num_meta
        np.random.seed(seed)
        if download:
            self.download()


        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
            ' You can use download=True to download it')


        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_labels_1 = []
            self.train_labels_2 = []
            self.train_labels_3 = []
            self.train_labels_4 = []
            self.train_coarse_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                    self.train_labels_1 += entry['labels']
                    self.train_labels_2 += entry['labels']
                    self.train_labels_3 += entry['labels']
                    self.train_labels_4 += entry['labels']
                    img_num_list = [int(self.num_meta/10)] * 10
                    num_classes = 10
                else:
                    self.train_labels += entry['fine_labels']
                    self.train_labels_1 += entry['fine_labels']
                    self.train_labels_2 += entry['fine_labels']
                    self.train_labels_3 += entry['fine_labels']
                    self.train_labels_4 += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    img_num_list = [int(self.num_meta/100)] * 100
                    num_classes = 100
                fo.close()


            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) # convert to HWC


            data_list_val = {}
            for j in range(num_classes):
                data_list_val[j] = [i for i, label in enumerate(self.train_labels) if label == j]

            idx_to_meta = []
            idx_to_train = []
            print(img_num_list)

            for cls_idx, img_id_list in data_list_val.items():
                np.random.shuffle(img_id_list)
                img_num = img_num_list[int(cls_idx)]
                idx_to_meta.extend(img_id_list[:img_num])
                idx_to_train.extend(img_id_list[img_num:])

            if meta is True:
                self.train_data = self.train_data[idx_to_meta]
                self.train_labels = list(np.array(self.train_labels)[idx_to_meta])
            else:
                self.train_data = self.train_data[idx_to_train]
                self.train_labels = list(np.array(self.train_labels)[idx_to_train])
                self.train_labels_1 = list(np.array(self.train_labels_1)[idx_to_train])
                self.train_labels_2 = list(np.array(self.train_labels_2)[idx_to_train])
                self.train_labels_3 = list(np.array(self.train_labels_3)[idx_to_train])
                self.train_labels_4 = list(np.array(self.train_labels_4)[idx_to_train])
            if corruption_type == 'cifar100_multi_3':
                self.train_coarse_labels = list(np.array(self.train_coarse_labels)[idx_to_train])


            if corruption_type == 'multi_2':
                C1 = flip_smi(self.corruption_prob, num_classes)
                C2 = flip_nei(self.corruption_prob, num_classes)
                print('smi-cm:\n', C1)
                print('nei-cm:\n', C2)
                self.C1 = C1
                self.C2 = C2
            elif corruption_type == 'special_4':
                C3 = uniform_mix_C(self.corruption_prob, num_classes)
                C2 = special_nei(self.corruption_prob, num_classes)
                C1 = special_smi(self.corruption_prob, num_classes)
                C4 = flip_adver(self.corruption_prob, num_classes)
                print('smi-cm:\n', C1)
                print('nei-cm:\n', C2)
                print('unif-CM:\n', C3)
                print('adver-cm:\n', C4)
                self.C1 = C1
                self.C2 = C2
                self.C3 = C3
                self.C4 = C4
            elif corruption_type == 'cifar10_multi_3':
                C1 = flip_smi(self.corruption_prob, num_classes)
                C2 = flip_nei(self.corruption_prob, num_classes)
                C3 = uniform_mix_C(self.corruption_prob, num_classes)
                # C4 = flip_adver(self.corruption_prob, num_classes)
                print('smi-cm:\n', C1)
                print('nei-cm:\n', C2)
                print('unif-CM:\n', C3)
                # print('adver-cm:\n', C4)
                self.C1 = C1
                self.C2 = C2
                self.C3 = C3
                # self.C4 = C4


            elif corruption_type == 'cifar100_multi_3':
                #粗标注类中以相同概率分类错误
                assert num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
                coarse_fine = []
                for i in range(20):
                    coarse_fine.append(set())
                for i in range(len(self.train_labels)):
                    coarse_fine[self.train_coarse_labels[i]].add(self.train_labels[i])
                for i in range(20):
                    coarse_fine[i] = list(coarse_fine[i])

                C1 = np.eye(num_classes) * (1 - corruption_prob)


                for i in range(20):
                    tmp = np.copy(coarse_fine[i])
                    for j in range(len(tmp)):
                        tmp2 = np.delete(np.copy(tmp), j) #删除第j列
                        C1[tmp[j], tmp2] += corruption_prob * 1/len(tmp2)
                self.C1 = C1
                np.save('./matrix/matrix_meta/cifar100_smi_%s.npy' % self.corruption_prob, C1)
                print('smi-cm:\n',C1)


                C2 = flip_nei(self.corruption_prob, num_classes)
                C3 = uniform_mix_C(self.corruption_prob, num_classes)
                # C4 = flip_adver(self.corruption_prob, num_classes)

                self.C2 = C2
                self.C3 = C3
                # self.C4 = C4

                print('nei-cm:\n', C2)
                print('unif-CM:\n', C3)
                # print('adver-cm:\n', C4)

            # elif corruption_type == 'clabels':
            # net = wrn.WideResNet(40, num_classes, 2, dropRate=0.3).cuda()
            # model_name = './cifar{}_labeler'.format(num_classes)
            # net.load_state_dict(torch.load(model_name))
            # net.eval()
            else:
                assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(corruption_type)


            if corruption_type == 'clabels':
                mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                std = [x / 255 for x in [63.0, 62.1, 66.7]]

                test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

                # obtain sampling probabilities
                sampling_probs = []
                print('Starting labeling')

                for i in range((len(self.train_labels) // 64) + 1):
                    current = self.train_data[i*64:(i+1)*64]
                    current = [Image.fromarray(current[i]) for i in range(len(current))]
                    current = torch.cat([test_transform(current[i]).unsqueeze(0) for i in range(len(current))], dim=0)

                    data = V(current).cuda()
                    logits = net(data)
                    smax = F.softmax(logits / 5) # temperature of 1
                    sampling_probs.append(smax.data.cpu().numpy())

                sampling_probs = np.concatenate(sampling_probs, 0)
                print('Finished labeling 1')


                new_labeling_correct = 0
                argmax_labeling_correct = 0
                for i in range(len(self.train_labels)):
                    old_label = self.train_labels[i]
                    new_label = np.random.choice(num_classes, p=sampling_probs[i])
                    self.train_labels[i] = new_label
                    if old_label == new_label:
                        new_labeling_correct += 1
                    if old_label == np.argmax(sampling_probs[i]):
                        argmax_labeling_correct += 1
                print('Finished labeling 2')
                print('New labeling accuracy:', new_labeling_correct / len(self.train_labels))
                print('Argmax labeling accuracy:', argmax_labeling_correct / len(self.train_labels))

            elif corruption_type == 'multi_2':
                for i in range(len(self.train_labels)):
                    self.train_labels_1[i] = np.random.choice(num_classes, p=C1[self.train_labels[i]])
                    self.train_labels_2[i] = np.random.choice(num_classes, p=C2[self.train_labels[i]])
                    self.corruption_matrix_1 = C1
                    self.corruption_matrix_2 = C2
            elif corruption_type == 'cifar10_multi_3' or corruption_type == 'cifar100_multi_3':
                for i in range(len(self.train_labels)):
                    self.train_labels_1[i] = np.random.choice(num_classes, p=C1[self.train_labels[i]])
                    self.train_labels_2[i] = np.random.choice(num_classes, p=C2[self.train_labels[i]])
                    self.train_labels_3[i] = np.random.choice(num_classes, p=C3[self.train_labels[i]])
                    self.corruption_matrix_1 = C1
                    self.corruption_matrix_2 = C2
                    self.corruption_matrix_3 = C3
            else:
                for i in range(len(self.train_labels)):
                    self.train_labels_1[i] = np.random.choice(num_classes, p=C1[self.train_labels[i]])
                    self.train_labels_2[i] = np.random.choice(num_classes, p=C2[self.train_labels[i]])
                    self.train_labels_3[i] = np.random.choice(num_classes, p=C3[self.train_labels[i]])
                    self.train_labels_4[i] = np.random.choice(num_classes, p=C4[self.train_labels[i]])
                    self.corruption_matrix_1 = C1
                    self.corruption_matrix_2 = C2
                    self.corruption_matrix_3 = C3
                    self.corruption_matrix_4 = C4


        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1)) # convert to HWC


    def __getitem__(self, index):
        if self.train:
            if self.meta:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                if self.corruption_type == 'multi_2':
                    img, target, target1, target2 = self.train_data[index], self.train_labels[index],\
                    self.train_labels_1[index], self.train_labels_2[index]
                elif self.corruption_type == 'cifar10_multi_3' or self.corruption_type == 'cifar100_multi_3':
                    img, target, target1, target2, target3 = self.train_data[index], self.train_labels[index],\
                    self.train_labels_1[index], self.train_labels_2[index],\
                    self.train_labels_3[index]
                elif self.corruption_type == 'multi_4' or self.corruption_type == 'special_4' or self.corruption_type == 'cifar100_multi_4':
                    img, target, target1, target2, target3, target4 = self.train_data[index], self.train_labels[index],\
                    self.train_labels_1[index], self.train_labels_2[index],\
                    self.train_labels_3[index], self.train_labels_4[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.train:
            if self.meta:
                return img, target
            else:
                if self.corruption_type == 'multi_2':
                    return img, target, target1, target2
                elif self.corruption_type == 'cifar10_multi_3' or self.corruption_type == 'cifar100_multi_3':
                    return img, target, target1, target2, target3
                elif self.corruption_type == 'multi_4' or self.corruption_type == 'special_4' or self.corruption_type == 'cifar100_multi_4':
                    return img, target, target1, target2, target3, target4
        else:
            return img, target


    def __len__(self):
        if self.train:
            if self.meta is True:
                return self.num_meta
            else:
                return 50000 - self.num_meta
        else:
            return 10000


    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)



class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    #url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    url = "file:///E:/meta-confusion-matrix/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
    ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]


    test_list = [
    ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
