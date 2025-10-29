"""

Contributed by Wenbin Li & Jinglin Xu

"""

from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time
from torch import autograd
from PIL import ImageFile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('Agg')

from dataset.AWADataset import animalAttrData
from dataset.BigEarthNetS1Dataset import BigEarthNetS1Dataset  # 新增：BigEarthNet-S1数据集
import models.network_MvNNcor as MultiviewNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Load the pre-trained model
data_name = 'AWA'
data_dir = './mvdata/AWA/Features'
model_trained = '/home/limn/mvnn/Mvnn/results/MvNNcor_AWA_Epochs_100_6.0/model_best.pth.tar'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default=data_dir, help='the path of data')
parser.add_argument('--data_name', default=data_name, help='The name of the data')
parser.add_argument('--mode', default='test', help='train|val|test')
parser.add_argument('--outf', default='./results/MvNNcor', help='初始输出目录（与训练一致）')
parser.add_argument('--resume', default=model_trained, help='use the saved model')
parser.add_argument('--basemodel', default='multiviewNet', help='multiviewNet')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=64, help='the mini-batch size of training')
parser.add_argument('--testSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100, help='训练时的epochs（用于生成路径，需与训练一致）')
parser.add_argument('--num_classes', type=int, default=50, help='the number of classes')
parser.add_argument('--num_view', type=int, default=6, help='the number of views')
parser.add_argument('--diffusion_K', type=int, default=10, help='扩散图的K值（需与训练时一致）')
parser.add_argument('--fea_out', type=int, default=200, help='the dimension of the first linear layer')
parser.add_argument('--fea_com', type=int, default=300, help='the dimension of the combination layer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gamma', type=float, default=6.0, help='the power of the weight for each view')
parser.add_argument('--loss_type', default='cross_entropy', help='loss type: cross_entropy | bce_with_logits')  # 新增

opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True
# 1. 复用训练代码的opt.outf生成逻辑（参数顺序、连接符完全一致）
opt.outf = f"{opt.outf}_{opt.data_name}_Epochs_{opt.epochs}_{opt.gamma}"
# 2. 动态拼接model_best.pth.tar的完整路径
opt.resume = os.path.join(opt.outf, 'model_best.pth.tar')
# 3. 打印生成的路径，方便调试
print(f"✅ 动态生成模型加载路径：{opt.resume}")
print(f"✅ 动态生成结果保存目录：{opt.outf}")
if not os.path.exists(opt.resume):
    print(f"❌ 动态生成的模型路径不存在：{opt.resume}")
    print(f"请检查：1. 是否已运行训练指令 2. 训练时的--epochs={opt.epochs}、--gamma={opt.gamma}是否与测试一致")
    print(f"训练建议指令：python MvNNcor_Train.py --dataset_dir=./mvdata/AWA/Features --data_name=AWA --num_classes=50 --num_view=6 --gamma={opt.gamma} --epochs={opt.epochs} --diffusion_K=10")
    exit(1)  # 路径不存在时退出，避免后续报错
else:
    print(f"✅ 模型路径存在，准备加载：{opt.resume}")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# save the opt and results to txt file
opt.outf = opt.outf+'_'+opt.data_name+'_Epochs_'+str(opt.epochs)+'_'+str(opt.gamma)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# 谱图保存目录（新增：训练结果目录下的 "spectrums" 子文件夹）
spectrum_save_dir = os.path.join(opt.outf, 'spectrums')
txt_save_path = os.path.join(opt.outf, 'test_results.txt')
F_txt = open(txt_save_path, 'a+')

# ======================================== 数据集配置字典（多数据集适配） ========================================
DATASET_CONFIG = {
    'AWA': {
        'dataset_class': animalAttrData,
        'loss_type': 'cross_entropy',
        'split_key': 'mode',
        'params': {
            'diffusion_K': 5,
            'diffusion_matrix': 'laplacian',
            'normalize_feats': True,
            'threshold_affinity': True,
            'use_cache': True,
            'cache_dir': './cache/eigenvalues',
            'use_parallel': True,
            'num_workers': 6
        }
    },
    'BigEarthNet-S1': {
        'dataset_class': BigEarthNetS1Dataset,
        'loss_type': 'bce_with_logits',
        'split_key': 'split',
        'params': {
            'normalize_feats': True,
            'which_matrix': 'affinity',
            'K': None,  # 运行时设置
            'use_cache': True,
            'cache_dir': './cache/bigearthnet_s1',
            'num_classes': None,
            'log_file': None
        }
    }
}

# ======================================== 动态加载数据集 ========================================
if opt.data_name not in DATASET_CONFIG:
    raise ValueError(f"不支持的数据集: {opt.data_name}。支持的数据集: {list(DATASET_CONFIG.keys())}")

config = DATASET_CONFIG[opt.data_name]
opt.loss_type = config['loss_type']

print(f"\n{'='*60}")
print(f"加载测试数据集: {opt.data_name}")
print(f"损失函数类型: {opt.loss_type}")
print(f"{'='*60}\n")

# 准备数据集参数
dataset_params = config['params'].copy()
if opt.data_name == 'BigEarthNet-S1':
    dataset_params['K'] = opt.fea_out
    dataset_params['num_classes'] = opt.num_classes
    dataset_params['save_dir'] = spectrum_save_dir
    dataset_params['log_file'] = 'test_dataset.log'
else:
    dataset_params['save_spectrum_dir'] = spectrum_save_dir

# 创建测试集
testset = config['dataset_class'](
    data_dir=opt.dataset_dir,
    **{config['split_key']: opt.mode},
    **dataset_params
)

print('Testset: %d' %len(testset))
print('Testset: %d' %len(testset), file=F_txt)

# ========================================== Load Datasets ==============================================
# 自定义collate_fn（通用多数据集适配）
def custom_collate_fn(batch):
    """'通用collate函数，根据数据集配置自动适配"""
    if isinstance(batch[0], dict):
        # 字典格式（如BigEarthNet-S1）
        sample_set = [[] for _ in range(opt.num_view)]
        sample_targets = []
        for item in batch:
            for v in range(opt.num_view):
                sample_set[v].append(item['sample_set'][v])
            sample_targets.append(item['sample_targets'])
        sample_set = [torch.stack(view_samples) for view_samples in sample_set]
        sample_targets = torch.stack(sample_targets)
        return sample_set, sample_targets
    else:
        # 元组格式（如AWA）
        return torch.utils.data.dataloader.default_collate(batch)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=opt.testSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True,
    collate_fn=custom_collate_fn  # 新增
    ) 
print(opt)
print(opt, file=F_txt)

# ========================================== Model config ===============================================
global best_prec1, epoch, weight_var
best_prec1 = 0
epoch = 0
weight_var = torch.ones(opt.num_view) * (1/opt.num_view)
weight_var = weight_var.to("cuda")

test_iter = iter(test_loader)
testdata, target = next(test_iter)
view_list = []
for v in range(len(testdata)):
    temp_size = testdata[v].size()
    view_list.append(temp_size[1])
# 打印view_list（各视图的特征维度）
print("Test view_list (各视图特征维度):", view_list)
# 打印视图数量
print("视图数量:", len(view_list))
print("测试集原始输入特征形状（每个视图）：")
for v in range(len(testdata)):
    print(f"视图{v+1}：{testdata[v].shape}")

ngpu = int(opt.ngpu)
model = MultiviewNet.define_MultiViewNet(which_model=opt.basemodel, norm='batch', init_type='normal', 
    use_gpu=opt.cuda, num_classes=opt.num_classes, num_view=opt.num_view, view_list=view_list,
    fea_out=opt.fea_out, fea_com=opt.fea_com)

# ======================================= 损失函数配置（多数据集适配） =======================================
LOSS_FUNCTIONS = {
    'cross_entropy': nn.CrossEntropyLoss().cuda(),
    'bce_with_logits': nn.BCEWithLogitsLoss().cuda()
}

if opt.loss_type not in LOSS_FUNCTIONS:
    raise ValueError(f"不支持的损失函数类型: {opt.loss_type}")

criterion = LOSS_FUNCTIONS[opt.loss_type]
print(f"\n使用损失函数: {opt.loss_type} - {criterion.__class__.__name__}")
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        weight_var = checkpoint['weight_var']
        model.load_state_dict(checkpoint['state_dict'])
        # print(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

if opt.ngpu > 1:
    model = nn.DataParallel(model, range(opt.ngpu))

print(model) 
print(model, file=F_txt)

# ======================================= Define functions =============================================
def visualize_features(model, test_loader, save_dir):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for sample_set, sample_targets in test_loader:
            input_var = [x.cuda() for x in sample_set]
            # 提取线性层输出的特征（修改根据模型结构调整）
            Fea_list = []
            for input_item, linear_item in zip(input_var, model.linear):
                fea_temp = linear_item(input_item)
                Fea_list.append(fea_temp)
            # 取第一个视图的特征作为示例
            fea = Fea_list[0].cpu().numpy()
            features.extend(fea)
            labels.extend(sample_targets.numpy())

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(np.array(features))

    # 绘图
    plt.figure(figsize=(10, 8))
    for c in range(opt.num_classes):
        mask = np.array(labels) == c
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f'Class {c}', s=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of Features (View 0)')
    save_path = os.path.join(save_dir, 'feature_tsne.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Feature t-SNE plot saved to {save_path}")

def validate(val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():

        end = time.time()

        for index, (sample_set, sample_targets) in enumerate(val_loader):

            input_var = [sample_set[i].cuda() for i in range(len(sample_set))]

            # deal with the target
            target_var = sample_targets.cuda()

            Output_list = model(input_var)
            loss = torch.zeros(1).to("cuda")
            
            # pdb.set_trace()
            for v in range(len(Output_list)):
                loss_temp = criterion(Output_list[v], target_var)
                loss += (weight_var[v] ** gamma) * loss_temp

            output_var = torch.stack(Output_list)
            # 保护性回退：若权重为全零，改为均匀权重
            if torch.sum(weight_var) == 0:
                weight_var = torch.ones_like(weight_var) / weight_var.numel()
            weight_var = weight_var.unsqueeze(1)
            weight_var = weight_var.unsqueeze(2)
            weight_var = weight_var.expand(weight_var.size(0), opt.testSize, opt.num_classes)
            output_weighted = weight_var * output_var
            output_weighted = torch.sum(output_weighted, 0)
            _, preds = output_weighted.topk(1, 1, True, True)  # 获取top1预测
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(sample_targets.numpy().flatten())

            weight_var = weight_var[:,:,1]
            weight_var = weight_var[:,1]

            # ======== 评估指标计算（多数据集适配） ========
            if opt.loss_type == 'bce_with_logits':
                # 多标签分类评估
                from dataset.BigEarthNetS1Dataset import multi_label_metrics
                micro_f1, macro_f1, precision, recall = multi_label_metrics(output_weighted, target_var)
                losses.update(loss.item(), target_var.size(0))
                top1.update(micro_f1, target_var.size(0))
                top5.update(macro_f1, target_var.size(0))
            else:
                # 单标签分类评估
                prec1, prec5 = accuracy(output_weighted, target_var, topk=(1, 5))
                losses.update(loss.item(), target_var.size(0))
                top1.update(prec1[0], target_var.size(0))
                top5.update(prec5[0], target_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % opt.print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, index, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5))

                print('Test: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, index, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5), file=F_txt)

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1, top5=top5, best=best_prec1))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1, top5=top5, best=best_prec1), file=F_txt)
        print(weight_var)
        print(weight_var, file=F_txt)
        # 计算混淆矩阵
    num_classes = opt.num_classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        confusion_matrix[t][p] += 1

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # 保存图片
    save_path = os.path.join(opt.outf, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ============================================ Testing phase ========================================
print('start testing.........')
start_time = time.time()  
gamma = torch.tensor(opt.gamma).to("cuda")
prec2 = validate(test_loader, model, weight_var, gamma, criterion, best_prec1, F_txt)
visualize_features(model, test_loader, opt.outf)
F_txt.close()

# ============================================ Testing End ========================================
