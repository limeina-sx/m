"""

Contributed by Wenbin Li & Jinglin Xu

"""

from __future__ import print_function
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # 非交互模式，支持后台保存图片
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

from dataset.AWADataset import animalAttrData
# BigEarthNet-S1Dataset 将在需要时延迟导入，避免影响AWA训练
import models.network_MvNNcor as MultiviewNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
# 在训练前初始化记录列表
# 在训练前初始化记录列表（删除冗余变量，规范命名）
train_losses = []
train_acc1s = []
# 验证集指标（规范命名：val_loss1s=验证集损失，val_prec1s=验证集准确率）
val_loss1s = []
val_prec1s = []
# 测试集指标（规范命名：test_losses=测试集损失，test_prec1s=测试集准确率）
test_losses = []
test_prec1s = []

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./mvdata/AWA/Features', help='the path of data')
parser.add_argument('--data_name', default='AWA', help='The name of the data')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/MvNNcor')
parser.add_argument('--net', default='', help='use the saved model')
parser.add_argument('--basemodel', default='multiviewNet', help='multiviewNet')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=64, help='the mini-batch size of training')
parser.add_argument('--testSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--num_classes', type=int, default=50, help='the number of classes')
parser.add_argument('--num_view', type=int, default=6, help='the number of views')
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
parser.add_argument('--loss_type', default='cross_entropy', help='loss type: cross_entropy (for AWA) | bce_with_logits (for BigEarthNet-S1 multi-label)')  # 新增：损失函数类型
parser.add_argument('--diffusion_K', type=int, default=10, help='扩散图的K值（特征图谱分析后确定）')
opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# save the opt and results to txt file
opt.outf = opt.outf+'_'+opt.data_name+'_Epochs_'+str(opt.epochs)+'_'+str(opt.gamma)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

txt_save_path = os.path.join(opt.outf, 'opt_results.txt')
F_txt = open(txt_save_path, 'a+')
spectrum_save_dir = os.path.join(opt.outf, 'spectrums')
# ======================================== 数据集配置字典（多数据集适配） ========================================
DATASET_CONFIG = {
    'AWA': {
        'dataset_class': animalAttrData,
        'loss_type': 'cross_entropy',
        'splits': ['train', 'val', 'test'],
        'split_key': 'mode',  # AWA使用'mode'参数
        'params': {
            'diffusion_K': 10,
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
        'dataset_class': None,  # 延迟导入，避免影响AWA
        'loss_type': 'bce_with_logits',
        'splits': ['train', 'val', 'test'],
        'split_key': 'split',  # BigEarthNet-S1使用'split'参数
        'params': {
            'normalize_feats': True,
            'which_matrix': 'affinity',
            'K': None,  # 将在运行时设置为opt.fea_out
            'use_cache': True,
            'cache_dir': './cache/bigearthnet_s1',
            'num_classes': None,  # 将在运行时设置为opt.num_classes
            'use_polar_features': False,  # 禁用极化特征（避免复杂计算）
            'use_texture_features': False,  # 禁用纹理特征（避免skimage依赖）
            'use_data_augmentation': False,  # 禁用数据增强（加速训练）
            'cache_format': 'pickle'  # 使用pickle而非h5py
            # log_file 将在创建数据集时动态添加
        }
    }
    # 在此添加更多数据集配置...
    # 'NewDataset': { ... }
}

# ======================================== 动态加载数据集 ========================================
if opt.data_name not in DATASET_CONFIG:
    raise ValueError(f"不支持的数据集: {opt.data_name}。支持的数据集: {list(DATASET_CONFIG.keys())}")

config = DATASET_CONFIG[opt.data_name]

# 延迟导入 BigEarthNet-S1Dataset（只在需要时导入）
if opt.data_name == 'BigEarthNet-S1' and config['dataset_class'] is None:
    try:
        from dataset.BigEarthNetS1Dataset import BigEarthNetS1Dataset
        config['dataset_class'] = BigEarthNetS1Dataset
        print("✅ 成功导入 BigEarthNetS1Dataset")
    except ImportError as e:
        print(f"❌ 无法导入 BigEarthNetS1Dataset: {e}")
        print("请安装依赖: pip install scikit-image h5py")
        raise

opt.loss_type = config['loss_type']  # 设置损失函数类型

print(f"\n{'='*60}")
print(f"加载数据集: {opt.data_name}")
print(f"损失函数类型: {opt.loss_type}")
print(f"{'='*60}\n")

# 准备数据集参数（运行时替换占位符）
dataset_params = config['params'].copy()
if opt.data_name == 'BigEarthNet-S1':
    dataset_params['K'] = opt.fea_out
    dataset_params['num_classes'] = opt.num_classes
    dataset_params['save_dir'] = spectrum_save_dir
else:
    dataset_params['save_spectrum_dir'] = spectrum_save_dir

# 动态创建train数据集
train_params = dataset_params.copy()
if opt.data_name == 'BigEarthNet-S1':
    train_params['log_file'] = 'train_dataset.log'
trainset = config['dataset_class'](
    data_dir=opt.dataset_dir,
    **{config['split_key']: 'train' if opt.data_name == 'BigEarthNet-S1' else opt.mode},
    **train_params
)

# 动态创建val数据集
val_params = dataset_params.copy()
if opt.data_name == 'BigEarthNet-S1':
    val_params['log_file'] = 'val_dataset.log'
valset = config['dataset_class'](
    data_dir=opt.dataset_dir,
    **{config['split_key']: 'val'},
    **val_params
)

# 动态创建test数据集
test_params = dataset_params.copy()
if opt.data_name == 'BigEarthNet-S1':
    test_params['log_file'] = 'test_dataset.log'
testset = config['dataset_class'](
    data_dir=opt.dataset_dir,
    **{config['split_key']: 'test'},
    **test_params
)

print('Traimmmmmmnset: %d' %len(trainset))
print('Valset: %d' %len(valset))
print('Testset: %d' %len(testset))
print('Trainset: %d' %len(trainset), file=F_txt)
print('Valset: %d' %len(valset), file=F_txt)
print('Testset: %d' %len(testset), file=F_txt)

# ========================================== Load Datasets ==============================================
# 自定义collate_fn（通用多数据集适配）
def custom_collate_fn(batch):
    """通用collate函数，根据数据集配置自动适配"""
    config = DATASET_CONFIG.get(opt.data_name)
    
    # 检查第一个样本的类型，判断是字典还是元组
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

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.batchSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True,
    collate_fn=custom_collate_fn  # 新增：自定义collate
    )
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=opt.testSize, shuffle=False, 
    num_workers=int(opt.workers), drop_last=False, pin_memory=True,
    collate_fn=custom_collate_fn
    ) 
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=opt.testSize, shuffle=False, 
    num_workers=int(opt.workers), drop_last=False, pin_memory=True,
    collate_fn=custom_collate_fn
    ) 
print(opt)
print(opt, file=F_txt)

# ========================================== Model config ===============================================
train_iter = iter(train_loader)
traindata, target = next(train_iter)
view_list = []

for v in range(len(traindata)):
    temp_size = traindata[v].size()
    print(type(temp_size))
    print(temp_size)
    #取第 v 个视图张量的形状（返回 torch.Size，类似元组）。例如 torch.Size([B, F])
    view_list.append(temp_size[1])
    #取形状的第二维 temp_size[1]（通常是单样本的 feature dimension F），追加到 view_list。结果例如 [200, 252, 200, ...]。
# 打印原始输入特征形状（每个视图）
print("原始输入特征形状（每个视图）：")
for v in range(len(traindata)):
    # traindata[v]形状为 [batch_size, 视图v的特征维度]
    print(f"视图{v+1}：{traindata[v].shape}")  # 例如：torch.Size([64, 200])
ngpu = int(opt.ngpu)
model = MultiviewNet.define_MultiViewNet(which_model=opt.basemodel, norm='batch', init_type='normal', 
    use_gpu=opt.cuda, num_classes=opt.num_classes, num_view=opt.num_view, view_list=view_list,
    fea_out=opt.fea_out, fea_com=opt.fea_com)

#num_classes是输出类别数 num_view比如同一个物体有 12 个视角图片，那 num_view=12  view_list每个视图输入的特征维度列表  fea_com多视图特征融合后的维度
# traindata[v].size() 形如 [batch_size, C, H, W]
if opt.net != '': 
    model.load_state_dict(torch.load(opt.net))

if opt.ngpu > 1:
    model = nn.DataParallel(model, range(opt.ngpu))

print(model) 
print(model, file=F_txt)

# ======================================= 损失函数配置（多数据集适配） =======================================
LOSS_FUNCTIONS = {
    'cross_entropy': nn.CrossEntropyLoss().cuda(),
    'bce_with_logits': nn.BCEWithLogitsLoss().cuda()
}

if opt.loss_type not in LOSS_FUNCTIONS:
    raise ValueError(f"不支持的损失函数类型: {opt.loss_type}。支持: {list(LOSS_FUNCTIONS.keys())}")

criterion = LOSS_FUNCTIONS[opt.loss_type]
print(f"\n使用损失函数: {opt.loss_type} - {criterion.__class__.__name__}")
optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr,
    betas=(opt.beta1, 0.9)
)

# ======================================= Define functions =============================================
def reset_grad():
    model.zero_grad()

def adjust_learning_rate(optimizer, epoch):
    """修改：学习率每30epoch衰减为原来的50%，避免下降过快"""
    lr = opt.lr * (0.1 ** (epoch // 30))  # 原0.05→改为0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 新增：打印学习率，方便初学者观察
    print(f"Epoch {epoch} - 调整后学习率：{lr:.6f}")

def train(train_loader, model, weight_var, gamma, criterion, optimizer, epoch, F_txt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for index, (sample_set, sample_targets) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = [sample_set[i].cuda() for i in range(len(sample_set))]

        # deal with the target
        target_var = sample_targets.to("cuda")

        # compute output
        Output_list = model(input_var)
        print("\n===== 模型输出与融合形状 =====")
        for v, out in enumerate(Output_list):
            print(f"视图{v + 1}的分类输出形状：{out.shape}")  # [batch_size, num_classes]
        weight_up_list = []
        loss = torch.zeros(1).to("cuda")

        for v in range(len(Output_list)):
            # 统一的损失计算（criterion会根据类型自动处理）
            loss_temp = criterion(Output_list[v], target_var)
            loss += (weight_var[v] ** gamma) * loss_temp
            loss = loss / len(Output_list)  # 关键：除以视图数量，得到总平均损失
            weight_up_temp = loss_temp ** (1/(1-gamma))
            weight_up_list.append(weight_up_temp)   

        output_var = torch.stack(Output_list)
        print(f"所有视图输出堆叠后形状：{output_var.shape}")  # [num_view, batch_size, num_classes]

        weight_var = weight_var.unsqueeze(1)
        weight_var = weight_var.unsqueeze(2)
        actual_batch_size = output_var.size(1)
        weight_var = weight_var.expand(weight_var.size(0), actual_batch_size, opt.num_classes)
        output_weighted = weight_var * output_var
        output_weighted = torch.sum(output_weighted, 0)
        print(f"加权融合后最终输出形状：{output_weighted.shape}")  # [batch_size, num_classes]

        weight_var = weight_var[:,:,1]
        weight_var = weight_var[:,1]
        weight_up_var = torch.FloatTensor(weight_up_list).to("cuda")
        weight_down_var = torch.sum(weight_up_var)
        weight_var = torch.div(weight_up_var, weight_down_var)
        
        # ======== 评估指标计算（多数据集适配） ========
        if opt.loss_type == 'bce_with_logits':
            # 多标签分类评估
            from dataset.BigEarthNetS1Dataset import multi_label_metrics
            micro_f1, macro_f1, precision, recall = multi_label_metrics(output_weighted, target_var)
            losses.update(loss.item(), target_var.size(0))
            top1.update(micro_f1, target_var.size(0))  # Micro-F1
            top5.update(macro_f1, target_var.size(0))  # Macro-F1
        else:
            # 单标签分类评估
            prec1, prec5 = accuracy(output_weighted, target_var, topk=(1, 5))
            losses.update(loss.item(), target_var.size(0))
            top1.update(prec1[0], target_var.size(0))
            top5.update(prec5[0], target_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % opt.print_freq == 0:
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, index, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, index, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5), file=F_txt)

    return weight_var, losses.avg, top1.avg

def validate(epoch, val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    linear0_weight_mean = model.linear[0][0].weight.mean().item()
    print(f"【验证中】Epoch {epoch} - 第一个线性层（linear_0）权重均值：{linear0_weight_mean:.6f}")
    with torch.no_grad():

        end = time.time()

        for index, (sample_set, sample_targets) in enumerate(val_loader):

            input_var = [sample_set[i].cuda() for i in range(len(sample_set))]

            # deal with the target
            target_var = sample_targets.cuda()

            Output_list = model(input_var)
            loss = torch.zeros(1).to("cuda")

            for v in range(len(Output_list)):
                loss_temp = criterion(Output_list[v], target_var)
                loss += (weight_var[v] ** gamma) * loss_temp
            loss = loss / len(Output_list)  # 关键：除以视图数量，得到总平均损失
            output_var = torch.stack(Output_list)
            weight_var = weight_var.unsqueeze(1)
            weight_var = weight_var.unsqueeze(2)
            actual_batch_size = output_var.size(1)
            weight_var = weight_var.expand(weight_var.size(0), actual_batch_size, opt.num_classes)
            output_weighted = weight_var * output_var
            output_weighted = torch.sum(output_weighted, 0)

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
    
    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        file_model_best = os.path.join(opt.outf, 'model_best.pth.tar')
        shutil.copyfile(filename, file_model_best)

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


# ============================================ Training phase ========================================
print('start training.........')
start_time = time.time()
best_prec1 = 0
weight_var = torch.ones(opt.num_view) * (1 / opt.num_view)
weight_var = weight_var.to("cuda")
gamma = torch.tensor(opt.gamma).to("cuda")

for epoch in range(opt.epochs):
    # 1. 调整学习率
    adjust_learning_rate(optimizer, epoch)
    # 2. 训练当前epoch，获取训练指标
    weight_var, train_loss, train_acc1 = train(train_loader, model, weight_var, gamma, criterion, optimizer, epoch,
                                               F_txt)
    train_losses.append(train_loss.cpu().item() if isinstance(train_loss, torch.Tensor) else train_loss)
    train_acc1s.append(train_acc1.cpu().item() if isinstance(train_acc1, torch.Tensor) else train_acc1)
    linear0_weight_mean = model.linear[0][0].weight.mean().item()
    print(f"\n【验证前】Epoch {epoch} - 第一个线性层（linear_0）权重均值：{linear0_weight_mean:.6f}")
    # 3. 验证集评估（记录指标，添加调试打印）
    print('=============== Testing in the validation set ===============')
    prec1, val_loss1 = validate(epoch, val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt)
    # 记录验证集指标（使用规范后的变量名）
    val_loss1s.append(val_loss1.cpu().item() if isinstance(val_loss1, torch.Tensor) else val_loss1)
    val_prec1s.append(prec1.cpu().item() if isinstance(prec1, torch.Tensor) else prec1)
    # 调试打印：确认验证集指标已记录（长度应随epoch递增）
    print(f"Epoch {epoch} - 验证集指标记录：损失列表长度={len(val_loss1s)}，准确率列表长度={len(val_prec1s)}")

    # 4. 测试集评估（若启用，需同步修改变量名；若注释则列表为空，绘图时不影响）
    print('================== Testing in the test set ==================')
    prec2, val_loss2 = validate(epoch, test_loader, model, weight_var, gamma, criterion, best_prec1, F_txt)
    # 记录测试集指标（使用规范后的变量名）
    test_losses.append(val_loss2.cpu().item() if isinstance(val_loss2, torch.Tensor) else val_loss2)
    test_prec1s.append(prec2.cpu().item() if isinstance(prec2, torch.Tensor) else prec2)
    # 调试打印：确认测试集指标已记录
    print(f"Epoch {epoch} - 测试集指标记录：损失列表长度={len(test_losses)}，准确率列表长度={len(test_prec1s)}")

    # 5. 保存最佳模型（逻辑不变）
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    print(
        f'Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc1:.2f}%; '
        f'Val Loss {val_loss1:.4f}, Val Acc {prec1:.2f}%; '
        f'Test Loss {val_loss2:.4f}, Test Acc {prec2:.2f}%'
    )
    # 保存 checkpoint（逻辑不变）
    filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' % epoch)
    if is_best:
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': opt.basemodel,
                'state_dict': model.state_dict(),
                'weight_var': weight_var,
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename)

# 训练结束后绘制并保存曲线
# 训练结束后绘制并保存曲线（修改后）
def plot_metrics(save_dir):
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if len(train_losses) > 0:
        plt.plot(range(1, len(train_losses)+1), train_losses,
                 label='Train Loss', color='red', linestyle='-', marker='o', markersize=3)
    # 验证集损失：蓝色虚线，带方块标记
    if len(val_loss1s) > 0:
        plt.plot(range(1, len(val_loss1s)+1), val_loss1s,
                 label='Val Loss', color='blue', linestyle='--', marker='s', markersize=3)
    # 测试集损失：绿色点线，带三角形标记
    if len(test_losses) > 0:
        plt.plot(range(1, len(test_losses)+1), test_losses,
                 label='Test Loss', color='green', linestyle='-.', marker='^', markersize=3)
    # 图表标注
    plt.xlabel('Epoch', fontsize=11)  # 放大字体
    plt.ylabel('Loss', fontsize=11)
    plt.title('Loss Curve', fontsize=12, fontweight='bold')  # 加粗标题
    plt.legend(fontsize=10)  # 调整图例字体
    plt.grid(True, alpha=0.3)  # 添加网格线，方便看数值
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    # 训练集准确率：红色实线
    if len(train_acc1s) > 0:
        plt.plot(range(1, len(train_acc1s)+1), train_acc1s,
                 label='Train Acc@1', color='red', linestyle='-', marker='o', markersize=3)
    # 验证集准确率：蓝色虚线
    if len(val_prec1s) > 0:
        plt.plot(range(1, len(val_prec1s)+1), val_prec1s,
                 label='Val Acc@1', color='blue', linestyle='--', marker='s', markersize=3)
    # 测试集准确率：绿色点线
    if len(test_prec1s) > 0:
        plt.plot(range(1, len(test_prec1s)+1), test_prec1s,
                 label='Test Acc@1', color='green', linestyle='-.', marker='^', markersize=3)
    # 图表优化
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Accuracy Curve', fontsize=12)

    # 保存图片
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'train_metrics.png')
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")


# 调用绘图函数（放在训练循环结束后）
plot_metrics(opt.outf)
print('======== Training END ========')
F_txt.close()

# ============================================ Training End ========================================
