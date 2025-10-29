"""

Contributed by Wenbin Li & Jinglin Xu

"""

import torch
import torch.nn as nn
from torch.nn import init
import functools

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_MultiViewNet(pretrained=False, model_root=None, which_model='multiviewNet', norm='batch', init_type='normal',
    use_gpu=True, num_classes=6, num_view=5, view_list=None, fea_out=200, fea_com=300, **kwargs):
    MultiviewNet = None
    norm_layer = get_norm_layer(norm_type=norm)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!进入define_MultiViewNet！！！！！！！！！！！！！！！！！")
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model == 'multiviewNet':
        MultiviewNet = MultiViewNet(num_classes=num_classes, num_view=num_view, view_list=view_list, 
            fea_out=fea_out, fea_com=fea_com, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(MultiviewNet, init_type=init_type)

    if use_gpu:
        MultiviewNet.cuda()

    if pretrained:
        MultiviewNet.load_state_dict(model_root)

    return MultiviewNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class MultiViewNet(nn.Module):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!进入MultiViewNet！！！！！！！！！！！！！！！！！！！！！")
    def __init__(self, num_classes, num_view, view_list, fea_out, fea_com):
        super(MultiViewNet, self).__init__()
        # list of the linear layer
        self.linear = []
        for i in range(len(view_list)):
            self.add_module('linear_'+str(i), nn.Sequential(
                nn.Linear(view_list[i], 2 * fea_out).cuda(),
                #view_list[i] → 就是 第 i 个视图输入特征的维度。它直接决定了 Linear(in_features=...) 的输入维度
                nn.BatchNorm1d(2 * fea_out).cuda(),           
                nn.ReLU(inplace=True).cuda(),
                nn.Dropout().cuda(),
                nn.Linear(2 * fea_out, fea_out).cuda(),
                nn.BatchNorm1d(fea_out).cuda(),              
                nn.ReLU(inplace=True).cuda()                  
                )
            )
        self.linear = AttrProxy(self, 'linear_') 

        self.relation_out = RelationBlock_Out()

        self.classifier_out = nn.Sequential(
            nn.Linear(num_view * fea_out, fea_com),
            nn.BatchNorm1d(fea_com),     
            nn.ReLU(inplace=True),
            nn.Dropout().cuda(),
            nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, input):
        Fea_list = []#初始化空列表用于存储结果
        print("\n===== 特征转换层（线性层）输出形状 =====")
        # 单视图特征提取
        for i, (input_item, linear_item) in enumerate(zip(input, self.linear)):
            # 步骤1：用线性层提取特征（输出形状：(批次大小, 5，比如64个样本，每个200维）
            # 打印输入到线性层的形状（每个视图）
            print(f"视图{i + 1}输入到线性层的形状：{input_item.shape}")
            fea_temp = linear_item(input_item)  # 形状：(64, 5)
        # for input_item, linear_item in zip(input, self.linear):#对输入列表中的每个元素应用不同的线性变换，并将结果收集到一个新列表中
        #     #使用zip()函数将两个可迭代对象(input和self.linear)配对，同时遍历输入数据列表和对应的线性变换层列表，每次迭代获取一对元素：input_item(输入数据)和linear_item(线性层)
        #     fea_temp = linear_item(input_item)#将当前输入数据input_item通过对应的线性层linear_item进行变换
        #     #这里的linear_item是一个PyTorch的nn.Linear层，执行的是矩阵乘法加偏置的操作：y = xA^T + b
        #     #每个视图先经过各自独立的两层 MLP（self.linear_i）得到单视图特征  先把各视图特征映射到同一维度
            # 打印线性层输出形状（应统一为 [batch_size, fea_out]，默认fea_out=200）
            print(f"视图{i + 1}线性层输出形状：{fea_temp.shape}")
            Fea_list.append(fea_temp)#将变换后的特征张量添加到结果列表中

        Relation_fea = self.relation_out(Fea_list)#对任意两视图 (i, j) 的特征做外积

        Fea_Relation_list = []
        print("\n===== 特征拼接与分类层输出形状 =====")
        for k in range(len(Fea_list)):
            # 打印原始特征和关联特征的形状
            print(f"视图{k + 1}原始特征形状：{Fea_list[k].shape}")  # [batch_size, 200]
            print(f"视图{k + 1}关联特征形状：{Relation_fea[k].shape}")  # 例如[batch_size, 600]
            # 拼接后形状：原始特征维度 + 关联特征维度
            Fea_Relation_temp = torch.cat((Fea_list[k], Relation_fea[k]), 1)
            print(f"视图{k + 1}拼接后特征形状：{Fea_Relation_temp.shape}")  # 例如[batch_size, 800]
            #torch.cat((Fea_list[k], Relation_fea[k]), dim=1)  dim=1 表示在特征维度（通道维）拼接
            #在 batch 内 对于第 i 个视图，把它与其他所有视图的关系向量拼接起来（剔除与自身的关系），再与它自己的特征拼接  每个视图的预测都参考了“它与其他所有视图的关系”，对多视图互补较友好。
            # 分类器输出（num_classes维度
            output = self.classifier_out(Fea_Relation_temp)
            print(f"视图{k + 1}分类器输出形状：{output.shape}")  # [batch_size, num_classes]
            Fea_Relation_list.append(output) #喂给一个共享的 classifier_out，得到每个视图各自的分类输出  逐视图分类
            print("####################################################################所有视图分类输出")
            print([x.shape for x in Fea_Relation_list])
        return Fea_Relation_list


class RelationBlock_Out(nn.Module):
    def __init__(self):
        super(RelationBlock_Out, self).__init__()

        self.linear_out = nn.Sequential(
            nn.Linear(200*200, 200),
            nn.BatchNorm1d(200),                       
            nn.ReLU(inplace=True)                                
        )

    def cal_relation(self, input1, input2):

        input1 = input1.unsqueeze(2)
        input2 = input2.unsqueeze(1)
        outproduct = torch.bmm(input1, input2)

        return outproduct    
 
    def forward(self, x):
        print("\n===== 关系特征计算层输出形状 =====")
        relation_eachview_list = []
        for i in range(len(x)):
            print(f"\n===== 以视图{i + 1}为中心计算关联特征 =====")
            relation_list = []
            for j in range(len(x)):
                # 打印输入到关联计算的两个视图特征形状
                print(f"视图{i + 1}与视图{j + 1}的特征形状：{x[i].shape}, {x[j].shape}")
                # 步骤1：计算外积（batch_size, fea_out, fea_out）
                relation_temp = self.cal_relation(x[i], x[j])
                print(f"外积结果形状：{relation_temp.shape}")  # 应为[batch_size, 200, 200]
                # 步骤2：展平为（batch_size, 200*200）
                relation_temp = relation_temp.view(relation_temp.size(0), 200*200)
                print(f"外积展平后形状：{relation_temp.shape}")  # 应为[batch_size, 40000]
                # 步骤3：线性层处理为（batch_size, 200）
                relation_temp = self.linear_out(relation_temp)
                print(f"线性层处理后关联特征形状：{relation_temp.shape}")  # 应为[batch_size, 200]
                relation_list.append(relation_temp)
             # 移除自身与自身的关联（i=j的情况）
            relation_list.pop(i)
            # 拼接其他所有视图的关联特征（(num_view-1)*200 维度）
            relation_eachview_temp = torch.cat(relation_list, 1)
            print(f"视图{i + 1}的所有关联特征拼接后形状：{relation_eachview_temp.shape}")  # 例如4个视图时为[batch_size, 600]
            relation_eachview_list.append(relation_eachview_temp)

        return relation_eachview_list   
