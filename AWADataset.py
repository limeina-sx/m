"""

Contributed by Wenbin Li & Jinglin Xu
多视图数据加载：处理具有多个特征视图的数据集

结构化数据组织：按照类别组织数据，确保每个样本有多个特征表示

数据划分：将每个类别的样本按比例划分为训练、验证和测试集

特征加载：从文本文件中加载特征并转换为PyTorch张量

接口标准化：实现了PyTorch数据集的标准接口（__len__和__getitem__）
"""

import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from PIL import Image
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import warnings
import sys
import pickle  # 新增:用于缓存
import hashlib  # 新增:用于生成缓存key
from multiprocessing import Pool, cpu_count  # 新增:用于并行处理
from functools import partial  # 新增:用于传递额外参数
warnings.filterwarnings('ignore')  # 忽略警告信息
torch.multiprocessing.set_sharing_strategy('file_system')
#设置PyTorch多进程共享策略为'file_system'，这是为了解决多进程数据加载中的共享内存问题

def process_single_view_wrapper(args):
    """
    并行处理的包装函数(用于multiprocessing.Pool)
    由于Pool.map只能传递一个参数,需要将所有参数打包
    """
    view_idx, feats_np, params = args
    
    # 转换numpy数组为tensor
    feats = torch.from_numpy(feats_np).float()
    
    # 调用原始处理函数
    result = process_features_to_diffusion(
        feats=feats,
        view_idx=view_idx,
        **params  # 解包参数字典
    )
    
    return view_idx, result

def compute_cache_key(feats, normalize_feats, threshold_affinity, k_ratio=0.005, which_matrix='laplacian'):
    """生成缓存唯一标识
    基于特征形状+参数生成缓存key(避免对全部数据计算hash)
    """
    # 只使用形状和参数生成key(计算快速)
    shape_str = f"shape{feats.shape[0]}x{feats.shape[1]}"
    params_str = f"norm{int(normalize_feats)}_thresh{int(threshold_affinity)}_k{k_ratio}_mat{which_matrix}"
    return f"{shape_str}_{params_str}"

def process_features_to_diffusion(feats,  # 输入特征 (N, D)
                                 which_matrix='laplacian',  # 矩阵类型
                                 K=5,  # 提取的特征向量数量
                                 normalize_feats=True,  # 是否归一化特征
                                 threshold_affinity=True,  # 是否过滤
                                 save_dir = './results/spectrums',  # 谱图保存目录
                                 view_idx = 0,  # 视图索引
                                 use_cache=True,  # 新增:是否启用缓存
                                 cache_dir='./cache/eigenvalues'):  # 新增:缓存目录
    """
    将输入特征通过扩散图处理，输出拉普拉斯/亲和矩阵的特征向量
    新增:支持缓存机制，首次计算后保存，后续直接加载
    """
    
    # ========== 缓存检查 ==========
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = compute_cache_key(feats, normalize_feats, threshold_affinity, 0.005, which_matrix)
        cache_file = os.path.join(cache_dir, f'view_{view_idx}_{cache_key}.pkl')
        
        # 尝试加载缓存
        if os.path.exists(cache_file):
            print(f"\n[视图{view_idx+1}] 🚀 从缓存加载特征值和特征向量")
            print(f"  缓存文件: {os.path.basename(cache_file)}")
            sys.stdout.flush()
            
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"  ✓ 缓存加载成功 (耗时 < 0.1秒)")
                print(f"  - 特征值形状: {cached_data['eigenvalues'].shape}")
                print(f"  - 特征向量形状: {cached_data['eigenvectors'].shape}")
                sys.stdout.flush()
                return cached_data
            except Exception as e:
                print(f"  ⚠️ 缓存加载失败: {e}")
                print(f"  重新计算...")
                sys.stdout.flush()
        else:
            print(f"\n[视图{view_idx+1}] 缓存不存在，开始计算...")
            sys.stdout.flush()
    # --------------------------
    # 1. 特征预处理
    # --------------------------
    if normalize_feats:
        # ========== 修改：使用标准化代替L2归一化 ==========
        # 原因：L2归一化后所有向量模长为1，点积范围[-1,1]，可能导致相似度分布过于集中
        # 新方案：使用标准化（均值0，方差1），保留特征的分布信息
        
        # 方案A：标准化（推荐，优先尝试）
        feats_mean = feats.mean(dim=0, keepdim=True)  # (1, D)
        feats_std = feats.std(dim=0, keepdim=True) + 1e-8  # (1, D)，避免除零
        feats = (feats - feats_mean) / feats_std  # (N, D)，标准化
        
        # 方案B：L2归一化（原方案，备选）
        # feats = F.normalize(feats, p=2, dim=-1)  # (N, D) -> (N, D)，每个特征向量模长为1
        
        print(f"\n[视图{view_idx+1}] 特征预处理:")
        print(f"  归一化方法: 标准化（Z-score）")
        print(f"  特征均值: {feats.mean():.6f}")
        print(f"  特征标准差: {feats.std():.6f}")
        sys.stdout.flush()  # 强制刷新输出

    # --------------------------
    # 2. 构建亲和矩阵 (Affinity Matrix)
    # --------------------------
    # 亲和矩阵表示特征间的相似度，通过点积计算
    affinity = torch.matmul(feats, feats.T)  # (N, N)，A_ij = 特征i与特征j的点积
    
    # ========== 新增：亲和矩阵归一化到[0, 1]范围 ==========
    # 问题：标准化后的点积范围是(-∞, +∞)，实际数据中出现0-2047的大数值
    # 解决：将亲和矩阵归一化到[0, 1]，避免度矩阵过大导致特征值退化
    affinity_min = affinity.min()
    affinity_max = affinity.max()
    affinity = (affinity - affinity_min) / (affinity_max - affinity_min + 1e-8)
    
    print(f"\n[视图{view_idx+1}] 亲和矩阵归一化:")
    print(f"  原始范围: [{affinity_min:.6f}, {affinity_max:.6f}]")
    print(f"  归一化后范围: [{affinity.min():.6f}, {affinity.max():.6f}]")
    print(f"  归一化后均值: {affinity.mean():.6f}")
    sys.stdout.flush()  # 强制刷新输出

    if threshold_affinity:
        # ========== 修复：使用Top-K稀疏化策略（真正减少连接） ==========
        # 根本问题：
        # 1. 之前的阈值过滤只是置零，矩阵仍然稠密（21311x21311全部元素）
        # 2. 度矩阵计算包含了所有0值，导致度值过大
        # 3. 对称归一化后对角线接近1，特征值全部聚集在1附近
        
        # 新策略：每个节点只保留K个最强连接（真正稀疏化）
        # 参数调优：从1%降低到0.5%，进一步降低度值，使对角线远离1
        k_neighbors = max(10, int(affinity.shape[0] * 0.005))  # 每个节点保留0.5%的连接（约106个）
        
        print(f"\n[视图{view_idx+1}] 阈值过滤策略:")
        print(f"  策略类型: Top-K稀疏化（每节点保留最强K个连接）")
        print(f"  K值: {k_neighbors} (样本总数的0.5%)")
        print(f"  过滤前非零元素: {(affinity > 0).sum().item()}")
        sys.stdout.flush()  # 强制刷新输出
        
        # 对每一行保留Top-K最大值
        topk_values, topk_indices = torch.topk(affinity, k_neighbors, dim=1)  # (N, K)
        
        # 创建稀疏矩阵（只保留Top-K连接）
        affinity_sparse = torch.zeros_like(affinity)  # (N, N)
        for i in range(affinity.shape[0]):
            affinity_sparse[i, topk_indices[i]] = topk_values[i]
        
        # 对称化（保证无向图）
        affinity = (affinity_sparse + affinity_sparse.T) / 2.0
        
        print(f"  过滤后非零元素: {(affinity > 0).sum().item()}")
        print(f"  稀疏度: {(affinity > 0).sum().item() / affinity.numel() * 100:.2f}%")
        print(f"  理论最大非零元素: {affinity.shape[0] * k_neighbors * 2} (对称化后)")
        sys.stdout.flush()  # 强制刷新输出

    # 转换为numpy数组用于后续计算
    affinity_np = affinity.cpu().numpy()  # (N, N)
    
    # ============ 诊断输出：亲和矩阵统计 ============
    print(f"\n[视图{view_idx+1}] 亲和矩阵统计:")
    print(f"  形状: {affinity_np.shape}")
    print(f"  最大值: {affinity_np.max():.6f}")
    print(f"  最小值: {affinity_np.min():.6f}")
    print(f"  均值: {affinity_np.mean():.6f}")
    print(f"  标准差: {affinity_np.std():.6f}")
    print(f"  非零元素比例: {(affinity_np > 0).mean()*100:.2f}%")
    print(f"  正值比例: {(affinity_np > 0).mean()*100:.2f}%")
    print(f"  负值比例: {(affinity_np < 0).mean()*100:.2f}%")
    sys.stdout.flush()  # 强制刷新输出

    # --------------------------
    # 3. 构建拉普拉斯矩阵 (Laplacian Matrix)
    # --------------------------
    if which_matrix == 'laplacian':
        # ========== 修复：正确计算度矩阵（只计算非零元素的和） ==========
        # 问题：之前的np.sum包含了所有0值，导致度值虚高
        # 解决：只对非零元素求和（稀疏矩阵的真实度）
        degree = np.sum(affinity_np, axis=1)  # (N,)，稀疏矩阵的度（0值会被自动忽略）
        degree_matrix = diags(degree)  # (N, N)，对角矩阵，对角线为度值
        
        # ============ 诊断输出：度矩阵统计 ============
        print(f"\n[视图{view_idx+1}] 度矩阵统计:")
        print(f"  最大度: {degree.max():.6f}")
        print(f"  最小度: {degree.min():.6f}")
        print(f"  平均度: {degree.mean():.6f}")
        print(f"  度标准差: {degree.std():.6f}")
        print(f"  零度节点数: {(degree == 0).sum()}")
        sys.stdout.flush()  # 强制刷新输出

        # 拉普拉斯矩阵 L = D - A（D为度矩阵，A为亲和矩阵）
        laplacian = degree_matrix - affinity_np  # (N, N)
        
        # ============ 诊断输出：未归一化拉普拉斯矩阵统计 ============
        print(f"\n[视图{view_idx+1}] 未归一化拉普拉斯矩阵统计:")
        print(f"  最大值: {laplacian.max():.6f}")
        print(f"  最小值: {laplacian.min():.6f}")
        print(f"  均值: {laplacian.mean():.6f}")
        print(f"  对角线均值: {np.diag(laplacian).mean():.6f}")
        sys.stdout.flush()  # 强制刷新输出

        # ========== 终极修复：使用未归一化拉普拉斯矩阵 ==========
        # 问题分析：
        # 1. 对称归一化：L_sym → I（单位矩阵），对角线≈1
        # 2. 随机游走归一化：L_rw → I，对角线=1-0/d=1（因为A_ii=0）
        # 3. 结论：任何归一化在稀疏图上都会导致对角线接近1，特征值退化
        # 
        # 解决方案：直接使用未归一化拉普拉斯矩阵 L = D - A
        # - 对角线：等于度值（不是1）
        # - 特征值范围：[0, 2*d_max]（有真实的分布）
        # - 缺点：不同视图尺度不同（但可以接受）
        
        # 不做任何归一化，直接使用 L = D - A
        # （laplacian已经在上面计算好了）
        
        print(f"\n[视图{view_idx+1}] 拉普拉斯矩阵归一化:")
        print(f"  归一化类型: 无归一化（使用原始拉普拉斯矩阵 L = D - A）")
        print(f"  理论特征值范围: [0, {2*degree.max():.2f}]（约为2倍最大度）")
        print(f"  优势: 避免对角线退化为1，保留真实的谱结构")
        sys.stdout.flush()  # 强制刷新输出
        
        # ============ 诊断输出：最终拉普拉斯矩阵统计 ============
        print(f"\n[视图{view_idx+1}] 最终拉普拉斯矩阵统计:")
        print(f"  最大值: {laplacian.max():.6f}")
        print(f"  最小值: {laplacian.min():.6f}")
        print(f"  均值: {laplacian.mean():.6f}")
        print(f"  对角线均值: {np.diag(laplacian).mean():.6f}")
        print(f"  对角线最大值: {np.diag(laplacian).max():.6f}")
        print(f"  对角线最小值: {np.diag(laplacian).min():.6f}")
        print(f"  预期特征值范围: [0, {laplacian.max():.2f}]")
        sys.stdout.flush()  # 强制刷新输出

        matrix_to_eig = laplacian
        # 拉普拉斯矩阵特征值从小到大排序，取前K个（含0特征值）
        which_eig = 'SM'  # 最小特征值

    elif which_matrix == 'affinity':
        # 直接使用亲和矩阵计算特征向量
        matrix_to_eig = affinity_np
        # 亲和矩阵特征值从大到小排序，取前K个
        which_eig = 'LM'  # 最大特征值

    else:
        raise ValueError("矩阵类型必须为 'laplacian' 或 'affinity'")

        # -------------------------- 修复：使用稀疏矩阵eigsh避免内存溢出 --------------------------
    # 问题：np.linalg.eigh计算全部21311个特征值需要15-25GB内存，导致OOM（进程被Killed）
    # 解决：使用scipy.sparse.linalg.eigsh只计算前k个特征值（约500个），内存需求降至<1GB
    
    n_eigenvalues_for_plot = min(300, matrix_to_eig.shape[0])  # 最多计算500个特征值用于绘图
    
    print(f"\n[视图{view_idx + 1}] 开始计算特征值谱（前{n_eigenvalues_for_plot}个特征值）")
    print(f"  矩阵大小: {matrix_to_eig.shape[0]}x{matrix_to_eig.shape[0]}")
    print(f"  内存优化: 使用稀疏特征值分解（避免OOM）")
    sys.stdout.flush()  # 强制刷新输出
    
    # 使用eigsh计算前k个特征值（内存友好）
    try:
        if which_matrix == 'laplacian':
            # ========== 性能优化：使用ARPACK算法只计算需要的特征值 ==========
            # 原方案：计算全部21311个特征值 -> 耗时2-5分钟
            # 优化方案：使用scipy.sparse.linalg.eigsh + shift-invert模式 -> 耗时10-30秒
            
            # 转换为稀疏矩阵格式（CSR格式，eigsh专用）
            from scipy.sparse import csr_matrix
            matrix_sparse = csr_matrix(matrix_to_eig) if not hasattr(matrix_to_eig, 'toarray') else matrix_to_eig
            
            print(f"  矩阵转换: 密集/稀疏混合 -> CSR稀疏矩阵（{matrix_sparse.shape}）")
            sys.stdout.flush()
            
            # 使用shift-invert模式加速计算最小特征值
            # sigma=0: 寻找接近0的特征值（拉普拉斯矩阵的最小特征值）
            # mode='normal': 标准模式（比shift-invert更稳定）
            print(f"  开始计算特征值（ARPACK算法）...")
            sys.stdout.flush()
            
            all_eigenvalues = eigsh(matrix_sparse, k=n_eigenvalues_for_plot, which='SM', 
                                   return_eigenvectors=False, maxiter=10000, tol=1e-3)
            all_eigenvalues = np.sort(all_eigenvalues)  # 升序排列
            
            print(f"  ✓ 特征值计算完成")
            sys.stdout.flush()
            
        else:
            # 亲和矩阵：计算最大的k个特征值
            from scipy.sparse import csr_matrix
            matrix_sparse = csr_matrix(matrix_to_eig) if not hasattr(matrix_to_eig, 'toarray') else matrix_to_eig
            print(f"  开始计算特征值（ARPACK算法）...")
            sys.stdout.flush()
            
            all_eigenvalues = eigsh(matrix_sparse, k=n_eigenvalues_for_plot, which='LM', 
                                   return_eigenvectors=False, maxiter=10000, tol=1e-3)
            all_eigenvalues = np.sort(all_eigenvalues)[::-1]  # 降序排列
            
            print(f"  ✓ 特征值计算完成")
            sys.stdout.flush()

        # 特征值排序处理
        if which_matrix == 'laplacian':
            # 拉普拉斯矩阵:已经是最小的300个,保持升序(从0开始)
            all_eigenvalues_sorted = all_eigenvalues  # 升序:[0, 2.3, 3.9, ..., 8.2]
        else:
            # 亲和矩阵:已经是最大的300个,保持降序
            all_eigenvalues_sorted = all_eigenvalues  # 降序

        print(f"  成功计算{len(all_eigenvalues_sorted)}个特征值")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"  警告：eigsh失败，尝试使用密集矩阵方法（可能内存不足）")
        print(f"  错误信息: {e}")
        sys.stdout.flush()
        
        # 备选方案：如果矩阵较小或eigsh失败，使用eigh
        if matrix_to_eig.shape[0] <= 5000:
            all_eigenvalues = np.linalg.eigh(matrix_to_eig)[0]
            all_eigenvalues_sorted = np.sort(all_eigenvalues)[::-1][:n_eigenvalues_for_plot]
        else:
            print(f"  错误：矩阵过大且eigsh失败，跳过特征谱计算")
            sys.stdout.flush()
            # 返回空结果，跳过绘图
            return {
                'eigenvalues': torch.zeros(K).float(),
                'eigenvectors': torch.zeros(K, feats.shape[0]).float(),
                'affinity': affinity,
                'laplacian': matrix_to_eig if which_matrix == 'laplacian' else None
            }
    
    # ============ 诊断输出:特征值统计 ============
    # 统一变量名：所有后续代码使用all_eigenvalues
    all_eigenvalues_sorted = all_eigenvalues
    
    print(f"\n[视图{view_idx+1}] 特征值统计(前{len(all_eigenvalues_sorted)}个):")
    print(f"  特征值数量(计算的): {len(all_eigenvalues_sorted)}")
    print(f"  特征值数量(总共): {matrix_to_eig.shape[0]}")
        
    if which_matrix == 'laplacian':
        # 拉普拉斯矩阵:升序排列,最小值在前
        print(f"  最小特征值: {all_eigenvalues_sorted[0]:.6f}  (应接近0)")
        print(f"  最大特征值(前{len(all_eigenvalues_sorted)}个中): {all_eigenvalues_sorted[-1]:.6f}")
    else:
        # 亲和矩阵:降序排列,最大值在前
        print(f"  最大特征值: {all_eigenvalues_sorted[0]:.6f}")
        print(f"  最小特征值(前{len(all_eigenvalues_sorted)}个中): {all_eigenvalues_sorted[-1]:.6f}")
        
    print(f"  特征值均值: {all_eigenvalues_sorted.mean():.6f}")
    print(f"  特征值标准差: {all_eigenvalues_sorted.std():.6f}")
    print(f"  接近1的特征值数量(0.99-1.01): {((all_eigenvalues_sorted > 0.99) & (all_eigenvalues_sorted < 1.01)).sum()}")
    print(f"  接近0的特征值数量(<0.01): {(np.abs(all_eigenvalues_sorted) < 0.01).sum()}")
    print(f"  前10个特征值: {all_eigenvalues_sorted[:10]}")
    print(f"  后10个特征值: {all_eigenvalues_sorted[-10:]}")
    sys.stdout.flush()  # 强制刷新输出
    
    # ============ 诊断输出:特征值分布统计 ============
    print(f"\n[视图{view_idx+1}] 特征值分布:")
    if which_matrix == 'laplacian':
        print(f"  第1个特征值(最小): {all_eigenvalues_sorted[0]:.6f}")
        print(f"  第2个特征值: {all_eigenvalues_sorted[1]:.6f}")
        print(f"  第10个特征值: {all_eigenvalues_sorted[min(9, len(all_eigenvalues_sorted)-1)]:.6f}")
        print(f"  第100个特征值: {all_eigenvalues_sorted[min(99, len(all_eigenvalues_sorted)-1)]:.6f}")
        print(f"  第{len(all_eigenvalues_sorted)}个特征值(最大): {all_eigenvalues_sorted[-1]:.6f}")
    else:
        print(f"  第1个特征值(最大): {all_eigenvalues_sorted[0]:.6f}")
        print(f"  第2个特征值: {all_eigenvalues_sorted[1]:.6f}")
        print(f"  第10个特征值: {all_eigenvalues_sorted[min(9, len(all_eigenvalues_sorted)-1)]:.6f}")
        print(f"  第100个特征值: {all_eigenvalues_sorted[min(99, len(all_eigenvalues_sorted)-1)]:.6f}")
        print(f"  第{len(all_eigenvalues_sorted)}个特征值(最小): {all_eigenvalues_sorted[-1]:.6f}")
    print(f"  特征值方差: {np.var(all_eigenvalues_sorted):.6f}")
    sys.stdout.flush()  # 强制刷新输出
    # 3. 创建保存目录（若不存在则自动创建）
    os.makedirs(save_dir, exist_ok=True)
    # 4. 绘制谱图
    plt.figure(figsize=(10, 6))
    # 绘制特征值分布曲线（拉普拉斯矩阵从小到大，直接绘制）
    plt.plot(range(1, len(all_eigenvalues_sorted) + 1),  # 横轴：特征值索引（1开始）
             all_eigenvalues_sorted,
             marker='.', linestyle='-', color='darkblue', alpha=0.7, markersize=2)
    # 添加零值辅助线（拉普拉斯矩阵特征值非负，零值线可辅助判断“聚类簇数”）
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Threshold')
    # 图表标注
    plt.xlabel('Eigenvalue Index (Sorted Descending)', fontsize=12)
    plt.ylabel('Eigenvalue Value', fontsize=12)
    plt.title(f'Feature Spectrum (View {view_idx + 1}, {which_matrix} Matrix)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    # 5. 保存谱图（文件名含视图索引，避免多视图覆盖）
    spectrum_path = os.path.join(save_dir,
                                 f'view_{view_idx + 1}_{which_matrix}_spectrum.png')
    plt.savefig(spectrum_path, dpi=300)  # dpi=300确保图片清晰度
    plt.close()  # 关闭画布，释放内存
    print(f"视图{view_idx + 1}：特征谱图已保存到 {spectrum_path}")
    sys.stdout.flush()  # 强制刷新输出
    
    # ========================== 新增：绘制中间矩阵热力图 ==========================
    print(f"\n[视图{view_idx + 1}] 开始绘制中间矩阵热力图...")
    sys.stdout.flush()  # 强制刷新输出
    
    # 为了可视化效果，对大矩阵进行采样（只显示部分样本）
    n_samples = affinity_np.shape[0]
    max_display = 100  # 最多显示100x100的热力图，避免图片过大
    
    if n_samples > max_display:
        # 随机采样索引
        sample_indices = np.sort(np.random.choice(n_samples, max_display, replace=False))
        affinity_display = affinity_np[np.ix_(sample_indices, sample_indices)]
        degree_display = degree[sample_indices]
        laplacian_display = matrix_to_eig[np.ix_(sample_indices, sample_indices)] if which_matrix == 'laplacian' else None
        display_info = f"(采样{max_display}/{n_samples}个样本)"
    else:
        affinity_display = affinity_np
        degree_display = degree
        laplacian_display = matrix_to_eig if which_matrix == 'laplacian' else None
        display_info = f"(全部{n_samples}个样本)"
    
    # 创建热力图保存目录
    heatmap_dir = os.path.join(save_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # ------- 1. 绘制亲和矩阵热力图 -------
    plt.figure(figsize=(10, 8))
    im = plt.imshow(affinity_display, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Affinity Value')
    plt.title(f'Affinity Matrix Heatmap - View {view_idx + 1}\n{display_info}', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.tight_layout()
    affinity_heatmap_path = os.path.join(heatmap_dir, f'view_{view_idx + 1}_affinity_heatmap.png')
    plt.savefig(affinity_heatmap_path, dpi=200)
    plt.close()
    print(f"  ✓ 亲和矩阵热力图已保存: {affinity_heatmap_path}")
    sys.stdout.flush()  # 强制刷新输出
    
    # ------- 2. 绘制度分布直方图 -------
    plt.figure(figsize=(10, 6))
    plt.hist(degree_display, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(degree_display.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {degree_display.mean():.2f}')
    plt.axvline(degree_display.min(), color='green', linestyle='--', linewidth=2, label=f'Min: {degree_display.min():.2f}')
    plt.axvline(degree_display.max(), color='orange', linestyle='--', linewidth=2, label=f'Max: {degree_display.max():.2f}')
    plt.xlabel('Degree Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Degree Distribution - View {view_idx + 1}\n{display_info}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    degree_hist_path = os.path.join(heatmap_dir, f'view_{view_idx + 1}_degree_distribution.png')
    plt.savefig(degree_hist_path, dpi=200)
    plt.close()
    print(f"  ✓ 度分布直方图已保存: {degree_hist_path}")
    sys.stdout.flush()  # 强制刷新输出
    
    # ------- 3. 绘制拉普拉斯矩阵热力图（仅当使用拉普拉斯矩阵时） -------
    if which_matrix == 'laplacian' and laplacian_display is not None:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(laplacian_display, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=0, vmax=2)
        plt.colorbar(im, label='Laplacian Value')
        plt.title(f'Normalized Laplacian Matrix Heatmap - View {view_idx + 1}\n{display_info}', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Sample Index', fontsize=12)
        plt.tight_layout()
        laplacian_heatmap_path = os.path.join(heatmap_dir, f'view_{view_idx + 1}_laplacian_heatmap.png')
        plt.savefig(laplacian_heatmap_path, dpi=200)
        plt.close()
        print(f"  ✓ 拉普拉斯矩阵热力图已保存: {laplacian_heatmap_path}")
        sys.stdout.flush()  # 强制刷新输出
    
    # ------- 4. 绘制亲和矩阵稀疏模式图 -------
    plt.figure(figsize=(10, 8))
    # 使用二值化显示稀疏模式（非零为1，零为0）
    sparse_pattern = (affinity_display > 0).astype(float)
    im = plt.imshow(sparse_pattern, cmap='binary', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Connection (0=No, 1=Yes)', ticks=[0, 1])
    plt.title(f'Affinity Sparsity Pattern - View {view_idx + 1}\n{display_info}\nNon-zero: {(sparse_pattern > 0).sum()}/{sparse_pattern.size} ({(sparse_pattern > 0).mean()*100:.2f}%)', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.tight_layout()
    sparsity_path = os.path.join(heatmap_dir, f'view_{view_idx + 1}_sparsity_pattern.png')
    plt.savefig(sparsity_path, dpi=200)
    plt.close()
    print(f"  ✓ 稀疏模式图已保存: {sparsity_path}")
    sys.stdout.flush()  # 强制刷新输出
    
    # ------- 5. 综合统计对比图 -------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 子图1：亲和矩阵值分布直方图
    axes[0, 0].hist(affinity_display.flatten(), bins=100, color='teal', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(affinity_display.mean(), color='red', linestyle='--', label=f'Mean: {affinity_display.mean():.4f}')
    axes[0, 0].set_xlabel('Affinity Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Affinity Value Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2：度值分布（箱线图）
    axes[0, 1].boxplot(degree_display, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='blue'),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_ylabel('Degree Value')
    axes[0, 1].set_title(f'Degree Distribution (Boxplot)\nMean={degree_display.mean():.2f}, Std={degree_display.std():.2f}')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 子图3：特征值分布（前50个）
    top_k_eig = min(50, len(all_eigenvalues_sorted))
    axes[1, 0].bar(range(1, top_k_eig + 1), all_eigenvalues_sorted[:top_k_eig], color='navy', alpha=0.7)
    axes[1, 0].set_xlabel('Eigenvalue Rank')
    axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].set_title(f'Top {top_k_eig} Eigenvalues')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 子图4：矩阵统计摘要（文本）
    axes[1, 1].axis('off')
    summary_text = f"""Matrix Statistics Summary (View {view_idx + 1})

【Affinity Matrix】
  Shape: {affinity_np.shape}
  Max: {affinity_np.max():.6f}
  Min: {affinity_np.min():.6f}
  Mean: {affinity_np.mean():.6f}
  Std: {affinity_np.std():.6f}
  Non-zero ratio: {(affinity_np > 0).mean()*100:.2f}%

【Degree Matrix】
  Max degree: {degree.max():.2f}
  Min degree: {degree.min():.2f}
  Mean degree: {degree.mean():.2f}
  Std degree: {degree.std():.2f}

【Eigenvalue Spectrum】
  Max eigenvalue: {all_eigenvalues.max():.6f}
  Min eigenvalue: {all_eigenvalues.min():.6f}
  Mean eigenvalue: {all_eigenvalues.mean():.6f}
  Std eigenvalue: {all_eigenvalues.std():.6f}
  Eigenvalues ≈ 1: {((all_eigenvalues > 0.99) & (all_eigenvalues < 1.01)).sum()}
    """
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    summary_path = os.path.join(heatmap_dir, f'view_{view_idx + 1}_summary.png')
    plt.savefig(summary_path, dpi=200)
    plt.close()
    print(f"  ✓ 综合统计图已保存: {summary_path}")
    sys.stdout.flush()  # 强制刷新输出
    
    print(f"\n[视图{view_idx + 1}] 所有热力图已保存至: {heatmap_dir}")
    print("=" * 70)
    sys.stdout.flush()  # 强制刷新输出
    # 4. 特征向量提取（扩散图核心）
    # --------------------------
    # 计算矩阵的前K个特征值和特征向量
    # eigsh适用于对称矩阵，效率高于普通特征值分解
    eigenvalues, eigenvectors = eigsh(matrix_to_eig, k=K, which=which_eig)  # (K,), (N, K)

    # 调整特征向量顺序（拉普拉斯矩阵按从小到大，亲和矩阵按从大到小）
    if which_matrix == 'laplacian':
        # 拉普拉斯矩阵特征值从小到大，对应扩散图的"低频"分量
        eigenvectors = eigenvectors[:, :K]  # (N, K)
    else:
        # 亲和矩阵特征值从大到小，对应最显著的关联模式
        eigenvectors = eigenvectors[:, ::-1]  # 反转顺序，确保从大到小

    # 转换为PyTorch张量并调整维度（K, N）
    eigenvectors = torch.from_numpy(eigenvectors.T).float()  # (K, N)

    # --------------------------
    # 5. 特征向量后处理（解决符号歧义）
    # --------------------------
    for k in range(K):
        # 若特征向量中正值占比超过50%，反转符号（确保一致性）
        if torch.mean((eigenvectors[k] > 0).float()) > 0.5:
            eigenvectors[k] = -eigenvectors[k]

    # 构建返回结果
    result_data = {
        'eigenvalues': torch.from_numpy(eigenvalues).float(),  # 特征值
        'eigenvectors': eigenvectors,  # 特征向量（扩散图的核心输出）
        'affinity': affinity,  # 亲和矩阵
        'laplacian': laplacian if which_matrix == 'laplacian' else None  # 拉普拉斯矩阵（若计算）
    }
    
    # ========== 保存到缓存 ==========
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n[视图{view_idx+1}] ✓ 特征值和特征向量已缓存")
            print(f"  缓存文件: {os.path.basename(cache_file)}")
            print(f"  文件大小: {os.path.getsize(cache_file) / (1024**2):.2f} MB")
            sys.stdout.flush()
        except Exception as e:
            print(f"  ⚠️ 缓存保存失败: {e}")
            sys.stdout.flush()

    return result_data

class animalAttrData(object):
    """
       Dataloader for animal attributes dataset.类定义和初始化
    """
    def __init__(self, data_dir='/home/limn/mvnn/Mvnn/mvdata/AWA/Features',
        mode='train',
        diffusion_K = 10,
        diffusion_matrix = 'laplacian',
        normalize_feats = True,
        threshold_affinity = True,
        save_spectrum_dir='./results/spectrums',
        use_cache=True,  # 新增:缓存控制
        cache_dir='./cache/eigenvalues',  # 新增:缓存目录
        use_parallel=True,  # 新增:是否启用并行处理
        num_workers=None):  # 新增:并行进程数(默认为CPU核数)

        print(type(object))
        super(animalAttrData, self).__init__()
        self.diffusion_K = diffusion_K
        self.diffusion_matrix = diffusion_matrix
        self.normalize_feats = normalize_feats
        self.threshold_affinity = threshold_affinity
        self.save_spectrum_dir = save_spectrum_dir  # 保存谱图目录到实例变量
        self.use_cache = use_cache  # 缓存控制
        self.cache_dir = cache_dir  # 缓存目录
        self.use_parallel = use_parallel  # 并行控制
        self.num_workers = num_workers if num_workers is not None else min(cpu_count(), 6)  # 默认最多6个进程
        """. 数据加载和划分逻辑    1. 收集所有样本的路径和标签"""
        data_list = []#初始化空列表data_list用于存储所有数据样本
        fea_name = os.listdir(data_dir)
        #获取特征文件夹名称列表（不同视图的特征）
        class_name = os.listdir(os.path.join(data_dir, fea_name[0]))
        #获取类别名称列表（从第一个特征文件夹中获取）

        count = -1   
        for class_item in class_name:
        #遍历每个类别，为每个类别分配一个唯一索引（count）
            count += 1
            class_list = []
            class_path_list = []
            #为当前类别创建空列表class_list和class_path_list
            for fea_item in fea_name:
                class_path_list.append(os.path.join(data_dir, fea_item, class_item))
                #构建每个特征视图的路径列表

            sample_name = os.listdir(class_path_list[0])

            # each sample have servel kinds of features
            for sample_item in sample_name:#获取当前类别下的所有样本名称
                
                sample_fea_all = [os.path.join(class_path_list[i], sample_item) for i in range(len(class_path_list))]
                #为每个样本构建所有特征视图的完整路径列表
                class_list.append((sample_fea_all, count))
                #将(特征路径列表, 类别索引)元组添加到类别列表中
            
            # divide the data into training set and testing set
            random.seed(int(100)) 
            train_part = random.sample(class_list, int(0.7*len(class_list)) )
            rem_part = [rem for rem in class_list if rem not in train_part]
            val_part = random.sample(rem_part, int(2/3.0*len(rem_part)))
            test_part = [te for te in rem_part if te not in val_part]
#设置随机种子确保可重复性
#将当前类别的样本划分为：训练集：70%   验证集：剩余样本的2/3（约20%）  测试集：剩余样本的1/3（约10%）
            if mode == 'train':
                data_list.extend(train_part)
            elif mode == 'val':
                data_list.extend(val_part)
            else:
                data_list.extend(test_part)

        self.data_list = data_list
        # 3. 预加载所有样本的原始特征，按视图组织
        num_views = len(fea_name)
        self.view_features = [[] for _ in range(num_views)]  # 每个视图的特征列表：[视图0特征, 视图1特征, ...]
        for sample_fea_all, _ in self.data_list:
            for v in range(num_views):
                fea_temp = np.loadtxt(sample_fea_all[v])  # 加载单个样本的视图v特征
                self.view_features[v].append(fea_temp)

        # 4. 对每个视图的特征进行扩散图处理（核心新增步骤）
        self.processed_view_features = []  # 存储处理后的特征
        
        # ========== 并行处理优化 ==========
        if self.use_parallel and num_views > 1:
            print(f"\n🚀 启用并行处理: {num_views}个视图，{self.num_workers}个进程")
            print(f"CPU核数: {cpu_count()}，使用进程数: {self.num_workers}")
            sys.stdout.flush()
            
            # 准备并行任务参数
            tasks = []
            params = {
                'which_matrix': self.diffusion_matrix,
                'K': self.diffusion_K,
                'normalize_feats': self.normalize_feats,
                'threshold_affinity': self.threshold_affinity,
                'save_dir': self.save_spectrum_dir,
                'use_cache': self.use_cache,
                'cache_dir': self.cache_dir
            }
            
            for v in range(num_views):
                feats_np = np.array(self.view_features[v])  # 转为numpy数组
                tasks.append((v, feats_np, params))
            
            # 并行执行
            try:
                print(f"\n开始并行处理...")
                print(f"提示：如果长时间无输出，说明正在计算特征值，请耐心等待")
                print(f"预计每个视图需要10-30秒（首次）或<1秒（缓存）")
                sys.stdout.flush()
                
                with Pool(self.num_workers) as pool:
                    # 使用imap_unordered获取结果
                    results = []
                    completed = 0
                    for result in pool.imap_unordered(process_single_view_wrapper, tasks):
                        results.append(result)
                        completed += 1
                        print(f"\n>>> 进度: {completed}/{num_views} 视图已完成")
                        sys.stdout.flush()
                
                # 按视图顺序排序结果
                results_sorted = sorted(results, key=lambda x: x[0])
                
                for view_idx, diffusion_result in results_sorted:
                    processed_feats = diffusion_result['eigenvectors'].T  # (N, K)
                    self.processed_view_features.append(processed_feats)
                
                print(f"\n✓ 并行处理完成! 所有{num_views}个视图已处理")
                sys.stdout.flush()
                
            except Exception as e:
                print(f"\n⚠️ 并行处理失败: {e}")
                print("回退到串行处理...")
                sys.stdout.flush()
                self.use_parallel = False  # 回退到串行模式
        
        # ========== 串行处理（备用） ==========
        if not self.use_parallel or num_views == 1:
            print(f"\n使用串行处理: {num_views}个视图")
            print(f"提示：如果长时间无输出，说明正在计算特征值，请耐心等待")
            sys.stdout.flush()
            
            for v in range(num_views):
                print(f"\n>>> 开始处理视图 {v+1}/{num_views}")
                sys.stdout.flush()
                
                # 转换为Tensor：(N, D)
                feats = torch.from_numpy(np.array(self.view_features[v])).float()
                # 扩散图处理
                diffusion_result = process_features_to_diffusion(
                    feats=feats,
                    which_matrix=self.diffusion_matrix,
                    K=self.diffusion_K,
                    normalize_feats=self.normalize_feats,
                    threshold_affinity=self.threshold_affinity,
                    save_dir=self.save_spectrum_dir,
                    view_idx=v,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir
                )
                # 提取特征向量：(K, N) -> 转置为 (N, K)
                processed_feats = diffusion_result['eigenvectors'].T  # (N, K)
                self.processed_view_features.append(processed_feats)
                
                print(f">>> 视图 {v+1}/{num_views} 处理完成")
                sys.stdout.flush()
#数据加载器方法
    def __len__(self):
        return len(self.data_list)
        #返回数据集的大小（样本数量）


    def __getitem__(self, index):
        '''
            Load an episode each time
        '''
        # 5. 返回处理后的特征（替代原始特征）
        _, target = self.data_list[index]  # 原始数据列表仅用于获取标签
        processed_fea = [self.processed_view_features[v][index]
                         for v in range(len(self.processed_view_features))]
        # 先把 list of 1-D numpy 拼成 2-D ndarray
        #fea_np = np.stack([self.processed_view_features[v][index].numpy()
        #                   for v in range(len(self.processed_view_features))], axis=0)
        # 按索引收集每个视图的处理后特征
        #processed_fea = torch.from_numpy(fea_np).float()   # shape (num_view, K)
        # ---- 调试代码 ----
        # for v, f in enumerate(processed_fea):
        #     print(f'view{v} shape={f.shape}', end='  ')
        # print()
        # -----------------
        #processed_fea = [self.processed_view_features[v][index] for v in range(len(self.processed_view_features))]
        return processed_fea, target
        # (sample_fea_all, target) = self.data_list[index]
        # #根据索引获取一个数据样本 从元组中提取特征路径列表和目标类别
        # Sample_Fea = []
        # for i in range(len(sample_fea_all)):#遍历所有特征视图，使用np.loadtxt加载特征文件
        #     fea_temp = np.loadtxt(sample_fea_all[i])
        #     fea_temp = torch.from_numpy(fea_temp)
        #     #将NumPy数组转换为PyTorch张量并确保为Float类型
        #     Sample_Fea.append(fea_temp.type(torch.FloatTensor))
        #
        # return (Sample_Fea, target)
        # #返回一个元组，包含所有特征视图的张量列表和目标类别
        

