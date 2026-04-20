import math
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def ssn_vis(true_values: np.ndarray,
                           pred_values: np.ndarray,
                           feature_names: list,
                           highlight_idx=None,
                           y_floor: float = 5.0,
                           highlight_alpha: float = 0.18,     # ✅ 固定透明度
                           highlight_color: str = "#f4b6c2",  # ✅ 固定颜色
                           highlight_width: int = 0,          # ✅ 扩宽：0=单点，1/2=更粗的块
                           idx_to_time_scale: int = 1) -> str:

    assert true_values.ndim == 2 and pred_values.ndim == 2, "Expect 2D arrays: [F, T]"
    assert true_values.shape[0] == pred_values.shape[0], "Feature dims must match"
    n_features = true_values.shape[0]
    assert len(feature_names) == n_features, "feature_names length must match F"

    # 对齐时间长度：取共同最短长度
    T = min(true_values.shape[1], pred_values.shape[1])
    true_values = true_values[:, -T:]
    pred_values = pred_values[:, -T:]

    # --------- 将 highlight_idx 合并成连续区间 segments ---------
    segments = []
    if highlight_idx is not None and len(highlight_idx) > 0:
        hi = np.asarray(highlight_idx)
        mask = np.zeros(T, dtype=bool)
        
        # 判断输入是不是 [start, end] 区间格式的二维数组
        if hi.ndim == 2 and hi.shape[1] == 2:
            # 1. 区间格式处理：[[324, 375], [377, 380]]
            for start_idx, end_idx in hi:
                L = int(start_idx * idx_to_time_scale) - int(highlight_width)
                R = int(end_idx * idx_to_time_scale) + int(highlight_width)
                # 边界截断，防止越界
                L = max(0, L)
                R = min(T - 1, R)
                if L <= R:
                    mask[L:R+1] = True
        else:
            # 2. 单点格式处理：万一输入是一维数组 [4, 5, 6, 61]
            hi = hi.reshape(-1)
            hi = hi * int(idx_to_time_scale)
            hi = hi[(hi >= 0) & (hi < T)]
            for p in hi:
                L = max(0, int(p) - int(highlight_width))
                R = min(T - 1, int(p) + int(highlight_width))
                mask[L:R+1] = True

        # 3. 将布尔掩码(mask)重新提取为干净、不重叠的连续区间
        idxs = np.where(mask)[0]
        if idxs.size > 0:
            start = idxs[0]
            prev = idxs[0]
            for p in idxs[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    segments.append((start, prev))
                    start = p  # ✅ 正确的缩进：只有断开时才重置起点
                    prev = p   # ✅
            segments.append((start, prev))
    # -----------------------------------------------------------

    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    dpi = 150
    max_width_px = 3300
    max_height_px = 2000 * 5
    fig_width = max_width_px / dpi
    fig_height = (max_height_px / (n_features / 3)) / dpi

    title_fontsize = max_width_px // 220
    label_fontsize = max_width_px // 280
    tick_fontsize = max_width_px // 350

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi)
    # ✅ 替换为面向对象的写法：
    fig = Figure(figsize=(fig_width, fig_height), dpi=dpi)
    canvas = FigureCanvasAgg(fig) # 绑定后端画布
    axes = fig.subplots(n_rows, n_cols) # 使用 fig 实例的方法创建子图
    
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    x_axis = np.arange(T)

    for i, ax in enumerate(axes):
        if i >= n_features:
            ax.axis('off')
            continue

        y_true = true_values[i]
        y_pred = pred_values[i]

        # spine 在下层，线在上层
        for spine in ax.spines.values():
            spine.set_zorder(0)

        # ✅ 先画统一背景块（固定 alpha，不做强度映射）
        for L, R in segments:
            ax.axvspan(L - 0.5, R + 0.5, color=highlight_color, alpha=highlight_alpha, zorder=1)

        # 再画曲线
        ax.plot(x_axis, y_true, label='True', color='black', linewidth=1.2, linestyle='-', zorder=3)
        ax.plot(x_axis, y_pred, label='Pred', color='red', linewidth=1.2, linestyle='--', zorder=3)

        ax.tick_params(axis='both', labelsize=tick_fontsize)

        # -----------------------------------------------------------
        # ✅ 修改后的 Y 轴逻辑
        # -----------------------------------------------------------
        ymin = float(np.min([y_true.min(), y_pred.min()]))
        ymax = float(np.max([y_true.max(), y_pred.max()]))
        y_range = ymax - ymin  # 计算真实数据与预测数据的最大波动范围

        # 强制关闭所有子图 Y 轴的科学计数法和偏移量（彻底消灭 1e-6 + 1）
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        # 如果波动的范围小于 y_floor (1.0)，说明线几乎是平的，强行拉宽 Y 轴
        if y_range < y_floor:
            y_center = (ymax + ymin) / 2.0
            step = float(y_floor)
            ax.set_ylim(y_center - step, y_center + step)
            # 强制设置 3 个刻度：下限、中心、上限
            ax.yaxis.set_major_locator(mticker.FixedLocator([y_center - step, y_center, y_center + step]))
        # -----------------------------------------------------------

        # x 轴刻度（保持原逻辑）
        orig_locs = ax.get_xticks()
        if len(orig_locs) > 1:
            xmin, xmax = ax.get_xlim()
            orig_locs = orig_locs[(orig_locs >= xmin) & (orig_locs <= xmax)]
            mid_locs = (orig_locs[:-1] + orig_locs[1:]) / 2
            mid_locs = mid_locs[(mid_locs > xmin) & (mid_locs < xmax)]
            new_locs = np.sort(np.concatenate([orig_locs, mid_locs]))
            ax.set_xticks(new_locs)
            ax.set_xlim(xmin, xmax)

        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel("Value", fontsize=label_fontsize)
        ax.set_title(f"{feature_names[i]}", fontsize=title_fontsize)

        ax.margins(x=0.02)
        ax.legend(fontsize=label_fontsize, loc='upper right', frameon=False)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_b64