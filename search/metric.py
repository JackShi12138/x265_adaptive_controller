import numpy as np
import bjontegaard as bd


def calculate_bd_vmaf(anchor_points, test_points):
    """
    计算两组 R-D 曲线之间的 BD-VMAF 分数 (Bjontegaard Delta VMAF)。
    正值表示 Test 组在相同码率下具有更高的 VMAF (质量提升)。

    Args:
        anchor_points (list of tuples): 基准曲线数据 [(bitrate, vmaf), ...]
        test_points (list of tuples):   测试曲线数据 [(bitrate, vmaf), ...]

    Returns:
        float: BD-VMAF 分数。如果计算失败或数据异常，返回 -9999.0
    """
    # 1. 数据量校验
    # BD 算法通常需要至少 4 个点来进行曲线拟合
    if len(anchor_points) < 4 or len(test_points) < 4:
        return -9999.0

    # 2. 排序 (Sorting)
    # R-D 曲线计算要求数据点必须按码率 (X轴) 升序排列
    anchor_sorted = sorted(anchor_points, key=lambda x: x[0])
    test_sorted = sorted(test_points, key=lambda x: x[0])

    # 3. 数据解包
    rate_anchor = [p[0] for p in anchor_sorted]
    vmaf_anchor = [p[1] for p in anchor_sorted]
    rate_test = [p[0] for p in test_sorted]
    vmaf_test = [p[1] for p in test_sorted]

    # 4. 单调性与有效性检查 (Sanity Check)
    # 理论上码率增加，VMAF 应该增加。
    # 如果曲线出现严重倒挂 (即码率变大画质反而大幅下降)，说明编码器状态异常，拟合结果不可信。
    # 这里允许 5 分以内的波动 (考虑到 CBR 码控的不稳定性)，超过则视为异常。
    if vmaf_test[-1] < vmaf_test[0] - 5.0:
        return -9999.0

    # 同样检查 Anchor 数据（虽然理论上 Anchor 是预先生成的，应该是好的，但为了保险）
    if vmaf_anchor[-1] < vmaf_anchor[0] - 5.0:
        return -9999.0

    try:
        # 5. 调用 bjontegaard 库计算 BD-VMAF
        # bd.bd_psnr 计算的是垂直方向的差异 (即相同码率下的质量差异)
        #
        # 参数详解:
        # method='akima':
        #   Akima 分段三次插值法。相比 cubic spline，它在数据点波动时产生的震荡更小，
        #   非常适合视频编码这种可能存在噪声的 R-D 曲线。

        score = bd.bd_psnr(
            rate_anchor,
            vmaf_anchor,
            rate_test,
            vmaf_test,
            method="akima",
        )

        # 结果截断保护 (防止出现极端的数值异常)
        if np.isnan(score) or np.isinf(score):
            return -9999.0

        return float(score)

    except Exception:
        # 捕获所有拟合过程中的数学错误 (如点重合导致的除零错误等)
        return -9999.0
