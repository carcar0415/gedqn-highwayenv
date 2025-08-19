# utils/reward_utils.py
import numpy as np
import math
from typing import Dict, Any

# === 全局常量 ===
EPSILON = 1e-5  # 避免除零
DEFAULT_CONFIG = {
    "v_max": 20,
    "a_comf": 1.0,
    "s_safe": 5,
    "lambda_I": 0.30,
    "lambda_v": 0.30,
    "lambda_a": 0.30,
    "lambda_d": 0.10,
    "T_min": 1.0,
    "T_max": 5.0,
}

# 修改后的compute_custom_reward函数
def compute_custom_reward(info: Dict[str, Any], vehicle: Any, config: Dict[str, Any]) -> float:
    """计算自定义奖励函数，强化安全性与智能决策"""
    # === 合并默认配置 ===
    config = {**DEFAULT_CONFIG, **config}

    # === 基本状态 ===
    v = getattr(vehicle, "speed", 0.0)
    v_T = getattr(vehicle, "target_speed", 15.0)
    v_M = config["v_max"]
    a = getattr(vehicle, "acc_value", 1.5)
    a_comf = config["a_comf"]
    s_CG = getattr(vehicle, "gap", 2.0)
    s_safe = config["s_safe"]
    
    # === 安全距离动态调整 ===
    dynamic_safe_gap = s_safe + 0.1 * v  # 高速需要更大安全距离
    
    # === 计算 I_T (基于 TTC + PET) ===
    I_T = _compute_conflict_index(v, v_T, s_CG, config["T_min"], config["T_max"])

    # === 权重参数调整 ===
    lambda_I = 0.40  # 提高冲突风险权重
    lambda_v = 0.20
    lambda_a = 0.20
    lambda_d = 0.20  # 提高安全距离权重

    # === 偏置项（惩罚/奖励）===
    w_1 = -2000 if info.get("crash", False) else 0               # 碰撞惩罚大幅提高
    w_2 = -5 if info.get("lane_change", False) else 0           # 变道惩罚降低
    w_3 = -20 if info.get("out_of_road", False) else 0            # 提高出界惩罚

    # === 核心奖励项 ===
    R_I = 1 / (I_T * abs(a) + EPSILON)        # 冲突-加速度平衡
    R_v = 1 - abs((v - v_T) / v_M)            # 跟车速度匹配
    R_a = math.exp(-abs(a) / a_comf)          # 舒适加速度
    R_d = s_CG / (dynamic_safe_gap + EPSILON)   # 安全距离（动态）
    
    # === 其他惩罚项 ===
    penalty_slow_front = -1 if v_T < 8 else 0      # 前车过慢惩罚
    penalty_gap_small = -5 if s_CG < dynamic_safe_gap else 0  # 距离过近惩罚加强
    
    # === 换道安全奖励 ===
    lane_change_safety = 0
    if info.get("lane_change", False):
        if s_CG > dynamic_safe_gap * 1.5:  # 安全换道奖励
            lane_change_safety = 10
        else:  # 危险换道额外惩罚
            lane_change_safety = -15
    
    # === 提前避让奖励 ===
    early_avoidance = 0
    if I_T > 0.5 and a < -0.5:  # 检测到高风险时主动减速
        early_avoidance = 5
    
    # === 加权奖励函数 ===
    reward = (
        lambda_I * R_I +
        lambda_v * R_v +
        lambda_a * R_a +
        lambda_d * R_d +
        w_1 + w_2 + w_3 +
        penalty_slow_front + penalty_gap_small +
        lane_change_safety + early_avoidance
    )

    return reward

def _compute_conflict_index(v: float, v_T: float, s_CG: float, T_min: float, T_max: float) -> float:
    """
    计算冲突风险指数 I_T。

    参数:
        v (float): 当前速度。
        v_T (float): 目标速度。
        s_CG (float): 当前车距。
        T_min (float): 最小时间阈值。
        T_max (float): 最大时间阈值。

    返回:
        float: 冲突风险指数 I_T。
    """
    TTC = s_CG / (abs(v_T - v) + EPSILON) if abs(v_T - v) > EPSILON else float('inf')
    PET = (s_CG / (v + EPSILON)) - (s_CG / (v_T + EPSILON)) if v_T > EPSILON else float('inf')
    omega_1, omega_2 = 0.5, 0.5  # 权重
    T = omega_1 * TTC + omega_2 * PET

    if T < T_min:
        return 1.0
    elif T > T_max:
        return 0.0
    else:
        return 1 - (T - T_min) / (T_max - T_min)
