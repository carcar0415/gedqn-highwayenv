import numpy as np
import torch
import math

def extract_state_sequence(history):
    """提取状态序列用于CLMHA模型输入"""
    state_seq = np.array(history)  # shape: [T, N, F]
    state_seq = np.transpose(state_seq, (1, 0, 2))  # 转为 [N, T, F]
    return torch.FloatTensor(state_seq)  # 输出：每辆车的状态序列

def preprocess_obs(obs_list, expected_num=3, state_dim=45):
    """预处理观测，支持扩展状态维度"""
    processed = []
    for i in range(expected_num):
        if i >= len(obs_list) or obs_list[i] is None or obs_list[i] == []:
            processed.append(np.zeros(state_dim))
            continue
        
        obs_arr = np.array(obs_list[i])
        flat = obs_arr.flatten() if obs_arr.ndim == 2 else obs_arr
        
        if flat.shape[0] < state_dim:
            flat = np.pad(flat, (0, state_dim - flat.shape[0]))
        elif flat.shape[0] > state_dim:
            flat = flat[:state_dim]
            
        processed.append(flat)
    return processed

def extract_structured_obs(road, expected_num, risk_scores, risk_trend=None):
    """提取结构化观测，增强状态表示"""
    structured = []

    # 确保风险分数长度匹配
    if len(risk_scores) < expected_num:
        risk_scores = list(risk_scores) + [0] * (expected_num - len(risk_scores))
        
    # 如果没有提供风险趋势，初始化为0
    if risk_trend is None:
        risk_trend = [0] * expected_num
    elif len(risk_trend) < expected_num:
        risk_trend = list(risk_trend) + [0] * (expected_num - len(risk_trend))

    # 为每个车辆提取状态
    for i in range(expected_num):
        if i >= len(road.vehicles):
            structured.append(np.zeros(45))  # 扩展状态维度
            continue

        # 获取本车
        ego = road.vehicles[i]
        
        # 计算本车与道路边界的距离（增强安全感知）
        lane = ego.lane if hasattr(ego, 'lane') else None
        dist_to_left = 999.0
        dist_to_right = 999.0
        
        if lane:
            lateral_pos = ego.position[1]
            lane_width = getattr(lane, 'width', 4.0)
            lane_index = ego.lane_index[2] if ego.lane_index else 0
            
            # 左侧距离：当前位置到车道左边界的距离 + 左侧车道数量 * 车道宽度
            dist_to_left = lateral_pos + lane_index * lane_width
            
            # 右侧距离：车道右边界到当前位置的距离 + 右侧车道数量 * 车道宽度
            total_lanes = getattr(road, 'lanes_count', 4)
            dist_to_right = (total_lanes - lane_index - 1) * lane_width - lateral_pos
        
        # 检查是否存在碰撞风险
        potential_collision = getattr(ego, 'potential_collision', False)
        
        # 计算当前车辆的安全间隙
        gap_front = getattr(ego, "gap", 10.0)
        
        # 基本状态扩展
        ego_obs = [
            ego.position[0], ego.position[1], ego.speed,
            ego.heading, ego.lane_index[2] if ego.lane_index else -1,
            risk_scores[i], ego.target_speed, 
            getattr(ego, "acc_value", 0),
            risk_trend[i],
            1.0 if potential_collision else 0.0,
            gap_front,            # 前方间隙
            dist_to_left,         # 左侧边界距离
            dist_to_right         # 右侧边界距离
        ]

        # 查找前方车辆的更详细信息
        front_vehicles = []
        for v in road.vehicles:
            if v != ego:
                # 计算相对位置
                rel_pos = v.position - ego.position
                rel_distance = np.linalg.norm(rel_pos)
                
                # 只考虑前方±30度扇区内的车辆
                if rel_pos[0] > 0 and abs(math.atan2(rel_pos[1], rel_pos[0])) < math.pi/6:
                    # 计算相对速度和时间-碰撞
                    rel_speed = v.speed - ego.speed
                    ttc = rel_distance / max(rel_speed, 1e-5) if rel_speed > 0 else 30.0
                    
                    front_vehicles.append((v, rel_distance, rel_speed, ttc))
        
        # 按距离排序
        front_vehicles.sort(key=lambda x: x[1])
        
        # 获取最近的3辆前车信息
        for j in range(min(3, len(front_vehicles))):
            v, distance, rel_speed, ttc = front_vehicles[j]
            rel_pos = v.position - ego.position
            
            # 计算相对横向位置（对换道决策很重要）
            rel_lat = rel_pos[1]  # 正值表示在右侧，负值表示在左侧
            
            # 计算该车是否在同一车道
            same_lane = 1.0 if (v.lane_index and ego.lane_index and v.lane_index[2] == ego.lane_index[2]) else 0.0
            
            ego_obs += [
                rel_pos[0],       # 相对纵向距离
                rel_pos[1],       # 相对横向距离
                v.speed,          # 目标车速度
                rel_speed,        # 相对速度
                v.lane_index[2] if v.lane_index else -1,  # 车道
                getattr(v, "acc_value", 0),      # 加速度
                v.heading - ego.heading,         # 相对航向
                distance,                        # 欧氏距离
                ttc,                             # 时间-碰撞
                same_lane,                       # 是否同车道
                1.0 if getattr(v, "potential_collision", False) else 0.0  # 潜在碰撞标志
            ]
            
        # 如果前方车辆不足3辆，填充占位值
        missing_vehicles = 3 - len(front_vehicles)
        if missing_vehicles > 0:
            ego_obs += [0.0] * (missing_vehicles * 11)
            
        # 确保状态维度统一
        if len(ego_obs) < 45:
            ego_obs += [0] * (45 - len(ego_obs))
        elif len(ego_obs) > 45:
            ego_obs = ego_obs[:45]

        structured.append(np.array(ego_obs))

    return structured

def _compute_conflict_index(front_speed, rear_speed, d_ij, d_i, d_j, T_min=1.0, T_max=5.0):
    """计算冲突指数（增强版），基于TTC和PET"""
    # 确保速度非零，避免除零错误
    rel_speed = max(abs(front_speed - rear_speed), 1e-5)
    front_speed = max(front_speed, 1e-5)
    rear_speed = max(rear_speed, 1e-5)
    
    # 计算TTC和PET
    ttc = d_ij / rel_speed  # 时间-碰撞
    pet = (d_j / rear_speed) - (d_i / front_speed)  # 后至先到时间
    
    # 动态调整风险权重 - 在不同情况下调整TTC和PET的权重
    if ttc < 1.5:  # 紧急情况下，TTC更重要
        w_ttc, w_pet = 0.8, 0.2
    elif pet < 1.0:  # PET很小时，也需要重视
        w_ttc, w_pet = 0.5, 0.5
    else:  # 正常情况
        w_ttc, w_pet = 0.6, 0.4
    
    # 将TTC和PET标准化到[0,1]范围
    ttc_norm = 1.0 if ttc < T_min else (0.0 if ttc > T_max else (T_max - ttc) / (T_max - T_min))
    pet_norm = 1.0 if pet < T_min else (0.0 if pet > T_max else (T_max - pet) / (T_max - T_min))
    
    # 添加指数衰减，使得小值的TTC/PET有更高风险
    ttc_exp = math.exp(-ttc/2) if ttc < 3.0 else ttc_norm
    pet_exp = math.exp(-pet/2) if pet < 2.0 else pet_norm
    
    # 计算综合冲突指数
    conflict_index = w_ttc * ttc_exp + w_pet * pet_exp
    
    # 添加额外危险系数 - 如果两者都很小，风险额外增加
    if ttc < 2.0 and pet < 1.5:
        conflict_index = min(1.0, conflict_index * 1.25)  # 加强风险提示，但上限为1
    
    return conflict_index, ttc, pet

def predict_conflict_point(pos_i, dir_i, v_i, pos_j, dir_j, v_j, T=5.0, dt=0.1):
    """预测两车可能的冲突点，用于风险评估"""
    # 初始化
    min_dist = float('inf')
    conflict_point, t_i_best, t_j_best = None, 0, 0
    
    # 考虑到车辆可能的加速/减速，扩展预测模式
    v_i_modes = [v_i, v_i * 0.8, v_i * 1.2]  # 匀速、减速、加速
    v_j_modes = [v_j, v_j * 0.8, v_j * 1.2]
    
    # 对多种速度模式进行冲突点搜索
    for v_i_mode in v_i_modes:
        for v_j_mode in v_j_modes:
            # 在时间窗口内预测轨迹
            for t in np.arange(0, T, dt):
                # 简单线性预测轨迹
                p_i = pos_i + v_i_mode * dir_i * t
                p_j = pos_j + v_j_mode * dir_j * t
                
                # 计算两车在此时刻的距离
                dist = np.linalg.norm(p_i - p_j)
                
                # 更新最小距离和对应的冲突点
                if dist < min_dist:
                    min_dist = dist
                    conflict_point = (p_i + p_j) / 2  # 冲突点取两车位置的中点
                    t_i_best, t_j_best = t, t
    
    # 如果两车轨迹方向接近平行，调整冲突风险
    cos_angle = np.dot(dir_i, dir_j) / (np.linalg.norm(dir_i) * np.linalg.norm(dir_j))
    if cos_angle > 0.9:  # 夹角小于约25度
        min_dist = min_dist * 0.8  # 增加风险（减小距离）
    
    return conflict_point, min_dist, t_i_best, t_j_best

def calc_lane_change_risk(ego, nearby_vehicles, left_change=True):
    """计算换道风险，用于智能换道决策"""
    target_lane = ego.lane_index[2] + (1 if left_change else -1)
    
    # 如果目标车道不存在，风险极高
    total_lanes = getattr(ego.road, 'lanes_count', 4)
    if target_lane < 0 or target_lane >= total_lanes:
        return 1.0
    
    # 筛选目标车道上的车辆
    target_lane_vehicles = [v for v in nearby_vehicles 
                           if v.lane_index and v.lane_index[2] == target_lane]
    
    if not target_lane_vehicles:
        return 0.1  # 无车辆，低风险
    
    # 计算最近前后车的风险
    max_risk = 0.0
    ego_pos = ego.position[0]  # 纵向位置
    
    for v in target_lane_vehicles:
        rel_pos = v.position[0] - ego_pos
        distance = abs(rel_pos)
        
        # 前车或后车
        is_front = rel_pos > 0
        
        # 计算相对速度
        rel_speed = v.speed - ego.speed
        
        # 计算TTC
        if (is_front and rel_speed < 0) or (not is_front and rel_speed > 0):
            # 接近中
            ttc = distance / max(abs(rel_speed), 1e-5)
            # 将TTC转换为风险值
            risk = 1.0 if ttc < 1.0 else (0.0 if ttc > 5.0 else (5.0 - ttc) / 4.0)
        else:
            # 相对远离，风险较低
            risk = max(0.0, 0.5 - distance / 50.0)
        
        # 距离过近直接增加风险
        if distance < 10.0:
            risk = max(risk, (10.0 - distance) / 10.0)
        
        max_risk = max(max_risk, risk)
    
    return max_risk

def analyze_traffic_density(vehicles, road_length=500.0, lanes_count=4):
    """分析交通密度和车流特性"""
    if not vehicles:
        return 0.0, 0.0, []
    
    # 计算总体密度
    density = len(vehicles) / (road_length * lanes_count)
    
    # 每个车道的车辆数
    lane_counts = [0] * lanes_count
    lane_speeds = [[] for _ in range(lanes_count)]
    
    for v in vehicles:
        if v.lane_index and v.lane_index[2] < lanes_count:
            lane_idx = v.lane_index[2]
            lane_counts[lane_idx] += 1
            lane_speeds[lane_idx].append(v.speed)
    
    # 计算车道密度不均匀性
    lane_densities = [count / road_length for count in lane_counts]
    density_std = np.std(lane_densities) if lane_densities else 0.0
    
    # 计算各车道平均速度
    avg_speeds = []
    for speeds in lane_speeds:
        if speeds:
            avg_speeds.append(np.mean(speeds))
        else:
            avg_speeds.append(0.0)
    
    return density, density_std, avg_speeds

def predict_vehicle_trajectory(vehicle, time_horizon=5.0, dt=0.1, simple=True):
    """预测车辆未来轨迹，用于冲突预测"""
    if simple:
        # 简单线性预测
        trajectory = []
        pos = vehicle.position.copy()
        heading = vehicle.heading
        speed = vehicle.speed
        
        for t in np.arange(0, time_horizon, dt):
            dx = speed * math.cos(heading) * dt
            dy = speed * math.sin(heading) * dt
            pos = pos + np.array([dx, dy])
            trajectory.append((pos.copy(), speed))
        
        return trajectory
    else:
        # 考虑加速度和路径的复杂预测
        # 此处可以扩展为更复杂的模型，如考虑车道跟随、IDM模型等
        # 当前使用简单版本
        return predict_vehicle_trajectory(vehicle, time_horizon, dt, True)