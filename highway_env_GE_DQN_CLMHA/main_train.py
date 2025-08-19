import os, sys, yaml, numpy as np, torch, pandas as pd
import matplotlib.pyplot as plt
from env.custom_merge_env import CustomMergeEnv
from model.dqn_agent import DQNAgent
from utils.risk_utils import extract_structured_obs, _compute_conflict_index, predict_conflict_point
from utils.game_utils import compute_nash_action
import time
import matplotlib
import time
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境

# 初始化参数
previous_avg_risk, previous_avg_lane_changes = 0.0, 0.0
window_size, eps_reward, eps_loss = 50, 0.02, 1e-3
reward_history, loss_history = [], []

# 实时可视化相关参数
live_plot_interval = 20  # 每20个episode更新一次实时图表
plot_window_size = 100   # 实时图表显示最近100个episode的数据

def check_convergence(rewards, losses, window, eps_r, eps_l):
    """检查训练是否收敛"""
    if len(rewards) < 2 * window or len(losses) < 2 * window:
        return False
    if np.array(rewards[-window:]).std() > eps_r:
        return False
    l1, l0 = np.array(losses[-window:]), np.array(losses[-2*window:-window])
    return abs(l1.mean() - l0.mean()) < eps_l

def create_live_plots(metrics, episode, save_dir="results/live_plots"):
    """创建并保存实时训练指标图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 确定要绘制的数据范围（最近 plot_window_size 个episode）
    start_idx = max(0, episode - plot_window_size)
    x_range = list(range(start_idx + 1, episode + 2))
    
    # 创建多子图布局
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Training Progress - Episode {episode+1}", fontsize=16)
    
    # 绘制各种指标
    plot_configs = [
        (metrics["episode_rewards"][start_idx:episode+1], "Reward", axs[0, 0], "Episode", "Reward", "tab:blue"),
        (metrics["avg_risks"][start_idx:episode+1], "Risk", axs[0, 1], "Episode", "Avg Risk", "tab:red"),
        (metrics["avg_lane_changes"][start_idx:episode+1], "Lane Changes", axs[0, 2], "Episode", "Avg Lane Changes", "tab:green"),
        (metrics["avg_speeds"][start_idx:episode+1], "Speed", axs[1, 0], "Episode", "Avg Speed (m/s)", "tab:orange"),
        (metrics["avg_gaps"][start_idx:episode+1], "Gap", axs[1, 1], "Episode", "Avg Gap (m)", "tab:purple"),
        (metrics["avg_collisions"][start_idx:episode+1], "Collisions", axs[1, 2], "Episode", "Collisions", "tab:red"),
        (metrics["avg_ttc"][start_idx:episode+1], "TTC", axs[2, 0], "Episode", "Avg TTC (s)", "tab:brown"),
        (metrics["avg_pet"][start_idx:episode+1], "PET", axs[2, 1], "Episode", "Avg PET (s)", "tab:pink"),
        (metrics["loss_history"][start_idx:episode+1], "Loss", axs[2, 2], "Episode", "Loss", "tab:gray")
    ]
    
    for data, title, ax, xlabel, ylabel, color in plot_configs:
        ax.plot(x_range, data, color=color, marker='.', alpha=0.7)
        # 添加平滑曲线
        if len(data) > 10:
            window_size = min(10, len(data)//5)
            if window_size > 1:
                smooth_data = pd.Series(data).rolling(window=window_size, center=True).mean()
                ax.plot(x_range, smooth_data, color=color, linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加最近值标签
        if len(data) > 0:
            last_val = data[-1]
            ax.text(0.95, 0.05, f"Latest: {last_val:.4f}", 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{save_dir}/training_progress_episode_{episode+1}.png", dpi=150)
    plt.close(fig)
    
    # 创建进度追踪图表 - 单独保存最新进度图
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, metrics["episode_rewards"][start_idx:episode+1], 'b-', label='Reward')
    plt.plot(x_range, metrics["avg_risks"][start_idx:episode+1], 'r-', label='Risk')
    plt.fill_between(x_range, metrics["avg_collisions"][start_idx:episode+1], alpha=0.3, color='r', label='Collisions')
    plt.title(f"Training Progress Overview - Episode {episode+1}")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/current_progress.png", dpi=150)
    plt.close()

# class NullWriter:
#     def write(self, arg): pass
import sys
real_stdout = sys.stdout
real_stderr = sys.stderr
# 保存控制台输出
real_stdout, real_stderr = sys.stdout, sys.stderr
# sys.stdout, sys.stderr = NullWriter(), NullWriter()

# 创建结果目录
os.makedirs("results", exist_ok=True)
os.makedirs("results/live_plots", exist_ok=True)

# 加载配置
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

config = {
    # 训练参数
    "max_train_episodes": 30000,
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "batch_size": 128,
    "memory_size": 50000,
    "train_freq": 1,
    "target_update_interval": 100,
    "tau": 0.005,  # 软更新系数
    "use_priority": True,  # 启用优先经验回放
    
    # 环境参数
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "vehicles_count": 10,
    "controlled_vehicles": 1,
    
    # 动作空间参数
    "action_lateral": True,
    "action_longitudinal": True,
    "continuous_actions": False,
    
    # 奖励函数参数
    "reward_scale": 2.0,  # 奖励缩放
    "v_max": 30,
    "a_comf": 2.0,
    "s_safe": 10,
    "lambda_I": 0.40,
    "lambda_v": 0.20,
    "lambda_a": 0.20,
    "lambda_d": 0.20,
    "T_min": 1.0,
    "T_max": 5.0,
    
    # 状态表示
    "state_dim": 45,  # 扩展状态维度
    
    # 辅助训练功能
    "save_best_model": True,
    "eval_episodes": 10,
    "eval_interval": 100
}
# 设置仿真时间步长
dt = 1.0 / config.get("simulation_frequency", 10)

# 课程学习设置
curriculum_stages = [
    {"episodes": 500, "traffic_density": 5, "crash_penalty": -500, "epsilon_start": 1.0},
    {"episodes": 1000, "traffic_density": 8, "crash_penalty": -1000, "epsilon_start": 0.5},
    {"episodes": 1500, "traffic_density": 10, "crash_penalty": -1500, "epsilon_start": 0.3},
    {"episodes": 2000, "traffic_density": 12, "crash_penalty": -2000, "epsilon_start": 0.1}
]

# 当前课程阶段
current_stage = 0
stage_config = curriculum_stages[current_stage]

# ε-贪婪探索参数
epsilon_start = stage_config["epsilon_start"]
epsilon_end = 0.05
epsilon_decay_episodes = config["max_train_episodes"]

# 初始化环境和智能体
env_config = {
    "vehicles_count": stage_config["traffic_density"],
    "controlled_vehicles": 1,
    "simulation_frequency": 20,
    "policy_frequency": 10,
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "render_agent": True,
    "offscreen_rendering": False
}
env = CustomMergeEnv(config=env_config)
agents = [DQNAgent(config) for _ in range(1)]

# 初始化记录指标
metrics = {
    "avg_speeds": [], 
    "avg_accels": [], 
    "avg_gaps": [],
    "avg_risks": [], 
    "avg_lane_changes": [], 
    "episode_rewards": [],
    "avg_ttc": [], 
    "avg_pet": [], 
    "avg_collisions": [],
    "loss_history": []
}

# 训练开始时间
training_start_time = time.time()
sys.stdout = real_stdout
print(f"训练开始于: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
print(f"课程学习第1阶段: traffic_density={stage_config['traffic_density']}, crash_penalty={stage_config['crash_penalty']}")
sys.stdout = real_stdout

# 主训练循环
for episode in range(config["max_train_episodes"]):
    # 更新课程阶段
    print(f"开始 Episode {episode+1}/{config['max_train_episodes']}...")
    if episode >= stage_config["episodes"] and current_stage < len(curriculum_stages) - 1:
        current_stage += 1
        stage_config = curriculum_stages[current_stage]
        env.config.update({"vehicles_count": stage_config["traffic_density"]})
        sys.stdout = real_stdout
        print(f"进入课程学习第 {current_stage + 1} 阶段，traffic_density={stage_config['traffic_density']}, crash_penalty={stage_config['crash_penalty']}")
        sys.stdout = real_stdout
    
    # 调整探索率
    epsilon_start = stage_config["epsilon_start"]
    epsilon = max(
        epsilon_end,
        epsilon_start - (episode / epsilon_decay_episodes) * (epsilon_start - epsilon_end)
    )
    
    # 动态调整碰撞惩罚
    crash_penalty = stage_config["crash_penalty"]
    
    # 重置环境
    obs, info = env.reset()
    done = False
    for v in env.road.vehicles: v.prev_speed = v.speed
    
    # 初始化本集指标
    speed_list, accel_list, gap_list, risk_list, ttc_list, pet_list = [], [], [], [], [], []
    lane_change_count, collision_count, episode_reward, step_losses, step_idx = 0, 0, 0, [], 0
    
    # 记住上一步动作，用于平滑
    previous_actions = [0] * len(agents)
    
    # 本集训练循环
    while not done:
        # 风险计算
        if step_idx == 0:
            risk_scores = [previous_avg_risk + previous_avg_lane_changes] * len(agents)
        else:
            risk_scores = [o[-1] if len(o) >= 8 else 0 for o in obs]
        
        # 状态构建与动作选择
        state = extract_structured_obs(env.road, expected_num=len(agents), risk_scores=risk_scores)
        nash_actions = compute_nash_action(agents, state)
        
        # 使用平滑动作选择
        flat_actions = []
        for i, agent in enumerate(agents):
            a = agent.choose_action(
                state[i], 
                nash_actions[i], 
                epsilon,
                prev_action=previous_actions[i] if i < len(previous_actions) else None
            )
            flat_actions.append(int(a))
        
        # 记住当前动作，供下一步使用
        previous_actions = flat_actions.copy()
        # 确保动作是元组形式
        action_tuple = tuple(flat_actions)
        # 记录换道前状态
        ego = env.road.vehicles[0]
        old_lane = ego.lane_index[2] if ego.lane_index else None
        
        # 环境步进
        next_obs, reward, terminated, truncated, info = env.step(tuple(flat_actions))
        done = terminated or truncated
        
        # 记录换道
        ego = env.road.vehicles[0]
        new_lane = ego.lane_index[2] if ego.lane_index else None
        if old_lane is not None and new_lane is not None and new_lane != old_lane:
            lane_change_count += 1
        
        # 记录车辆状态
        for v in env.road.vehicles:
            accel = (v.speed - getattr(v, 'prev_speed', v.speed)) / dt
            accel_list.append(accel)
            v.prev_speed = v.speed
            speed_list.append(v.speed)
            gap_list.append(getattr(v, "gap", 0))
        
        # 计算冲突指标
        if len(env.road.vehicles) >= 2:
            front_car, rear_car = env.road.vehicles[0], env.road.vehicles[1]
            pos_i, dir_i, v_i = front_car.position, np.array([np.cos(front_car.heading), np.sin(front_car.heading)]), front_car.speed
            pos_j, dir_j, v_j = rear_car.position, np.array([np.cos(rear_car.heading), np.sin(rear_car.heading)]), rear_car.speed
            conflict_point, min_dist, _, _ = predict_conflict_point(pos_i, dir_i, v_i, pos_j, dir_j, v_j)
            d_ij, d_i, d_j = min_dist, np.linalg.norm(pos_i - conflict_point), np.linalg.norm(pos_j - conflict_point)
            conflict_index, ttc, pet = _compute_conflict_index(v_i, v_j, d_ij, d_i, d_j)
        else:
            conflict_index, ttc, pet = 0, 0, 0
        
        # 累计指标
        risk_list.append(conflict_index)
        ttc_list.append(ttc)
        pet_list.append(pet)
        if getattr(ego, 'crashed', False): 
            collision_count += 1
        
        # 奖励缩放
        raw_r = sum(reward) if isinstance(reward, (list, tuple, np.ndarray)) else reward
        
        # 如果发生碰撞，应用课程阶段的碰撞惩罚
        if info.get("crash", False):
            raw_r += crash_penalty
            
        scaled_r = np.tanh(raw_r / config.get("reward_scale", 1.0))
        episode_reward += scaled_r
        
        # 检测危险场景，增强经验回放优先级
        danger_detected = info.get("collision_warning", False) or (
            len(risk_list) >= 5 and np.mean(risk_list[-5:]) > 0.7
        )
        priority_factor = 2.0 if danger_detected else 1.0
        
        # 经验回放与训练
        for i, agent in enumerate(agents):
            s = np.array(state[i]).flatten()
            s_ = np.array(next_obs[i]).flatten() if i < len(next_obs) else np.array(state[i]).flatten()
            
            # 使用优先级添加经验
            agent.replay_buffer_add(s, flat_actions[i], scaled_r, s_, done, priority_factor)
            
            # 执行训练步骤
            loss = agent.train_step()
            if loss is not None: 
                step_losses.append(loss)
        
        obs, step_idx = next_obs, step_idx + 1
    
    # 本集结束，计算平均指标
    metrics["avg_speeds"].append(np.mean(speed_list) if speed_list else 0)
    metrics["avg_accels"].append(np.mean(accel_list) if accel_list else 0)
    metrics["avg_gaps"].append(np.mean(gap_list) if gap_list else 0)
    metrics["avg_risks"].append(np.mean(risk_list) if risk_list else 0)
    metrics["avg_lane_changes"].append(lane_change_count)
    metrics["episode_rewards"].append(episode_reward)
    metrics["avg_ttc"].append(np.mean(ttc_list) if ttc_list else 0)
    metrics["avg_pet"].append(np.mean(pet_list) if pet_list else 0)
    metrics["avg_collisions"].append(collision_count)
    metrics["loss_history"].append(np.mean(step_losses) if step_losses else np.nan)
    
    # 更新前一集指标（用于下一集）
    previous_avg_risk, previous_avg_lane_changes = metrics["avg_risks"][-1], metrics["avg_lane_changes"][-1]
    
    # 收敛检测数据
    reward_history.append(episode_reward)
    loss_history.append(np.mean(step_losses) if step_losses else np.nan)
    
    # 输出训练进度
    if (episode + 1) % 5 == 0 or episode == 0:  # 每5个episode输出一次
        sys.stdout = real_stdout
        current_time = time.time()
        elapsed_time = current_time - training_start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"[Episode {episode+1}/{config['max_train_episodes']}] "
              f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s | "
              f"Reward: {episode_reward:.2f} | "
              f"Risk: {previous_avg_risk:.3f} | "
              f"LaneChanges: {previous_avg_lane_changes} | "
              f"Collisions: {collision_count} | "
              f"ε: {epsilon:.2f}")
        sys.stdout = real_stdout
    
    # 绘制实时图表
    if (episode + 1) % live_plot_interval == 0 or episode == 0:
        sys.stdout = real_stdout
        print(f"创建实时图表... Episode {episode+1}")
        create_live_plots(metrics, episode)
        sys.stdout = real_stdout
    
    # 收敛检测
    if (episode+1) % window_size == 0:
        if check_convergence(reward_history, loss_history, window_size, eps_reward, eps_loss):
            sys.stdout = real_stdout
            print(f"收敛检测通过，训练在 Episode {episode+1} 时提前结束。")
            # 保存当前的实时图表作为最终图表
            create_live_plots(metrics, episode)
            sys.stdout = real_stdout
            break

# 训练结束，恢复stdout
sys.stdout = real_stdout
training_end_time = time.time()
training_duration = training_end_time - training_start_time
hours, remainder = divmod(training_duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"✅ 训练完成，总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

# 保存模型
for i, agent in enumerate(agents):
    torch.save(agent.net.state_dict(), f"results/dqn_agent_{i}.pth")
print("✅ 模型已保存至 results/")

# 创建数据帧并保存指标
df = pd.DataFrame({
    "episode": list(range(1, len(metrics["avg_speeds"])+1)),
    "avg_speed": metrics["avg_speeds"], 
    "avg_accel": metrics["avg_accels"], 
    "avg_gap": metrics["avg_gaps"],
    "avg_risk": metrics["avg_risks"], 
    "avg_lane_change": metrics["avg_lane_changes"], 
    "episode_reward": metrics["episode_rewards"],
    "avg_ttc": metrics["avg_ttc"], 
    "avg_pet": metrics["avg_pet"], 
    "avg_collision": metrics["avg_collisions"],
    "loss": metrics["loss_history"]
})
df.to_csv("results/training_metrics.csv", index=False)
print("✅ 指标已保存至 results/training_metrics.csv")

# 绘制最终聚合图表
def aggregate(data, w, mode='mean'):
    """聚合数据，支持平均值和求和"""
    if len(data) == 0:
        return []
    return [np.sum(data[i:i+w]) if mode == 'sum' else np.mean(data[i:i+w]) for i in range(0, len(data), w)]

# 设置聚合窗口大小
window = 100
x_vals = list(range(window, len(metrics["avg_speeds"])+1, window))

# 定义要绘制的图表
plots = [
    (metrics["avg_speeds"], "平均速度趋势图", "Speed (m/s)", "results/agg_avg_speed_curve.png", 'mean'),
    (metrics["avg_accels"], "平均加速度趋势图", "Acceleration (m/s²)", "results/agg_avg_accel_curve.png", 'mean'),
    (metrics["avg_gaps"], "平均车距趋势图", "Gap (m)", "results/agg_avg_gap_curve.png", 'mean'),
    (metrics["avg_risks"], "平均风险趋势图", "Risk", "results/agg_avg_risk_curve.png", 'mean'),
    (metrics["avg_lane_changes"], "平均换道次数趋势图", "Lane Changes", "results/agg_avg_lane_change_curve.png", 'mean'),
    (metrics["episode_rewards"], "平均奖励趋势图", "Total Reward", "results/agg_avg_reward_curve.png", 'mean'),
    (metrics["avg_ttc"], "平均TTC趋势图", "TTC (s)", "results/agg_avg_ttc_curve.png", 'mean'),
    (metrics["avg_pet"], "平均PET趋势图", "PET (s)", "results/agg_avg_pet_curve.png", 'mean'),
    (metrics["avg_collisions"], "碰撞总数趋势图", "Collisions", "results/agg_sum_collision_curve.png", 'sum')
]

# 绘制并保存各个指标图表
for data, title, ylabel, fname, mode in plots:
    if len(data) > 0:  # 确保有数据可用
        y = aggregate(data, window, mode)
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y, linestyle='-', marker='o', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

# 创建一个综合图表
plt.figure(figsize=(12, 8))
reward_agg = aggregate(metrics["episode_rewards"], window, 'mean')
risk_agg = aggregate(metrics["avg_risks"], window, 'mean')
collision_agg = aggregate(metrics["avg_collisions"], window, 'sum')

plt.subplot(2, 1, 1)
plt.plot(x_vals, reward_agg, 'b-', marker='o', label='Reward', linewidth=2)
plt.title("Training Progress Summary", fontsize=16)
plt.ylabel("Reward", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(x_vals, risk_agg, 'r-', marker='o', label='Risk', linewidth=2)
plt.bar(x_vals, collision_agg, alpha=0.3, color='purple', label='Collisions', width=window*0.8)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig("results/training_summary.png", dpi=150)
plt.close()

print("✅ 聚合图表已保存至 results/")
print("✅ 训练完成！")