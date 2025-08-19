# main.py
from env.custom_merge_env import CustomMergeEnv
from model.dqn_agent import DQNAgent
from utils.risk_utils import extract_structured_obs
from utils.game_utils import compute_nash_action
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


os.makedirs("results", exist_ok=True)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

env = CustomMergeEnv()
env.spec = type("spec", (), {"id": "CustomMerge-v0"})()

# agents = [DQNAgent(config) for _ in range(config["num_agents"])]
agents = [DQNAgent(config) for _ in range(1)] 

for agent in agents:
    agent.memory.clear()

# ✅ 初始化趋势数据
episode_rewards = []
mainlane_speeds = []
mainlane_accels = []
episode_crashes = []

for episode in range(config["max_episodes"]):
    obs = env.reset()
    done = False
    episode_reward = 0
    frames = []

    while not done:
        # ✅ 使用结构化状态提取函数（包含目标车+最近车辆+风险）
        risk_scores = [o[-1] if len(o) >= 8 else 0 for o in obs]
        state = extract_structured_obs(env.road, expected_num=len(agents), risk_scores=risk_scores)
        nash_actions = compute_nash_action(agents, state)

        actions = []
        for i, agent in enumerate(agents):
            if i >= len(state):
                actions.append(0)
                continue
            action = agent.choose_action(state[i], nash_actions[i],epsilon=0.1)
            action = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
            actions.append(action)

        flat_actions = [int(a[0]) if isinstance(a, (list, np.ndarray)) else int(a) for a in actions]
        next_obs, reward, terminated, truncated, info = env.step(tuple(flat_actions))
        done = terminated or truncated
        
        # 统计变道动作
        num_lane_changes = sum(1 for a in actions if a in [3, 4])
        print(f"变道动作数: {num_lane_changes}")
        if action in [3, 4]:
            print(f"🚗 Agent {i} 尝试变道 -> Action = {action}")


        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(Image.fromarray(frame))

        if isinstance(reward, (list, tuple, np.ndarray)):
            episode_reward += sum(reward)
        else:
            episode_reward += reward

        for i, agent in enumerate(agents):
            r = reward[i] if isinstance(reward, (list, tuple, np.ndarray)) else reward
            s = np.array(state[i]).flatten()
            s_ = np.array(state[i]).flatten() if i >= len(next_obs) else np.array(next_obs[i]).flatten()
            agent.replay_buffer_add(s, flat_actions[i], r, s_, done)
            agent.train_step()
        

        obs = next_obs

    episode_rewards.append(episode_reward)
    episode_crashes.append(1 if info.get("crash", False) else 0)

    main_speeds = [
        v.speed for v in env.road.vehicles
        if getattr(v, "lane_index", [None]*3)[2] == 0
    ]
    main_accels = []
    for v in env.road.vehicles:
        if getattr(v, "lane_index", [None]*3)[2] == 0:
            try:
                acc = v.acceleration() if callable(v.acceleration) else v.acceleration
                if isinstance(acc, (int, float)):
                    main_accels.append(acc)
            except Exception:
                continue

    mainlane_speeds.append(np.mean(main_speeds) if main_speeds else 0)
    mainlane_accels.append(np.mean(main_accels) if main_accels else 0)

    if episode == 0 and frames:
        gif_path = f"results/trajectory_episode{episode}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )

    print(f"[Episode {episode + 1:>3}/{config['max_episodes']}] Reward: {episode_reward:.2f}")


import torch
q_vals = agent.net(torch.FloatTensor(state[i]).unsqueeze(0))
print(f"Q-values Agent {i}:", q_vals.detach().numpy())


# ✅ 趋势图绘制
def plot_metric(data, title, ylabel, filename):
    plt.figure()
    plt.plot(data, marker='o')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_metric(episode_rewards, "奖励趋势图", "Total Reward", "results/reward_curve.png")
plot_metric(mainlane_speeds, "主线速度趋势图", "Speed (m/s)", "results/speed_curve.png")
plot_metric(mainlane_accels, "主线加速度趋势图", "Acceleration (m/s²)", "results/accel_curve.png")
plot_metric(episode_crashes, "碰撞趋势图", "Crash (0 or 1)", "results/crash_curve.png")
