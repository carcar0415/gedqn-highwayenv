# main_test.py (Modified to plot metrics every 10 episodes, fixed syntax errors)
from env.custom_merge_env import CustomMergeEnv
from model.dqn_agent import DQNAgent
from utils.risk_utils import extract_structured_obs
from utils.reward_utils import _compute_conflict_index
import yaml
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

# ==== 配置加载 ====
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs("results", exist_ok=True)

# ==== 测试参数 ====
dt = 1.0 / config.get("simulation_frequency", 10)
plot_interval = 10  # 每10个episode输出一次图
max_episodes = config.get("max_test_episodes", 50)

# ==== 初始化环境 ====
env = CustomMergeEnv()
env.spec = type("spec", (), {"id": "CustomMerge-v0"})()

# ==== 加载模型 ====
agents = [DQNAgent(config) for _ in range(1)]
for i, agent in enumerate(agents):
    model_path = f"results/dqn_agent_{i}.pth"
    agent.net.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.net.eval()
print("✅ 模型已加载")

# ==== 测试循环 ====
for episode in range(1, max_episodes + 1):
    # 重置环境，获取观测
    obs, _ = env.reset()
    done = False

    # 初始化记录
    speed_list, accel_list, risk_list = [], [], []
    ego = env.road.vehicles[0]
    prev_speed = ego.speed
    frames = []
    step = 0

    while not done:
        # 构造状态
        risk_scores = [o[-1] if len(o) >= 8 else 0 for o in obs]
        state = extract_structured_obs(env.road, expected_num=len(agents), risk_scores=risk_scores)

        # 选择动作（纯 exploitation）
        actions = [int(agent.choose_action(state[i], epsilon=0.0)) for i, agent in enumerate(agents)]

        # 环境步进
        next_obs, reward, terminated, truncated, info = env.step(tuple(actions))
        done = terminated or truncated

        # 累积车辆状态与指标
        for v in env.road.vehicles:
            speed_list.append(v.speed)
            accel_list.append(getattr(v, 'acc_value', 0))
            I_T = _compute_conflict_index(
                v.speed,
                v.target_speed,
                getattr(v, 'gap', config.get('s_safe', 5)),
                config.get('T_min', 1.0),
                config.get('T_max', 5.0)
            )
            risk_list.append(I_T)

        # 渲染 GIF 帧
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(Image.fromarray(frame))

        obs = next_obs
        step += 1

    # 计算本集平均指标
    avg_speed = np.mean(speed_list) if speed_list else 0
    avg_accel = np.mean(accel_list) if accel_list else 0
    avg_risk = np.mean(risk_list) if risk_list else 0
    print(f"[Episode {episode}/{max_episodes}] 完成, steps={step}, "
          f"AvgSpeed={avg_speed:.2f} m/s, "
          f"AvgAccel={avg_accel:.2f} m/s², "
          f"AvgRisk={avg_risk:.3f}")

    # 每 plot_interval episodes 输出三张折线图
    if episode % plot_interval == 0:
        t = list(range(len(speed_list)))
        # 速度曲线
        plt.figure()
        plt.plot(t, speed_list, '-o', markersize=3)
        plt.title(f"Episode {episode} 速度随时间变化")
        plt.xlabel("Step")
        plt.ylabel("Speed (m/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/speed_episode_{episode}.png")
        plt.close()

        # 加速度曲线
        plt.figure()
        plt.plot(t, accel_list, '-o', markersize=3)
        plt.title(f"Episode {episode} 加速度随时间变化")
        plt.xlabel("Step")
        plt.ylabel("Accel (m/s²)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/accel_episode_{episode}.png")
        plt.close()

        # 风险曲线
        plt.figure()
        plt.plot(t, risk_list, '-o', markersize=3)
        plt.title(f"Episode {episode} Risk 随时间变化")
        plt.xlabel("Step")
        plt.ylabel("Risk")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/risk_episode_{episode}.png")
        plt.close()

        print(f"✅ 已生成 Episode {episode} 的速度、加速度和风险曲线图")

    # 保存 GIF
    if frames:
        gif_path = f"results/test_episode_{episode}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        print(f"✅ 保存 GIF: {gif_path}")

print("测试完成。")
