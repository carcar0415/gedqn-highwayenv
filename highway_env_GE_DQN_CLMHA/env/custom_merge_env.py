# env/custom_merge_env.py

import gymnasium as gym
import numpy as np
import torch
import random
from highway_env.envs.merge_env import MergeEnv
from model.clmha_model import CLMHA

class CustomMergeEnv(MergeEnv):
    """自定义合流环境，适用于highway-env 1.10.1"""
    
    def __init__(self, config=None):
        # 使用默认配置初始化
        super().__init__(render_mode="human")
        
        # 更新配置
        default_config = {
            "controlled_vehicles": 1,
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction"
                }
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "absolute": True
            },
            "collision_reward": -5,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lanes_count": 4,
            "vehicles_count": 10,
            "policy_frequency": 10,
            "simulation_frequency": 20,
            "duration": 40,
            "acceleration_range": [-3.0, 3.0],
            "max_acceleration": 3.0,
            "max_deceleration": 3.0,
        }
        
        # 如果提供了配置，合并它
        if config:
            default_config.update(config)
            
        # 应用配置
        self.configure(default_config)
        
        # 初始化CLMHA模型
        try:
            self.clmha = CLMHA(input_dim=8, hidden_dim=64)
            self.clmha.load_state_dict(torch.load("model/clmha_model.pth", map_location="cpu"))
            self.clmha.eval()
        except Exception as e:
            print(f"初始化CLMHA模型时出错: {e}")
            self.clmha = None
            
        # 初始化历史记录和速度
        self.seq_len = 5
        self.history = []
        self._last_speeds = {}
    
    def reset(self, seed=None, options=None):
        """
        重置环境，适应新版API
        """
        # 使用父类的reset方法
        observation, info = super().reset(seed=seed, options=options)
        
        # 重置历史和速度记录
        self.history = []
        self._last_speeds = {id(v): v.speed for v in self.road.vehicles}
        
        return observation, info
    
    def step(self, action):
        """
        执行环境步进，处理观测和风险计算
        """
        if not isinstance(action, tuple):
            action = (action,)
        # 调用父类的step方法
        observation, reward, terminated, truncated, info = super().step(action)
        
        # 更新历史记录
        self.history.append(observation)
        
        # 计算风险分数
        risk_scores = np.zeros((len(observation),))
        if self.clmha is not None and len(self.history) >= self.seq_len:
            from utils.risk_utils import extract_state_sequence
            try:
                input_seq = extract_state_sequence(self.history[-self.seq_len:])
                with torch.no_grad():
                    risk_scores = self.clmha(input_seq).squeeze(-1).numpy()
            except Exception as e:
                print(f"计算风险分数时出错: {e}")
        
        # 处理观测
        from utils.risk_utils import extract_structured_obs
        try:
            processed_obs = extract_structured_obs(self.road, self.config["controlled_vehicles"], risk_scores)
        except Exception as e:
            print(f"处理观测时出错: {e}")
            processed_obs = observation
        
        # 更新车辆加速度
        for v in self.road.vehicles:
            last_v = self._last_speeds.get(id(v), v.speed)
            current_v = v.speed
            v.acc_value = (current_v - last_v) / (1.0 / self.config.get("policy_frequency", 10))
            self._last_speeds[id(v)] = current_v
            
            # 检测潜在碰撞风险
            v.potential_collision = False
            for other in self.road.vehicles:
                if v != other:
                    distance = np.linalg.norm(v.position - other.position)
                    relative_speed = v.speed - other.speed
                    ttc = distance / max(relative_speed, 1e-5) if relative_speed > 0 else float('inf')
                    
                    if ttc < 2.0 and distance < 20:  # 风险阈值
                        v.potential_collision = True
                        break
        
        # 计算奖励
        new_rewards = []
        if isinstance(reward, (list, tuple, np.ndarray)):
            from utils.reward_utils import compute_custom_reward
            for i, v in enumerate(self.road.vehicles[:self.config["controlled_vehicles"]]):
                try:
                    r = compute_custom_reward(info, v, self.config)
                    new_rewards.append(r)
                except Exception as e:
                    print(f"计算自定义奖励时出错: {e}")
                    new_rewards.append(reward[i] if i < len(reward) else 0)
        else:
            new_rewards = reward
            
        # 更新信息字典
        info["risk"] = float(np.mean(risk_scores))
        info["collision_warning"] = any(getattr(v, "potential_collision", False) 
                                     for v in self.road.vehicles[:self.config["controlled_vehicles"]])
        
        return processed_obs, new_rewards, terminated, truncated, info