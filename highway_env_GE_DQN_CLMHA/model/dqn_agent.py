import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.BatchNorm1d(256),  # 增加批归一化
            nn.Dropout(0.2),      # 增加Dropout防止过拟合
            nn.Linear(256, 256), nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # 处理批次大小为1的情况
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

class DQNAgent:
    """
    GE-DQN 智能体类（融合博弈策略 + DQN结构），支持软更新、梯度裁剪、ε-贪婪探索和优先经验回放。
    """
    def __init__(self, config):
        # 检查配置参数
        required_keys = ["learning_rate", "gamma", "state_dim", 
                         "batch_size", "train_freq", "target_update_interval"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置缺少必要参数: {key}")

        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]
        self.action_space = [0, 1, 2, 3, 4]
        
        self.state_dim = config["state_dim"]
        self.action_dim = len(self.action_space)

        # Q 网络与目标网络
        self.net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # 优先经验回放参数
        self.memory_size = config.get("memory_size", 10000)
        self.use_priority = config.get("use_priority", True)
        
        if self.use_priority:
            # 优先经验回放缓存和相关参数
            self.alpha = config.get("priority_alpha", 0.6)  # 优先级指数
            self.beta = config.get("priority_beta", 0.4)    # 重要性采样指数
            self.beta_increment = config.get("beta_increment", 0.001)
            self.epsilon_priority = 1e-5  # 防止优先级为0
            
            # 初始化优先回放存储
            self.memory = []  # 存储经验
            self.priorities = np.zeros((self.memory_size,), dtype=np.float32)  # 存储优先级
            self.memory_pos = 0  # 当前存储位置
            self.memory_full = False  # 标记缓存是否已满
        else:
            # 普通经验回放缓存
            self.memory = deque(maxlen=self.memory_size)

        self.batch_size = config["batch_size"]
        self.train_freq = config["train_freq"]
        self.step_counter = 0
        self.target_update_interval = config["target_update_interval"]

        # 软更新系数 τ
        self.tau = config.get("tau", 0.005)
        
        # 记录上一个动作，用于平滑动作选择
        self.prev_action = None

    def choose_action(self, state, nash_action=None, epsilon=0.1, prev_action=None):
        """选择动作，支持 ε-贪婪策略、纳什动作融合和平滑动作过渡"""
        self.net.eval()
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            if np.random.rand() < epsilon:
                # ε-贪婪探索：随机选动作
                dqn_action = np.random.choice(self.action_space)
            else:
                # 正常 Q 网络决策
                q_vals = self.net(state)
                
                # 如果有前一个动作，添加动作平滑约束
                if prev_action is not None:
                    # 添加惩罚，使动作变化不会过大
                    action_diff = np.abs(np.array(self.action_space) - prev_action)
                    smoothing_penalty = action_diff * 0.5  # 平滑系数
                    
                    # 应用惩罚
                    for i, penalty in enumerate(smoothing_penalty):
                        q_vals[0, i] -= penalty
                        
                dqn_action = q_vals.argmax().item()

        # 融合纳什动作（如提供）
        if nash_action is not None:
            final_action = int(0.7 * dqn_action + 0.3 * nash_action)  # 调整权重
        else:
            final_action = dqn_action
            
        # 与前一动作平滑过渡
        if prev_action is not None:
            # 限制单步动作变化
            max_change = 1  # 最大允许变化幅度
            action_change = final_action - prev_action
            
            if abs(action_change) > max_change:
                final_action = prev_action + np.sign(action_change) * max_change

        # 保存此次动作作为下次的prev_action
        self.prev_action = final_action
        
        return int(final_action)

    def replay_buffer_add(self, s, a, r, s_, done, priority_factor=1.0):
        """将经验存入回放缓冲区，带优先级"""
        if self.use_priority:
            # 优先经验回放存储逻辑
            # 获取当前最大优先级，用于新加入的经验
            max_priority = np.max(self.priorities) if self.memory else 1.0
            
            # 根据priority_factor调整优先级（用于高风险场景）
            adjusted_priority = max_priority * priority_factor
            
            # 如果缓存未满，直接添加
            if len(self.memory) < self.memory_size:
                self.memory.append((s, a, r, s_, done))
            else:
                # 缓存已满，覆盖旧经验
                self.memory[self.memory_pos] = (s, a, r, s_, done)
                self.memory_full = True
                
            # 设置优先级    
            self.priorities[self.memory_pos] = adjusted_priority
            
            # 更新存储位置
            self.memory_pos = (self.memory_pos + 1) % self.memory_size
        else:
            # 普通经验回放存储逻辑
            self.memory.append((s, a, r, s_, done))

    def _pad_state(self, state, target_dim):
        """自动补齐状态维度"""
        if isinstance(state, np.ndarray) and state.ndim > 1 and state.shape[1] != target_dim:
            pad = target_dim - state.shape[1]
            state = np.pad(state, ((0, 0), (0, pad)), mode='constant')
        return state

    def train_step(self):
        """执行一次训练步骤，使用优先经验回放"""
        if self.use_priority:
            # 检查经验回放缓冲区是否有足够样本
            memory_size = len(self.memory) if not self.memory_full else self.memory_size
            if memory_size < self.batch_size:
                return None
                
            # 优先经验回放采样
            # 计算采样概率
            priorities = self.priorities[:memory_size]
            probs = priorities ** self.alpha
            probs /= np.sum(probs) + self.epsilon_priority
            
            # 采样索引和计算重要性权重
            indices = np.random.choice(memory_size, self.batch_size, p=probs)
            samples = [self.memory[idx] for idx in indices]
            
            # 计算重要性采样权重
            weights = (memory_size * probs[indices]) ** (-self.beta)
            weights /= np.max(weights) + self.epsilon_priority
            
            # Beta参数随着训练进行递增
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # 解包经验样本
            s, a, r, s_, d = zip(*samples)
            
            # 将权重转为张量
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1)
        else:
            # 普通回放缓冲区采样
            if len(self.memory) < self.batch_size:
                return None
                
            batch = random.sample(self.memory, self.batch_size)
            s, a, r, s_, d = zip(*batch)
            indices = None  # 普通回放不需要索引
            weights_tensor = torch.ones((self.batch_size, 1))  # 均匀权重
            
        # 转为 numpy 加速转换
        s = np.array(s)
        s_ = np.array(s_)

        # 自动修复状态维度
        s = self._pad_state(s, self.state_dim)
        s_ = self._pad_state(s_, self.state_dim)

        # 转为 Tensor
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        d = torch.FloatTensor(d).unsqueeze(1)

        # Q-Learning 更新
        q_eval = self.net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s_).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * (1 - d) * q_next

        # 使用权重调整损失（对于优先回放）
        td_errors = torch.abs(q_eval - q_target).detach()
        loss = torch.mean(weights_tensor * nn.MSELoss(reduction='none')(q_eval, q_target))
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新优先级（仅优先回放）
        if self.use_priority and indices is not None:
            td_errors_np = td_errors.numpy()
            for i, idx in enumerate(indices):
                self.priorities[idx] = td_errors_np[i][0] + self.epsilon_priority

        # 软更新目标网络 θ_target ← τ θ_online + (1-τ) θ_target
        for p, p_targ in zip(self.net.parameters(), self.target_net.parameters()):
            p_targ.data.mul_(1.0 - self.tau)
            p_targ.data.add_(self.tau * p.data)

        self.step_counter += 1
        return loss.item()