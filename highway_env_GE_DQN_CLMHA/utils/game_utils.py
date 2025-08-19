# utils/game_utils.py
# 纳什均衡近似求解（Best Response）

import numpy as np
import torch

# 修改utils/game_utils.py中的compute_nash_action函数

def compute_nash_action(agents, state_list):
    actions = []
    for i, agent in enumerate(agents):
        # 存储当前训练状态
        training_mode = agent.net.training
        
        # 切换到评估模式
        agent.net.eval()
        
        best_q = -np.inf
        best_a = 0
        
        with torch.no_grad():  # 不计算梯度
            state_array = np.array(state_list[i]).flatten()
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            
            q_values = agent.net(state_tensor).cpu().numpy().squeeze()
            best_a = np.argmax(q_values)
                
        # 恢复原来的训练状态
        if training_mode:
            agent.net.train()
            
        actions.append(int(best_a))
    return actions


