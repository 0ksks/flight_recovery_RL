import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


# 定义Q网络（Q-Network），用于估计Q值
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(128, 128)        # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(128, action_dim)  # 隐藏层2到输出层（每个动作的Q值）

    def forward(self, x):
        # 定义前向传播过程，使用ReLU激活函数
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        return self.fc3(x)  # 输出动作的Q值


# 回放缓冲区（Replay Buffer），用于存储经验数据
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用deque存储数据，设定最大容量

    def push(self, state, action, reward, next_state, done):
        # 向缓冲区中添加经验
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机抽取一个小批量数据
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        # 返回缓冲区中的经验数量
        return len(self.buffer)


# DDQN智能体（Agent）
class DDQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, tau=0.005, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_dim = self.env.n_nodes + 1  # 状态维度：已访问状态+当前状态
        self.action_dim = self.env.action_space.n  # 动作维度

        # 初始化行为网络和目标网络
        self.behaviour_q_network = QNetwork(self.state_dim, self.action_dim).to(th.device('cpu'))
        self.target_q_network = QNetwork(self.state_dim, self.action_dim).to(th.device('cpu'))

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.behaviour_q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # 初始化回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新因子
        self.epsilon = epsilon_start  # ε-greedy策略的初始ε值
        self.epsilon_end = epsilon_end  # ε的最小值
        self.epsilon_decay = epsilon_decay  # ε的衰减率

        # 初始化目标网络的参数
        self.update_target_network()

    def update_target_network(self):
        # 将行为网络的参数复制到目标网络
        self.target_q_network.load_state_dict(self.behaviour_q_network.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            # ε-greedy策略中的随机选择，在可访问的动作中随机选择
            return self.env.action_space.sample()
        else:
            state = th.FloatTensor(state).unsqueeze(0)  # 将状态转换为张量
            q_values = self.behaviour_q_network(state)  # 获取所有动作的Q值
            # 在可访问的动作中选择Q值最大的动作
            return q_values.max(1)[1].item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # 当缓冲区中的经验不足一个批次时，不进行训练

        # 从缓冲区中采样数据
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        state = th.FloatTensor(state)
        action = th.LongTensor(action)
        reward = th.FloatTensor(reward)
        next_state = th.FloatTensor(next_state)
        done = th.FloatTensor(done)

        # 计算行为网络的Q值
        q_values = self.behaviour_q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        next_q_values = self.behaviour_q_network(next_state)
        next_q_state_values = self.target_q_network(next_state)
        next_q_action = next_q_values.max(1)[1]
        target_q_values = (reward
                           + (1 - done) * self.gamma
                           * next_q_state_values
                           .gather(1, next_q_action.unsqueeze(1))
                           .squeeze(1))

        # 计算损失并更新行为网络
        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新ε值（逐渐减少探索）
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        # 保存模型
        th.save(self.behaviour_q_network.state_dict(), path)

    def load_model(self, path):
        # 加载模型并更新目标网络
        self.behaviour_q_network.load_state_dict(th.load(path))
        self.update_target_network()


# 训练循环
def train_ddqn(env, agent: DDQNAgent, num_episodes=1000, log_every_episode=100):
    for episode in range(num_episodes):
        state, _ = env.reset()  # 重置环境并获取初始状态
        done = False
        total_reward = 0

        while not done:
            # 将访问状态和当前状态合并作为输入
            state_input = np.concatenate([state["visited_state"], [state["current_state"]]])
            action = agent.select_action(state_input)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作，获取下一个状态和奖励
            next_state_input = np.concatenate([next_state["visited_state"], [next_state["current_state"]]])

            # 将经验推入缓冲区
            agent.replay_buffer.push(state_input, action, reward, next_state_input, done)
            agent.train()  # 训练智能体

            state = next_state  # 更新状态
            total_reward += reward  # 累积奖励

        # 软更新目标网络
        agent.update_target_network()
        if episode % log_every_episode == log_every_episode - 1:
            print(f"Episode {episode + 1}: Total Reward({int(total_reward)})")


# 获取最佳的一次episode
def get_best_episode(buffer: ReplayBuffer):
    start = 0
    end = 0
    best_episode_len = 0
    best_episode = []
    curr_episode_len = 0
    curr_episode = []
    for experience in buffer.buffer:
        action = experience[1]
        done = experience[4]
        if done:
            start += 1
            if curr_episode_len > best_episode_len:
                best_episode_len = curr_episode_len
                best_episode = curr_episode
            curr_episode = []
            curr_episode_len = 0
        else:
            end += 1
            curr_episode.append(action)
            curr_episode_len += 1
    return best_episode, best_episode_len


# 示例代码
if __name__ == "__main__":
    from dag_env import DagEnv
    from adj_list import ADJList

    # 初始化环境和智能体
    adj_list = ADJList(20, density=0.3)
    env = DagEnv(adj_list.inner_adj_list)
    agent = DDQNAgent(env)

    agent.load_model("model.pt")

    # 开始训练
    train_ddqn(env, agent, num_episodes=1000, log_every_episode=100)

    agent.save_model("model.pt")

    # 获取最佳的动作序列（episode）
    best_episode, _ = get_best_episode(agent.replay_buffer)

    # 可视化最佳路径
    fig, ax = adj_list.register_drawer(6, 1.5, 0.1)
    adj_list.config_font_from_toml("font_config.toml")
    adj_list.ini_elements()
    adj_list.ini_fig(ax, add_arrow=False)
    adj_list.add_trajectory_elements(ax, [0,] + best_episode, add_arrow=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("example.png")
