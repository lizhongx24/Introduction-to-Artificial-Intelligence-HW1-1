from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame, EPS

import numpy as np

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree
        possible_index = np.flatnonzero(node.action_mask)
        N_visit = node.child_N_visit[possible_index]
        unvisited = possible_index[N_visit == 0]
        if unvisited.size > 0:
            return int(np.random.choice(unvisited))
        N_np = N_visit.astype(np.float32)
        Exploit = node.child_V_total[possible_index] / N_np
        N_parent = node.child_N_visit.sum()
        Explore = self.config.C * np.sqrt(np.log(N_parent) / N_np)
        UCB = Exploit + Explore
        return int(possible_index[np.argmax(UCB)])

    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        # 提示：传入的 value 是叶节点“当前行动方”视角的价值，而父节点边上的
        # 统计量属于在父节点选择该动作的一方。每向上经过一条边都要转换一次
        # 玩家视角，并让该边的访问次数恰好增加 1。
        current_node = node
        while(current_node.parent != None):
            value = -value
            current_node.parent.child_N_visit[current_node.action] += 1
            current_node.parent.child_V_total[current_node.action] += value
            current_node = current_node.parent
    
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!
        # 提示：必须先 fork()，再从合法动作中随机落子直到终局，不能修改树节点
        # 保存的环境。step() 的终局 reward 属于刚刚落子的玩家；返回前应将它
        # 转换成 rollout 开始时 node.env.current_player 的视角。和棋的 EPS
        # 是中性标记，无论落子奇偶都直接返回 EPS。
        node_env = node.env.fork()
        done = node_env.ended
        if(done):
            return 0.0
        count = 0
        reward = 0.0
        while(done == False):
            possible_index = np.flatnonzero(node_env._valid_action_mask)
            action = int(np.random.choice(possible_index))
            _,reward, done = node_env.step(action,return_obs = False)
            count += 1
        if(reward == EPS):
            return EPS
        if(count % 2 == 0):
            reward = -reward
        return reward
    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        # 提示：从根节点开始用 uct_action_select() 选择动作；若对应孩子已经
        # 存在就继续向下，否则用 add_child() 扩展并立即返回新节点。若到达
        # 终局节点，则直接返回该终局节点。
        current_node = self.root
        if(current_node.done):
            return current_node
        action = self.uct_action_select(current_node)
        while(current_node.has_child(action)):
            next_node = current_node.children[action]
            current_node = next_node
            if(current_node.done):
                return current_node
            action = self.uct_action_select(current_node)
        return current_node.add_child(action)
    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy comes from the visit count of each action
        # 提示：本题直接将合法动作的访问次数归一化为概率；非法动作概率为 0。
        # 本次作业不要求使用 config 中的 temperature 或噪声相关参数。
        if(node == None):
            node = self.root 
        possible_index = np.flatnonzero(node.action_mask)
        N_visit = node.child_N_visit[possible_index]
        N_total_visit = N_visit.sum()
        distribution = np.zeros(node.n_action, dtype = float)
        if (N_total_visit == 0):
            if(len(possible_index) > 0):
                distribution[possible_index] = 1.0 / len (possible_index)
            return distribution
        distribution[possible_index] = N_visit / N_total_visit
        return distribution

    def search(self):
        # search the tree for n_search times
        # each time, pick a leaf node, rollout the game (if game is not ended)
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                # 提示：leaf.reward 属于生成该叶节点时刚刚落子的玩家，而此时
                # leaf.env.current_player 已经切换为另一方；传给 backup 前需要
                # 转换为叶节点当前行动方视角。
                if(leaf.reward != EPS):
                    value += - leaf.reward
                else:
                    value += EPS
            else:
                # 提示：独立执行 n_rollout 次 rollout，取算术平均作为本轮叶节点
                # 价值，然后只调用一次 backup。因此每轮搜索的访问次数只增加 1。
                count = 0
                total_value = 0.0
                while (count < self.config.n_rollout):
                    total_value += self.rollout(leaf)
                    count += 1
                value += (total_value / self.config.n_rollout)
            self.backup(leaf, value)

        return self.get_policy(self.root)
