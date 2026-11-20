"""Depth-limited negamax with Alpha-Beta pruning for 5x5 Go."""

import numpy as np

from env.base_env import BaseGame, NOTEND
from other_algo.heuristic import go_heuristic_evaluation


class AlphaBetaSearch:
    """A short Alpha-Beta exercise; the visualizer consumes the emitted events."""

    def __init__(self, depth=2):
        self.depth_limit = depth
        self.trace_callback = None
        self._next_node_id = 0

    def _emit(self, phase, **data):
        if self.trace_callback is not None:
            self.trace_callback({"phase": phase, **data})

    def _child(self, parent_id, action, depth, state, alpha, beta):
        node_id = self._next_node_id
        self._next_node_id += 1
        self._emit(
            "node", node_id=node_id, parent_id=parent_id, action=int(action),
            depth=depth, alpha=alpha, beta=beta, board=state.observation,
        )
        return node_id

    def max_value(self, state: BaseGame, alpha, beta, depth, node_id=0, level=0):
        if depth <= 0:
            value = float(go_heuristic_evaluation(state))
            self._emit("leaf", node_id=node_id, value=value, alpha=alpha, beta=beta)
            return value

        value = -np.inf
        valid_moves = np.where(state.action_mask == 1)[0]
        for index, action in enumerate(valid_moves):
            child_state = state.fork()
            _, reward, _ = child_state.step(action, return_obs=False)
            child_id = self._child(node_id, action, level + 1, child_state, -beta, -alpha)
            if reward != NOTEND:
                child_value = float(reward)
                self._emit("terminal", node_id=child_id, value=child_value, alpha=-beta, beta=-alpha)
            else:
                child_value = -self.max_value(child_state, -beta, -alpha, depth - 1, node_id = child_id, level = level + 1)
            value = max(value, child_value)
            alpha = max(value, alpha)
            self._emit(
                "update", node_id=node_id, action=int(action), value=float(value),
                action_value=float(child_value), alpha=float(alpha), beta=float(beta),
            )
            should_prune = (alpha >= beta)  
            if should_prune:
                self._emit(
                    "prune", node_id=node_id, alpha=float(alpha), beta=float(beta),
                    skipped=[int(item) for item in valid_moves[index + 1:]],
                )
                break
        return value

    def get_best_move(self, state: BaseGame):
        self._next_node_id = 1
        self._emit(
            "node", node_id=0, parent_id=None, action=None, depth=0,
            alpha=-np.inf, beta=np.inf, board=state.observation,
        )
        best_move, best_value = -1, -np.inf
        alpha, beta = -np.inf, np.inf
        valid_moves = np.where(state.action_mask == 1)[0]
        for index, action in enumerate(valid_moves):
            child_state = state.fork()
            _, reward, _ = child_state.step(action, return_obs=False)
            child_id = self._child(0, action, 1, child_state, -beta, -alpha)
            value = float(reward) if reward != NOTEND else -self.max_value(
                child_state, -beta, -alpha, self.depth_limit - 1, child_id, 1
            )
            if reward != NOTEND:
                self._emit("terminal", node_id=child_id, value=value, alpha=-beta, beta=-alpha)
            if value > best_value:
                best_value, best_move = value, int(action)
            alpha = max(alpha, best_value)
            self._emit(
                "update", node_id=0, action=int(action), value=float(best_value),
                action_value=float(value), alpha=float(alpha), beta=float(beta), best_move=best_move,
            )
        self._emit("finish", node_id=0, action=best_move, value=float(best_value))
        return best_move
