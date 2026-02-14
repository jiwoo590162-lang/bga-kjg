import math
import random

class MCTS:
    def __init__(self, net, simulations=100):
        self.net = net
        self.simulations = simulations
        self.last_trace = []
        self.last_policy = {}
        self.temperature = 0.9
        self.capture_bonus = 0.25

    def run(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None

        # Heuristic: prefer higher material gain and better mobility.
        best_move = None
        best_score = -10**9
        for mv in legal_moves:
            score = self._score_move(board, mv)
            if score > best_score:
                best_score = score
                best_move = mv

        if best_move is None:
            return random.choice(legal_moves)
        return best_move

    def run_with_score(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None, None

        best_move = None
        best_score = -10**9
        for mv in legal_moves:
            score = self._score_move(board, mv)
            if score > best_score:
                best_score = score
                best_move = mv

        if best_move is None:
            return random.choice(legal_moves), None
        return best_move, best_score

    def run_with_stats(self, board):
        move, score, stats = self.run_with_trace(board)
        return move, score, stats

    def run_with_trace(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            self.last_trace = ["no legal moves"]
            self.last_policy = {}
            return None, None, {"candidates": 0, "evaluated": 0, "best_score": None}

        root = _Node()
        priors, root_value = self.net.policy_value(board, legal_moves)
        root.expand(legal_moves, priors)

        self.last_trace = []
        for sim in range(self.simulations):
            node = root
            sim_board = board.clone()
            path = []

            while node.is_expanded():
                move, node = node.select_child()
                sim_board.move(move)
                path.append((node, move))

            legal = sim_board.get_legal_moves()
            if legal:
                priors, value = self.net.policy_value(sim_board, legal)
                node.expand(legal, priors)
            else:
                value = 0.0

            for n, _ in path:
                n.update(value)

            self.last_trace.append(
                f"sim {sim+1}: value={value:.3f}, expanded={len(legal)}"
            )

        best_move = root.best_move()
        best_score = root.children[best_move].q if best_move in root.children else root_value
        self.last_policy = root.policy_distribution()
        # Diversity: sample from policy distribution with temperature + capture bonus
        sampled = self._sample_move(board, self.last_policy)
        if sampled is not None:
            best_move = sampled
        stats = {
            "candidates": len(legal_moves),
            "evaluated": root.total_visits(),
            "best_score": round(best_score, 3) if best_score is not None else None
        }
        return best_move, best_score, stats

    def _sample_move(self, board, policy_dist):
        if not policy_dist:
            return None
        moves = list(policy_dist.keys())
        weights = []
        for mv in moves:
            w = policy_dist[mv]
            y, x, ny, nx = mv
            if board.board[ny][nx] is not None:
                w *= (1.0 + self.capture_bonus)
            weights.append(max(w, 1e-8))
        # apply temperature
        if self.temperature and self.temperature != 1.0:
            weights = [w ** (1.0 / self.temperature) for w in weights]
        return random.choices(moves, weights=weights, k=1)[0]


class _Node:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.children = {}
        self.n = 0
        self.w = 0.0
        self.q = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, legal_moves, priors):
        for mv in legal_moves:
            p = priors.get(mv, 1e-6)
            self.children[mv] = _Node(prior=p)

    def select_child(self, c_puct=1.2):
        best_score = -1e9
        best_move = None
        best_child = None
        sqrt_n = math.sqrt(self.n + 1)
        for mv, child in self.children.items():
            u = c_puct * child.prior * sqrt_n / (1 + child.n)
            score = child.q + u
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child
        return best_move, best_child

    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w / self.n

    def best_move(self):
        if not self.children:
            return None
        return max(self.children.items(), key=lambda kv: kv[1].n)[0]

    def total_visits(self):
        return self.n

    def policy_distribution(self):
        total = sum(child.n for child in self.children.values())
        if total == 0:
            return {}
        return {mv: (child.n / total) for mv, child in self.children.items()}

    def _score_move(self, board, move):
        test = board.clone()
        test.move(move)
        color = "r" if board.turn == "red" else "b"
        material = self._material_score(test, color)
        mobility = len(test.get_legal_moves())
        return material * 10 + mobility

    def _material_score(self, board, color):
        values = {
            "K": 10000,
            "G": 3,
            "R": 9,
            "H": 5,
            "E": 4,
            "C": 7,
            "S": 2
        }
        score = 0
        for row in board.board:
            for piece in row:
                if not piece:
                    continue
                piece_color = piece[0]
                kind = piece[1]
                value = values.get(kind, 0)
                if piece_color == color:
                    score += value
                else:
                    score -= value
        return score
