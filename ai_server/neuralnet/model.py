import threading
import time
import random
from collections import deque
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None


PIECE_TO_PLANE = {
    "rK": 0,
    "rG": 1,
    "rR": 2,
    "rH": 3,
    "rE": 4,
    "rC": 5,
    "rS": 6,
    "bK": 7,
    "bG": 8,
    "bR": 9,
    "bH": 10,
    "bE": 11,
    "bC": 12,
    "bS": 13
}

ACTION_SIZE = 90 * 90


def move_to_index(move):
    y, x, ny, nx = move
    return (y * 9 + x) * 90 + (ny * 9 + nx)


def index_to_move(idx):
    from_sq = idx // 90
    to_sq = idx % 90
    y, x = divmod(from_sq, 9)
    ny, nx = divmod(to_sq, 9)
    return (y, x, ny, nx)


def encode_board(board):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    planes = torch.zeros((15, 10, 9), dtype=torch.float32)
    for y in range(10):
        for x in range(9):
            piece = board.board[y][x]
            if piece is None:
                continue
            plane = PIECE_TO_PLANE.get(piece)
            if plane is not None:
                planes[plane, y, x] = 1.0
    turn_plane = 14
    planes[turn_plane, :, :] = 1.0 if board.turn == "red" else -1.0
    return planes


if nn is not None and F is not None and torch is not None:
    class JanggiNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(15, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
            self.policy_fc = nn.Linear(4 * 10 * 9, ACTION_SIZE)

            self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
            self.value_fc1 = nn.Linear(2 * 10 * 9, 128)
            self.value_fc2 = nn.Linear(128, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))

            p = self.policy_conv(x)
            p = p.view(p.size(0), -1)
            p = self.policy_fc(p)

            v = self.value_conv(x)
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v))

            return p, v
else:
    class JanggiNet:
        def __init__(self):
            raise RuntimeError("PyTorch is not installed.")


class NeuralNetWrapper:
    def __init__(self, device="cpu"):
        self.device = device
        self._lock = threading.Lock()
        self._train_state = {
            "running": False,
            "step": 0,
            "max_steps": 0,
            "loss": None,
            "status": "idle",
            "auto": False,
            "target_loss": None
        }
        self._trace = deque(maxlen=200)
        self._dataset_maxlen = 20000
        self._dataset = deque(maxlen=self._dataset_maxlen)
        self._model = None
        self._optimizer = None
        self._auto_thread = None
        self._data_path = Path("ai_server/neuralnet/data.pt")
        self._auto_save_every = 50
        self._auto_save_counter = 0
        self._auto_save_every_samples = 200
        self._data_added_since_save = 0
        self._try_autoload()
        if len(self._dataset) > 0:
            self.set_auto_train(True)

    def _ensure_model(self):
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install torch to enable training.")
        if self._model is None:
            self._model = JanggiNet().to(self.device)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

    def _try_autoload(self):
        if torch is None:
            return
        if self._data_path.exists():
            try:
                obj = torch.load(self._data_path, map_location="cpu")
                data = obj.get("dataset", [])
                self._dataset = deque(data, maxlen=self._dataset_maxlen)
            except Exception:
                pass

    def predict(self, board):
        return None, 0

    def policy_value(self, board, legal_moves):
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            x = encode_board(board).unsqueeze(0).to(self.device)
            logits, value = self._model(x)
            logits = logits.squeeze(0).cpu()
            value = float(value.squeeze(0).item())

        priors = {}
        if legal_moves:
            indices = [move_to_index(mv) for mv in legal_moves]
            legal_logits = logits[indices]
            probs = torch.softmax(legal_logits, dim=0).cpu().tolist()
            for mv, p in zip(legal_moves, probs):
                priors[mv] = float(p)
        return priors, value

    def get_train_state(self):
        with self._lock:
            state = dict(self._train_state)
        state["dataset_size"] = len(self._dataset)
        return state

    def get_trace(self):
        with self._lock:
            return list(self._trace)

    def record_step(self, board, move):
        self._ensure_model()
        state = encode_board(board).clone()
        policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        policy_target[move_to_index(move)] = 1.0
        value_target = self._material_value(board)
        self._dataset.append((state, policy_target, value_target))
        self._maybe_autosave_on_data()

    def add_game(self, history, winner_color):
        # history: list of (state_tensor, policy_target, player_color)
        if winner_color not in ("red", "blue", None):
            winner_color = None
        for state, policy_target, player in history:
            if winner_color is None:
                value_target = 0.0
            else:
                value_target = 1.0 if player == winner_color else -1.0
            self._dataset.append((state, policy_target, float(value_target)))
        if history:
            self._maybe_autosave_on_data(len(history))

    def add_game_from_moves(self, moves, winner_color):
        from ai_server.engine.board import Board
        b = Board()
        history = []
        for from_pos, to_pos in moves:
            move = b.move_tuple_from_pos(from_pos, to_pos)
            state = encode_board(b).clone()
            policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
            policy_target[move_to_index(move)] = 1.0
            history.append((state, policy_target, b.turn))
            b.apply_move_by_pos(from_pos, to_pos)
            if b.game_result() is not None:
                break
        self.add_game(history, winner_color)

    def set_auto_train(self, enabled: bool):
        with self._lock:
            self._train_state["auto"] = bool(enabled)
        if enabled and (self._auto_thread is None or not self._auto_thread.is_alive()):
            self._auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
            self._auto_thread.start()

    def set_target_loss(self, target_loss):
        with self._lock:
            self._train_state["target_loss"] = target_loss

    def _auto_loop(self):
        while True:
            with self._lock:
                enabled = self._train_state["auto"]
                target_loss = self._train_state.get("target_loss")
            if not enabled:
                time.sleep(0.2)
                continue
            try:
                self._ensure_model()
            except Exception:
                time.sleep(0.5)
                continue
            if len(self._dataset) < 64:
                time.sleep(0.2)
                continue
            loss = self._train_step()
            with self._lock:
                self._train_state["status"] = "training"
                self._train_state["step"] = self._train_state.get("step", 0) + 1
                self._train_state["max_steps"] = 0
                self._train_state["loss"] = round(loss, 5)
                self._trace.append(f"auto step: loss={loss:.5f}")
                if target_loss is not None and loss <= target_loss:
                    self._train_state["auto"] = False
                    self._train_state["status"] = "idle"
                    self._trace.append(f"auto stop: target_loss {target_loss} reached")
                self._auto_save_counter += 1
                if self._auto_save_counter >= self._auto_save_every:
                    self._auto_save_counter = 0
                    self.save_dataset(self._data_path)
            time.sleep(0.05)

    def start_training(self, steps=100, episodes=4, max_moves=60):
        self._ensure_model()
        with self._lock:
            if self._train_state["running"]:
                return False
            self._train_state.update({
                "running": True,
                "step": 0,
                "max_steps": steps,
                "loss": None,
                "status": "training"
            })

        thread = threading.Thread(
            target=self._train_loop,
            args=(steps, episodes, max_moves),
            daemon=True
        )
        thread.start()
        return True

    def _train_loop(self, steps, episodes, max_moves):
        try:
            for step in range(1, steps + 1):
                if len(self._dataset) < 64:
                    self._generate_self_play(episodes, max_moves)
                loss = self._train_step()
                with self._lock:
                    self._train_state["step"] = step
                    self._train_state["loss"] = round(loss, 5)
                    self._train_state["status"] = "training"
                    self._trace.append(f"train step {step}: loss={loss:.5f}")
                    if step % self._auto_save_every == 0:
                        self.save_dataset(self._data_path)
                time.sleep(0.05)
        finally:
            with self._lock:
                self._train_state["running"] = False
                self._train_state["status"] = "idle"
                self.save_dataset(self._data_path)

    def _generate_self_play(self, episodes, max_moves):
        from ai_server.engine.board import Board
        for ep in range(episodes):
            b = Board()
            history = []
            for _ in range(max_moves):
                legal = b.get_legal_moves()
                if not legal:
                    break
                priors, _ = self.policy_value(b, legal)
                moves = list(priors.keys())
                probs = [priors[mv] for mv in moves]
                mv = random.choices(moves, weights=probs, k=1)[0]
                state = encode_board(b).clone()
                policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
                for move, prob in priors.items():
                    policy_target[move_to_index(move)] = float(prob)
                player = b.turn
                history.append((state, policy_target, player))
                b.move(mv)
                winner = b.game_result()
                if winner is not None:
                    self.add_game(history, winner)
                    break
            else:
                self.add_game(history, None)

    def _material_value(self, board):
        values = {
            "K": 1.0,
            "G": 0.2,
            "R": 0.5,
            "H": 0.3,
            "E": 0.25,
            "C": 0.4,
            "S": 0.1
        }
        score = 0.0
        for row in board.board:
            for piece in row:
                if not piece:
                    continue
                val = values.get(piece[1], 0.0)
                score += val if piece[0] == "r" else -val
        # Check pressure bonus/penalty
        try:
            if board.is_in_check("r"):
                score -= 0.3
            if board.is_in_check("b"):
                score += 0.3
        except Exception:
            pass
        return max(-1.0, min(1.0, score))

    def _train_step(self):
        self._ensure_model()
        batch = random.sample(list(self._dataset), k=min(32, len(self._dataset)))
        states = torch.stack([s for s, _, _ in batch]).to(self.device)
        policy_targets = torch.stack([p for _, p, _ in batch]).to(self.device)
        value_targets = torch.tensor([v for _, _, v in batch], dtype=torch.float32).to(self.device)

        self._model.train()
        logits, values = self._model(states)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values.squeeze(1), value_targets)
        loss = policy_loss + value_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss.item())

    def save_dataset(self, path):
        self._ensure_model()
        path = Path(path)
        data = list(self._dataset)
        torch.save({"dataset": data}, path)
        return str(path)

    def load_dataset(self, path):
        self._ensure_model()
        path = Path(path)
        obj = torch.load(path, map_location="cpu")
        data = obj.get("dataset", [])
        self._dataset = deque(data, maxlen=self._dataset_maxlen)
        return len(self._dataset)

    def _maybe_autosave_on_data(self, added_count=1):
        self._data_added_since_save += added_count
        if self._data_added_since_save >= self._auto_save_every_samples:
            self._data_added_since_save = 0
            self.save_dataset(self._data_path)
