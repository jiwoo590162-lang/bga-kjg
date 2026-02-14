from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

from ai_server.engine.board import Board
from ai_server.mcts.mcts import MCTS
import json
import re
import random
from ai_server.neuralnet.model import NeuralNetWrapper, encode_board, move_to_index, ACTION_SIZE, torch

app = FastAPI(
    title="Janggi AI Server",
    description="Human vs AI Janggi API",
    version="0.1"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class MoveRequest(BaseModel):
    from_pos: str
    to_pos: str


board = Board()
net = NeuralNetWrapper()
mcts = MCTS(net)
last_ai_stats = {}
last_ai_trace = []
turn_counter = 0
game_history = []
gen_progress = {"running": False, "current": 0, "total": 0}
gen_lock = False


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return {"status": "Janggi AI Server Running"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return FileResponse(STATIC_DIR / "ui.html")


@app.get("/state")
def state():
    return board.to_dict()


@app.post("/reset")
def reset():
    board.setup()
    board.turn = "red"
    game_history.clear()
    return {"status": "ok"}

@app.get("/legal")
def legal(from_pos: str):
    try:
        moves = board.legal_moves_from_pos(from_pos)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"moves": [board.move_to_pos(mv)[1] for mv in moves]}


@app.get("/ai_status")
def ai_status(light: bool = True):
    state = {
        "mcts": last_ai_stats,
        "train": net.get_train_state()
    }
    if not light:
        state["trace"] = last_ai_trace
        state["train_trace"] = net.get_trace()
    return state


@app.get("/ai_status_light")
def ai_status_light():
    return {
        "mcts": last_ai_stats,
        "train": net.get_train_state()
    }


@app.post("/train/start")
def train_start(steps: int = 50):
    ok = net.start_training(steps=steps)
    return {"started": ok, "state": net.get_train_state()}


@app.post("/train/auto/start")
def train_auto_start(target_loss: float = None):
    if target_loss is not None and target_loss < 0:
        target_loss = None
    net.set_target_loss(target_loss)
    net.set_auto_train(True)
    return {"auto": True, "state": net.get_train_state()}


@app.post("/train/auto/stop")
def train_auto_stop():
    net.set_auto_train(False)
    return {"auto": False, "state": net.get_train_state()}


@app.post("/train/save")
def train_save(path: str = "ai_server/neuralnet/data.pt"):
    saved = net.save_dataset(path)
    return {"saved": saved}


@app.post("/train/load")
def train_load(path: str = "ai_server/neuralnet/data.pt"):
    count = net.load_dataset(path)
    return {"loaded": count}


@app.post("/train/records/load")
def train_records_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    loaded = 0
    for game in data:
        moves = game.get("moves", [])
        winner = game.get("winner")
        if not moves:
            continue
        try:
            net.add_game_from_moves(moves, winner)
            loaded += 1
        except Exception:
            continue
    return {"loaded": loaded}


def _kif_piece_to_kind(piece):
    mapping = {
        "차": "R",
        "마": "H",
        "상": "E",
        "사": "G",
        "포": "C",
        "졸": "S",
        "왕": "K",
        "장": "K"
    }
    return mapping.get(piece)


def _side_to_color(side):
    # User chose: 초=top, 한=bottom. Our board has red bottom, blue top.
    return "blue" if side == "초" else "red"


def _file_num_to_pos(file_num, y):
    files = "abcdefghi"
    x = file_num - 1
    return f"{files[x]}{y}"


def _parse_kif_moves(text):
    lines = text.splitlines()
    winner = None
    moves = []
    b = Board()
    applied = 0
    skipped = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") and "Result" in line:
            if "초" in line and "승리" in line:
                winner = "blue"
            elif "한" in line and "승리" in line:
                winner = "red"
            continue

        m = re.match(r"^\d+\.\s*(초|한)\s+(\d)(차|마|상|사|포|졸|왕|장)\s*→\s*(\d)(차|마|상|사|포|졸|왕|장)?", line)
        if not m:
            skipped += 1
            continue

        side, from_file_s, piece_s, to_file_s, _to_piece = m.groups()
        from_file = int(from_file_s)
        to_file = int(to_file_s)
        kind = _kif_piece_to_kind(piece_s)
        if kind is None:
            skipped += 1
            continue

        # Set turn to match side
        b.turn = "red" if _side_to_color(side) == "red" else "blue"
        legal = b.get_legal_moves()
        # filter by piece kind and file
        candidates = []
        for mv in legal:
            y, x, ny, nx = mv
            piece = b.board[y][x]
            if not piece:
                continue
            if piece[1] != kind:
                continue
            if x != (from_file - 1):
                continue
            if nx != (to_file - 1):
                continue
            candidates.append(mv)

        if len(candidates) != 1:
            skipped += 1
            continue

        mv = candidates[0]
        b.move(mv)
        frm, to = b.move_to_pos(mv)
        moves.append([frm, to])
        applied += 1

    return {"winner": winner, "moves": moves, "applied": applied, "skipped": skipped}


@app.post("/train/kif/load")
def train_kif_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parsed = _parse_kif_moves(text)
    if parsed["moves"]:
        net.add_game_from_moves(parsed["moves"], parsed["winner"])
    return {
        "moves_applied": parsed["applied"],
        "moves_skipped": parsed["skipped"],
        "winner": parsed["winner"],
        "loaded": len(parsed["moves"])
    }


@app.post("/records/generate")
def records_generate(
    count: int = 200,
    max_moves: int = 120,
    sims: int = 15,
    path: str = "ai_server/neuralnet/records_generated.json",
    auto_load: bool = False,
    auto_train_steps: int = 0
):
    global gen_lock
    if gen_lock:
        return {"error": "generation already running", "progress": dict(gen_progress)}
    gen_lock = True
    local_mcts = MCTS(net, simulations=sims)
    games = []
    gen_progress["running"] = True
    gen_progress["current"] = 0
    gen_progress["total"] = count
    for _ in range(count):
        gen_progress["current"] += 1
        b = Board()
        moves = []
        for _ in range(max_moves):
            mv, _, _ = local_mcts.run_with_stats(b)
            if mv is None:
                break
            frm, to = b.move_to_pos(mv)
            moves.append([frm, to])
            b.move(mv)
            winner = b.game_result()
            if winner is not None:
                games.append({"winner": winner, "moves": moves})
                break
        else:
            games.append({"winner": None, "moves": moves})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False, indent=2)
    gen_progress["running"] = False
    if gen_progress["current"] > gen_progress["total"]:
        gen_progress["current"] = gen_progress["total"]
    response = {"saved": path, "games": len(games)}
    gen_lock = False
    if auto_load:
        try:
            loaded = net.load_dataset(path)
            response["loaded"] = loaded
        except Exception:
            response["loaded"] = 0
    if auto_train_steps and auto_train_steps > 0:
        try:
            net.start_training(steps=auto_train_steps)
            response["training_started"] = True
        except Exception:
            response["training_started"] = False
    return response


@app.get("/records/progress")
def records_progress():
    return dict(gen_progress)


@app.get("/eval")
def eval_ai(games: int = 20, sims: int = 200):
    local_mcts = MCTS(net, simulations=sims)
    wins = {"mcts": 0, "random": 0, "draw": 0}
    for g in range(games):
        b = Board()
        mcts_color = "red" if g % 2 == 0 else "blue"
        for _ in range(200):
            legal = b.get_legal_moves()
            if not legal:
                break
            if (b.turn == "red" and mcts_color == "red") or (b.turn == "blue" and mcts_color == "blue"):
                mv, _, _ = local_mcts.run_with_stats(b)
                if mv is None:
                    break
            else:
                mv = random.choice(legal)
            b.move(mv)
            winner = b.game_result()
            if winner is not None:
                if winner == mcts_color:
                    wins["mcts"] += 1
                else:
                    wins["random"] += 1
                break
        else:
            wins["draw"] += 1
    total = wins["mcts"] + wins["random"] + wins["draw"]
    return {
        "games": total,
        "mcts_win_rate": wins["mcts"] / total if total else 0,
        "random_win_rate": wins["random"] / total if total else 0,
        "draw_rate": wins["draw"] / total if total else 0
    }


@app.post("/move")
def make_move(move: MoveRequest):
    pre_board = board.clone()
    move_tuple = board.move_tuple_from_pos(move.from_pos, move.to_pos)
    try:
        board.apply_move_by_pos(move.from_pos, move.to_pos)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        state = encode_board(pre_board).clone()
        policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        policy_target[move_to_index(move_tuple)] = 1.0
        game_history.append((state, policy_target, pre_board.turn))
    except Exception:
        pass

    legal = board.get_legal_moves()
    if not legal:
        return {
            "human_move": f"{move.from_pos} -> {move.to_pos}",
            "ai_move": None,
            "status": "game over"
        }

    ai_move, ai_score, stats = mcts.run_with_stats(board)
    global last_ai_stats
    last_ai_stats = stats
    global last_ai_trace, turn_counter
    turn_counter += 1
    last_ai_trace.append(f"--- Turn {turn_counter} ---")
    last_ai_trace.extend(mcts.last_trace)
    if ai_move is None:
        winner = board.game_result()
        if winner is not None:
            try:
                net.add_game(game_history, winner)
            except Exception:
                pass
            game_history.clear()
        return {
            "human_move": f"{move.from_pos} -> {move.to_pos}",
            "ai_move": None,
            "status": "game over",
            "ai_eval": None,
            "mcts": stats
        }

    pre_ai = board.clone()
    board.move(ai_move)
    try:
        state = encode_board(pre_ai).clone()
        policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        for mv, prob in (mcts.last_policy or {}).items():
            policy_target[move_to_index(mv)] = float(prob)
        if policy_target.sum().item() == 0:
            policy_target[move_to_index(ai_move)] = 1.0
        game_history.append((state, policy_target, pre_ai.turn))
    except Exception:
        pass
    ai_from, ai_to = board.move_to_pos(ai_move)
    winner = board.game_result()
    if winner is not None:
        try:
            net.add_game(game_history, winner)
        except Exception:
            pass
        game_history.clear()
    return {
        "human_move": f"{move.from_pos} -> {move.to_pos}",
        "ai_move": f"{ai_from} -> {ai_to}",
        "ai_eval": ai_score,
        "mcts": stats
    }
