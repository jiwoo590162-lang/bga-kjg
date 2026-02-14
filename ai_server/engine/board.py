FILES = "abcdefghi"
RANKS = [str(n) for n in range(1, 11)]


def pos_to_coord(pos):
    if not isinstance(pos, str):
        raise ValueError("position must be a string like 'e3'")
    pos = pos.strip().lower()
    if len(pos) < 2 or len(pos) > 3:
        raise ValueError("position must be like 'e3' or 'e10'")
    file_char = pos[0]
    rank_str = pos[1:]
    if file_char not in FILES or rank_str not in RANKS:
        raise ValueError("position out of range")
    x = FILES.index(file_char)
    y = int(rank_str) - 1
    return y, x


def coord_to_pos(y, x):
    return f"{FILES[x]}{y + 1}"


class Board:
    def __init__(self):
        self.board = [[None for _ in range(9)] for _ in range(10)]
        self.turn = "red"
        self.setup()

    def setup(self):
        self.board = [[None for _ in range(9)] for _ in range(10)]
        # Red (bottom)
        self._place("rR", "a1")
        self._place("rH", "b1")
        self._place("rE", "c1")
        self._place("rG", "d1")
        self._place("rG", "f1")
        self._place("rE", "g1")
        self._place("rH", "h1")
        self._place("rR", "i1")
        self._place("rK", "e2")
        self._place("rC", "b3")
        self._place("rC", "h3")
        for pos in ["a4", "c4", "e4", "g4", "i4"]:
            self._place("rS", pos)

        # Blue (top)
        self._place("bR", "a10")
        self._place("bH", "b10")
        self._place("bE", "c10")
        self._place("bG", "d10")
        self._place("bG", "f10")
        self._place("bE", "g10")
        self._place("bH", "h10")
        self._place("bR", "i10")
        self._place("bK", "e9")
        self._place("bC", "b8")
        self._place("bC", "h8")
        for pos in ["a7", "c7", "e7", "g7", "i7"]:
            self._place("bS", pos)

    def _place(self, piece, pos):
        y, x = pos_to_coord(pos)
        self.board[y][x] = piece

    def in_bounds(self, y, x):
        return 0 <= y < 10 and 0 <= x < 9

    def piece_at(self, y, x):
        return self.board[y][x]

    def _color_of(self, piece):
        return piece[0]

    def _kind_of(self, piece):
        return piece[1]

    def _is_enemy(self, piece, other):
        return piece is not None and other is not None and self._color_of(piece) != self._color_of(other)

    def _palace(self, color):
        if color == "r":
            return (0, 2, 3, 5)
        return (7, 9, 3, 5)

    def _in_palace(self, y, x, color):
        y1, y2, x1, x2 = self._palace(color)
        return y1 <= y <= y2 and x1 <= x <= x2

    def get_legal_moves(self):
        color = "r" if self.turn == "red" else "b"
        pseudo = self._pseudo_legal_moves_for_color(color)
        legal = []
        for mv in pseudo:
            test = self.clone()
            test.move(mv)
            if test._is_facing_kings():
                continue
            if test.is_in_check(color):
                continue
            legal.append(mv)
        return legal

    def _pseudo_legal_moves_for_color(self, color):
        moves = []
        for y in range(10):
            for x in range(9):
                piece = self.board[y][x]
                if piece is None:
                    continue
                if self._color_of(piece) != color:
                    continue
                moves.extend(self._piece_moves(y, x, piece))
        return moves

    def _find_king(self, color):
        target = f"{color}K"
        for y in range(10):
            for x in range(9):
                if self.board[y][x] == target:
                    return (y, x)
        return None

    def winner_color(self):
        red = self._find_king("r") is not None
        blue = self._find_king("b") is not None
        if red and blue:
            return None
        if red and not blue:
            return "red"
        if blue and not red:
            return "blue"
        return None

    def game_result(self):
        winner = self.winner_color()
        if winner:
            return winner
        # No legal moves means the side to move loses.
        if not self.get_legal_moves():
            return "blue" if self.turn == "red" else "red"
        return None

    def _is_facing_kings(self):
        rk = self._find_king("r")
        bk = self._find_king("b")
        if not rk or not bk:
            return False
        ry, rx = rk
        by, bx = bk
        if rx != bx:
            return False
        step = 1 if by > ry else -1
        y = ry + step
        while y != by:
            if self.board[y][rx] is not None:
                return False
            y += step
        return True

    def is_in_check(self, color):
        king_pos = self._find_king(color)
        if not king_pos:
            return False
        enemy = "b" if color == "r" else "r"
        for mv in self._pseudo_legal_moves_for_color(enemy):
            _, _, ny, nx = mv
            if (ny, nx) == king_pos:
                return True
        return False

    def _add_if_valid(self, moves, y, x, ny, nx, piece):
        if not self.in_bounds(ny, nx):
            return
        target = self.board[ny][nx]
        if target is None or self._is_enemy(piece, target):
            moves.append((y, x, ny, nx))

    def _piece_moves(self, y, x, piece):
        kind = self._kind_of(piece)
        color = self._color_of(piece)
        moves = []

        if kind == "K":
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not self.in_bounds(ny, nx):
                        continue
                    if not self._in_palace(ny, nx, color):
                        continue
                    self._add_if_valid(moves, y, x, ny, nx, piece)

        elif kind == "G":
            for dy in [-1, 1]:
                for dx in [-1, 1]:
                    ny, nx = y + dy, x + dx
                    if not self.in_bounds(ny, nx):
                        continue
                    if not self._in_palace(ny, nx, color):
                        continue
                    self._add_if_valid(moves, y, x, ny, nx, piece)

        elif kind == "R":
            moves.extend(self._line_moves(y, x, piece))

        elif kind == "C":
            moves.extend(self._cannon_moves(y, x, piece))

        elif kind == "H":
            for dy, dx in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                ny, nx = y + dy, x + dx
                if not self.in_bounds(ny, nx):
                    continue
                if abs(dy) == 2:
                    leg = (y + dy // 2, x)
                else:
                    leg = (y, x + dx // 2)
                if self.board[leg[0]][leg[1]] is not None:
                    continue
                self._add_if_valid(moves, y, x, ny, nx, piece)

        elif kind == "E":
            for dy, dx in [(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)]:
                ny, nx = y + dy, x + dx
                if not self.in_bounds(ny, nx):
                    continue
                if abs(dy) == 3:
                    leg1 = (y + dy // 3, x)
                    leg2 = (y + 2 * (dy // 3), x + dx // 2)
                else:
                    leg1 = (y, x + dx // 3)
                    leg2 = (y + dy // 2, x + 2 * (dx // 3))
                if self.board[leg1[0]][leg1[1]] is not None:
                    continue
                if self.board[leg2[0]][leg2[1]] is not None:
                    continue
                self._add_if_valid(moves, y, x, ny, nx, piece)

        elif kind == "S":
            forward = 1 if color == "r" else -1
            for dy, dx in [(forward, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not self.in_bounds(ny, nx):
                    continue
                self._add_if_valid(moves, y, x, ny, nx, piece)
            for dy, dx in [(forward, -1), (forward, 1)]:
                ny, nx = y + dy, x + dx
                if not self.in_bounds(ny, nx):
                    continue
                if self._in_palace(ny, nx, color):
                    self._add_if_valid(moves, y, x, ny, nx, piece)

        return moves

    def legal_moves_from_pos(self, pos):
        y, x = pos_to_coord(pos)
        piece = self.board[y][x]
        if piece is None:
            return []
        color = self._color_of(piece)
        pseudo = self._piece_moves(y, x, piece)
        legal = []
        for mv in pseudo:
            test = self.clone()
            test.move(mv)
            if test._is_facing_kings():
                continue
            if test.is_in_check(color):
                continue
            legal.append(mv)
        return legal

    def _line_moves(self, y, x, piece):
        moves = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            while self.in_bounds(ny, nx):
                target = self.board[ny][nx]
                if target is None:
                    moves.append((y, x, ny, nx))
                else:
                    if self._is_enemy(piece, target):
                        moves.append((y, x, ny, nx))
                    break
                ny += dy
                nx += dx
        return moves

    def _cannon_moves(self, y, x, piece):
        moves = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            screen_count = 0
            ny, nx = y + dy, x + dx
            while self.in_bounds(ny, nx):
                target = self.board[ny][nx]
                if target is None:
                    if screen_count == 1:
                        moves.append((y, x, ny, nx))
                else:
                    if screen_count == 0:
                        # Cannon cannot jump over another cannon as a screen
                        if self._kind_of(target) == "C":
                            break
                        screen_count = 1
                    elif screen_count == 1:
                        if self._is_enemy(piece, target) and self._kind_of(target) != "C":
                            moves.append((y, x, ny, nx))
                        break
                    else:
                        break
                ny += dy
                nx += dx
        return moves

    def move(self, move):
        y, x, ny, nx = move
        self.board[ny][nx] = self.board[y][x]
        self.board[y][x] = None
        self.turn = "blue" if self.turn == "red" else "red"

    def apply_move_by_pos(self, from_pos, to_pos):
        y, x = pos_to_coord(from_pos)
        ny, nx = pos_to_coord(to_pos)
        piece = self.board[y][x]
        if piece is None:
            raise ValueError("no piece at from_pos")
        if (self.turn == "red" and self._color_of(piece) != "r") or (
            self.turn == "blue" and self._color_of(piece) != "b"
        ):
            raise ValueError("not your turn")
        legal = self.get_legal_moves()
        if (y, x, ny, nx) not in legal:
            raise ValueError("illegal move")
        self.move((y, x, ny, nx))
        return (y, x, ny, nx)

    def move_to_pos(self, move):
        y, x, ny, nx = move
        return coord_to_pos(y, x), coord_to_pos(ny, nx)

    def move_tuple_from_pos(self, from_pos, to_pos):
        y, x = pos_to_coord(from_pos)
        ny, nx = pos_to_coord(to_pos)
        return (y, x, ny, nx)

    def clone(self):
        new_board = Board.__new__(Board)
        new_board.board = [row[:] for row in self.board]
        new_board.turn = self.turn
        return new_board

    def to_dict(self):
        return {
            "turn": self.turn,
            "board": self.board
        }
