import pygame
import chess
import math
import os
import random
import threading
import time


class ChessProblem:
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    PST = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.KNIGHT: [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30
        ]
    }

    def __init__(self):
        self.board = chess.Board()

    def is_game_over(self):
        return self.board.is_game_over()

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def make_move(self, move):
        self.board.push(move)

    def undo_move(self):
        self.board.pop()

    def evaluate(self):
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -100000
            else:
                return 100000

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        evaluation = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]

                if piece.color == chess.WHITE:
                    pst_bonus = self.PST[piece.piece_type][square]
                    evaluation += (value + pst_bonus)
                else:
                    mirrored_square = chess.square_mirror(square)
                    pst_bonus = self.PST[piece.piece_type][mirrored_square]
                    evaluation -= (value + pst_bonus)

        return evaluation


class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.heuristic_value = 0.0
        self.untried_moves = self._order_moves(list(self.board.legal_moves))

    def _order_moves(self, moves):
        captures = []
        checks = []
        others = []
        for m in moves:
            if self.board.is_capture(m):
                captures.append(m)
            elif self.board.gives_check(m):
                checks.append(m)
            else:
                others.append(m)
        random.shuffle(captures)
        random.shuffle(checks)
        random.shuffle(others)
        return captures + checks + others

    def select_child(self):
        c = 1.41
        return max(self.children, key=lambda child:
        child.wins / child.visits + c * math.sqrt(math.log(self.visits) / child.visits)
        + child.heuristic_value / (child.visits + 1))

    def add_child(self, move, heuristic_value=0.0):
        board_copy = self.board.copy()
        board_copy.push(move)
        child = MCTSNode(board_copy, parent=self, move=move)
        child.heuristic_value = heuristic_value
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result


class Solver:
    def __init__(self, search_depth=4, algorithm="minimax"):
        self.search_depth = search_depth
        self.algorithm = algorithm
        self.mcts_iterations = 1000

    def minimax(self, problem, depth, alpha, beta, is_max):
        if depth == 0 or problem.is_game_over():
            return problem.evaluate()

        if is_max:
            max_eval = -math.inf
            for move in problem.get_legal_moves():
                problem.make_move(move)
                eval_score = self.minimax(problem, depth - 1, alpha, beta, False)
                problem.undo_move()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in problem.get_legal_moves():
                problem.make_move(move)
                eval_score = self.minimax(problem, depth - 1, alpha, beta, True)
                problem.undo_move()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move_minimax(self, problem):
        best_move = None
        best_score = -math.inf if problem.board.turn == chess.WHITE else math.inf
        alpha = -math.inf
        beta = math.inf

        for move in problem.get_legal_moves():
            problem.make_move(move)
            score = self.minimax(problem, self.search_depth - 1, alpha, beta,
                                 problem.board.turn == chess.WHITE)
            problem.undo_move()

            if problem.board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)

        return best_move

    def _eval_to_score(self, board, player_who_just_moved):
        problem = ChessProblem()
        problem.board = board
        raw = problem.evaluate()

        if player_who_just_moved == chess.BLACK:
            raw = -raw

        return 1.0 / (1.0 + math.exp(-raw / 400.0))

    def mcts_simulate(self, board):
        sim_board = board.copy()
        moves_played = 0
        max_rollout_depth = 10

        while not sim_board.is_game_over() and moves_played < max_rollout_depth:
            moves = list(sim_board.legal_moves)
            if not moves:
                break

            captures = [m for m in moves if sim_board.is_capture(m)]
            if captures:
                move = random.choice(captures)
            else:
                move = random.choice(moves)

            sim_board.push(move)
            moves_played += 1

        if sim_board.is_checkmate():
            return 1.0 if sim_board.turn == board.turn else 0.0
        if sim_board.is_game_over():
            return 0.5

        return self._eval_to_score(sim_board, not board.turn)

    def get_best_move_mcts(self, problem):
        root = MCTSNode(problem.board)

        for _ in range(self.mcts_iterations):
            node = root
            board = problem.board.copy()

            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                board.push(node.move)

            if node.untried_moves != []:
                move = node.untried_moves[0]
                board.push(move)
                heuristic = self._eval_to_score(board, not board.turn)
                node = node.add_child(move, heuristic_value=heuristic)

            result = self.mcts_simulate(board)

            while node is not None:
                node.update(result)
                result = 1 - result
                node = node.parent

        return max(root.children, key=lambda c: c.visits).move if root.children else None

    def get_best_move(self, problem):
        if self.algorithm == "mcts":
            return self.get_best_move_mcts(problem)
        else:
            return self.get_best_move_minimax(problem)


class ChessGUI:
    BOARD_SIZE = 512
    DIMENSION = 8
    SQ_SIZE = BOARD_SIZE // DIMENSION
    TOP_PANEL = 50
    BOTTOM_PANEL = 50
    WIDTH = BOARD_SIZE
    HEIGHT = BOARD_SIZE + TOP_PANEL + BOTTOM_PANEL
    MAX_FPS = 30

    COLOR_THEMES = {
        "Classic": {
            "light": (240, 217, 181),
            "dark": (181, 136, 99),
            "highlight": (186, 202, 68),
            "select": (246, 246, 130),
        },
        "Blue": {
            "light": (222, 235, 250),
            "dark": (100, 140, 190),
            "highlight": (80, 180, 220),
            "select": (160, 210, 250),
        },
        "Green": {
            "light": (238, 238, 210),
            "dark": (118, 150, 86),
            "highlight": (186, 202, 68),
            "select": (214, 230, 130),
        },
        "Purple": {
            "light": (230, 220, 240),
            "dark": (140, 100, 170),
            "highlight": (180, 130, 210),
            "select": (210, 180, 240),
        },
        "Coral": {
            "light": (250, 228, 218),
            "dark": (200, 120, 100),
            "highlight": (240, 160, 100),
            "select": (250, 200, 160),
        },
    }

    COLOR_LIGHT = (240, 217, 181)
    COLOR_DARK = (181, 136, 99)
    COLOR_HIGHLIGHT = (186, 202, 68)
    COLOR_SELECT = (246, 246, 130)

    EVAL_BAR_GREEN = (118, 180, 58)
    EVAL_BAR_DARK = (58, 58, 58)
    PANEL_BG = (30, 30, 30)
    TIMER_COLOR = (220, 220, 220)

    MENU_BG_COLOR = (45, 35, 30)
    BUTTON_COLOR = (140, 95, 65)
    BUTTON_HOVER = (181, 136, 99)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Chess - Main Menu")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.msg_font = pygame.font.SysFont("Arial", 48, bold=True)
        self.btn_font = pygame.font.SysFont("Arial", 32, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.timer_font = pygame.font.SysFont("Arial", 18, bold=True)

        self.images = {}
        self.load_images()

        self.problem = ChessProblem()
        self.solver_white = Solver(search_depth=3)
        self.solver_black = Solver(search_depth=3)
        self.selected_square = None

        self.white_time = 0.0
        self.black_time = 0.0
        self.last_tick = None
        self.current_eval = 0

        self.state = "MODE_SELECT"
        self.game_mode = None

        btn_width, btn_height = 280, 60
        center_x = self.WIDTH // 2 - btn_width // 2

        self.player_vs_player_btn = pygame.Rect(center_x, 100, btn_width, btn_height)
        self.player_vs_ai_btn = pygame.Rect(center_x, 180, btn_width, btn_height)
        self.ai_vs_ai_btn = pygame.Rect(center_x, 260, btn_width, btn_height)

        self.minimax_btn = pygame.Rect(center_x, 120, btn_width, btn_height)
        self.mcts_btn = pygame.Rect(center_x, 200, btn_width, btn_height)

        small_btn_width = 120
        small_center = self.WIDTH // 2 - small_btn_width // 2
        self.easy_btn = pygame.Rect(small_center, 300, small_btn_width, 50)
        self.med_btn = pygame.Rect(small_center, 360, small_btn_width, 50)
        self.hard_btn = pygame.Rect(small_center, 420, small_btn_width, 50)

        self.selected_algorithm = None
        self.selected_algorithm_white = None
        self.selected_algorithm_black = None

        self.ai_thread = None
        self.ai_move_ready = False
        self.best_ai_move = None

        theme_names = list(self.COLOR_THEMES.keys())
        theme_btn_width, theme_btn_height = 280, 50
        theme_center_x = self.WIDTH // 2 - theme_btn_width // 2
        self.color_buttons = {}
        for i, name in enumerate(theme_names):
            self.color_buttons[name] = pygame.Rect(
                theme_center_x, 100 + i * 70, theme_btn_width, theme_btn_height
            )

        self.ava_step_requested = False
        step_btn_w, step_btn_h = 140, 36
        btn_y = self.TOP_PANEL + self.BOARD_SIZE + (self.BOTTOM_PANEL - step_btn_h) // 2
        self.step_back_btn = pygame.Rect(self.WIDTH // 2 - step_btn_w - 10, btn_y, step_btn_w, step_btn_h)
        self.step_forward_btn = pygame.Rect(self.WIDTH // 2 + 10, btn_y, step_btn_w, step_btn_h)

        self.pending_caption = ""

    def load_images(self):
        pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
        for piece in pieces:
            try:
                image_path = os.path.join("images", f"{piece}.png")
                image = pygame.image.load(image_path)
                image = image.convert_alpha()
                self.images[piece] = pygame.transform.smoothscale(image, (self.SQ_SIZE, self.SQ_SIZE))
            except FileNotFoundError:
                print(f"Warning: Image {image_path} not found.")

    def draw_mode_select(self):
        self.screen.fill(self.MENU_BG_COLOR)

        title = self.msg_font.render("Select Game Mode", True, self.COLOR_LIGHT)
        self.screen.blit(title, title.get_rect(center=(self.WIDTH // 2, 40)))

        mouse_pos = pygame.mouse.get_pos()

        buttons = [
            (self.player_vs_player_btn, "Player vs Player"),
            (self.player_vs_ai_btn, "Player vs AI"),
            (self.ai_vs_ai_btn, "AI vs AI")
        ]

        for btn, text in buttons:
            color = self.BUTTON_HOVER if btn.collidepoint(mouse_pos) else self.BUTTON_COLOR

            pygame.draw.rect(self.screen, color, btn, border_radius=15)
            pygame.draw.rect(self.screen, self.COLOR_LIGHT, btn, width=2, border_radius=15)

            txt_surface = self.small_font.render(text, True, self.COLOR_LIGHT)
            self.screen.blit(txt_surface, txt_surface.get_rect(center=btn.center))

    def draw_ai_select_menu(self, player_label=""):
        self.screen.fill(self.MENU_BG_COLOR)

        title_text = f"Select Algorithm {player_label}"
        title = self.msg_font.render(title_text, True, self.COLOR_LIGHT)
        self.screen.blit(title, title.get_rect(center=(self.WIDTH // 2, 50)))

        mouse_pos = pygame.mouse.get_pos()

        algo_buttons = [
            (self.minimax_btn, "Minimax"),
            (self.mcts_btn, "MCTS")
        ]

        for btn, text in algo_buttons:
            if self.selected_algorithm == text:
                color = self.COLOR_HIGHLIGHT
            elif btn.collidepoint(mouse_pos):
                color = self.BUTTON_HOVER
            else:
                color = self.BUTTON_COLOR

            pygame.draw.rect(self.screen, color, btn, border_radius=15)
            pygame.draw.rect(self.screen, self.COLOR_LIGHT, btn, width=2, border_radius=15)

            txt_surface = self.btn_font.render(text, True, self.COLOR_LIGHT)
            self.screen.blit(txt_surface, txt_surface.get_rect(center=btn.center))

        if self.selected_algorithm:
            diff_title = self.small_font.render("Select Difficulty", True, self.COLOR_LIGHT)
            self.screen.blit(diff_title, diff_title.get_rect(center=(self.WIDTH // 2, 270)))

            diff_buttons = [
                (self.easy_btn, "Easy"),
                (self.med_btn, "Medium"),
                (self.hard_btn, "Hard")
            ]

            for btn, text in diff_buttons:
                color = self.BUTTON_HOVER if btn.collidepoint(mouse_pos) else self.BUTTON_COLOR

                pygame.draw.rect(self.screen, color, btn, border_radius=10)
                pygame.draw.rect(self.screen, self.COLOR_LIGHT, btn, width=2, border_radius=10)

                txt_surface = self.small_font.render(text, True, self.COLOR_LIGHT)
                self.screen.blit(txt_surface, txt_surface.get_rect(center=btn.center))

    def draw_color_select(self):
        self.screen.fill(self.MENU_BG_COLOR)

        title = self.msg_font.render("Board Colour", True, self.COLOR_LIGHT)
        self.screen.blit(title, title.get_rect(center=(self.WIDTH // 2, 40)))

        mouse_pos = pygame.mouse.get_pos()

        for name, btn in self.color_buttons.items():
            theme = self.COLOR_THEMES[name]
            hover = btn.collidepoint(mouse_pos)

            preview_left = pygame.Rect(btn.x, btn.y, btn.width // 2, btn.height)
            preview_right = pygame.Rect(btn.x + btn.width // 2, btn.y, btn.width // 2, btn.height)
            pygame.draw.rect(self.screen, theme["light"], preview_left, border_radius=15)
            pygame.draw.rect(self.screen, theme["dark"], preview_right, border_radius=15)

            if hover:
                pygame.draw.rect(self.screen, (255, 255, 255), btn, width=3, border_radius=15)
            else:
                pygame.draw.rect(self.screen, self.COLOR_LIGHT, btn, width=2, border_radius=15)

            txt_surface = self.small_font.render(name, True, (255, 255, 255))
            shadow = self.small_font.render(name, True, (0, 0, 0))
            center = btn.center
            self.screen.blit(shadow, shadow.get_rect(center=(center[0] + 1, center[1] + 1)))
            self.screen.blit(txt_surface, txt_surface.get_rect(center=center))

    def apply_theme(self, theme_name):
        theme = self.COLOR_THEMES[theme_name]
        self.COLOR_LIGHT = theme["light"]
        self.COLOR_DARK = theme["dark"]
        self.COLOR_HIGHLIGHT = theme["highlight"]
        self.COLOR_SELECT = theme["select"]

    def draw_board(self):
        colors = [self.COLOR_LIGHT, self.COLOR_DARK]
        for row in range(self.DIMENSION):
            for col in range(self.DIMENSION):
                color = colors[((row + col) % 2)]
                rect = pygame.Rect(col * self.SQ_SIZE, self.TOP_PANEL + row * self.SQ_SIZE, self.SQ_SIZE, self.SQ_SIZE)
                pygame.draw.rect(self.screen, color, rect)

    def draw_pieces(self):
        piece_mapping = {
            chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
            chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"
        }
        board = self.problem.board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                prefix = "w" if piece.color == chess.WHITE else "b"
                piece_str = prefix + piece_mapping[piece.piece_type]

                if piece_str in self.images:
                    rect = pygame.Rect(col * self.SQ_SIZE, self.TOP_PANEL + row * self.SQ_SIZE, self.SQ_SIZE, self.SQ_SIZE)
                    self.screen.blit(self.images[piece_str], rect)

    def draw_highlights(self):
        board = self.problem.board
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = 7 - chess.square_rank(self.selected_square)
            rect = pygame.Rect(col * self.SQ_SIZE, self.TOP_PANEL + row * self.SQ_SIZE, self.SQ_SIZE, self.SQ_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 5)

            for move in board.legal_moves:
                if move.from_square == self.selected_square:
                    dest_col = chess.square_file(move.to_square)
                    dest_row = 7 - chess.square_rank(move.to_square)
                    dest_rect = pygame.Rect(dest_col * self.SQ_SIZE, self.TOP_PANEL + dest_row * self.SQ_SIZE, self.SQ_SIZE,
                                            self.SQ_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, dest_rect, 5)

    def draw_game_over(self):
        board = self.problem.board
        if board.is_game_over():
            msg = "Game Over!"
            if board.is_checkmate():
                msg = "Checkmate! " + ("Black Wins!" if board.turn == chess.WHITE else "White Wins!")
            elif board.is_stalemate():
                msg = "Stalemate!"
            elif board.is_fivefold_repetition():
                msg = "Five Fold Repetition!"

            overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))

            text_surface = self.msg_font.render(msg, True, (255, 50, 50))
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.TOP_PANEL + self.BOARD_SIZE // 2))
            self.screen.blit(text_surface, text_rect)

    def update_timers(self):
        now = time.time()
        if self.last_tick is not None and not self.problem.is_game_over():
            dt = now - self.last_tick
            if self.problem.board.turn == chess.WHITE:
                self.white_time += dt
            else:
                self.black_time += dt
        self.last_tick = now

    def format_time(self, seconds):
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    def draw_top_panel(self):
        pygame.draw.rect(self.screen, self.PANEL_BG, (0, 0, self.WIDTH, self.TOP_PANEL))

        self.current_eval = self.problem.evaluate()

        eval_clamped = max(-2000, min(2000, self.current_eval))
        white_ratio = (eval_clamped + 2000) / 4000.0
        green_width = int(self.WIDTH * white_ratio)

        bar_y = 2
        bar_h = 18
        pygame.draw.rect(self.screen, self.EVAL_BAR_DARK, (0, bar_y, self.WIDTH, bar_h))
        pygame.draw.rect(self.screen, self.EVAL_BAR_GREEN, (0, bar_y, green_width, bar_h))

        eval_display = self.current_eval / 100.0
        sign = "+" if eval_display >= 0 else ""
        eval_text = self.timer_font.render(f"{sign}{eval_display:.1f}", True, (255, 255, 255))
        eval_rect = eval_text.get_rect(center=(self.WIDTH // 2, bar_y + bar_h // 2))
        self.screen.blit(eval_text, eval_rect)

        timer_y = bar_y + bar_h + 4
        white_str = f"White: {self.format_time(self.white_time)}"
        black_str = f"Black: {self.format_time(self.black_time)}"

        active_color = (180, 255, 180)
        inactive_color = self.TIMER_COLOR

        w_color = active_color if self.problem.board.turn == chess.WHITE else inactive_color
        b_color = active_color if self.problem.board.turn == chess.BLACK else inactive_color

        w_surface = self.timer_font.render(white_str, True, w_color)
        b_surface = self.timer_font.render(black_str, True, b_color)
        self.screen.blit(w_surface, (10, timer_y))
        self.screen.blit(b_surface, (self.WIDTH - b_surface.get_width() - 10, timer_y))

    def draw_bottom_panel(self):
        panel_y = self.TOP_PANEL + self.BOARD_SIZE
        pygame.draw.rect(self.screen, self.PANEL_BG, (0, panel_y, self.WIDTH, self.BOTTOM_PANEL))

        if self.game_mode != "AVA":
            return

        mouse_pos = pygame.mouse.get_pos()
        computing = self.ai_thread is not None and not self.ai_move_ready

        for btn, label in [(self.step_back_btn, "< Back"), (self.step_forward_btn, "Forward >")]:
            if computing and label == "Forward >":
                color = (80, 80, 80)
            elif btn.collidepoint(mouse_pos):
                color = self.BUTTON_HOVER
            else:
                color = self.BUTTON_COLOR

            pygame.draw.rect(self.screen, color, btn, border_radius=10)
            pygame.draw.rect(self.screen, self.COLOR_LIGHT, btn, width=2, border_radius=10)

            txt = self.timer_font.render(label, True, self.COLOR_LIGHT)
            self.screen.blit(txt, txt.get_rect(center=btn.center))

    def handle_player_move(self, x, y):
        y -= self.TOP_PANEL
        if y < 0:
            return
        col = x // self.SQ_SIZE
        row = y // self.SQ_SIZE
        clicked_square = chess.square(col, 7 - row)
        board = self.problem.board

        if self.selected_square is None:
            piece = board.piece_at(clicked_square)
            if piece and piece.color == board.turn:
                self.selected_square = clicked_square
        else:
            move = chess.Move(self.selected_square, clicked_square)

            piece_at_selected = board.piece_at(self.selected_square)
            if piece_at_selected and piece_at_selected.piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) == 7 or chess.square_rank(clicked_square) == 0:
                    move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)

            if move in board.legal_moves:
                self.problem.make_move(move)
                self.selected_square = None
            else:
                piece = board.piece_at(clicked_square)
                if piece and piece.color == board.turn:
                    self.selected_square = clicked_square
                else:
                    self.selected_square = None

    def configure_solver(self, solver, algorithm, difficulty):
        if algorithm == "Minimax":
            solver.algorithm = "minimax"
            if difficulty == "Easy":
                solver.search_depth = 2
            elif difficulty == "Medium":
                solver.search_depth = 3
            else:
                solver.search_depth = 4
        else:
            solver.algorithm = "mcts"
            if difficulty == "Easy":
                solver.mcts_iterations = 600
            elif difficulty == "Medium":
                solver.mcts_iterations = 1500
            else:
                solver.mcts_iterations = 3000

    def run(self):
        if not hasattr(self, 'ai_thread'):
            self.ai_thread = None
            self.ai_move_ready = False
            self.best_ai_move = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    if self.state == "MODE_SELECT":
                        if self.player_vs_player_btn.collidepoint(pos):
                            self.game_mode = "PVP"
                            self.state = "COLOR_SELECT"
                            pygame.display.set_caption("Chess - Player vs Player")

                        elif self.player_vs_ai_btn.collidepoint(pos):
                            self.game_mode = "PVA"
                            self.state = "AI_SELECT"
                            self.selected_algorithm = None

                        elif self.ai_vs_ai_btn.collidepoint(pos):
                            self.game_mode = "AVA"
                            self.state = "AI_SELECT_WHITE"
                            self.selected_algorithm = None

                    elif self.state == "AI_SELECT":
                        if self.minimax_btn.collidepoint(pos):
                            self.selected_algorithm = "Minimax"
                        elif self.mcts_btn.collidepoint(pos):
                            self.selected_algorithm = "MCTS"

                        if self.selected_algorithm:
                            if self.easy_btn.collidepoint(pos):
                                self.configure_solver(self.solver_black, self.selected_algorithm, "Easy")
                                self.pending_caption = f"Chess - Player vs {self.selected_algorithm} (Easy)"
                                self.state = "COLOR_SELECT"

                            elif self.med_btn.collidepoint(pos):
                                self.configure_solver(self.solver_black, self.selected_algorithm, "Medium")
                                self.pending_caption = f"Chess - Player vs {self.selected_algorithm} (Medium)"
                                self.state = "COLOR_SELECT"

                            elif self.hard_btn.collidepoint(pos):
                                self.configure_solver(self.solver_black, self.selected_algorithm, "Hard")
                                self.pending_caption = f"Chess - Player vs {self.selected_algorithm} (Hard)"
                                self.state = "COLOR_SELECT"

                    elif self.state == "AI_SELECT_WHITE":
                        if self.minimax_btn.collidepoint(pos):
                            self.selected_algorithm = "Minimax"
                        elif self.mcts_btn.collidepoint(pos):
                            self.selected_algorithm = "MCTS"

                        if self.selected_algorithm:
                            if self.easy_btn.collidepoint(pos):
                                self.configure_solver(self.solver_white, self.selected_algorithm, "Easy")
                                self.selected_algorithm_white = self.selected_algorithm + " (Easy)"
                                self.state = "AI_SELECT_BLACK"
                                self.selected_algorithm = None

                            elif self.med_btn.collidepoint(pos):
                                self.configure_solver(self.solver_white, self.selected_algorithm, "Medium")
                                self.selected_algorithm_white = self.selected_algorithm + " (Medium)"
                                self.state = "AI_SELECT_BLACK"
                                self.selected_algorithm = None

                            elif self.hard_btn.collidepoint(pos):
                                self.configure_solver(self.solver_white, self.selected_algorithm, "Hard")
                                self.selected_algorithm_white = self.selected_algorithm + " (Hard)"
                                self.state = "AI_SELECT_BLACK"
                                self.selected_algorithm = None

                    elif self.state == "AI_SELECT_BLACK":
                        if self.minimax_btn.collidepoint(pos):
                            self.selected_algorithm = "Minimax"
                        elif self.mcts_btn.collidepoint(pos):
                            self.selected_algorithm = "MCTS"

                        if self.selected_algorithm:
                            if self.easy_btn.collidepoint(pos):
                                self.configure_solver(self.solver_black, self.selected_algorithm, "Easy")
                                self.selected_algorithm_black = self.selected_algorithm + " (Easy)"
                                self.pending_caption = f"Chess - {self.selected_algorithm_white} vs {self.selected_algorithm_black}"
                                self.state = "COLOR_SELECT"

                            elif self.med_btn.collidepoint(pos):
                                self.configure_solver(self.solver_black, self.selected_algorithm, "Medium")
                                self.selected_algorithm_black = self.selected_algorithm + " (Medium)"
                                self.pending_caption = f"Chess - {self.selected_algorithm_white} vs {self.selected_algorithm_black}"
                                self.state = "COLOR_SELECT"

                            elif self.hard_btn.collidepoint(pos):
                                self.configure_solver(self.solver_black, self.selected_algorithm, "Hard")
                                self.selected_algorithm_black = self.selected_algorithm + " (Hard)"
                                self.pending_caption = f"Chess - {self.selected_algorithm_white} vs {self.selected_algorithm_black}"
                                self.state = "COLOR_SELECT"

                    elif self.state == "COLOR_SELECT":
                        for theme_name, btn in self.color_buttons.items():
                            if btn.collidepoint(pos):
                                self.apply_theme(theme_name)
                                self.state = "PLAYING"
                                self.last_tick = time.time()
                                if hasattr(self, 'pending_caption') and self.pending_caption:
                                    pygame.display.set_caption(self.pending_caption)
                                break

                    elif self.state == "PLAYING":
                        if self.game_mode == "PVP":
                            if not self.problem.is_game_over():
                                x, y = pygame.mouse.get_pos()
                                self.handle_player_move(x, y)
                        elif self.game_mode == "PVA":
                            if self.problem.board.turn == chess.WHITE and not self.problem.is_game_over():
                                x, y = pygame.mouse.get_pos()
                                self.handle_player_move(x, y)
                        elif self.game_mode == "AVA":
                            if self.step_forward_btn.collidepoint(pos):
                                if not self.problem.is_game_over() and self.ai_thread is None:
                                    self.ava_step_requested = True
                            elif self.step_back_btn.collidepoint(pos):
                                if len(self.problem.board.move_stack) > 0:
                                    self.ai_thread = None
                                    self.ai_move_ready = False
                                    self.ava_step_requested = False
                                    self.problem.undo_move()

            if self.state == "MODE_SELECT":
                self.draw_mode_select()

            elif self.state == "AI_SELECT":
                self.draw_ai_select_menu()

            elif self.state == "AI_SELECT_WHITE":
                self.draw_ai_select_menu("(White)")

            elif self.state == "AI_SELECT_BLACK":
                self.draw_ai_select_menu("(Black)")

            elif self.state == "COLOR_SELECT":
                self.draw_color_select()

            elif self.state == "PLAYING":
                if self.game_mode == "PVA":
                    if self.problem.board.turn == chess.BLACK and not self.problem.is_game_over():
                        if self.ai_thread is None:
                            self.ai_move_ready = False

                            def calc_move():
                                problem_copy = ChessProblem()
                                problem_copy.board = self.problem.board.copy()
                                self.best_ai_move = self.solver_black.get_best_move(problem_copy)
                                self.ai_move_ready = True

                            self.ai_thread = threading.Thread(target=calc_move)
                            self.ai_thread.start()

                        elif self.ai_move_ready:
                            if self.best_ai_move:
                                self.problem.make_move(self.best_ai_move)
                            self.ai_thread = None

                elif self.game_mode == "AVA":
                    if not self.problem.is_game_over() and self.ava_step_requested:
                        if self.ai_thread is None:
                            self.ai_move_ready = False

                            def calc_move_ava():
                                problem_copy = ChessProblem()
                                problem_copy.board = self.problem.board.copy()
                                if self.problem.board.turn == chess.WHITE:
                                    self.best_ai_move = self.solver_white.get_best_move(problem_copy)
                                else:
                                    self.best_ai_move = self.solver_black.get_best_move(problem_copy)
                                self.ai_move_ready = True

                            self.ai_thread = threading.Thread(target=calc_move_ava)
                            self.ai_thread.start()

                        elif self.ai_move_ready:
                            if self.best_ai_move:
                                self.problem.make_move(self.best_ai_move)
                            self.ai_thread = None
                            self.ava_step_requested = False

                self.update_timers()
                self.draw_top_panel()
                self.draw_board()
                if self.game_mode != "AVA":
                    self.draw_highlights()
                self.draw_pieces()
                self.draw_game_over()
                self.draw_bottom_panel()

            self.clock.tick(self.MAX_FPS)
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    game = ChessGUI()
    game.run()
