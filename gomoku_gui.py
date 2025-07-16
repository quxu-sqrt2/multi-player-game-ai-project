"""
五子棋独立窗口
"""

import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any
from games.gomoku import GomokuGame, GomokuEnv
from agents import RandomBot, MinimaxBot, MCTSBot, HumanAgent
import config

COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "BROWN": (139, 69, 19),
    "LIGHT_BROWN": (205, 133, 63),
    "RED": (255, 0, 0),
    "YELLOW": (255, 255, 0),
    "GREEN": (0, 255, 0),
    "LIGHT_GRAY": (211, 211, 211),
    "DARK_GRAY": (64, 64, 64),
    "BLUE": (0, 0, 255),
    "ORANGE": (255, 165, 0),
}

class GomokuGUI:
    def __init__(self):
        pygame.init()
        self.font_path = self._get_chinese_font()
        self.font_large = pygame.font.Font(self.font_path, 28)
        self.font_medium = pygame.font.Font(self.font_path, 20)
        self.font_small = pygame.font.Font(self.font_path, 16)
        self.window_width = 900
        self.window_height = 700
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Gomoku - 独立窗口")
        self.clock = pygame.time.Clock()
        self.env = GomokuEnv(board_size=15, win_length=5)
        self.human_agent = HumanAgent(name="Human Player", player_id=1)
        self.selected_ai = "RandomBot"
        self.ai_agent = RandomBot(name="Random AI", player_id=2)
        self.current_agent = self.human_agent
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.paused = False
        self.cell_size = 30
        self.margin = 50
        self.last_update = time.time()
        self.update_interval = 1.0
        self.buttons = self._create_buttons()
        self.reset_game()

    def _get_chinese_font(self):
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
        return None

    def _create_buttons(self):
        button_width = 120
        button_height = 30
        start_x = 650
        buttons = {
            "random_ai": {
                "rect": pygame.Rect(start_x, 150, button_width, button_height),
                "text": "Random AI",
                "color": COLORS["YELLOW"],
            },
            "minimax_ai": {
                "rect": pygame.Rect(start_x, 190, button_width, button_height),
                "text": "Minimax AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            "mcts_ai": {
                "rect": pygame.Rect(start_x, 230, button_width, button_height),
                "text": "MCTS AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            "new_game": {
                "rect": pygame.Rect(start_x, 290, button_width, button_height),
                "text": "New Game",
                "color": COLORS["GREEN"],
            },
            "pause": {
                "rect": pygame.Rect(start_x, 330, button_width, button_height),
                "text": "Pause",
                "color": COLORS["ORANGE"],
            },
            "quit": {
                "rect": pygame.Rect(start_x, 370, button_width, button_height),
                "text": "Quit",
                "color": COLORS["RED"],
            },
        }
        return buttons

    def reset_game(self):
        self.env.reset()
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.current_agent = self.human_agent
        self.last_update = time.time()
        self.paused = False
        self.buttons["pause"]["text"] = "Pause"

    def _handle_button_click(self, mouse_pos):
        for button_name, button_info in self.buttons.items():
            if button_info["rect"].collidepoint(mouse_pos):
                if button_name == "new_game":
                    self.reset_game()
                elif button_name == "quit":
                    pygame.quit()
                    sys.exit()
                elif button_name == "pause":
                    self.paused = not self.paused
                    self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
                elif button_name.endswith("_ai"):
                    old_ai = f"{self.selected_ai.lower()}_ai"
                    if old_ai in self.buttons:
                        self.buttons[old_ai]["color"] = COLORS["LIGHT_GRAY"]
                    if button_name == "random_ai":
                        self.selected_ai = "RandomBot"
                        self.ai_agent = RandomBot(name="Random AI", player_id=2)
                    elif button_name == "minimax_ai":
                        self.selected_ai = "MinimaxBot"
                        self.ai_agent = MinimaxBot(name="Minimax AI", player_id=2, max_depth=3)
                    elif button_name == "mcts_ai":
                        self.selected_ai = "MCTSBot"
                        self.ai_agent = MCTSBot(name="MCTS AI", player_id=2, simulation_count=300)
                    self.buttons[button_name]["color"] = COLORS["YELLOW"]
                    self.reset_game()
                return True
        return False

    def _handle_gomoku_click(self, mouse_pos):
        x, y = mouse_pos
        board_x = x - self.margin
        board_y = y - self.margin
        if board_x < 0 or board_y < 0:
            return
        col = round(board_x / self.cell_size)
        row = round(board_y / self.cell_size)
        if 0 <= row < 15 and 0 <= col < 15:
            action = (row, col)
            if action in self.env.get_valid_actions():
                self._make_move(action)

    def _make_move(self, action):
        if self.game_over or self.paused:
            return
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.last_move = action
            if terminated or truncated:
                self.game_over = True
                self.winner = self.env.get_winner()
            else:
                self._switch_player()
        except Exception as e:
            print(f"Move execution failed: {e}")

    def _switch_player(self):
        if isinstance(self.current_agent, HumanAgent):
            self.current_agent = self.ai_agent
            self.thinking = True
        else:
            self.current_agent = self.human_agent

    def update_game(self):
        if self.game_over or self.paused:
            return
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        self.last_update = current_time
        if not isinstance(self.current_agent, HumanAgent) and self.thinking:
            try:
                observation = self.env._get_observation()
                action = self.ai_agent.get_action(observation, self.env)
                if action:
                    self._make_move(action)
                self.current_agent = self.human_agent
                self.thinking = False
            except Exception as e:
                print(f"AI thinking failed: {e}")
                self.current_agent = self.human_agent
                self.thinking = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                pass  # 五子棋不需要键盘输入
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    click_result = self._handle_button_click(mouse_pos)
                    if click_result:
                        return
                    if not self.game_over and isinstance(self.current_agent, HumanAgent) and not self.thinking and not self.paused:
                        self._handle_gomoku_click(mouse_pos)

    def draw(self):
        self.screen.fill(COLORS["WHITE"])
        self._draw_gomoku()
        self._draw_ui()
        self._draw_game_status()
        pygame.display.flip()

    def _draw_gomoku(self):
        board_size = 15
        board = self.env.game.board
        board_rect = pygame.Rect(
            self.margin - 20,
            self.margin - 20,
            board_size * self.cell_size + 40,
            board_size * self.cell_size + 40,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_BROWN"], board_rect)
        for i in range(board_size):
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (
                self.margin + i * self.cell_size,
                self.margin + (board_size - 1) * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (
                self.margin + (board_size - 1) * self.cell_size,
                self.margin + i * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["BLACK"], center, 4)
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] != 0:
                    center = (
                        self.margin + col * self.cell_size,
                        self.margin + row * self.cell_size,
                    )
                    if board[row, col] == 1:
                        color = COLORS["BLACK"]
                        border_color = COLORS["WHITE"]
                    else:
                        color = COLORS["WHITE"]
                        border_color = COLORS["BLACK"]
                    pygame.draw.circle(self.screen, color, center, 12)
                    pygame.draw.circle(self.screen, border_color, center, 12, 2)
        if (
            self.last_move
            and isinstance(self.last_move, tuple)
            and len(self.last_move) == 2
        ):
            row, col = self.last_move
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["RED"], center, 6, 3)

    def _draw_ui(self):
        for button_name, button_info in self.buttons.items():
            pygame.draw.rect(self.screen, button_info["color"], button_info["rect"])
            pygame.draw.rect(self.screen, COLORS["BLACK"], button_info["rect"], 2)
            text_surface = self.font_medium.render(
                button_info["text"], True, COLORS["BLACK"]
            )
            text_rect = text_surface.get_rect(center=button_info["rect"].center)
            self.screen.blit(text_surface, text_rect)
        title_text = self.font_medium.render("AI Selection:", True, COLORS["BLACK"])
        self.screen.blit(title_text, (self.buttons["random_ai"]["rect"].x, 125))
        instructions = [
            "Gomoku Controls:",
            "• Click to place stone",
            "• Connect 5 to win",
        ]
        start_y = 420
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS["DARK_GRAY"])
            self.screen.blit(
                text, (self.buttons["new_game"]["rect"].x, start_y + i * 20)
            )

    def _draw_game_status(self):
        status_x = 20
        status_y = self.window_height - 100
        if self.paused:
            status_text = "Game Paused..."
            color = COLORS["ORANGE"]
        elif self.game_over:
            if self.winner == 1:
                status_text = "Congratulations! You Win!"
                color = COLORS["GREEN"]
            elif self.winner == 2:
                status_text = "AI Wins! Try Again!"
                color = COLORS["RED"]
            else:
                status_text = "Draw!"
                color = COLORS["ORANGE"]
        else:
            if isinstance(self.current_agent, HumanAgent):
                status_text = "Your Turn - Click to Place Stone"
                color = COLORS["BLUE"]
            else:
                if self.thinking:
                    status_text = f"{self.ai_agent.name} is Thinking..."
                    color = COLORS["ORANGE"]
                else:
                    status_text = f"{self.ai_agent.name}'s Turn"
                    color = COLORS["RED"]
        text_surface = self.font_large.render(status_text, True, color)
        self.screen.blit(text_surface, (status_x, status_y))
        info_y = status_y + 40
        player_info = f"Black: Human Player  White: {self.ai_agent.name if self.ai_agent else 'AI'}"
        info_surface = self.font_small.render(player_info, True, COLORS["DARK_GRAY"])
        self.screen.blit(info_surface, (status_x, info_y))

    def run(self):
        running = True
        while running:
            self.handle_events()
            self.update_game()
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()

def main():
    print("Starting Gomoku Standalone GUI...")
    try:
        game = GomokuGUI()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 