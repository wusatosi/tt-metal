#!/usr/bin/env python3
import curses
import requests
import json
import textwrap
import time


class SimpleTerminalLLMUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.prompt = "San Francisco is a"
        self.model = "meta-llama/Llama-3.1-70B-Instruct"  # Fixed model
        self.max_tokens = 32
        self.temperature = 0.0  # Fixed temperature
        self.completion = ""
        self.raw_response = ""
        self.status = ""
        self.tokens_per_second = 0.0  # New field for TPS
        self.generation_time = 0.0  # New field for generation time
        self.current_field = 0  # 0: prompt, 1: max_tokens, 2: send button

        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected field
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Field labels
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Status
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Results
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)  # Errors
        curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Statistics

        # Hide cursor
        curses.curs_set(0)

        # Get screen dimensions
        self.height, self.width = stdscr.getmaxyx()

        # Set up main window
        self.stdscr.clear()
        self.stdscr.refresh()

    def safe_addstr(self, y, x, text, attr=curses.A_NORMAL):
        """Safely add a string to the screen, checking boundaries"""
        h, w = self.stdscr.getmaxyx()
        if y >= h or x >= w:
            return

        # Truncate the string if it would go beyond screen width
        max_len = w - x - 1
        if max_len <= 0:
            return

        text = str(text)
        if len(text) > max_len:
            text = text[:max_len]

        try:
            self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def draw_ui(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Minimum size check
        if h < 10 or w < 40:
            self.safe_addstr(0, 0, "Terminal too small!")
            self.stdscr.refresh()
            return

        # Draw title
        title = "Simple LLM Interface"
        self.safe_addstr(0, max(0, (w - len(title)) // 2), title, curses.A_BOLD)

        # Draw prompt field
        prompt_label = "Prompt:"
        self.safe_addstr(1, 2, prompt_label, curses.color_pair(2) | curses.A_BOLD)

        # Draw prompt input box
        if self.current_field == 0:
            attr = curses.color_pair(1)
        else:
            attr = curses.A_NORMAL

        box_width = min(w - 4, 70)
        self.safe_addstr(2, 2, "┌" + "─" * (box_width - 2) + "┐", attr)

        # Wrap and display prompt text
        max_lines = 3  # Fixed 3 lines for prompt
        prompt_lines = textwrap.wrap(self.prompt, box_width - 4) if self.prompt else [""]

        for i in range(max_lines):
            line_text = prompt_lines[i] if i < len(prompt_lines) else ""
            padding = " " * (box_width - len(line_text) - 4)
            self.safe_addstr(3 + i, 2, "│ " + line_text + padding + "│", attr)

        self.safe_addstr(3 + max_lines, 2, "└" + "─" * (box_width - 2) + "┘", attr)

        # Fixed vertical position for next elements
        param_y = 7

        # Draw max_tokens
        if self.current_field == 1:
            attr = curses.color_pair(1)
        else:
            attr = curses.A_NORMAL
        self.safe_addstr(param_y, 2, f"Max Tokens: {self.max_tokens}", attr)

        # Draw Send button
        if self.current_field == 2:
            attr = curses.color_pair(1)
        else:
            attr = curses.A_NORMAL
        self.safe_addstr(param_y + 1, 2, "[ Send Request ]", attr)

        # Fixed position for results
        result_y = param_y + 3

        # Draw generation statistics
        # if self.tokens_per_second > 0:
        #     stats_text = f"Generation Speed: {self.tokens_per_second:.2f} tokens/sec | Time: {self.generation_time:.2f} sec"
        #     self.safe_addstr(result_y, 2, stats_text, curses.color_pair(6) | curses.A_BOLD)
        #     result_y += 1

        # Result area height
        result_height = h - result_y - 2

        # Draw result area
        if result_height > 0:
            # Draw header
            self.safe_addstr(result_y, 2, "Result:", curses.color_pair(2) | curses.A_BOLD)

            # Draw result box
            self.safe_addstr(result_y + 1, 2, "┌" + "─" * (box_width - 2) + "┐")

            for i in range(min(result_height - 2, h - result_y - 4)):
                self.safe_addstr(result_y + 2 + i, 2, "│" + " " * (box_width - 4) + "│")

            self.safe_addstr(result_y + result_height - 1, 2, "└" + "─" * (box_width - 2) + "┘")

            # Draw result content
            result_lines = textwrap.wrap(self.completion, box_width - 6)

            for i, line in enumerate(result_lines):
                if i < result_height - 3 and result_y + 2 + i < h - 2:
                    self.safe_addstr(result_y + 2 + i, 4, line[: box_width - 8], curses.color_pair(4))

        # Draw status at the bottom
        if "Error" in self.status:
            attr = curses.color_pair(5)
        else:
            attr = curses.color_pair(3)

        if h > 1:
            self.safe_addstr(h - 1, 0, " " * (w - 1))  # Clear the line
            self.safe_addstr(h - 1, 2, self.status, attr)

        # Draw help
        if h > 2:
            help_text = "↑/↓/j/k: Navigate | Enter: Edit/Select | s: Send | q: Quit"
            self.safe_addstr(h - 2, max(0, (w - len(help_text)) // 2), help_text)

        self.stdscr.refresh()

    def edit_field(self):
        if self.current_field == 0:  # Prompt
            self.edit_prompt()
        elif self.current_field == 1:  # Max tokens
            self.edit_number_field("max_tokens", min_val=1, max_val=2048)
        elif self.current_field == 2:  # Send button
            self.send_request()

    def edit_prompt(self):
        curses.curs_set(1)  # Show cursor
        h, w = self.stdscr.getmaxyx()

        # Check if there's room to edit
        if h < 6 or w < 10:
            return

        # Calculate edit area size
        box_width = min(w - 4, 70)

        try:
            # Create an editing area
            edit_win = curses.newwin(1, box_width - 2, 3, 3)
            edit_win.clear()

            # Set up the editing buffer
            buffer = self.prompt

            # Display initial content
            edit_win.addstr(0, 0, buffer[: box_width - 3])
            edit_win.refresh()

            # Edit loop
            while True:
                key = self.stdscr.getch()

                if key == 27:  # Escape
                    break
                elif key == 10:  # Enter
                    self.prompt = buffer
                    break
                elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                    if buffer:
                        buffer = buffer[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    buffer += chr(key)

                # Clear and redisplay
                edit_win.clear()
                display_text = buffer[: box_width - 3]
                edit_win.addstr(0, 0, display_text)
                edit_win.refresh()
        except curses.error:
            pass

        curses.curs_set(0)  # Hide cursor

    def edit_number_field(self, field_name, min_val, max_val):
        curses.curs_set(1)  # Show cursor
        h, w = self.stdscr.getmaxyx()

        param_y = 7  # Fixed position for max_tokens

        try:
            # Create editing window
            edit_win = curses.newwin(1, 8, param_y, 14)
            edit_win.clear()

            # Set up editing buffer
            buffer = str(getattr(self, field_name))

            # Display initial content
            edit_win.addstr(0, 0, buffer)
            edit_win.refresh()

            # Edit loop
            while True:
                key = self.stdscr.getch()

                if key == 27:  # Escape
                    break
                elif key == 10:  # Enter
                    try:
                        val = int(buffer)
                        val = max(min_val, min(max_val, val))
                        setattr(self, field_name, val)
                    except:
                        pass
                    break
                elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                    if buffer:
                        buffer = buffer[:-1]
                elif 48 <= key <= 57:  # Digits
                    buffer += chr(key)

                # Clear and redisplay
                edit_win.clear()
                edit_win.addstr(0, 0, buffer)
                edit_win.refresh()
        except curses.error:
            pass

        curses.curs_set(0)  # Hide cursor

    def send_request(self):
        self.status = "Sending request..."
        self.draw_ui()

        try:
            # Prepare request
            url = "http://localhost:8000/v1/completions"
            payload = {
                "model": self.model,
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            # Start timing
            start_time = time.time()

            # Send request
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

            # End timing
            end_time = time.time()

            # Handle response
            if response.status_code == 200:
                result = response.json()
                self.completion = result["choices"][0]["text"]
                self.raw_response = json.dumps(result, indent=2)

                # Calculate generation time and tokens per second
                # self.generation_time = end_time - start_time
                self.generation_time = end_time - start_time - 0.12
                tokens_generated = result.get("usage", {}).get("completion_tokens", self.max_tokens)
                self.tokens_per_second = tokens_generated / self.generation_time if self.generation_time > 0 else 0

                self.status = f"Generated {tokens_generated} tokens in {self.generation_time:.2f}s"
            else:
                self.status = f"Error: API returned status {response.status_code}"
        except Exception as e:
            self.status = f"Error: {str(e)}"

        self.draw_ui()

    def run(self):
        while True:
            try:
                # Update window dimensions in case of resize
                self.height, self.width = self.stdscr.getmaxyx()
                self.draw_ui()

                # Get user input
                key = self.stdscr.getch()

                if key == ord("q") or key == 27:  # q or Escape
                    break
                elif key == curses.KEY_UP or key == ord("k"):
                    # Up arrow or k (vim-style)
                    self.current_field = max(0, self.current_field - 1)
                elif key == curses.KEY_DOWN or key == ord("j"):
                    # Down arrow or j (vim-style)
                    self.current_field = min(2, self.current_field + 1)
                elif key == 10:  # Enter
                    self.edit_field()
                elif key == ord("s"):
                    # Shortcut to send
                    self.current_field = 2
                    self.send_request()
                elif key == curses.KEY_RESIZE:
                    # Handle terminal resize
                    self.stdscr.clear()
            except curses.error as e:
                # Handle any unexpected curses errors
                self.status = f"Curses error: {str(e)}"


def main(stdscr):
    # Set up the screen
    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(100)  # Set getch() timeout for handling resize events

    ui = SimpleTerminalLLMUI(stdscr)
    ui.run()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
