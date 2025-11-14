#!/usr/bin/env python3
"""
Wordle Solver with Enhanced Tkinter GUI

Beautiful, user-friendly GUI with modern styling and visual feedback.

Usage:
  python wordle_solver_gui.py --words valid-wordle-words.txt

Dependencies: Python 3.x, tkinter
"""

from collections import Counter, defaultdict
import math
import argparse
import sys
import os
from functools import lru_cache
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- constants & tuning ----------
DEFAULT_WORD_FILE = "valid-wordle-words.txt"
BEST_STARTING_WORDS = ["salet", "roate", "raise", "crane", "slate", "crate", "trace", "carte"]
GET_FEEDBACK_CACHE = 200_000
HEURISTIC_TOP_K = 400
CANDIDATE_BONUS = 1e-3

# Color scheme
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'accent': '#0f3460',
    'highlight': '#533483',
    'text': '#eee',
    'gray': '#787c7e',
    'yellow': '#c9b458',
    'green': '#6aaa64',
    'button': '#818384',
    'button_hover': '#9a9b9c',
    'success': '#538d4e',
    'error': '#d94744'
}

# ---------- feedback utilities ----------
@lru_cache(maxsize=GET_FEEDBACK_CACHE)
def get_feedback(guess: str, solution: str):
    feedback = [0] * 5
    solution_chars = list(solution)

    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = 2
            solution_chars[i] = None

    remaining = Counter(ch for ch in solution_chars if ch is not None)

    for i in range(5):
        if feedback[i] == 2:
            continue
        ch = guess[i]
        if remaining.get(ch, 0) > 0:
            feedback[i] = 1
            remaining[ch] -= 1

    return tuple(feedback)


def feedback_to_key(feedback):
    return ''.join(str(int(f)) for f in feedback)

# ---------- solver core ----------
class WordleSolver:
    def __init__(self, candidates):
        self.candidates = list(dict.fromkeys(w.lower() for w in candidates if len(w) == 5 and w.isalpha()))
        self.candidates_set = set(self.candidates)
        self.history = []
        self.initial_size = len(self.candidates)
        self._entropy_cache = {}

    def filter_candidates(self, guess: str, feedback):
        new_candidates = [w for w in self.candidates if get_feedback(guess, w) == feedback]
        removed = len(self.candidates) - len(new_candidates)
        self.candidates = new_candidates
        self.candidates_set = set(new_candidates)
        self._entropy_cache.clear()
        return removed

    def _compute_entropy(self, guess, candidates):
        if guess in self._entropy_cache:
            return self._entropy_cache[guess]

        counts = defaultdict(int)
        for sol in candidates:
            key = feedback_to_key(get_feedback(guess, sol))
            counts[key] += 1

        total = len(candidates)
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            entropy -= p * math.log2(p)

        result = (entropy, len(counts))
        self._entropy_cache[guess] = result
        return result

    def _heuristic_score(self, guess, candidates):
        freq = Counter()
        for w in candidates:
            freq.update(set(w))
        return sum(freq[ch] for ch in set(guess))

    def _get_best_starting_words(self, allowed_guesses, top_n=5):
        available = [w for w in BEST_STARTING_WORDS if w in allowed_guesses]
        if len(available) < top_n:
            extras = ["adieu", "audio", "ouija", "arose", "irate", "stare", "tears", "store"]
            for w in extras:
                if w in allowed_guesses and w not in available:
                    available.append(w)
                if len(available) >= top_n:
                    break
        return [(w, 5.0) for w in available[:top_n]]

    def suggest_next(self, allowed_guesses=None, top_n=1):
        if allowed_guesses is None:
            allowed_guesses = self.candidates

        if not self.candidates:
            return []

        if len(self.candidates) == 1:
            return [(self.candidates[0], float('inf'))]

        if len(self.candidates) <= 2:
            return [(w, 1.0) for w in self.candidates[:top_n]]

        if len(self.candidates) == self.initial_size and not self.history:
            return self._get_best_starting_words(allowed_guesses, top_n)

        pool = allowed_guesses

        if len(pool) > 2000 and len(self.candidates) > 50:
            priority_pool = [w for w in pool if w in self.candidates_set]
            other_pool = [w for w in pool if w not in self.candidates_set]
            pool = priority_pool + other_pool[:800]
        elif len(pool) > 5000:
            pool = [w for w in pool if w in self.candidates_set][:1500]

        compute_pool = pool
        if len(pool) > HEURISTIC_TOP_K:
            scored = []
            for w in pool:
                scored.append((w, self._heuristic_score(w, self.candidates)))
            scored.sort(key=lambda x: -x[1])
            top_k = min(HEURISTIC_TOP_K, len(scored))
            compute_pool = [w for w, sc in scored[:top_k]]

        results = []
        for guess in compute_pool:
            entropy, partition_count = self._compute_entropy(guess, self.candidates)
            bonus = CANDIDATE_BONUS if guess in self.candidates_set else 0.0
            score = entropy + bonus
            results.append((guess, score, partition_count))

        if compute_pool is not pool:
            missing_candidates = [w for w in self.candidates if w not in compute_pool]
            for w in missing_candidates[:50]:
                entropy, partition_count = self._compute_entropy(w, self.candidates)
                score = entropy + CANDIDATE_BONUS
                results.append((w, score, partition_count))

        results.sort(key=lambda x: (-x[1], -x[2]))
        return [(w, s) for (w, s, parts) in results[:top_n]]

    def add_history(self, guess: str, feedback):
        self.history.append((guess, feedback))
        removed = self.filter_candidates(guess, feedback)
        return removed

# ---------- I/O helpers ----------

def load_words_from_file(path):
    words = []
    seen = set()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == 5 and w.isalpha() and w not in seen:
                words.append(w)
                seen.add(w)
    return words

# ---------- Custom Widgets ----------

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, width=120, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['panel'], 
                        highlightthickness=0, **kwargs)
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        
        self.rect = self.create_rectangle(2, 2, width-2, height-2, 
                                         fill=COLORS['accent'], outline='')
        self.text_id = self.create_text(width//2, height//2, text=text, 
                                       fill=COLORS['text'], font=('Arial', 11, 'bold'))
        
        self.bind('<Button-1>', lambda e: self.on_click())
        self.bind('<Enter>', lambda e: self.on_hover())
        self.bind('<Leave>', lambda e: self.on_leave())
        
    def on_hover(self):
        self.itemconfig(self.rect, fill=COLORS['highlight'])
        
    def on_leave(self):
        self.itemconfig(self.rect, fill=COLORS['accent'])
        
    def on_click(self):
        if self.command:
            self.command()

class FeedbackBox(tk.Canvas):
    def __init__(self, parent, index, callback, **kwargs):
        super().__init__(parent, width=60, height=60, bg=COLORS['panel'], 
                        highlightthickness=0, **kwargs)
        self.index = index
        self.callback = callback
        self.state = 0  # 0=gray, 1=yellow, 2=green
        
        self.rect = self.create_rectangle(5, 5, 55, 55, fill=COLORS['gray'], 
                                         outline='', width=2)
        self.text_id = self.create_text(30, 30, text='', fill=COLORS['text'], 
                                       font=('Arial', 20, 'bold'))
        
        self.bind('<Button-1>', lambda e: self.toggle())
        self.bind('<Enter>', lambda e: self.on_hover())
        self.bind('<Leave>', lambda e: self.on_leave())
        
    def toggle(self):
        self.state = (self.state + 1) % 3
        self.update_display()
        if self.callback:
            self.callback(self.index, self.state)
            
    def set_state(self, state):
        self.state = state
        self.update_display()
        
    def set_letter(self, letter):
        self.itemconfig(self.text_id, text=letter.upper())
        
    def update_display(self):
        colors = [COLORS['gray'], COLORS['yellow'], COLORS['green']]
        self.itemconfig(self.rect, fill=colors[self.state])
        
    def on_hover(self):
        self.itemconfig(self.rect, outline='white', width=2)
        
    def on_leave(self):
        self.itemconfig(self.rect, outline='', width=2)

# ---------- GUI ----------
class WordleGUI:
    def __init__(self, root, words=None):
        self.root = root
        self.root.title("Wordle Solver")
        self.root.configure(bg=COLORS['bg'])
        self.words = words or []
        self.solver = None
        self.allowed_guesses = None

        self._build_ui()
        if self.words:
            self._init_solver(self.words)

    def _build_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg=COLORS['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(main_container, bg=COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(header, text="🔤 WORDLE SOLVER", 
                        font=('Arial', 24, 'bold'), 
                        fg=COLORS['text'], bg=COLORS['bg'])
        title.pack(side=tk.LEFT)
        
        # File controls
        file_frame = tk.Frame(header, bg=COLORS['bg'])
        file_frame.pack(side=tk.RIGHT)
        
        self.btn_load = ModernButton(file_frame, "📁 Load Words", 
                                     command=self._on_load_file, width=120, height=35)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.lbl_wordfile = tk.Label(file_frame, text="No file loaded", 
                                     font=('Arial', 10), 
                                     fg=COLORS['button'], bg=COLORS['bg'])
        self.lbl_wordfile.pack(side=tk.LEFT, padx=10)
        
        # Main content area
        content = tk.Frame(main_container, bg=COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input
        left_panel = tk.Frame(content, bg=COLORS['panel'], relief=tk.FLAT, bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Guess input section
        input_section = tk.Frame(left_panel, bg=COLORS['panel'])
        input_section.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(input_section, text="Enter Your Guess", 
                font=('Arial', 14, 'bold'), 
                fg=COLORS['text'], bg=COLORS['panel']).pack(anchor=tk.W, pady=(0, 10))
        
        self.entry_guess = tk.Entry(input_section, font=('Arial', 16), 
                                    bg=COLORS['accent'], fg=COLORS['text'],
                                    insertbackground=COLORS['text'], relief=tk.FLAT,
                                    justify=tk.CENTER)
        self.entry_guess.pack(fill=tk.X, ipady=8)
        self.entry_guess.bind('<KeyRelease>', self._on_guess_change)
        
        # Feedback boxes
        tk.Label(input_section, text="Click boxes to set feedback", 
                font=('Arial', 12), 
                fg=COLORS['button'], bg=COLORS['panel']).pack(pady=(15, 5))
        
        fb_container = tk.Frame(input_section, bg=COLORS['panel'])
        fb_container.pack(pady=10)
        
        self.fb_boxes = []
        for i in range(5):
            box = FeedbackBox(fb_container, i, self._on_feedback_change)
            box.pack(side=tk.LEFT, padx=3)
            self.fb_boxes.append(box)
        
        # Action buttons
        btn_frame = tk.Frame(input_section, bg=COLORS['panel'])
        btn_frame.pack(pady=(20, 0))
        
        self.btn_apply = ModernButton(btn_frame, "✓ Apply Feedback", 
                                      command=self._on_apply_feedback, 
                                      width=180, height=45)
        self.btn_apply.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = ModernButton(btn_frame, "↻ Reset", 
                                      command=self._on_reset, 
                                      width=100, height=45)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.lbl_status = tk.Label(input_section, text="", 
                                   font=('Arial', 11), 
                                   fg=COLORS['success'], bg=COLORS['panel'])
        self.lbl_status.pack(pady=(15, 0))
        
        # History section
        history_section = tk.Frame(left_panel, bg=COLORS['panel'])
        history_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        tk.Label(history_section, text="History", 
                font=('Arial', 14, 'bold'), 
                fg=COLORS['text'], bg=COLORS['panel']).pack(anchor=tk.W, pady=(0, 10))
        
        history_scroll_frame = tk.Frame(history_section, bg=COLORS['accent'])
        history_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_canvas = tk.Canvas(history_scroll_frame, bg=COLORS['accent'], 
                                       highlightthickness=0)
        self.history_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Suggestions
        right_panel = tk.Frame(content, bg=COLORS['panel'], relief=tk.FLAT, bd=0)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Candidates count
        count_frame = tk.Frame(right_panel, bg=COLORS['panel'])
        count_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(count_frame, text="Remaining Candidates", 
                font=('Arial', 12), 
                fg=COLORS['button'], bg=COLORS['panel']).pack(anchor=tk.W)
        
        self.lbl_count = tk.Label(count_frame, text="0", 
                                 font=('Arial', 32, 'bold'), 
                                 fg=COLORS['text'], bg=COLORS['panel'])
        self.lbl_count.pack(anchor=tk.W)
        
        # Suggestions section
        suggest_section = tk.Frame(right_panel, bg=COLORS['panel'])
        suggest_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        suggest_header = tk.Frame(suggest_section, bg=COLORS['panel'])
        suggest_header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(suggest_header, text="Top Suggestions", 
                font=('Arial', 14, 'bold'), 
                fg=COLORS['text'], bg=COLORS['panel']).pack(side=tk.LEFT)
        
        tk.Label(suggest_header, text="Show:", 
                font=('Arial', 10), 
                fg=COLORS['button'], bg=COLORS['panel']).pack(side=tk.LEFT, padx=(20, 5))
        
        self.spin_suggest = tk.Spinbox(suggest_header, from_=1, to=20, width=3,
                                      font=('Arial', 10), bg=COLORS['accent'],
                                      fg=COLORS['text'], buttonbackground=COLORS['accent'],
                                      relief=tk.FLAT)
        self.spin_suggest.delete(0, tk.END)
        self.spin_suggest.insert(0, "8")
        self.spin_suggest.pack(side=tk.LEFT)
        
        refresh_btn = ModernButton(suggest_header, "↻", command=self.update_ui, 
                                  width=40, height=30)
        refresh_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Suggestions list
        suggest_scroll_frame = tk.Frame(suggest_section, bg=COLORS['accent'])
        suggest_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.suggest_canvas = tk.Canvas(suggest_scroll_frame, bg=COLORS['accent'], 
                                       highlightthickness=0)
        self.suggest_canvas.pack(fill=tk.BOTH, expand=True)

    def _on_guess_change(self, event):
        guess = self.entry_guess.get().lower()
        for i, box in enumerate(self.fb_boxes):
            if i < len(guess):
                box.set_letter(guess[i])
            else:
                box.set_letter('')

    def _on_feedback_change(self, index, state):
        pass  # Feedback updated by the box itself

    def _on_load_file(self):
        path = filedialog.askopenfilename(
            title="Select wordlist file", 
            filetypes=[("Text files", "*.txt"), ("All files","*")])
        if not path:
            return
        try:
            words = load_words_from_file(path)
            if not words:
                messagebox.showerror("Error", f"No valid 5-letter words found in {path}")
                return
            self.lbl_wordfile.config(text=f"✓ {os.path.basename(path)}", fg=COLORS['success'])
            self.words = words
            self._init_solver(words)
            self.update_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load words: {e}")

    def _init_solver(self, words):
        self.allowed_guesses = words
        self.solver = WordleSolver(candidates=words)
        self.entry_guess.delete(0, tk.END)
        for box in self.fb_boxes:
            box.set_state(0)
            box.set_letter('')
        self.update_ui()

    def update_ui(self):
        if not self.solver:
            return
        
        cnt = len(self.solver.candidates)
        self.lbl_count.config(text=str(cnt))
        
        try:
            topn = int(self.spin_suggest.get())
        except Exception:
            topn = 8
        
        top = self.solver.suggest_next(allowed_guesses=self.allowed_guesses, top_n=topn)
        
        # Clear and redraw suggestions
        self.suggest_canvas.delete('all')
        y_pos = 10
        for i, (word, score) in enumerate(top):
            # Create clickable suggestion box
            box_height = 50
            rect = self.suggest_canvas.create_rectangle(
                10, y_pos, 380, y_pos + box_height,
                fill=COLORS['bg'], outline='', tags=f'sug_{i}')
            
            word_text = self.suggest_canvas.create_text(
                20, y_pos + 25, text=word.upper(), 
                font=('Arial', 16, 'bold'), fill=COLORS['text'], 
                anchor=tk.W, tags=f'sug_{i}')
            
            score_text = self.suggest_canvas.create_text(
                370, y_pos + 25, text=f'{score:.3f}', 
                font=('Arial', 11), fill=COLORS['button'], 
                anchor=tk.E, tags=f'sug_{i}')
            
            # Bind click event
            self.suggest_canvas.tag_bind(f'sug_{i}', '<Button-1>', 
                                        lambda e, w=word: self._use_word(w))
            self.suggest_canvas.tag_bind(f'sug_{i}', '<Enter>', 
                                        lambda e, r=rect: self.suggest_canvas.itemconfig(r, fill=COLORS['accent']))
            self.suggest_canvas.tag_bind(f'sug_{i}', '<Leave>', 
                                        lambda e, r=rect: self.suggest_canvas.itemconfig(r, fill=COLORS['bg']))
            
            y_pos += box_height + 5
        
        # Update history
        self.history_canvas.delete('all')
        y_pos = 10
        for guess, fb in reversed(self.solver.history[-10:]):  # Show last 10
            # Draw word boxes
            for i, (letter, state) in enumerate(zip(guess, fb)):
                colors = [COLORS['gray'], COLORS['yellow'], COLORS['green']]
                x = 10 + i * 55
                self.history_canvas.create_rectangle(
                    x, y_pos, x + 50, y_pos + 50,
                    fill=colors[state], outline='')
                self.history_canvas.create_text(
                    x + 25, y_pos + 25, text=letter.upper(),
                    font=('Arial', 18, 'bold'), fill='white')
            y_pos += 60

    def _use_word(self, word):
        self.entry_guess.delete(0, tk.END)
        self.entry_guess.insert(0, word)
        for i, letter in enumerate(word):
            self.fb_boxes[i].set_letter(letter)

    def _on_apply_feedback(self):
        guess = self.entry_guess.get().strip().lower()
        if not guess or len(guess) != 5 or not guess.isalpha():
            self.lbl_status.config(text="⚠ Guess must be a 5-letter word", fg=COLORS['error'])
            return
        
        fb = tuple(box.state for box in self.fb_boxes)
        removed = self.solver.add_history(guess, fb)
        
        remaining = len(self.solver.candidates)
        self.lbl_status.config(
            text=f"✓ Pruned {removed} words, {remaining} remain", 
            fg=COLORS['success'])
        
        # Reset input
        self.entry_guess.delete(0, tk.END)
        for box in self.fb_boxes:
            box.set_state(0)
            box.set_letter('')
        
        self.update_ui()
        
        if remaining == 1:
            self.lbl_status.config(
                text=f"🎉 Solved! The word is: {self.solver.candidates[0].upper()}", 
                fg=COLORS['success'])

    def _on_reset(self):
        if not self.words:
            return
        if messagebox.askyesno("Reset", "Reset solver to initial state?"):
            self._init_solver(self.words)
            self.lbl_status.config(text="")

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Wordle solver GUI")
    parser.add_argument("--words", help="Path to word list file", default=None)
    args = parser.parse_args()

    word_file = None
    if args.words:
        if not os.path.exists(args.words):
            print(f"Word file '{args.words}' not found.")
            sys.exit(1)
        word_file = args.words
    elif os.path.exists(DEFAULT_WORD_FILE):
        word_file = DEFAULT_WORD_FILE

    words = []
    if word_file:
        try:
            words = load_words_from_file(word_file)
            print(f"Loaded {len(words)} words from {word_file}")
        except Exception as e:
            print(f"Failed to load words: {e}")

    root = tk.Tk()
    app = WordleGUI(root, words=words)
    root.geometry('1000x700')
    root.minsize(900, 650)
    root.mainloop()

if __name__ == "__main__":
    main()