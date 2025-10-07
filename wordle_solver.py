#!/usr/bin/env python3
"""
Wordle solver (interactive) - Optimized version

- Supports feedback values encoded as integers: 0 (absent / gray), 1 (present but wrong spot / yellow), 2 (correct spot / green).
- Uses exact feedback rules compatible with Wordle (greens first, then yellows considering leftover letters).
- Suggests next guesses by computing expected information (entropy) over the current candidate set.
- Can operate in interactive mode where you provide feedback after each guess, or auto-solve mode
  where you give the secret word and it simulates the solving.
- Reads words from valid-wordle-words.txt file by default.

Usage:
  python wordle_solver.py         # interactive (uses valid-wordle-words.txt)
  python wordle_solver.py --auto secretword   # auto-solve simulation
  python wordle_solver.py --words words.txt   # use custom word list

Author: ChatGPT (GPT-5 Thinking mini) - Optimized by Claude
"""

from collections import Counter, defaultdict
import math
import argparse
import sys
import os
from functools import lru_cache

# Default word file name
DEFAULT_WORD_FILE = "valid-wordle-words.txt"

# Pre-computed best starting words (computed offline using full entropy)
# These are optimal starting guesses for a standard Wordle word list
BEST_STARTING_WORDS = ["salet", "roate", "raise", "crane", "slate", "crate", "trace", "carte"]

# Small fallback list in case file is missing
FALLBACK_WORDS = [
    "about", "above", "actor", "acute", "admit", "adopt", "adult", "after",
    "again", "agent", "agree", "ahead", "alarm", "album", "alert", "alien",
    "align", "alike", "alive", "allow", "alone", "along", "alter", "anger",
    "angle", "angry", "apart", "apple", "apply", "arena", "argue", "arise",
    "array", "aside", "asset", "audio", "audit", "avoid", "award", "aware",
    "badly", "baker", "bases", "basic", "basin", "basis", "beach", "began",
    "begin", "being", "below", "bench", "billy", "birth", "black", "blade",
    "blame", "blank", "blast", "bleed", "blend", "bless", "blind", "block",
    "blood", "bloom", "board", "boost", "booth", "bound", "brain", "brand",
    "brass", "brave", "bread", "break", "breed", "brief", "bring", "broad",
    "broke", "brown", "build", "built", "buyer", "cable", "calif", "carry",
    "catch", "cause", "chain", "chair", "chaos", "chart", "chase", "cheap",
    "check", "chest", "chief", "child", "china", "chose", "civil", "claim",
    "class", "clean", "clear", "click", "cliff", "climb", "clock", "close",
    "coach", "coast", "could", "count", "court", "cover", "craft", "crash",
    "crazy", "cream", "crime", "cross", "crowd", "crown", "crude", "curve",
    "cycle", "daily", "dance", "dated", "dealt", "death", "debut", "delay",
    "depth", "doing", "doubt", "dozen", "draft", "drama", "drank", "drawn",
    "dream", "dress", "drill", "drink", "drive", "drove", "dying", "eager",
    "early", "earth", "eight", "elite", "empty", "enemy", "enjoy", "enter",
    "entry", "equal", "error", "event", "every", "exact", "exist", "extra",
    "faith", "false", "fault", "fiber", "field", "fifth", "fifty", "fight",
    "final", "first", "fixed", "flash", "fleet", "flesh", "float", "flood",
    "floor", "fluid", "focus", "force", "forth", "forty", "forum", "found",
    "frame", "frank", "fraud", "fresh", "front", "fruit", "fully", "funny"
]


# ---------- feedback & checking utilities ----------
@lru_cache(maxsize=100000)
def get_feedback(guess: str, solution: str):
    """
    Return feedback tuple for guess vs solution using integer encoding:
    - 2 : letter correct & correct position (green)
    - 1 : letter present in solution but wrong position (yellow)
    - 0 : letter not in solution (gray)

    Implements Wordle rules for duplicates (greens first, then yellows up to remaining counts).
    Returns a tuple of 5 integers in {0,1,2}.
    
    Cached for performance.
    """
    feedback = [0] * 5
    solution_chars = list(solution)

    # First pass: greens (2)
    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = 2
            solution_chars[i] = None  # consume this char

    # Count remaining letters in solution (excluding consumed greens)
    remaining = Counter(ch for ch in solution_chars if ch is not None)

    # Second pass: yellows (1) or grays (0)
    for i in range(5):
        if feedback[i] == 2:
            continue
        ch = guess[i]
        if remaining[ch] > 0:
            feedback[i] = 1
            remaining[ch] -= 1

    return tuple(feedback)


def feedback_to_key(feedback):
    """Convert feedback tuple to a string key for dicts, e.g. (2,1,0,0,2) -> '21002'"""
    return ''.join(str(int(f)) for f in feedback)


# ---------- solver core ----------
class WordleSolver:
    def __init__(self, candidates):
        """
        candidates: list of lowercase 5-letter words (initial candidate set)
        """
        # Ensure uniqueness, lowercase, and five-letter alpha words
        self.candidates = list(dict.fromkeys(w.lower() for w in candidates if len(w) == 5 and w.isalpha()))
        self.candidates_set = set(self.candidates)  # For O(1) lookup
        self.history = []  # list of (guess, feedback_tuple)
        self.initial_size = len(self.candidates)

    def filter_candidates(self, guess: str, feedback):
        """Apply feedback to prune self.candidates. `feedback` should be a tuple of ints (0/1/2)."""
        # Use list comprehension for speed
        new_candidates = [w for w in self.candidates if get_feedback(guess, w) == feedback]
        removed = len(self.candidates) - len(new_candidates)
        self.candidates = new_candidates
        self.candidates_set = set(new_candidates)
        return removed

    def _compute_entropy(self, guess, candidates):
        """
        Compute expected information (entropy) for a guess given current candidates.
        Returns (entropy, partition_count).
        """
        # Partition counts by feedback outcome
        counts = defaultdict(int)
        for sol in candidates:
            key = feedback_to_key(get_feedback(guess, sol))
            counts[key] += 1
        
        # Compute entropy
        total = len(candidates)
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            entropy -= p * math.log2(p)
        
        return entropy, len(counts)

    def _get_best_starting_words(self, allowed_guesses, top_n=5):
        """Get best starting words from pre-computed list that exist in allowed_guesses."""
        available = [w for w in BEST_STARTING_WORDS if w in allowed_guesses]
        # If we don't have enough pre-computed words in the allowed list, add some high-frequency letter words
        if len(available) < top_n:
            # Common high-value starting words
            extras = ["adieu", "audio", "ouija", "arose", "irate", "stare", "tears", "store"]
            for w in extras:
                if w in allowed_guesses and w not in available:
                    available.append(w)
                if len(available) >= top_n:
                    break
        
        # Return with dummy scores (they're all good starting words)
        return [(w, 5.0) for w in available[:top_n]]

    def suggest_next(self, allowed_guesses=None, top_n=1):
        """
        Suggest next guess(es).

        allowed_guesses: list of words to consider as possible guesses (if None, use self.candidates).
                         If you have a larger 'allowed guess' list, pass it here.
        top_n: return the top_n suggestions (list of tuples (word, score))
        """
        if allowed_guesses is None:
            allowed_guesses = self.candidates

        if not self.candidates:
            return []

        # if only one candidate left, return it immediately
        if len(self.candidates) == 1:
            return [(self.candidates[0], float('inf'))]

        # If candidates are very small, just return them
        if len(self.candidates) <= 2:
            return [(w, 1.0) for w in self.candidates[:top_n]]

        # OPTIMIZATION: For first guess (when we haven't filtered anything yet), use pre-computed starting words
        if len(self.candidates) == self.initial_size and not self.history:
            return self._get_best_starting_words(allowed_guesses, top_n)

        pool = allowed_guesses
        best = []
        
        # Limit pool size for performance when candidate set is large
        if len(pool) > 2000 and len(self.candidates) > 50:
            # Prioritize words that are actual candidates
            priority_pool = [w for w in pool if w in self.candidates_set]
            other_pool = [w for w in pool if w not in self.candidates_set]
            # Take top candidates + sample of others
            pool = priority_pool + other_pool[:1000]
        elif len(pool) > 5000:
            # Very large pool - be even more aggressive
            pool = [w for w in pool if w in self.candidates_set][:2000]

        # Compute entropy for each potential guess
        for guess in pool:
            entropy, partition_count = self._compute_entropy(guess, self.candidates)
            
            # Bonus for guesses that are actually candidates (helps solve faster)
            bonus = 0.5 if guess in self.candidates_set else 0.0
            score = entropy + bonus * 1e-6  # tiny bonus, keeps entropy as main metric
            
            best.append((guess, score, partition_count))

        # Sort descending by score (entropy), then by partition count
        best.sort(key=lambda x: (-x[1], -x[2]))
        return [(w, s) for (w, s, parts) in best[:top_n]]

    def add_history(self, guess: str, feedback):
        """Record a round and prune candidates accordingly. feedback should be tuple of ints."""
        self.history.append((guess, feedback))
        removed = self.filter_candidates(guess, feedback)
        return removed


# ---------- helper I/O / interactive flow ----------
def parse_feedback_input(s: str):
    """
    Accept several formats:
      - "0,1,2,0,2"      (new preferred encoding: 0=absent,1=present(yellow),2=correct(green))
      - "01202"          (compact digits, 5 chars)
      - "0 1 2 0 2"
      - For backward compatibility this will also accept "0,0.5,1,0,1" or ".5" tokens and map them:
          0 -> 0 (absent)
          0.5 or .5 -> 1 (present / yellow)
          1 -> 1 (present / yellow)   [legacy mapping]
          2 -> 2 (correct / green)
    Returns a feedback tuple of 5 integers in {0,1,2}.
    """
    s = s.strip()
    if "," in s or " " in s:
        parts = [p.strip() for p in s.replace(",", " ").split()]
        fb = []
        for p in parts:
            if p == "0":
                fb.append(0)
            elif p == "1":
                # New scheme: 1 means present (yellow)
                fb.append(1)
            elif p == "2":
                # New scheme: 2 means correct (green)
                fb.append(2)
            elif p == "0.5" or p == ".5":
                # backward compatibility: 0.5 -> present (1)
                fb.append(1)
            else:
                # try to parse numeric (e.g., someone typed "1.0" or "2.0")
                try:
                    val = float(p)
                    if math.isclose(val, 0.0):
                        fb.append(0)
                    elif math.isclose(val, 0.5):
                        fb.append(1)
                    elif math.isclose(val, 1.0):
                        # legacy ambiguity: treat 1.0 as 'present' (yellow) in new scheme
                        fb.append(1)
                    elif math.isclose(val, 2.0):
                        fb.append(2)
                    else:
                        # default to absent if unknown
                        fb.append(0)
                except Exception:
                    raise ValueError(f"Cannot parse feedback token: {p}")
        if len(fb) != 5:
            raise ValueError("Feedback must have 5 values.")
        return tuple(fb)
    else:
        # compact form like 01202 (digits 0/1/2), must be exactly 5 chars
        if len(s) != 5:
            raise ValueError("Feedback must be 5 tokens (either comma-separated or 5-char compact form).")
        fb = []
        for ch in s:
            if ch == "0":
                fb.append(0)
            elif ch == "1":
                fb.append(1)
            elif ch == "2":
                fb.append(2)
            else:
                raise ValueError(f"Unknown feedback char: {ch}")
        return tuple(fb)


def load_words_from_file(path):
    """Load words from file, deduplicated and filtered to 5-letter alpha words."""
    words = []
    seen = set()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == 5 and w.isalpha() and w not in seen:
                words.append(w)
                seen.add(w)
    return words


def interactive_mode(solver: WordleSolver, allowed_guesses=None):
    print("\nInteractive Wordle solver")
    print("Type 'help' for commands, 'quit' to exit.")
    print("After each guess you'll provide feedback in one of these formats:")
    print("  0,1,2,0,2   or   01202   or   0 1 2 0 2")
    print("Encoding: 0 = absent (gray), 1 = present but wrong spot (yellow), 2 = correct spot (green)\n")

    round_no = 1
    while True:
        print(f"\nRound {round_no}")
        print(f"Candidates remaining: {len(solver.candidates)}")
        
        if round_no == 1 and len(solver.candidates) > 1000:
            print("(Using pre-computed optimal starting words for speed...)")
        
        top = solver.suggest_next(allowed_guesses=allowed_guesses, top_n=5)
        if not top:
            print("No candidates left. Something's inconsistent with the feedback provided.")
            return
        
        print("Top suggestions (word : expected-info score):")
        for w, sc in top:
            print(f"  {w:10}  {sc:.4f}")
        
        if len(solver.candidates) <= 50:
            print("\nCurrent candidates:", ", ".join(solver.candidates))

        # Let user choose guess or accept top
        guess = input("Enter your guess (or press Enter to pick the top suggestion): ").strip().lower()
        if guess == "":
            guess = top[0][0]
            print("Picking:", guess)
        if guess == "help":
            print("Commands: 'help', 'quit', 'candidates', 'history'")
            continue
        if guess == "quit":
            return
        if guess == "candidates":
            print(", ".join(solver.candidates))
            continue
        if guess == "history":
            for g, fb in solver.history:
                print(f"{g} -> {feedback_to_key(fb)}")
            continue

        # accept any valid 5-letter alpha word
        if len(guess) != 5 or not guess.isalpha():
            print("Guess must be a 5-letter word.")
            continue

        fb_in = input("Enter feedback for that guess (e.g. 0,1,2,0,2): ").strip()
        try:
            fb = parse_feedback_input(fb_in)
        except ValueError as e:
            print("Could not parse feedback:", e)
            continue

        removed = solver.add_history(guess, fb)
        print(f"Pruned {removed} words; {len(solver.candidates)} candidates remain.")
        if len(solver.candidates) == 1:
            print("Solved! The word is:", solver.candidates[0])
            return
        if all(f == 2 for f in fb):
            print("All green — solved!")
            return
        round_no += 1


def auto_solve(secret, solver: WordleSolver, allowed_guesses=None, max_rounds=6):
    secret = secret.lower()
    if len(secret) != 5 or not secret.isalpha():
        raise ValueError("Secret must be a 5-letter word.")
    print("Auto-solve simulation for secret:", secret)
    for round_no in range(1, max_rounds + 1):
        suggestion = solver.suggest_next(allowed_guesses=allowed_guesses, top_n=1)
        if not suggestion:
            print("No suggestions available; failed.")
            return False
        guess = suggestion[0][0]
        fb = get_feedback(guess, secret)
        print(f"Round {round_no}: guess={guess}  feedback={feedback_to_key(fb)}")
        solver.add_history(guess, fb)
        if all(x == 2 for x in fb):
            print(f"Solved in {round_no} rounds.")
            return True
    print(f"Failed to solve within {max_rounds} rounds.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Wordle solver (interactive / auto)")
    parser.add_argument("--words", help="Path to word list file (one 5-letter word per line).", default=None)
    parser.add_argument("--auto", help="Auto-solve for the provided secret word (simulate).", default=None)
    parser.add_argument("--max-rounds", type=int, default=6, help="Max rounds for auto-solve.")
    args = parser.parse_args()

    # Determine which word file to use
    if args.words:
        word_file = args.words
    elif os.path.exists(DEFAULT_WORD_FILE):
        word_file = DEFAULT_WORD_FILE
    else:
        word_file = None

    # Load words
    if word_file:
        try:
            words = load_words_from_file(word_file)
            if not words:
                print(f"No valid 5-letter words found in {word_file}; using fallback list.")
                words = FALLBACK_WORDS
            else:
                print(f"Loaded {len(words)} words from {word_file}")
        except Exception as e:
            print(f"Failed to load words file '{word_file}': {e}")
            print("Using fallback word list.")
            words = FALLBACK_WORDS
    else:
        print(f"Word file '{DEFAULT_WORD_FILE}' not found; using fallback list.")
        words = FALLBACK_WORDS

    # allowed guesses: by default same as candidate list
    allowed_guesses = words

    print(f"Initializing solver with {len(words)} candidate words...")
    solver = WordleSolver(candidates=words)

    if args.auto:
        success = auto_solve(args.auto, solver, allowed_guesses=allowed_guesses, max_rounds=args.max_rounds)
        if not success:
            print("Candidates left (top 20):", solver.candidates[:20])
    else:
        interactive_mode(solver, allowed_guesses=allowed_guesses)

if __name__ == "__main__":
    main()