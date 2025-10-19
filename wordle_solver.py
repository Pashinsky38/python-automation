#!/usr/bin/env python3
"""
Wordle solver (interactive) - Optimized version (improved)

Changes vs original:
 - Removed unused numpy import.
 - Use a per-candidate-set entropy cache keyed by guess (avoid expensive sorting).
 - Add a fast letter-frequency heuristic prefilter to limit expensive entropy computations.
 - Use a small meaningful candidate bonus as tie-breaker.
 - Minor tuning of pooling thresholds.
Author: ChatGPT (GPT-5 Thinking mini). Edited by assistant.
"""

from collections import Counter, defaultdict
import math
import argparse
import sys
import os
from functools import lru_cache

# Default word file name
DEFAULT_WORD_FILE = "valid-wordle-words.txt"

# Pre-computed best starting words
BEST_STARTING_WORDS = ["salet", "roate", "raise", "crane", "slate", "crate", "trace", "carte"]

# Small fallback list (kept shortened here; expand if you want full fallback)
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
    "broke", "brown", "build", "built", "buyer", "cable", "calif", "carry"
]

# ---------- tuning ----------
GET_FEEDBACK_CACHE = 200_000   # lru_cache for get_feedback
HEURISTIC_TOP_K = 400         # compute exact entropy for top-K by heuristic
CANDIDATE_BONUS = 1e-3        # small tie-breaker bonus for guesses that are also candidates

# ---------- feedback & checking utilities ----------
@lru_cache(maxsize=GET_FEEDBACK_CACHE)
def get_feedback(guess: str, solution: str):
    """
    Return feedback tuple for guess vs solution using integer encoding:
    2 : correct position (green)
    1 : present wrong position (yellow)
    0 : not present (gray)
    """
    feedback = [0] * 5
    solution_chars = list(solution)

    # Greens first
    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = 2
            solution_chars[i] = None

    remaining = Counter(ch for ch in solution_chars if ch is not None)

    # Yellows / grays
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
        # normalize/dedupe
        self.candidates = list(dict.fromkeys(w.lower() for w in candidates if len(w) == 5 and w.isalpha()))
        self.candidates_set = set(self.candidates)
        self.history = []
        self.initial_size = len(self.candidates)
        # entropy cache keyed by guess for the CURRENT candidate set
        self._entropy_cache = {}

    def filter_candidates(self, guess: str, feedback):
        new_candidates = [w for w in self.candidates if get_feedback(guess, w) == feedback]
        removed = len(self.candidates) - len(new_candidates)
        self.candidates = new_candidates
        self.candidates_set = set(new_candidates)
        # candidate set changed -> clear entropy cache
        self._entropy_cache.clear()
        return removed

    def _compute_entropy(self, guess, candidates):
        """
        Compute expected information (entropy) for a guess given current candidates.
        Cache results by guess for the current candidate set (cache cleared when candidates change).
        """
        # simple cache: entropy depends only on guess and current candidate set
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
        """
        Fast heuristic: sum of letter frequencies across current candidates (unique letters per guess).
        Cheap and effective to prefilter guesses for full entropy computation.
        """
        freq = Counter()
        for w in candidates:
            freq.update(set(w))  # count each letter once per candidate word
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
        """
        Suggest next guesses. Uses heuristic prefilter when the allowed guess pool is large.
        Returns list of (word, score).
        """
        if allowed_guesses is None:
            allowed_guesses = self.candidates

        if not self.candidates:
            return []

        if len(self.candidates) == 1:
            return [(self.candidates[0], float('inf'))]

        if len(self.candidates) <= 2:
            return [(w, 1.0) for w in self.candidates[:top_n]]

        # starting guess optimization
        if len(self.candidates) == self.initial_size and not self.history:
            return self._get_best_starting_words(allowed_guesses, top_n)

        pool = allowed_guesses

        # reduce pool aggressively for very large allowed-guess lists
        if len(pool) > 2000 and len(self.candidates) > 50:
            priority_pool = [w for w in pool if w in self.candidates_set]
            other_pool = [w for w in pool if w not in self.candidates_set]
            pool = priority_pool + other_pool[:800]
        elif len(pool) > 5000:
            pool = [w for w in pool if w in self.candidates_set][:1500]

        # If pool still large, use heuristic to pick top-K fastest candidates for entropy calculation
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

        # If we only computed on a subset, ensure a few actual candidate words not in compute_pool are considered
        if compute_pool is not pool:
            missing_candidates = [w for w in self.candidates if w not in compute_pool]
            for w in missing_candidates[:50]:
                entropy, partition_count = self._compute_entropy(w, self.candidates)
                score = entropy + CANDIDATE_BONUS
                results.append((w, score, partition_count))

        # sort by score then by partition count
        results.sort(key=lambda x: (-x[1], -x[2]))
        return [(w, s) for (w, s, parts) in results[:top_n]]

    def add_history(self, guess: str, feedback):
        self.history.append((guess, feedback))
        removed = self.filter_candidates(guess, feedback)
        return removed


# ---------- helper I/O / interactive flow ----------
def parse_feedback_input(s: str):
    s = s.strip()
    if "," in s or " " in s:
        parts = [p.strip() for p in s.replace(",", " ").split()]
        fb = []
        for p in parts:
            if p == "0":
                fb.append(0)
            elif p == "1":
                fb.append(1)
            elif p == "2":
                fb.append(2)
            elif p == "0.5" or p == ".5":
                fb.append(1)
            else:
                try:
                    val = float(p)
                    if math.isclose(val, 0.0):
                        fb.append(0)
                    elif math.isclose(val, 0.5):
                        fb.append(1)
                    elif math.isclose(val, 1.0):
                        fb.append(1)
                    elif math.isclose(val, 2.0):
                        fb.append(2)
                    else:
                        fb.append(0)
                except Exception:
                    raise ValueError(f"Cannot parse feedback token: {p}")
        if len(fb) != 5:
            raise ValueError("Feedback must have 5 values.")
        return tuple(fb)
    else:
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

    if args.words:
        word_file = args.words
    elif os.path.exists(DEFAULT_WORD_FILE):
        word_file = DEFAULT_WORD_FILE
    else:
        word_file = None

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
