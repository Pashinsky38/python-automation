#!/usr/bin/env python3
"""
Wordle solver (interactive)

- Supports feedback values encoded as integers: 0 (absent / gray), 1 (present but wrong spot / yellow), 2 (correct spot / green).
- Uses exact feedback rules compatible with Wordle (greens first, then yellows considering leftover letters).
- Suggests next guesses by computing expected information (entropy) over the current candidate set.
- Can operate in interactive mode where you provide feedback after each guess, or auto-solve mode
  where you give the secret word and it simulates the solving.
- By default it uses a small built-in 5-letter word list (for quick testing). For best results,
  provide a larger word list file (one 5-letter word per line) using the --words PATH option.

Usage:
  python wordle_solver.py         # interactive
  python wordle_solver.py --auto secretword   # auto-solve simulation
  python wordle_solver.py --words words.txt   # use custom word list

Author: ChatGPT (GPT-5 Thinking mini)
"""

from collections import Counter, defaultdict
import math
import argparse
import sys
import textwrap

# ---------- built-in small-ish wordlist (for quick testing) ----------
# You can replace with your own list file (one word per line) for stronger performance.
# NOTE: only 5-letter words will be used from this list (the code filters them).
BUILTIN_WORDS = """
abuse amuse arise arena babel baton beach binge blaze brave
brink brand brick chain chair charm chase cheep cheek chest chess chime
crane crate crave crisp daily daisy dance debar debug demon
eager eagle earth eater elbow elude empty epoch equip erase evict
fable faint forage forge frame freak fresh fruit fudge gauge ghost giant
glare glaze giver grasp great grime grove grunt habit hatch haunt hazel
heavy heir hello hence higher hinge honey honor hover ideal image imbue
imply input inward ironi issue jacket jumbo judas judge juice juicy
kappa knead label labor laden laity lambda lapse large lathe layer learnt
leave lemon lessen light liner linen lingo lithe lodge lonely lover lower
magma magic maker maple march mauve maybe modal model money month
movie motif mount mouse muse music nasty navy never niche night noble
noise north novel ocean oculi offer often olive orbit order organ other
outer ounce palace panic paper parade patch pause peace pearl pedal pence
phase phone pivot pixel place plain plane plank plate plenty plush
pound power praise prime print prized probe proud prove prune pulse punch
pupil puzzle quake quail quaint queen query quest quiet quite radial radio
raise range rapid razor reach realm rebel recap relax relay relish rely
rhyme ridge right rival river roast robin robot rogue roomy rotary rough
round route royal ruin rumor runner rural sable sack safe saint salad samey
saner satin scale scary scowl scour scout seedy seize
shake shame shank sharp sheaf sheath sheep shelf shell shift shine shire
shore short shout shown shred shrug siege sieve sight sigma silent silver
slice slick slide slope small smart smear smell smile smoke snare sneak
sneer snake snuck solar solid solve sorry sound space spade spare spark
spice spike spill spine spirit spite split spoil spore sport spray sprint
squad staff stale stamp stand stark start stash state steel steep stern stick
stiff still sting stink stint stock stone stood stool story storm stork strap
straw stray strip strive study stuff stump style sugar super surge swirl
table tacit taken tally turbo teach team tearteen teeth tender token tone
topic total touch tough tower trace trade trail train tramp trunk trust truth
tulip twice twin type ultrs umpire unbox under union unity upper upset urban
usage usher usurp value valid vapor vault vector velvet vendor verse vicer
vigil vocal vogue voida voice volley vowel waste watch water weave weird
where which while white whole widget width wield willing winter witch woman
worse worth would wound writer wrong yardy yacht yearly yeast yield young youth
zesty zebra
""".split()

# Filter to ensure only five-letter lowercase words
DEFAULT_WORDS = [w.lower() for w in BUILTIN_WORDS if len(w) == 5 and w.isalpha()]


# ---------- feedback & checking utilities ----------
def get_feedback(guess: str, solution: str):
    """
    Return feedback tuple for guess vs solution using integer encoding:
    - 2 : letter correct & correct position (green)
    - 1 : letter present in solution but wrong position (yellow)
    - 0 : letter not in solution (gray)

    Implements Wordle rules for duplicates (greens first, then yellows up to remaining counts).
    Returns a tuple of 5 integers in {0,1,2}.
    """
    guess = guess.lower()
    solution = solution.lower()
    if len(guess) != 5 or len(solution) != 5:
        raise ValueError("Both guess and solution must be 5-letter words.")

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
        else:
            feedback[i] = 0

    return tuple(feedback)


def feedback_to_key(feedback):
    """Convert feedback tuple to a string key for dicts, e.g. (2,1,0,0,2) -> '2,1,0,0,2'"""
    # Accept ints or floats but represent as integers in the key
    return ",".join(str(int(f)) for f in feedback)


# ---------- solver core ----------
class WordleSolver:
    def __init__(self, candidates):
        """
        candidates: list of lowercase 5-letter words (initial candidate set)
        """
        # Ensure uniqueness, lowercase, and five-letter alpha words
        self.candidates = list(dict.fromkeys(w.lower() for w in candidates if len(w) == 5 and w.isalpha()))
        self.history = []  # list of (guess, feedback_tuple)

    def filter_candidates(self, guess: str, feedback):
        """Apply feedback to prune self.candidates. `feedback` should be a tuple of ints (0/1/2)."""
        new_candidates = []
        for w in self.candidates:
            if get_feedback(guess, w) == feedback:
                new_candidates.append(w)
        removed = len(self.candidates) - len(new_candidates)
        self.candidates = new_candidates
        return removed

    def suggest_next(self, allowed_guesses=None, top_n=1, use_entropy=True):
        """
        Suggest next guess(es).

        allowed_guesses: list of words to consider as possible guesses (if None, use self.candidates).
                         If you have a larger 'allowed guess' list, pass it here.
        top_n: return the top_n suggestions (list of tuples (word, score))
        use_entropy: compute expected information (entropy). If False, use a heuristic frequency score.
        """
        if allowed_guesses is None:
            allowed_guesses = self.candidates

        if not self.candidates:
            return []

        # if only one candidate left, return it immediately
        if len(self.candidates) == 1:
            return [(self.candidates[0], float('inf'))]

        pool = allowed_guesses

        best = []
        # iterate guesses and compute expected entropy over current candidates
        for guess in pool:
            # partition counts by feedback outcome
            counts = defaultdict(int)
            for sol in self.candidates:
                key = feedback_to_key(get_feedback(guess, sol))
                counts[key] += 1
            # compute entropy
            total = len(self.candidates)
            entropy = 0.0
            for cnt in counts.values():
                p = cnt / total
                entropy -= p * math.log2(p)
            # Optionally score by entropy; to break ties prefer guesses that are actually candidates
            bonus = 0.5 if guess in self.candidates else 0.0
            score = entropy + bonus * 1e-6  # tiny bonus, keeps entropy as main metric
            best.append((guess, score, len(counts)))  # include partitions count for info

        # sort descending by score (entropy)
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
    words = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == 5 and w.isalpha():
                words.append(w)
    return list(dict.fromkeys(words))


def interactive_mode(solver: WordleSolver, allowed_guesses=None):
    print("\nInteractive Wordle solver")
    print("Type 'help' for commands, 'quit' to exit.")
    print("After each guess you'll provide feedback in one of these formats:")
    print("  0,1,2,0,2   or   01202   or   0 1 2 0 2")
    print("Encoding: 0 = absent (gray), 1 = present but wrong spot (yellow), 2 = correct spot (green)\n")

    round_no = 1
    while True:
        print("\nRound", round_no)
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
            print("Commands: 'help', 'quit', 'candidates', 'history', 'pick N' (choose Nth suggestion)")
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

        # no warning if guess isn't in allowed lists; accept any valid 5-letter alpha word
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
            print("Solved in", round_no, "rounds.")
            return True
    print("Failed to solve within", max_rounds, "rounds.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Wordle solver (interactive / auto)")
    parser.add_argument("--words", help="Path to word list file (one 5-letter word per line).", default=None)
    parser.add_argument("--auto", help="Auto-solve for the provided secret word (simulate).", default=None)
    parser.add_argument("--max-rounds", type=int, default=6, help="Max rounds for auto-solve.")
    args = parser.parse_args()

    if args.words:
        try:
            words = load_words_from_file(args.words)
            if not words:
                print("No valid 5-letter words found in file; falling back to builtin list.")
                words = DEFAULT_WORDS
        except Exception as e:
            print("Failed to load words file:", e)
            words = DEFAULT_WORDS
    else:
        words = DEFAULT_WORDS

    # allowed guesses: by default same as candidate list; if you have a bigger allowed list, load and pass it.
    allowed_guesses = words

    solver = WordleSolver(candidates=words)

    if args.auto:
        success = auto_solve(args.auto, solver, allowed_guesses=allowed_guesses, max_rounds=args.max_rounds)
        if not success:
            print("Candidates left (top 20):", solver.candidates[:20])
    else:
        interactive_mode(solver, allowed_guesses=allowed_guesses)


if __name__ == "__main__":
    main()
