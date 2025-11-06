/*
 * Wordle solver (interactive) - C++ version (optimized)
 * 
 * Converted from Python for maximum performance.
 * Compile: g++ -std=c++17 -O3 -o wordle_solver wordle_solver.cpp
 * Or with Clang: clang++ -std=c++17 -O3 -o wordle_solver wordle_solver.cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <list>

using namespace std;

// Configuration constants
const string DEFAULT_WORD_FILE = "valid-wordle-words.txt";
const size_t GET_FEEDBACK_CACHE_SIZE = 200000;
const size_t HEURISTIC_TOP_K = 400;
const double CANDIDATE_BONUS = 1e-3;

// Pre-computed best starting words
const vector<string> BEST_STARTING_WORDS = {
    "salet", "roate", "raise", "crane", "slate", "crate", "trace", "carte"
};

// Fallback word list
const vector<string> FALLBACK_WORDS = {
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
};

// Feedback type: 0=absent, 1=present wrong position, 2=correct position
using Feedback = array<int, 5>;

// Hash function for Feedback
struct FeedbackHash {
    size_t operator()(const Feedback& fb) const {
        size_t hash = 0;
        for (int i = 0; i < 5; i++) {
            hash = hash * 3 + fb[i];
        }
        return hash;
    }
};

// Hash function for pair<string, string>
struct PairHash {
    size_t operator()(const pair<string, string>& p) const {
        return hash<string>{}(p.first) ^ (hash<string>{}(p.second) << 1);
    }
};

// LRU Cache implementation for get_feedback
template<typename K, typename V>
class LRUCache {
private:
    size_t capacity;
    list<pair<K, V>> items;
    unordered_map<K, typename list<pair<K, V>>::iterator> cache;

public:
    LRUCache(size_t cap) : capacity(cap) {}

    bool get(const K& key, V& value) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            return false;
        }
        items.splice(items.begin(), items, it->second);
        value = it->second->second;
        return true;
    }

    void put(const K& key, const V& value) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            items.erase(it->second);
            cache.erase(it);
        }
        items.push_front({key, value});
        cache[key] = items.begin();
        if (cache.size() > capacity) {
            auto last = items.back();
            cache.erase(last.first);
            items.pop_back();
        }
    }
};

// Global cache for get_feedback
LRUCache<pair<string, string>, Feedback> feedback_cache(GET_FEEDBACK_CACHE_SIZE);

// Get feedback for guess vs solution
Feedback get_feedback(const string& guess, const string& solution) {
    auto key = make_pair(guess, solution);
    Feedback cached;
    if (feedback_cache.get(key, cached)) {
        return cached;
    }

    Feedback feedback = {0, 0, 0, 0, 0};
    array<char, 5> solution_chars;
    for (int i = 0; i < 5; i++) {
        solution_chars[i] = solution[i];
    }

    // Mark greens
    for (int i = 0; i < 5; i++) {
        if (guess[i] == solution[i]) {
            feedback[i] = 2;
            solution_chars[i] = '\0';
        }
    }

    // Count remaining characters
    unordered_map<char, int> remaining;
    for (char ch : solution_chars) {
        if (ch != '\0') {
            remaining[ch]++;
        }
    }

    // Mark yellows
    for (int i = 0; i < 5; i++) {
        if (feedback[i] == 2) continue;
        char ch = guess[i];
        if (remaining[ch] > 0) {
            feedback[i] = 1;
            remaining[ch]--;
        }
    }

    feedback_cache.put(key, feedback);
    return feedback;
}

// Convert feedback to string key
string feedback_to_key(const Feedback& fb) {
    string s;
    for (int val : fb) {
        s += to_string(val);
    }
    return s;
}

// Convert string to lowercase
string to_lower(string s) {
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

// Trim whitespace
string trim(const string& s) {
    auto start = s.begin();
    while (start != s.end() && isspace(*start)) start++;
    auto end = s.end();
    do { end--; } while (distance(start, end) > 0 && isspace(*end));
    return string(start, end + 1);
}

class WordleSolver {
private:
    vector<string> candidates;
    unordered_set<string> candidates_set;
    vector<pair<string, Feedback>> history;
    size_t initial_size;
    unordered_map<string, pair<double, int>> entropy_cache;

    pair<double, int> compute_entropy(const string& guess, const vector<string>& cands) {
        if (entropy_cache.count(guess)) {
            return entropy_cache[guess];
        }

        unordered_map<string, int> counts;
        for (const auto& sol : cands) {
            string key = feedback_to_key(get_feedback(guess, sol));
            counts[key]++;
        }

        double total = cands.size();
        double entropy = 0.0;
        for (const auto& [key, cnt] : counts) {
            double p = cnt / total;
            entropy -= p * log2(p);
        }

        auto result = make_pair(entropy, (int)counts.size());
        entropy_cache[guess] = result;
        return result;
    }

    double heuristic_score(const string& guess, const vector<string>& cands) {
        unordered_map<char, int> freq;
        for (const auto& w : cands) {
            set<char> unique_chars(w.begin(), w.end());
            for (char ch : unique_chars) {
                freq[ch]++;
            }
        }
        
        set<char> guess_chars(guess.begin(), guess.end());
        double score = 0.0;
        for (char ch : guess_chars) {
            score += freq[ch];
        }
        return score;
    }

    vector<pair<string, double>> get_best_starting_words(
        const vector<string>& allowed_guesses, int top_n) {
        
        unordered_set<string> allowed_set(allowed_guesses.begin(), allowed_guesses.end());
        vector<pair<string, double>> available;
        
        for (const auto& w : BEST_STARTING_WORDS) {
            if (allowed_set.count(w)) {
                available.push_back({w, 5.0});
                if (available.size() >= (size_t)top_n) break;
            }
        }
        
        if (available.size() < (size_t)top_n) {
            vector<string> extras = {"adieu", "audio", "ouija", "arose", "irate", 
                                    "stare", "tears", "store"};
            for (const auto& w : extras) {
                if (allowed_set.count(w)) {
                    bool already_added = false;
                    for (const auto& [word, score] : available) {
                        if (word == w) {
                            already_added = true;
                            break;
                        }
                    }
                    if (!already_added) {
                        available.push_back({w, 5.0});
                        if (available.size() >= (size_t)top_n) break;
                    }
                }
            }
        }
        
        return available;
    }

public:
    WordleSolver(const vector<string>& words) {
        unordered_set<string> seen;
        for (const auto& w : words) {
            string lower = to_lower(w);
            if (lower.length() == 5 && all_of(lower.begin(), lower.end(), ::isalpha)) {
                if (!seen.count(lower)) {
                    candidates.push_back(lower);
                    seen.insert(lower);
                }
            }
        }
        candidates_set = seen;
        initial_size = candidates.size();
    }

    size_t filter_candidates(const string& guess, const Feedback& feedback) {
        vector<string> new_candidates;
        for (const auto& w : candidates) {
            if (get_feedback(guess, w) == feedback) {
                new_candidates.push_back(w);
            }
        }
        size_t removed = candidates.size() - new_candidates.size();
        candidates = new_candidates;
        candidates_set = unordered_set<string>(candidates.begin(), candidates.end());
        entropy_cache.clear();
        return removed;
    }

    void add_history(const string& guess, const Feedback& feedback) {
        history.push_back({guess, feedback});
        filter_candidates(guess, feedback);
    }

    vector<pair<string, double>> suggest_next(const vector<string>* allowed_guesses_ptr, int top_n) {
        const vector<string>& allowed_guesses = allowed_guesses_ptr ? *allowed_guesses_ptr : candidates;

        if (candidates.empty()) {
            return {};
        }

        if (candidates.size() == 1) {
            return {{candidates[0], INFINITY}};
        }

        if (candidates.size() <= 2) {
            vector<pair<string, double>> result;
            for (size_t i = 0; i < min(candidates.size(), (size_t)top_n); i++) {
                result.push_back({candidates[i], 1.0});
            }
            return result;
        }

        // Starting guess optimization
        if (candidates.size() == initial_size && history.empty()) {
            return get_best_starting_words(allowed_guesses, top_n);
        }

        vector<string> pool = allowed_guesses;

        // Reduce pool for large guess lists
        if (pool.size() > 2000 && candidates.size() > 50) {
            vector<string> priority_pool, other_pool;
            for (const auto& w : pool) {
                if (candidates_set.count(w)) {
                    priority_pool.push_back(w);
                } else {
                    other_pool.push_back(w);
                    if (other_pool.size() >= 800) break;
                }
            }
            pool = priority_pool;
            pool.insert(pool.end(), other_pool.begin(), other_pool.end());
        } else if (pool.size() > 5000) {
            vector<string> filtered;
            for (const auto& w : pool) {
                if (candidates_set.count(w)) {
                    filtered.push_back(w);
                    if (filtered.size() >= 1500) break;
                }
            }
            pool = filtered;
        }

        // Use heuristic prefilter for large pools
        vector<string> compute_pool = pool;
        if (pool.size() > HEURISTIC_TOP_K) {
            vector<pair<string, double>> scored;
            for (const auto& w : pool) {
                scored.push_back({w, heuristic_score(w, candidates)});
            }
            sort(scored.begin(), scored.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
            
            compute_pool.clear();
            size_t top_k = min(HEURISTIC_TOP_K, scored.size());
            for (size_t i = 0; i < top_k; i++) {
                compute_pool.push_back(scored[i].first);
            }
        }

        // Compute entropy for selected words
        vector<tuple<string, double, int>> results;
        for (const auto& guess : compute_pool) {
            auto [entropy, partition_count] = compute_entropy(guess, candidates);
            double bonus = candidates_set.count(guess) ? CANDIDATE_BONUS : 0.0;
            double score = entropy + bonus;
            results.push_back({guess, score, partition_count});
        }

        // Add some candidate words if we filtered
        if (compute_pool.size() < pool.size()) {
            int added = 0;
            for (const auto& w : candidates) {
                if (find(compute_pool.begin(), compute_pool.end(), w) == compute_pool.end()) {
                    auto [entropy, partition_count] = compute_entropy(w, candidates);
                    double score = entropy + CANDIDATE_BONUS;
                    results.push_back({w, score, partition_count});
                    if (++added >= 50) break;
                }
            }
        }

        // Sort by score then partition count
        sort(results.begin(), results.end(),
             [](const auto& a, const auto& b) {
                 if (get<1>(a) != get<1>(b)) return get<1>(a) > get<1>(b);
                 return get<2>(a) > get<2>(b);
             });

        vector<pair<string, double>> output;
        for (size_t i = 0; i < min((size_t)top_n, results.size()); i++) {
            output.push_back({get<0>(results[i]), get<1>(results[i])});
        }
        return output;
    }

    const vector<string>& get_candidates() const { return candidates; }
    const vector<pair<string, Feedback>>& get_history() const { return history; }
    size_t get_initial_size() const { return initial_size; }
};

// Parse feedback input
Feedback parse_feedback_input(const string& input) {
    string s = trim(input);
    vector<int> fb;
    
    if (s.find(',') != string::npos || s.find(' ') != string::npos) {
        // Parse comma or space separated
        stringstream ss(s);
        string token;
        while (ss >> token) {
            if (token.find(',') != string::npos) {
                size_t pos = 0;
                while ((pos = token.find(',')) != string::npos) {
                    string part = token.substr(0, pos);
                    if (!part.empty()) fb.push_back(stoi(part));
                    token.erase(0, pos + 1);
                }
                if (!token.empty()) fb.push_back(stoi(token));
            } else {
                fb.push_back(stoi(token));
            }
        }
    } else {
        // Parse 5-character compact form
        if (s.length() != 5) {
            throw runtime_error("Feedback must be 5 characters");
        }
        for (char ch : s) {
            if (ch == '0') fb.push_back(0);
            else if (ch == '1') fb.push_back(1);
            else if (ch == '2') fb.push_back(2);
            else throw runtime_error("Unknown feedback char: " + string(1, ch));
        }
    }
    
    if (fb.size() != 5) {
        throw runtime_error("Feedback must have 5 values");
    }
    
    Feedback result;
    copy(fb.begin(), fb.end(), result.begin());
    return result;
}

// Load words from file
vector<string> load_words_from_file(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + path);
    }
    
    vector<string> words;
    unordered_set<string> seen;
    string line;
    
    while (getline(file, line)) {
        string w = trim(to_lower(line));
        if (w.length() == 5 && all_of(w.begin(), w.end(), ::isalpha) && !seen.count(w)) {
            words.push_back(w);
            seen.insert(w);
        }
    }
    
    return words;
}

// Interactive mode
void interactive_mode(WordleSolver& solver, const vector<string>& allowed_guesses) {
    cout << "\nInteractive Wordle solver\n";
    cout << "Type 'help' for commands, 'quit' to exit.\n";
    cout << "After each guess you'll provide feedback in one of these formats:\n";
    cout << "  0,1,2,0,2   or   01202   or   0 1 2 0 2\n";
    cout << "Encoding: 0 = absent (gray), 1 = present but wrong spot (yellow), 2 = correct spot (green)\n\n";

    int round_no = 1;
    while (true) {
        cout << "\nRound " << round_no << "\n";
        cout << "Candidates remaining: " << solver.get_candidates().size() << "\n";

        if (round_no == 1 && solver.get_candidates().size() > 1000) {
            cout << "(Using pre-computed optimal starting words for speed...)\n";
        }

        auto top = solver.suggest_next(&allowed_guesses, 5);
        if (top.empty()) {
            cout << "No candidates left. Something's inconsistent with the feedback provided.\n";
            return;
        }

        cout << "Top suggestions (word : expected-info score):\n";
        for (const auto& [w, sc] : top) {
            cout << "  " << w << "      " << fixed << sc << "\n";
        }

        if (solver.get_candidates().size() <= 50) {
            cout << "\nCurrent candidates: ";
            for (size_t i = 0; i < solver.get_candidates().size(); i++) {
                if (i > 0) cout << ", ";
                cout << solver.get_candidates()[i];
            }
            cout << "\n";
        }

        string guess;
        cout << "Enter your guess (or press Enter to pick the top suggestion): ";
        getline(cin, guess);
        guess = trim(to_lower(guess));

        if (guess.empty()) {
            guess = top[0].first;
            cout << "Picking: " << guess << "\n";
        }
        if (guess == "help") {
            cout << "Commands: 'help', 'quit', 'candidates', 'history'\n";
            continue;
        }
        if (guess == "quit") {
            return;
        }
        if (guess == "candidates") {
            for (const auto& w : solver.get_candidates()) {
                cout << w << " ";
            }
            cout << "\n";
            continue;
        }
        if (guess == "history") {
            for (const auto& [g, fb] : solver.get_history()) {
                cout << g << " -> " << feedback_to_key(fb) << "\n";
            }
            continue;
        }

        if (guess.length() != 5 || !all_of(guess.begin(), guess.end(), ::isalpha)) {
            cout << "Guess must be a 5-letter word.\n";
            continue;
        }

        string fb_in;
        cout << "Enter feedback for that guess (e.g. 0,1,2,0,2): ";
        getline(cin, fb_in);

        Feedback fb;
        try {
            fb = parse_feedback_input(fb_in);
        } catch (const exception& e) {
            cout << "Could not parse feedback: " << e.what() << "\n";
            continue;
        }

        solver.add_history(guess, fb);
        cout << "Pruned words; " << solver.get_candidates().size() << " candidates remain.\n";

        if (solver.get_candidates().size() == 1) {
            cout << "Solved! The word is: " << solver.get_candidates()[0] << "\n";
            return;
        }
        if (all_of(fb.begin(), fb.end(), [](int x) { return x == 2; })) {
            cout << "All green — solved!\n";
            return;
        }
        round_no++;
    }
}

// Auto-solve simulation
bool auto_solve(const string& secret, WordleSolver& solver, 
                const vector<string>& allowed_guesses, int max_rounds) {
    string secret_lower = to_lower(secret);
    if (secret_lower.length() != 5 || !all_of(secret_lower.begin(), secret_lower.end(), ::isalpha)) {
        throw runtime_error("Secret must be a 5-letter word");
    }

    cout << "Auto-solve simulation for secret: " << secret_lower << "\n";
    
    for (int round_no = 1; round_no <= max_rounds; round_no++) {
        auto suggestion = solver.suggest_next(&allowed_guesses, 1);
        if (suggestion.empty()) {
            cout << "No suggestions available; failed.\n";
            return false;
        }
        
        string guess = suggestion[0].first;
        Feedback fb = get_feedback(guess, secret_lower);
        cout << "Round " << round_no << ": guess=" << guess 
             << "  feedback=" << feedback_to_key(fb) << "\n";
        
        solver.add_history(guess, fb);
        
        if (all_of(fb.begin(), fb.end(), [](int x) { return x == 2; })) {
            cout << "Solved in " << round_no << " rounds.\n";
            return true;
        }
    }
    
    cout << "Failed to solve within " << max_rounds << " rounds.\n";
    return false;
}

int main(int argc, char* argv[]) {
    string word_file;
    string auto_word;
    int max_rounds = 6;

    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--words" && i + 1 < argc) {
            word_file = argv[++i];
        } else if (arg == "--auto" && i + 1 < argc) {
            auto_word = argv[++i];
        } else if (arg == "--max-rounds" && i + 1 < argc) {
            max_rounds = stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --words FILE       Path to word list file\n";
            cout << "  --auto WORD        Auto-solve for given word\n";
            cout << "  --max-rounds N     Max rounds for auto-solve (default: 6)\n";
            return 0;
        }
    }

    // Load words
    vector<string> words;
    
    if (!word_file.empty()) {
        try {
            words = load_words_from_file(word_file);
            if (words.empty()) {
                cout << "No valid 5-letter words found in " << word_file << "; using fallback list.\n";
                words = FALLBACK_WORDS;
            } else {
                cout << "Loaded " << words.size() << " words from " << word_file << "\n";
            }
        } catch (const exception& e) {
            cout << "Failed to load words file '" << word_file << "': " << e.what() << "\n";
            cout << "Using fallback word list.\n";
            words = FALLBACK_WORDS;
        }
    } else {
        ifstream test_file(DEFAULT_WORD_FILE);
        if (test_file.good()) {
            try {
                words = load_words_from_file(DEFAULT_WORD_FILE);
                cout << "Loaded " << words.size() << " words from " << DEFAULT_WORD_FILE << "\n";
            } catch (...) {
                cout << "Word file '" << DEFAULT_WORD_FILE << "' not found; using fallback list.\n";
                words = FALLBACK_WORDS;
            }
        } else {
            cout << "Word file '" << DEFAULT_WORD_FILE << "' not found; using fallback list.\n";
            words = FALLBACK_WORDS;
        }
    }

    cout << "Initializing solver with " << words.size() << " candidate words...\n";
    WordleSolver solver(words);

    if (!auto_word.empty()) {
        bool success = auto_solve(auto_word, solver, words, max_rounds);
        if (!success) {
            cout << "Candidates left (top 20): ";
            for (size_t i = 0; i < min((size_t)20, solver.get_candidates().size()); i++) {
                if (i > 0) cout << ", ";
                cout << solver.get_candidates()[i];
            }
            cout << "\n";
        }
    } else {
        interactive_mode(solver, words);
    }

    return 0;
}