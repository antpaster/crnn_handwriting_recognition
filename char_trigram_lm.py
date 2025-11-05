import os
import math
from collections import Counter, defaultdict
import heapq

class CharTrigramLM:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.vocab = set()
        self.log_probs_tri = {}
        self.log_probs_bi = {}
        self.log_probs_uni = {}
        self.total_unigrams = 0

    def train(self, texts):
        """
        texts: iterable of strings (your training labels).
        """
        for text in texts:
            # Add start tokens to give context at beginning
            seq = ["<s>", "<s>"] + list(text) + ["</s>"]
            for c in seq:
                self.unigram_counts[c] += 1
                self.vocab.add(c)

            for i in range(1, len(seq)):
                self.bigram_counts[(seq[i-1], seq[i])] += 1
            for i in range(2, len(seq)):
                self.trigram_counts[(seq[i-2], seq[i-1], seq[i])] += 1

        self.total_unigrams = sum(self.unigram_counts.values())
        self._compute_log_probs()

    def _compute_log_probs(self):
        V = len(self.vocab)
        a = self.alpha

        # Unigram P(c)
        for c in self.vocab:
            count = self.unigram_counts[c]
            num = count + a
            den = self.total_unigrams + a * V
            self.log_probs_uni[c] = math.log(num / den)

        # Bigram P(c2|c1)
        bigram_den = defaultdict(lambda: 0)
        for (c1, c2), cnt in self.bigram_counts.items():
            bigram_den[c1] += cnt

        for (c1, c2), cnt in self.bigram_counts.items():
            num = cnt + a
            den = bigram_den[c1] + a * V
            self.log_probs_bi[(c1, c2)] = math.log(num / den)

        # Trigram P(c3|c1,c2)
        trigram_den = defaultdict(lambda: 0)
        for (c1, c2, c3), cnt in self.trigram_counts.items():
            trigram_den[(c1, c2)] += cnt

        for (c1, c2, c3), cnt in self.trigram_counts.items():
            num = cnt + a
            den = trigram_den[(c1, c2)] + a * V
            self.log_probs_tri[(c1, c2, c3)] = math.log(num / den)

    def log_prob_next(self, prev2, prev1, c):
        """
        Log P(c | prev2, prev1), with backoff to bigram/unigram.
        prev2, prev1, c are characters (including ' ').
        """
        # Trigram
        key3 = (prev2, prev1, c)
        if key3 in self.log_probs_tri:
            return self.log_probs_tri[key3]

        # Bigram
        key2 = (prev1, c)
        if key2 in self.log_probs_bi:
            return self.log_probs_bi[key2]

        # Unigram
        if c in self.log_probs_uni:
            return self.log_probs_uni[c]

        # Unknown char: small penalty
        return -20.0

    def sentence_log_prob(self, text):
        """
        Total LM log-prob of a sequence (for re-ranking, if needed).
        """
        seq = ["<s>", "<s>"] + list(text) + ["</s>"]
        lp = 0.0
        for i in range(2, len(seq)):
            lp += self.log_prob_next(seq[i-2], seq[i-1], seq[i])
        return lp
    
def load_train_texts_from_labels(labels_file="labels.txt"):
    texts = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            path, text = line.split("\t", 1)
            texts.append(text)
    return texts

def ctc_beam_search_with_char_lm(
    log_probs,     # (T, C) torch.Tensor (log-softmaxed)
    codec,         # your CTCCodec
    lm: CharTrigramLM,
    beam_size=10,
    lm_weight=0.5,
    blank_idx=0,
):
    """
    Single-utterance beam search with char trigram LM.
    Returns best decoded string.
    """
    T, C = log_probs.shape
    log_probs = log_probs.cpu().numpy()

    # Each beam: (text_str, ctc_logp, lm_logp, prev2, prev1, last_token_idx)
    beams = [("", 0.0, 0.0, "<s>", "<s>", blank_idx)]

    for t in range(T):
        frame = log_probs[t]  # (C,)
        new_beams = {}

        for (text, ctc_lp, lm_lp, p2, p1, last_tok) in beams:
            # 1) Extend with blank
            lp_blank = ctc_lp + frame[blank_idx]
            key = (text, p2, p1, blank_idx)
            # Keep best ctc-only + lm (lm doesn't change for blank)
            if key not in new_beams or lp_blank + lm_lp > new_beams[key][0]:
                new_beams[key] = (lp_blank + lm_lp, lm_lp)

            # 2) Extend with characters
            for c_idx in range(1, C):
                char = codec.idx2char.get(c_idx, "")
                if char == "":
                    continue

                lp_char = ctc_lp + frame[c_idx]

                # LM contribution: only when adding a non-blank char
                lm_lp_new = lm_lp + lm.log_prob_next(p2, p1, char)

                # CTC rule: collapse repeated chars only when separated by blank.
                # Here we just append char; CTC collapsing happens after.
                new_text = text + char
                new_p2, new_p1 = p1, char

                key = (new_text, new_p2, new_p1, c_idx)
                score = lp_char + lm_weight * lm_lp_new

                if key not in new_beams or score > new_beams[key][0]:
                    new_beams[key] = (score, lm_lp_new)

        # Keep top beam_size beams
        beams = []
        for (text, p2, p1, last_tok), (score, lm_lp) in new_beams.items():
            beams.append((text, score - lm_weight * lm_lp, lm_lp, p2, p1, last_tok))

        beams.sort(key=lambda x: x[1] + lm_weight * x[2], reverse=True)
        beams = beams[:beam_size]

    # Select best final beam by combined score
    best = max(beams, key=lambda x: x[1] + lm_weight * x[2])
    best_text = best[0]

    # Optional: collapse repeated chars like CTC; but since we appended chars only
    # when reading non-blank, and we didn't explicitly keep previous non-blank token,
    # we won't see typical "aa" repeats from CTC here; still, you can:
    collapsed = []
    prev = ""
    for c in best_text:
        if c != prev:
            collapsed.append(c)
        prev = c
    return "".join(collapsed)

if __name__ == "__main__":
    # Example:
    texts = load_train_texts_from_labels(os.path.join("images", "labels.txt"))
    lm = CharTrigramLM(alpha=0.1)
    lm.train(texts)