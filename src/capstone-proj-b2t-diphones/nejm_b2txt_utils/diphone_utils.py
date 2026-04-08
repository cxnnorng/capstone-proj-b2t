"""
Utility functions for diphone-based brain-to-text decoding.

Diphones capture context between consecutive phonemes by representing
each phoneme as a (previous_phoneme, current_phoneme) pair.

Functions:
    build_diphone_vocab       -- scan training sequences, return vocab dicts
    mono_seq_to_diphone_seq   -- convert label sequence at data-load time
    diphone_seq_to_mono_seq   -- convert decoded sequence back for PER calc
    diphone_logits_to_mono_logits -- logsumexp-aggregate at eval time
    save_diphone_vocab        -- persist vocab to JSON
    load_diphone_vocab        -- reload vocab from JSON
"""

import os
import json
import numpy as np

# BLANK token index (index 0 in LOGIT_TO_PHONEME)
BLANK_IDX = 0

# Silence / word-boundary token index (index 40 in LOGIT_TO_PHONEME)
SIL_IDX = 40

# Total number of monophone classes (BLANK + 39 phones + SIL)
N_MONO = 41


# ---------------------------------------------------------------------------
# Vocab building
# ---------------------------------------------------------------------------

def build_diphone_vocab(all_mono_seqs):
    """
    Build a diphone vocabulary from a list of monophone ID sequences.

    Only (prev, curr) pairs that actually appear in training data are included,
    keeping the output layer small.  The first phoneme in each sequence is
    paired with SIL_IDX (40) as its predecessor.

    The BLANK self-diphone (BLANK_IDX, BLANK_IDX) is always assigned ID 0 so
    CTC loss works correctly (blank=0 matches the model's output layer index 0).

    Args:
        all_mono_seqs: list of array-like, each containing int monophone IDs (0-40)

    Returns:
        diphone_to_id (dict): (prev_id, curr_id) -> int
        id_to_diphone (dict): int -> (prev_id, curr_id)
    """
    blank_diphone = (BLANK_IDX, BLANK_IDX)  # self-diphone for CTC blank

    diphone_set = set()
    for seq in all_mono_seqs:
        seq = list(seq)
        for i in range(len(seq)):
            prev = SIL_IDX if i == 0 else int(seq[i - 1])
            curr = int(seq[i])
            diphone_set.add((prev, curr))

    # Ensure blank self-diphone is not in the sorted set (it gets ID 0 separately)
    diphone_set.discard(blank_diphone)

    # Sort for determinism; BLANK self-diphone always gets index 0
    sorted_diphones = sorted(diphone_set)

    diphone_to_id = {blank_diphone: 0}
    id_to_diphone = {0: blank_diphone}
    for idx, diphone in enumerate(sorted_diphones):
        diphone_to_id[diphone] = idx + 1
        id_to_diphone[idx + 1] = diphone

    return diphone_to_id, id_to_diphone


# ---------------------------------------------------------------------------
# Sequence conversion (used in dataset and validation)
# ---------------------------------------------------------------------------

def mono_seq_to_diphone_seq(mono_ids, diphone_to_id):
    """
    Convert a sequence of monophone IDs to diphone IDs.

    Args:
        mono_ids:       array-like of int, monophone IDs (0-40)
        diphone_to_id:  dict mapping (prev_id, curr_id) -> diphone_id

    Returns:
        np.ndarray of int64 diphone IDs, same length as mono_ids
    """
    blank_diphone = (BLANK_IDX, BLANK_IDX)
    mono_ids = list(mono_ids)
    diphone_ids = []
    for i in range(len(mono_ids)):
        prev = SIL_IDX if i == 0 else int(mono_ids[i - 1])
        curr = int(mono_ids[i])
        key = (prev, curr)
        # Fall back to BLANK diphone (0) for any unseen pair (graceful degradation)
        diphone_ids.append(diphone_to_id.get(key, diphone_to_id.get(blank_diphone, 0)))

    return np.array(diphone_ids, dtype=np.int64)


def diphone_seq_to_mono_seq(diphone_ids, id_to_diphone):
    """
    Convert a sequence of diphone IDs back to monophone IDs.

    Each diphone (prev, curr) -> curr phoneme.
    The BLANK self-diphone (0, 0) -> curr=0 (BLANK monophone).

    Args:
        diphone_ids:    array-like of int, diphone IDs
        id_to_diphone:  dict mapping int -> (prev_id, curr_id)

    Returns:
        np.ndarray of int64 monophone IDs
    """
    blank_diphone = (BLANK_IDX, BLANK_IDX)
    mono_ids = []
    for did in diphone_ids:
        _prev, curr = id_to_diphone.get(int(did), blank_diphone)
        mono_ids.append(curr)

    return np.array(mono_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Logit aggregation (used in evaluate_model.py before Redis send)
# ---------------------------------------------------------------------------

def _log_softmax(x):
    """Numerically stable log-softmax along the last axis. No scipy needed."""
    x_max = np.max(x, axis=-1, keepdims=True)
    shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


def diphone_logits_to_mono_logits(diphone_logits, id_to_diphone, n_mono=N_MONO):
    """
    Aggregate (T x N_diphones) raw logits -> (T x n_mono) log-probs via logsumexp.

    Steps:
      1. Apply log-softmax to convert raw diphone logits to log-probabilities.
      2. For each monophone m, accumulate log-probs of all diphones whose
         current (right) phoneme is m using logsumexp.

    Args:
        diphone_logits: np.ndarray of shape (T, N_diphones) — raw logits from model
        id_to_diphone:  dict mapping int -> (prev_id, curr_id) or int -> 'BLANK'
        n_mono:         int, number of monophone output classes (default 41)

    Returns:
        np.ndarray of float32, shape (T, n_mono) — log-probabilities suitable
        for passing to rearrange_speech_logits_pt and then the language model.
    """
    diphone_logits = diphone_logits.astype(np.float32)  # (T, N_diphones)
    N_diphones = diphone_logits.shape[1]

    # Step 1: raw logits -> log-probs (numerically stable, no scipy)
    log_probs = _log_softmax(diphone_logits)  # (T, N_diphones)

    # Step 2: accumulate into monophone bins via logsumexp
    # Initialise to -inf (log(0)) so logaddexp identity works correctly
    mono_log_probs = np.full(
        (diphone_logits.shape[0], n_mono), -np.inf, dtype=np.float32
    )

    for diphone_id, diphone in id_to_diphone.items():
        if diphone_id >= N_diphones:
            continue  # guard against vocab/model size mismatch
        _prev, curr_mono = diphone  # (0,0)->curr=0=BLANK; (p,c)->curr=c

        col = log_probs[:, diphone_id]  # (T,)
        mono_log_probs[:, curr_mono] = np.logaddexp(
            mono_log_probs[:, curr_mono], col
        )

    return mono_log_probs  # (T, n_mono) log-probs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_diphone_vocab(diphone_to_id, id_to_diphone, save_dir):
    """
    Save diphone vocab dicts to <save_dir>/diphone_vocab.json.

    Tuple keys are serialised as "prev_curr" strings; int keys as strings.

    Returns:
        str: full path to the saved JSON file
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'diphone_vocab.json')

    # All keys are (int, int) tuples — serialise as "prev_curr" strings
    serializable_to_id = {
        f'{k[0]}_{k[1]}': v for k, v in diphone_to_id.items()
    }
    serializable_id_to = {
        str(k): [v[0], v[1]] for k, v in id_to_diphone.items()
    }

    with open(path, 'w') as f:
        json.dump(
            {'diphone_to_id': serializable_to_id, 'id_to_diphone': serializable_id_to},
            f,
            indent=2,
        )

    return path


def load_diphone_vocab(vocab_path):
    """
    Load diphone vocab from a JSON file previously written by save_diphone_vocab.

    Returns:
        diphone_to_id (dict): (prev_id, curr_id) -> int  OR  'BLANK' -> 0
        id_to_diphone (dict): int -> (prev_id, curr_id)  OR  int -> 'BLANK'
    """
    with open(vocab_path, 'r') as f:
        data = json.load(f)

    # All keys serialised as "prev_curr" strings — parse back to (int, int) tuples
    diphone_to_id = {}
    for k, v in data['diphone_to_id'].items():
        parts = k.split('_')
        diphone_to_id[(int(parts[0]), int(parts[1]))] = v

    id_to_diphone = {}
    for k, v in data['id_to_diphone'].items():
        id_to_diphone[int(k)] = (v[0], v[1])

    return diphone_to_id, id_to_diphone
