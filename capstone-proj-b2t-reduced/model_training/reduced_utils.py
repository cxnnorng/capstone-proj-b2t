import torch
import re

#----------PHONEME REDUCTION UTILS-----------
#these will be used in the data retrieval file to get change the phonemes to their new classes !

OLD_LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ', #CONSIDER EDITING HERE?
]

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AW',
    'AY', 'B',  'CH', 'D',
    'EH', 'EY', 'F', 'G',
    'HH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UW', 'V',
    'W', 'Y', 
    ' | ', 
]
#- **ZH → SH**
# - **DH → TH**
# - **AO → AA**
# - **ER → AH**
# - **Z → S**
# - **UH → UW**
# - **IH → IY**


def reduce_phonemes(sentence):
    sentence = re.sub(r'\bZH\b', 'SH', sentence)
    sentence = re.sub(r'\bDH\b', 'TH', sentence)
    sentence = re.sub(r'\bAO\b', 'AA', sentence)
    sentence = re.sub(r'\bER\b', 'AH', sentence)
    sentence = re.sub(r'\bUH\b', 'UW', sentence)
    sentence = re.sub(r'\bIH\b', 'IY', sentence)
    sentence = re.sub(r'\bZ\b',  'S',  sentence)  
    return sentence


PHONEME_TO_LOGIT = {p: i for i, p in enumerate(LOGIT_TO_PHONEME)}

PHONEME_REDUCTION = {
    'AO': 'AA',
    'ER': 'AH',
    'DH': 'TH',
    'IH': 'IY',
    'UH': 'UW',
    'Z':  'S',
    'ZH': 'SH',
}

def build_index_remap():
    remap = []
    for old_phoneme in OLD_LOGIT_TO_PHONEME:
        canonical = PHONEME_REDUCTION.get(old_phoneme, old_phoneme)
        remap.append(PHONEME_TO_LOGIT[canonical])
    return torch.tensor(remap, dtype=torch.long)

INDEX_REMAP = build_index_remap()


# ------------------------------------------