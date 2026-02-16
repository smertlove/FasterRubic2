
# соответствия 1в1 ud и opencorpora
UD2OPENCORPORA_POS = {
    'NOUN': {'NOUN',},
    'ADJ': {'ADJF', 'ADJS',},
    'ADV': {'ADVB', 'COMP',},
    'VERB': {'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND',},
    'NUM': {'NUMR',},
    'ADP': {'PREP',},
    'PART': {'PRCL',},
    'INTJ': {'INTJ',},
    'PRED': {'PRED',},
    'PRON': {'NPRO',},
    'CCONJ': {'CONJ',},
    'SCONJ': {'CONJ',},
}

NON_PRODUCTIVE_GRAMMEMES = {
    'NUMR',
    'NPRO',
    'PRED',
    'PREP',
    'CONJ',
    'PRCL',
    'INTJ',
    'Apro',
}