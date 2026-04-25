from functools import lru_cache
import pymorphy3 as pm


def None_if_error(func):
    def wraps(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception:
            result = None
        return result
    return wraps


def rm_archaic(func):
    def wraps(text: str):
        result = func(text)
        if result is None:
            return result
        result = result.replace("ѣ", "е")
        result = result.replace("і", "и")
        result = result.replace("ï", "и")
        result = result.replace("ї", "и")
        result = result.replace("ѳ", "ф")
        result = result.replace("_", "")
        if result and result[-1] == "ъ":
            result = result[:-1]
        return result
    return wraps


class _Parser:
    morph = pm.MorphAnalyzer()

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

    @classmethod
    @lru_cache(123123123)
    def parse(cls, word: str):
        return cls.morph.parse(word)

    @classmethod
    def try_to_find(cls, word: str, pos: str):

        oc_pos_set = cls.UD2OPENCORPORA_POS.get(pos)
        if oc_pos_set is None:
            return None

        parses = cls.parse(word)
        try:
            matching_parses = [
                p for p in parses 
                if p.tag.POS in oc_pos_set and p.is_known
            ]
        except ValueError:
            return None
        if len(matching_parses) == 1:
            return matching_parses[0].normal_form

        elif len(matching_parses) > 1:
            normal_forms = {p.normal_form for p in matching_parses}

            if len(normal_forms) == 1:
                return matching_parses[0].normal_form

        return None

@None_if_error
def foreign_or_None(s: str):
    if "Foreign:Yes" in s:
        return s.split(maxsplit=1)[0].lower()
    else:
        return None

@None_if_error
@rm_archaic
def sing_nomn_noun_or_None(s: str):
    if "NOUN" in s and "Case:Nom" in s and "Number:Sing" in s:
        return s.split(maxsplit=1)[0].lower()
    else:
        return None

@None_if_error
@rm_archaic
def infn_verb_or_None(s: str):
    if "VERB" in s and "VerbForm:Inf" in s:
        return s.split(maxsplit=1)[0].lower()
    else:
        return None

@None_if_error
def punct_or_None(s: str):
    if "PUNCT" in s:
        return s.split(maxsplit=1)[0].lower()
    else:
        return None

@None_if_error
def sym_or_None(s: str):
    if "SYM" in s:
        return s.split(maxsplit=1)[0].lower()
    else:
        return None

@None_if_error
def num_anum_isalpha_or_None(s: str):
    split = s.split(maxsplit=1)[0].lower()
    if "NUM" in s and not split.isalpha():
        return split
    else:
        return None

@None_if_error
def pymorphy_or_None(s: str):
    word, pos_feats = s.split(maxsplit=1)
    pos = pos_feats.split(maxsplit=1)[0]
    return _Parser.try_to_find(word, pos)
