"""The linguistics extension, version 0.1 (spec/extensions/linguistics-v0.1.md). Never edit once released: a new
version of the extension is a new module."""

NAME = "linguistics"
VERSION = "0.1"
OBJECTS = {"result"}

FIELDS = {"gloss", "segmented", "judgement"}
JUDGEMENTS = {"*", "?", "??", "#", "%"}


def _words(text):
    return text.split()


def check(blocks, doc, r):
    for where, _typ, obj, block in blocks:
        value = obj.get("value")
        if not (isinstance(value, dict) and "text" in value):
            r.err("linguistics.1", where, "a linguistics block is on a result whose value is an excerpt (a "
                                          "linguistic example)")
            continue
        extra = set(block) - FIELDS
        if extra:
            r.err("linguistics.2", where, f"only gloss, segmented and judgement; not {', '.join(sorted(extra))}")
        gloss = block.get("gloss")
        if not (isinstance(gloss, str) and gloss.strip()):
            r.err("linguistics.2", f"{where}.gloss", "required: the interlinear gloss, word by word (Leipzig "
                                                    "Glossing Rules)")
            continue
        seg = block.get("segmented")
        if seg is not None and not (isinstance(seg, str) and seg.strip()):
            r.err("linguistics.2", f"{where}.segmented", "the example split into morphemes, as text")
            seg = None
        if seg is not None:
            sw, gw = _words(seg), _words(gloss)
            if len(sw) != len(gw):
                r.err("linguistics.3", where, f"segmented has {len(sw)} words and gloss {len(gw)}: each word of "
                                              f"the example has one word of gloss (Leipzig rule 1)")
            else:
                for i, (s, g) in enumerate(zip(sw, gw), 1):
                    if s.count("-") != g.count("-"):
                        r.err("linguistics.3", where, f"word {i}: '{s}' has {s.count('-') + 1} morphemes and its "
                                                      f"gloss '{g}' {g.count('-') + 1}: each morpheme is glossed "
                                                      f"(Leipzig rule 2)")
        elif isinstance(value.get("text"), str) and len(_words(value["text"])) != len(_words(gloss)):
            r.warn("linguistics.w1", where, f"the excerpt has {len(_words(value['text']))} words and the gloss "
                                            f"{len(_words(gloss))}: give the example split into morphemes in "
                                            f"segmented, aligned with the gloss word by word")
        if "judgement" in block and block["judgement"] not in JUDGEMENTS:
            r.err("linguistics.4", f"{where}.judgement", f"one of {' '.join(sorted(JUDGEMENTS))}: the "
                                                        f"acceptability mark of the example")
