# Claimset extension: linguistics — v0.1 (draft)

Status: working draft · Date: 2026-11-06 · Core: claimset v0.38 and later · Name in files: `linguistics`

Linguistic examples are observations whose value is an excerpt (claimset §3.4): words of a language, as the source gives them, with a `language` tag and an English `translation` in the core. This extension adds what linguistics needs besides: the interlinear gloss, the example's segmentation into morphemes, and the acceptability judgement the paper gives it.

## 1. Where

A `linguistics` block appears only on a **result entry whose value is an excerpt** (`"ext": {"linguistics": {…}}` on the result). Tables do not carry it: a table of examples is recorded as result entries, one per example.

## 2. Fields

| Field | Obligation | Type | Notes |
|---|---|---|---|
| `gloss` | required | string | The interlinear gloss, word by word and morpheme by morpheme, after the Leipzig Glossing Rules: one gloss word per word of the example, separated by spaces; within a word, morphemes separated by hyphens as in the segmentation; grammatical categories in capitals (`1SG`, `PST`, `ACC`) |
| `segmented` | recommended | string | The example split into morphemes with hyphens, word by word, as the gloss aligns with it (line 1 of an interlinear example): `ni-ta-ku-penda`. The excerpt's `text` keeps the words as the source gives them |
| `judgement` | optional | enum | The acceptability the paper gives the example, as the conventional mark: `*` ungrammatical · `?` marginal · `??` strongly marginal · `#` pragmatically or semantically anomalous · `%` acceptable to some speakers only. Absent: the example is given as acceptable |

```json
{"id": "R4", "variable": "D3", "conditions": {"D1": "D7"},
 "value": {"text": "nitakupenda", "locator": "fieldnotes 2019, p. 14", "language": "sw",
           "translation": "I will love you"},
 "ext": {"linguistics": {"segmented": "ni-ta-ku-penda", "gloss": "1SG-FUT-2SG.OBJ-love"}}}
```

The variety the example belongs to is a definition of the paper (a `population` or `setting` naming the language or dialect), with its Glottolog code in `identifiers` (`glottolog:swah1253`), and named in the result's conditions.

## 3. Rules

linguistics.1. A `linguistics` block is on a result entry whose value is an excerpt.
linguistics.2. The block has only `gloss`, `segmented` and `judgement`; `gloss` is a non-empty string; `segmented`, if present, is a non-empty string.
linguistics.3. When `segmented` is given, it has as many words (separated by spaces) as `gloss`, and each word of it as many morphemes (separated by hyphens) as the word of `gloss` it aligns with (Leipzig rules 1 and 2).
linguistics.4. `judgement`, if present, is one of `*`, `?`, `??`, `#`, `%`.

## 4. Warnings

linguistics.w1. No `segmented`, and the excerpt's text and the gloss have different numbers of words: give the segmentation the gloss aligns with.
