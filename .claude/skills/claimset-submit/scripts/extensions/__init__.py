"""Extensions of the claimset format (spec §2.7, rule 80, warning 33).

An extension adds fields, in blocks of its own (`ext.<name>`) on the objects it concerns; the core never depends on
them. Rule 80 itself (where blocks may appear, how they are declared) is a core rule, in validate.py; this
package is the registry of extensions and holds their own rules. Each version of an extension is a module here,
`<name>_v<major>_<minor>.py`, listed in KNOWN. A released version's module is never edited (spec §2.6): a new
version of an extension is a new module.

A module defines:
    NAME, VERSION        the extension and the version it checks
    OBJECTS              the object types its blocks may appear on (validate.EXT_OBJECT_TYPES)
    check(blocks, doc, r)   its own rules, on [(where, object type, object, block)], reported as `<name>.<n>`
                            (errors) and `<name>.w<n>` (warnings)
"""

import importlib

#: Registered extensions: name -> {version: module}. Mirrors spec/extensions/README.md.
KNOWN = {
    "linguistics": {"0.1": "linguistics_v0_1"},
    "qualitative": {"0.1": "qualitative_v0_1"},
}


def module(name, version):
    """The module that checks `version` of extension `name`, or None if this package does not know it."""
    mod = KNOWN.get(name, {}).get(version)
    return importlib.import_module(f"{__name__}.{mod}") if mod else None


def latest(name):
    """The latest version of an extension this package knows, or None."""
    versions = KNOWN.get(name, {})
    return max(versions, key=lambda v: tuple(int(x) for x in v.split("."))) if versions else None
