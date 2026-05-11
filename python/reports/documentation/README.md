# Smooth Package Documentation

Super-detailed documentation of the R and Python smooth forecasting package. Designed for agents and developers navigating both implementations and translating features between them.

## Structure

| Folder | Contents |
|--------|----------|
| [r_package/](r_package/) | R package: file index, function registry, ADAM flow, per-model docs |
| [python_package/](python_package/) | Python package: file index, function registry, ADAM flow, per-module docs |
| [cpp_shared/](cpp_shared/) | Shared C++: architecture, headers/sources, API reference |
| [translation/](translation/) | R-to-Python mapping: function index, parameters, data structures, coverage |
| [validation/](validation/) | Scan proof: SOURCES_SCANNED.md, CHANGELOG.md |
| [two_stage_initialization/](two_stage_initialization/) | Two-stage init: bug fixes, implementation notes |

## How to Use (for Agents)

1. **Find where a function lives**: Use `r_package/02_FUNCTION_REGISTRY.md` or `python_package/02_FUNCTION_REGISTRY.md`
2. **Understand flow**: Use `r_package/03_ADAM_FLOW.md` or `python_package/03_ADAM_FLOW.md`
3. **Translate R → Python**: Use `translation/00_TRANSLATION_INDEX.md` and `translation/01_PARAMETER_MAPPING.md`
4. **Check coverage**: Use `translation/03_COVERAGE_MATRIX.md`
5. **Translate a new feature**: Follow `translation/04_TRANSLATION_CHECKLIST.md`

## Anti-Hallucination

- All function/file references are verifiable. See [validation/SOURCES_SCANNED.md](validation/SOURCES_SCANNED.md) for scanned files and grep patterns.
- Uncertain mappings are marked "unverified" or "TODO".
