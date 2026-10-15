"""The low-rank factor of the covariance in the filter: timing (E23).

Claim (INTERMITTENT_LIKELIHOOD.tex, Section 7 "Computation and implementation",
paragraph "Cost", E23): "The package uses the factor when four times the number of its
columns at most is below the number of cells, and the whole covariance otherwise, as for
intermittent demand, where the unobserved periods are many: with lags 24 and 168 (193
cells) and 1 to 48 missing values among 1344 hourly observations, the filter is 8.6 to
1.3 times faster with the factor, with the same results." Target: speed-ups (time with
the whole covariance / time with the factor) from 8.6x down to 1.3x over 1..48 missing
values, with the same likelihood and fitted values.

The two variants. The choice is internal to the C++ core (src/headers/adamCore.h,
adamCore::filter(): state.lowRank = 4 * gaps * (...) <= state.cells.n_elem; with 193
cells the factor is used up to 48 unobserved periods) and has no switch, so the package
is not changed: build_lowrank_variants.sh (next to this script) copies src/ and
python/CMakeLists.txt of the repository to a directory outside it, patches the copy of
that line to SMOOTH_FORCE_LOWRANK (1: always the factor, 0: always the whole
covariance), and builds _adamCore twice with the package's CMakeLists (Release, -O3
-DNDEBUG -flto: the flags of the package's own Python build, which has no
-ffp-contract option; the flags are printed by the build script). Each build also gets
PYBIND11_INTERNALS_KIND="_smooth_<variant>", which only separates the pybind11 type
registries, so that both variants and the package's own _adamCore load in one process
and the timings can be interleaved. Rebuild (about 3 minutes):

    bash python/docs/claimset/scripts/build_lowrank_variants.sh [OUTPUT_DIR]

OUTPUT_DIR (default $LOWRANK_DIR, else smooth-lowrank in the temporary directory:
VARIANT_DIR_DEFAULT below) must be outside the repository; this script reads the same
LOWRANK_DIR and refuses to run if OUTPUT_DIR/variants/{lowrank,full}/_adamCore*.so
are missing, or if they were built from another commit than the one checked out
(warning only when the working tree's src/ is dirty).

Model and data (settings fixed below):
- The paper does not say which hourly data it used, and the speed-up depends on them
  (through the estimated persistence, which sets how many cells the covariance spreads
  over and how fast it decays), so two data-generating processes are run, labelled
  `dgp` in the CSV, both t = 0..1343 (1344 hourly observations, 8 weeks), with
  e_t ~ N(0, SIGMA^2), SIGMA = 2, drawn with np.random.default_rng(SEED):
  * "sine": y_t = 100 + 10 sin(2 pi t/24) + 5 sin(2 pi t/168) + e_t, a fixed
    seasonality (the estimated alpha and gamma_24 are zero, gamma_168 is 0.07: the
    covariance stays on the weekly cells and is sparse);
  * "ets": ETS(A,N,A) itself, y_t = l_{t-1} + s_{t-24} + w_{t-168} + e_t, with
    persistence (alpha, gamma_24, gamma_168) = (0.1, 0.1, 0.1), the level starting at
    100 and the seasonal states at the two sine patterns above (the estimates are near
    0.08: the covariance fills all 193 cells).
- Missing values: one permutation of 0..1343 drawn with the same generator after the
  errors (the same positions for both processes); the k missing positions are its
  first k entries (nested sets, k = 1..48). The positions are written to the CSV
  (column `positions`). Also columns: the first missing position (`first_missing`),
  from which the covariance is non-zero; the number of periods whose one-step variance
  f_t exceeds one (`periods_f_above_1`: the covariance reaches the cells used then)
  and the last of them (`last_f_above_1`).
- Model: ADAM ETS(A,N,A) with lags [24, 168] (level + daily + weekly seasonal states:
  1 + 24 + 168 = 193 cells of the profile, checked at run time), the normal
  distribution, the likelihood loss, initial="backcasting" (the default, two
  iterations). initial="optimal" was not used: ADAM(model="ANA", lags=[24, 168],
  initial="optimal") fails in the Python package (TypeError in creator/initialiser.py,
  "'bool' object is not subscriptable"), with or without missing values.
- Fixed parameters: B (alpha, gamma_24, gamma_168) estimated once by the package on the
  data with the 48 missing values. For each k the model is fitted by the package on the
  data with k missing values (only to obtain its internal dictionaries, built by the
  package's own checker), and everything below is evaluated at the same B.

What is timed, for each k and each variant (lowrank, full, and the package's own
_adamCore, which picks the factor for k <= 48 and serves as a control of the build):
- `filter_pass`: adamCore.filter() with backcast=False, nIterations=1: one forward pass
  in which the filter runs from the first unobserved period on. This is "the filter" of
  the claim. Its inputs are those the package's cost function passes to the filter at
  B (captured by a recording proxy), with only the two backcasting arguments changed.
- `package_call`: adamCore.filter() exactly as the cost function calls it (backcasting:
  backcast=True, nIterations=2, so the backcasting passes, which do not run the filter,
  are included).
- `likelihood`: a whole evaluation of the cost function CF() (filler, bounds checks,
  the filter as in package_call, the scale and the log-likelihood), with the variant's
  adamCore passed as adam_cpp.
Each case is run REPETITIONS times, the variants interleaved within a repetition (the
order rotated between repetitions), each call timed alone with time.perf_counter_ns
(inputs copied before the clock starts, gc disabled); one untimed warm-up call per
variant and case first. Reported: median and IQR (75th - 25th percentile) in
milliseconds, and the speed-up = median(full) / median(lowrank).

Same results: the maximum absolute differences, lowrank against full, of the
log-likelihood (-CF), the fitted values, the errors and the variances f_t of the filter
(over all k), and the same for the package's core against the lowrank variant (expected
0: the same code). Check: -CF(B) with the package's core at k = 48 equals the model's
own log-likelihood.

Threads: OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1, set before numpy
is imported. Printed at run time and written to the CSV: the CPU model, the OS, the
Python, numpy, scipy, smooth and greybox versions, `git rev-parse HEAD` of the
repository and its dirty state, and the commit the variants were built from.

Development runs used fewer repetitions (--repetitions); the stated settings are
REPETITIONS = 25 and k = 1..48 (the defaults). Run (from any directory):

    python/.venv/bin/python python/docs/claimset/scripts/lowrank_timing.py

Output: a table on stdout and results/lowrank_timing.csv next to this script (one row
per k and timed call, settings as columns).
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import importlib.metadata  # noqa: E402
import importlib.util  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smooth import ADAM  # noqa: E402
from smooth.adam_general.core.utils.cost_functions import CF  # noqa: E402

SEED = 2310
N_OBS = 1344
LAGS = [24, 168]
MODEL = "ANA"
DISTRIBUTION = "dnorm"
INITIAL = "backcasting"
DGPS = ("sine", "ets")
SIGMA = 2.0
ETS_PERSISTENCE = (0.1, 0.1, 0.1)
MISSING = list(range(1, 49))
REPETITIONS = 25
CELLS_EXPECTED = 1 + 24 + 168
TARGET_FIRST, TARGET_LAST = 8.6, 1.3
VARIANTS = ("lowrank", "full", "package")
MODES = ("filter_pass", "package_call", "likelihood")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VARIANT_DIR_DEFAULT = str(Path(tempfile.gettempdir()) / "smooth-lowrank")
VARIANT_DIR = Path(os.environ.get("LOWRANK_DIR", VARIANT_DIR_DEFAULT))
CSV = HERE / "results" / "lowrank_timing.csv"


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout.strip()


def load_variants():
    """The two builds of _adamCore, refused when missing or from another commit."""
    modules = {}
    for variant in ("lowrank", "full"):
        found = glob.glob(str(VARIANT_DIR / "variants" / variant / "_adamCore*.so"))
        if not found:
            sys.exit(
                f"Variant '{variant}' not found under {VARIANT_DIR}/variants. Build "
                f"it with: bash {HERE / 'build_lowrank_variants.sh'} {VARIANT_DIR}"
            )
        spec = importlib.util.spec_from_file_location(
            f"variant_{variant}._adamCore", found[0]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[variant] = module
    built = (VARIANT_DIR / "variants" / "commit.txt").read_text().split("\n")[0]
    if built != git("rev-parse", "HEAD"):
        sys.exit(f"The variants were built from {built}: rebuild them.")
    if git("status", "--porcelain", "--", "src"):
        warnings.warn("src/ is dirty: the variants were built from a copy of it")
    return modules, built


def make_data(dgp):
    """The series of a data-generating process and the order of the missing values."""
    rng = np.random.default_rng(SEED)
    t = np.arange(N_OBS)
    daily = 10 * np.sin(2 * np.pi * np.arange(24) / 24)
    weekly = 5 * np.sin(2 * np.pi * np.arange(168) / 168)
    errors = rng.normal(0, SIGMA, N_OBS)
    if dgp == "sine":
        y = 100 + daily[t % 24] + weekly[t % 168] + errors
    else:
        # ETS(A,N,A): y_t = l_{t-1} + s_{t-24} + w_{t-168} + e_t, each state moving
        # by its smoothing parameter times e_t
        alpha, gamma_24, gamma_168 = ETS_PERSISTENCE
        level, y = 100.0, np.empty(N_OBS)
        daily, weekly = daily.copy(), weekly.copy()
        for i in range(N_OBS):
            y[i] = level + daily[i % 24] + weekly[i % 168] + errors[i]
            level += alpha * errors[i]
            daily[i % 24] += gamma_24 * errors[i]
            weekly[i % 168] += gamma_168 * errors[i]
    order = rng.permutation(N_OBS)
    return y, order


def with_missing(y, positions):
    y_k = y.copy()
    y_k[positions] = np.nan
    return y_k


def fit(y_k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ADAM(
            model=MODEL, lags=LAGS, distribution=DISTRIBUTION, initial=INITIAL
        ).fit(y_k)


def cf_arguments(model, B):
    """The arguments of CF() at B for a fitted model, as its OPG covariance makes."""
    return dict(
        B=np.asarray(B, dtype=float),
        model_type_dict=model._model_type,
        components_dict=model._components,
        lags_dict=model._lags_model,
        matrices_dict=model._adam_created_seeded(),
        persistence_checked=model._persistence,
        initials_checked=model._initials,
        arima_checked=model._arima,
        explanatory_checked=model._explanatory,
        phi_dict=model._phi_internal,
        constants_checked=model._constant,
        observations_dict=model._observations,
        profile_dict=model._profile,
        general=model._general,
    )


def make_core(module, model):
    """adamCore of a module with the constructor arguments of the model's own."""
    components = model._components
    core = module.adamCore(
        lags=np.array(model._lags_model["lags_model_all"], dtype=np.uint64),
        E=model._model_type["error_type"],
        T=model._model_type["trend_type"],
        S=model._model_type["season_type"],
        nNonSeasonal=components["components_number_ets_non_seasonal"],
        nSeasonal=components["components_number_ets_seasonal"],
        nETS=components["components_number_ets"],
        nArima=components.get("components_number_arima", 0),
        nXreg=model._explanatory.get("xreg_number", 0),
        nComponents=components["components_number_all"],
        constant=model._constant.get("constant_required", False),
        adamETS=model._general.get("ets") == "adam",
    )
    for field in ("flipConstant", "headLength", "sparseTransition"):
        setattr(core, field, getattr(model._adam_cpp, field))
    return core


class Recorder:
    """The core, recording the arguments of the calls of its filter."""

    def __init__(self, core):
        self.core = core
        self.calls = []

    def filter(self, **kwargs):
        self.calls.append({key: copy(value) for key, value in kwargs.items()})
        return self.core.filter(**kwargs)

    def __getattr__(self, name):
        return getattr(self.core, name)


def copy(value):
    if isinstance(value, np.ndarray):
        return np.array(value, order="F", copy=True)
    return value


def filter_outputs(result):
    return {
        name: np.ravel(getattr(result, name)).copy()
        for name in ("fitted", "errors", "variances")
    }


def time_case(cores, call, repetitions):
    """Times call(core) for each core, interleaved; returns times (ms) and outputs."""
    outputs = {variant: call(core) for variant, core in cores.items()}
    times = {variant: [] for variant in cores}
    names = list(cores)
    gc.disable()
    try:
        for rep in range(repetitions):
            shift = rep % len(names)
            for variant in names[shift:] + names[:shift]:
                prepared = call(cores[variant], prepare_only=True)
                start = time.perf_counter_ns()
                prepared()
                times[variant].append((time.perf_counter_ns() - start) / 1e6)
    finally:
        gc.enable()
    return times, outputs


def filter_call(arguments):
    def call(core, prepare_only=False):
        copies = {key: copy(value) for key, value in arguments.items()}

        def run():
            return core.filter(**copies)

        return run if prepare_only else filter_outputs(run())

    return call


def likelihood_call(arguments):
    def call(core, prepare_only=False):
        def run():
            return CF(**arguments, adam_cpp=core)

        return run if prepare_only else run()

    return call


def summary(times):
    q25, q50, q75 = np.percentile(times, [25, 50, 75])
    return q50, q75 - q25


def max_difference(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def environment():
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "threads": os.environ["OMP_NUM_THREADS"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "smooth": importlib.metadata.version("smooth"),
        "greybox": importlib.metadata.version("greybox"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--missing", type=str, default=None, help="e.g. 1,24,48")
    args = parser.parse_args()
    missing = (
        MISSING if args.missing is None else [int(k) for k in args.missing.split(",")]
    )
    started = time.time()
    modules, built = load_variants()
    env = environment()
    for key, value in env.items():
        print(f"{key:>10}: {value}")
    print(f"  variants: {VARIANT_DIR / 'variants'} (built from {built})")
    print(f"repetitions: {args.repetitions}, missing: {missing[0]}..{missing[-1]}")
    if args.repetitions != REPETITIONS or missing != MISSING:
        print("NOTE: development settings, not the stated ones")

    rows = []
    for dgp in DGPS:
        rows += run_dgp(dgp, modules, missing, args.repetitions)
    table = pd.DataFrame(rows)
    settings = dict(
        seed=SEED,
        n_obs=N_OBS,
        lags="24;168",
        model=f"ETS({MODEL[0]},{MODEL[1]},{MODEL[2]})",
        distribution=DISTRIBUTION,
        initial=INITIAL,
        sigma=SIGMA,
        repetitions=args.repetitions,
        variants_commit=built,
        **env,
    )
    for key, value in settings.items():
        table[key] = value
    CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV, index=False)
    report(table)
    print(f"CSV: {CSV}\nrun time: {time.time() - started:.0f} s")


def run_dgp(dgp, modules, missing, repetitions):
    """The timings of one data-generating process over the numbers of missing values."""
    y, order = make_data(dgp)
    model_48 = fit(with_missing(y, np.sort(order[:48])))
    B = np.asarray(model_48._adam_estimated["B"], dtype=float)
    cells = len(np.unique(model_48._profile["index_lookup_table"]))
    check_cf = -CF(**cf_arguments(model_48, B), adam_cpp=model_48._adam_cpp)
    print(f"\nDGP {dgp}: B (alpha, gamma_24, gamma_168) = {np.round(B, 6).tolist()}")
    print(f"cells of the profile: {cells} (expected {CELLS_EXPECTED})")
    print(f"check: -CF(B) = {check_cf:.10f}, model loglik = {model_48.loglik:.10f}")
    if cells != CELLS_EXPECTED:
        sys.exit("unexpected number of cells")
    print(
        f"{'k':>3} {'first':>5} {'f>1':>5} {'mode':>12} | {'lowrank ms (IQR)':>17} | "
        f"{'full ms (IQR)':>17} | {'package ms':>10} | {'speed-up':>8}"
    )
    rows = []
    for k in missing:
        positions = np.sort(order[:k])
        for row in run_case(y, positions, B, modules, repetitions):
            row.update(
                dgp=dgp,
                alpha=B[0],
                gamma_24=B[1],
                gamma_168=B[2],
                cells=cells,
                loglik_check_diff=abs(check_cf - model_48.loglik),
            )
            rows.append(row)
    return rows


def run_case(y, positions, B, modules, repetitions):
    """The three timed calls at k = len(positions) missing values."""
    model = fit(with_missing(y, positions))
    arguments = cf_arguments(model, B)
    cores = {
        "lowrank": make_core(modules["lowrank"], model),
        "full": make_core(modules["full"], model),
        "package": model._adam_cpp,
    }
    recorder = Recorder(model._adam_cpp)
    CF(**arguments, adam_cpp=recorder)
    if len(recorder.calls) != 1:
        sys.exit(f"expected one call of the filter, got {len(recorder.calls)}")
    captured = recorder.calls[0]
    calls = {
        "filter_pass": filter_call(dict(captured, backcast=False, nIterations=1)),
        "package_call": filter_call(captured),
        "likelihood": likelihood_call(arguments),
    }
    rows = []
    for mode in MODES:
        times, outputs = time_case(cores, calls[mode], repetitions)
        row = {
            "k": len(positions),
            "first_missing": int(positions[0]),
            "mode": mode,
            "backcast": mode != "filter_pass",
            "n_iterations": 1 if mode == "filter_pass" else captured["nIterations"],
        }
        for variant in VARIANTS:
            row[f"{variant}_median_ms"], row[f"{variant}_iqr_ms"] = summary(
                times[variant]
            )
        row["speedup"] = row["full_median_ms"] / row["lowrank_median_ms"]
        row["package_vs_lowrank"] = row["package_median_ms"] / row["lowrank_median_ms"]
        if mode == "likelihood":
            row["loglik"] = -outputs["lowrank"]
            row["max_diff_loglik"] = abs(outputs["lowrank"] - outputs["full"])
            row["max_diff_loglik_package"] = abs(
                outputs["package"] - outputs["lowrank"]
            )
        else:
            # The periods whose one-step variance carries the covariance of the states
            above = np.flatnonzero(outputs["lowrank"]["variances"] > 1)
            row["periods_f_above_1"] = len(above)
            row["last_f_above_1"] = int(above[-1]) if len(above) else -1
            for name in ("fitted", "errors", "variances"):
                row[f"max_diff_{name}"] = max_difference(
                    outputs["lowrank"][name], outputs["full"][name]
                )
                row[f"max_diff_{name}_package"] = max_difference(
                    outputs["package"][name], outputs["lowrank"][name]
                )
        row["positions"] = ";".join(str(p) for p in positions)
        rows.append(row)
        active = row.get("periods_f_above_1", "")
        print(
            f"{row['k']:>3} {positions[0]:>5} {active:>5} {mode:>12} | "
            f"{row['lowrank_median_ms']:8.3f} ({row['lowrank_iqr_ms']:6.3f}) | "
            f"{row['full_median_ms']:8.3f} ({row['full_iqr_ms']:6.3f}) | "
            f"{row['package_median_ms']:10.3f} | {row['speedup']:8.2f}"
        )
    return rows


def report(table):
    """The headline speed-ups and the differences between the variants."""
    for dgp in DGPS:
        part = table[table["dgp"] == dgp]
        print(f"\nDGP {dgp}, speed-up = median full / median lowrank:")
        for mode in MODES:
            speedup = part[part["mode"] == mode].set_index("k")["speedup"]
            print(
                f"  {mode:>12}: k={speedup.index[0]}: {speedup.iloc[0]:.2f}x, "
                f"k={speedup.index[-1]}: {speedup.iloc[-1]:.2f}x, range "
                f"{speedup.min():.2f}x..{speedup.max():.2f}x "
                f"(paper: {TARGET_FIRST}x to {TARGET_LAST}x)"
            )
    filt = table[table["mode"] != "likelihood"]
    lik = table[table["mode"] == "likelihood"]
    for label, suffix in (("lowrank - full", ""), ("package - lowrank", "_package")):
        print(f"Max |{label}| over all cases:")
        print(f"  log-likelihood: {lik[f'max_diff_loglik{suffix}'].max():.3e}")
        for name in ("fitted", "errors", "variances"):
            print(f"  {name}: {filt[f'max_diff_{name}{suffix}'].max():.3e}")
    print(f"Max |-CF(B) - loglik| (k = 48): {table['loglik_check_diff'].max():.3e}")


if __name__ == "__main__":
    main()
