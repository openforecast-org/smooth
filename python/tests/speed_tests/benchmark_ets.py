"""
ETS Benchmark Comparison: smooth (Python) vs statsforecast

This script benchmarks ETS model performance on M1, M3, and Tourism datasets.

Setup:
    1. Run: Rscript extract_data_from_R.R  (creates benchmark_data/ folder)
    2. Run: pip install statsforecast tqdm

Usage:
    python benchmark_ets.py --dataset m3_monthly --n_series 10
    python benchmark_ets.py --dataset m3_monthly --output results.csv
    python benchmark_ets.py --all
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_benchmark import (
    compute_all_metrics,
    get_seasonality,
    list_available_datasets,
    load_dataset,
    print_available_datasets,
)

warnings.filterwarnings('ignore')


def benchmark_smooth_python(y_train, h, seasonality, initial='backcasting'):
    """
    Benchmark smooth Python ADAM model.

    Returns:
        dict with 'forecast', 'time', 'model_type', 'success'
    """
    from smooth.adam_general.core.adam import ADAM

    t0 = time.perf_counter()
    try:
        lags = [seasonality] if seasonality > 1 else [1]
        model = ADAM(model='ZZZ', lags=lags, initial=initial)
        model.fit(y_train)
        forecast_result = model.predict(h=h)

        # Handle different return types
        if isinstance(forecast_result, dict):
            forecast = forecast_result.get('mean', forecast_result.get('forecast'))
            if hasattr(forecast, 'values'):
                forecast = forecast.values
        elif hasattr(forecast_result, 'values'):
            forecast = forecast_result.values
        else:
            forecast = np.array(forecast_result)

        elapsed = time.perf_counter() - t0
        model_type = getattr(model, 'model_', 'ZZZ')

        return {
            'forecast': forecast.flatten()[:h],
            'time': elapsed,
            'model_type': str(model_type),
            'success': True,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            'forecast': np.full(h, np.nan),
            'time': elapsed,
            'model_type': 'ERROR',
            'success': False,
            'error': str(e),
        }


def benchmark_statsforecast(y_train, h, seasonality):
    """
    Benchmark statsforecast AutoETS model.

    Returns:
        dict with 'forecast', 'time', 'model_type', 'success'
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    t0 = time.perf_counter()
    try:
        df = pd.DataFrame({
            'unique_id': 'series',
            'ds': range(len(y_train)),
            'y': y_train
        })

        season_length = seasonality if seasonality > 1 else 1
        sf = StatsForecast(
            models=[AutoETS(season_length=season_length)],
            freq=1
        )
        sf_forecast = sf.forecast(df=df, h=h)
        forecast = sf_forecast['AutoETS'].values

        elapsed = time.perf_counter() - t0

        return {
            'forecast': forecast.flatten()[:h],
            'time': elapsed,
            'model_type': 'AutoETS',
            'success': True,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            'forecast': np.full(h, np.nan),
            'time': elapsed,
            'model_type': 'ERROR',
            'success': False,
            'error': str(e),
        }


def benchmark_single_series(y_train, y_test, h, seasonality, methods=None):
    """
    Benchmark a single series across all specified methods.

    Args:
        y_train: Training data (numpy array)
        y_test: Test data (numpy array)
        h: Forecast horizon
        seasonality: Seasonal period
        methods: List of methods to benchmark (default: all)

    Returns:
        dict with results for each method
    """
    if methods is None:
        methods = ['smooth_py_back', 'smooth_py_opt', 'statsforecast']

    results = {}

    # smooth Python - backcasting
    if 'smooth_py_back' in methods:
        res = benchmark_smooth_python(y_train, h, seasonality, initial='backcasting')
        if res['success']:
            fc_len = min(len(res['forecast']), len(y_test), h)
            metrics = compute_all_metrics(
                y_test[:fc_len], res['forecast'][:fc_len], y_train, seasonality
            )
            res.update(metrics)
        results['smooth_py_back'] = res

    # smooth Python - optimal
    if 'smooth_py_opt' in methods:
        res = benchmark_smooth_python(y_train, h, seasonality, initial='optimal')
        if res['success']:
            fc_len = min(len(res['forecast']), len(y_test), h)
            metrics = compute_all_metrics(
                y_test[:fc_len], res['forecast'][:fc_len], y_train, seasonality
            )
            res.update(metrics)
        results['smooth_py_opt'] = res

    # statsforecast AutoETS
    if 'statsforecast' in methods:
        res = benchmark_statsforecast(y_train, h, seasonality)
        if res['success']:
            fc_len = min(len(res['forecast']), len(y_test), h)
            metrics = compute_all_metrics(
                y_test[:fc_len], res['forecast'][:fc_len], y_train, seasonality
            )
            res.update(metrics)
        results['statsforecast'] = res

    return results


def run_benchmark(dataset_name, n_series=None, methods=None, data_dir=None):
    """
    Run full benchmark on a dataset.

    Args:
        dataset_name: Name of dataset (e.g., 'm3_monthly')
        n_series: Number of series to benchmark (None = all)
        methods: List of methods to benchmark
        data_dir: Directory containing benchmark data

    Returns:
        DataFrame with results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    ds = load_dataset(dataset_name, data_dir)
    seasonality = get_seasonality(dataset_name)

    print(f"Series: {ds.n_series}, Horizon: {ds.horizon}, Seasonality: {seasonality}")

    # Get series IDs
    series_ids = ds.get_series_ids()
    if n_series is not None:
        series_ids = series_ids[:n_series]

    print(f"Benchmarking {len(series_ids)} series...")

    all_results = []

    for series_id in tqdm(series_ids, desc=f"Processing {dataset_name}"):
        series = ds.get_series(series_id)
        y_train = series['train']
        y_test = series['test']
        h = series['horizon']

        # Skip if insufficient data
        if len(y_train) < seasonality + 2 or len(y_test) < 1:
            continue

        result = benchmark_single_series(y_train, y_test, h, seasonality, methods)

        # Flatten results for DataFrame
        for method, res in result.items():
            row = {
                'dataset': dataset_name,
                'series_id': series_id,
                'method': method,
                'time': res.get('time', np.nan),
                'mae': res.get('mae', np.nan),
                'rmse': res.get('rmse', np.nan),
                'smape': res.get('smape', np.nan),
                'mase': res.get('mase', np.nan),
                'model_type': res.get('model_type', ''),
                'success': res.get('success', False),
            }
            all_results.append(row)

    return pd.DataFrame(all_results)


def aggregate_results(results_df):
    """Compute summary statistics by method."""
    successful = results_df[results_df['success']]

    if len(successful) == 0:
        return pd.DataFrame()

    summary = successful.groupby(['dataset', 'method']).agg({
        'mae': 'mean',
        'rmse': 'mean',
        'smape': 'mean',
        'mase': 'mean',
        'time': ['mean', 'sum'],
        'success': 'sum',
    }).round(4)

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    return summary


def print_summary(summary_df):
    """Print formatted summary table."""
    if len(summary_df) == 0:
        print("\nNo successful results to summarize.")
        return

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for dataset in summary_df.index.get_level_values('dataset').unique():
        print(f"\nDataset: {dataset}")
        print("-"*70)
        ds_summary = summary_df.loc[dataset]

        print(f"{'Method':<18} {'MAE':>10} {'RMSE':>10} {'sMAPE':>10} {'MASE':>8} {'Avg Time':>10}")
        print("-"*70)

        for method in ds_summary.index:
            row = ds_summary.loc[method]
            print(f"{method:<18} {row['mae']:>10.2f} {row['rmse']:>10.2f} "
                  f"{row['smape']:>9.2f}% {row['mase']:>8.3f} {row['time_mean']:>9.4f}s")

        print("-"*70)


def main():
    parser = argparse.ArgumentParser(
        description='ETS Benchmark: smooth (Python) vs statsforecast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
    1. Rscript extract_data_from_R.R   # Extract data from R packages
    2. pip install statsforecast tqdm  # Install Python dependencies

Examples:
    python benchmark_ets.py --list
    python benchmark_ets.py --dataset m3_monthly --n_series 10
    python benchmark_ets.py --dataset m3_monthly --output results.csv
    python benchmark_ets.py --all --n_series 50
        """
    )

    parser.add_argument('--dataset', type=str, default='m3_monthly',
                        help='Dataset name (default: m3_monthly)')
    parser.add_argument('--n_series', type=int, default=None,
                        help='Number of series to benchmark (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: ./benchmark_data)')
    parser.add_argument('--all', action='store_true',
                        help='Run benchmark on all available datasets')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    parser.add_argument('--methods', nargs='+',
                        default=['smooth_py_back', 'smooth_py_opt', 'statsforecast'],
                        help='Methods to benchmark')

    args = parser.parse_args()

    if args.list:
        print_available_datasets(args.data_dir)
        return

    # Check data exists
    available = list_available_datasets(args.data_dir)
    if not available:
        print("No datasets found!")
        print("Run 'Rscript extract_data_from_R.R' first to extract data from R packages.")
        return

    # Determine which datasets to run
    if args.all:
        datasets = available
    else:
        if args.dataset not in available:
            print(f"Dataset '{args.dataset}' not found.")
            print(f"Available: {available}")
            return
        datasets = [args.dataset]

    # Run benchmarks
    all_results = []
    for dataset in datasets:
        results = run_benchmark(
            dataset,
            n_series=args.n_series,
            methods=args.methods,
            data_dir=args.data_dir
        )
        all_results.append(results)

    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Print summary
    summary = aggregate_results(combined_results)
    print_summary(summary)

    # Save results
    if args.output:
        combined_results.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")

        summary_path = args.output.replace('.csv', '_summary.csv')
        summary.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
