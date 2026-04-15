"""Run the full analysis for the U.S. trade-balance regime-instability case study.

Outputs are written to ``results/figures`` and ``results/tables``.
"""

import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import levene
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "USTradeBalances.csv"
ACTUALS_PATH = REPO_ROOT / "data" / "external" / "trade_balance_actuals_2025.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
for p in [FIG_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2000-01-01")
MAX_WORKERS = max(1, min(8, os.cpu_count() or 1))

MODELS = {
    "rw_naive": {"kind": "rw", "label": "Naive (random walk)"},
    "seasonal_naive12": {"kind": "snaive", "label": "Seasonal naive"},
    "arima_011": {"kind": "arima", "label": "ARIMA(0,1,1)", "order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
    "arima_012": {"kind": "arima", "label": "ARIMA(0,1,2)", "order": (0, 1, 2), "seasonal_order": (0, 0, 0, 0)},
    "sarima_011_001_12": {"kind": "arima", "label": "SARIMA(0,1,1)x(0,0,1,12)", "order": (0, 1, 1), "seasonal_order": (0, 0, 1, 12)},
    "sarima_012_001_12": {"kind": "arima", "label": "SARIMA(0,1,2)x(0,0,1,12)", "order": (0, 1, 2), "seasonal_order": (0, 0, 1, 12)},
    "ucm_local_level_seasonal12": {"kind": "ucm", "label": "UCM local level + seasonal(12)"},
}
PARAMETRIC_MODELS = {"arima_011", "arima_012", "sarima_011_001_12", "sarima_012_001_12", "ucm_local_level_seasonal12"}


def load_series():
    df = pd.read_csv(DATA_PATH)
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["DateIndex"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    df = df.sort_values(["Year", "Month"]).set_index("DateIndex").asfreq("MS")
    series = df["Total"].astype(float)

    actual_2025 = pd.read_csv(ACTUALS_PATH)
    actual_2025["date"] = pd.to_datetime(actual_2025["date"])
    actual_2025 = actual_2025.set_index("date")["total_trade_balance"].astype(float)
    actual_2025.name = "Total"

    combined = pd.concat([series, actual_2025]).sort_index()
    return series, combined, actual_2025


SERIES_1992_2024, SERIES_1992_2025, ACTUAL_2025 = load_series()


def mase_scale(train: pd.Series, seasonality: int = 12) -> float:
    train = train.dropna()
    if len(train) > seasonality:
        scale = np.mean(np.abs(train.values[seasonality:] - train.values[:-seasonality]))
    else:
        scale = np.mean(np.abs(np.diff(train.values)))
    return float(scale) if np.isfinite(scale) and scale != 0 else np.nan


def future_index(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS")


def forecast_rw(train: pd.Series, steps: int):
    idx = future_index(train.index[-1], steps)
    mean = pd.Series([train.iloc[-1]] * steps, index=idx)
    sigma = np.std(np.diff(train.values), ddof=1)
    z = 1.959963984540054
    se = pd.Series([math.sqrt(h) * sigma for h in range(1, steps + 1)], index=idx)
    return pd.DataFrame({"mean": mean, "mean_ci_lower": mean - z * se, "mean_ci_upper": mean + z * se})


def forecast_snaive(train: pd.Series, steps: int, m: int = 12):
    idx = future_index(train.index[-1], steps)
    vals = [train.iloc[-m + (h - 1)] for h in range(1, steps + 1)]
    mean = pd.Series(vals, index=idx)
    resid = train[m:] - train.shift(m).dropna() if len(train) > m else pd.Series(np.diff(train.values))
    sigma = resid.std(ddof=1)
    z = 1.959963984540054
    se = pd.Series([sigma] * steps, index=idx)
    return pd.DataFrame({"mean": mean, "mean_ci_lower": mean - z * se, "mean_ci_upper": mean + z * se})


def forecast_model(train: pd.Series, model_name: str, steps: int = 12):
    spec = MODELS[model_name]
    if spec["kind"] == "rw":
        return None, forecast_rw(train, steps)
    if spec["kind"] == "snaive":
        return None, forecast_snaive(train, steps)
    if spec["kind"] == "arima":
        fit = ARIMA(train, order=spec["order"], seasonal_order=spec["seasonal_order"], trend="n").fit(cov_type="none")
        sf = fit.get_forecast(steps=steps).summary_frame(alpha=0.05)
        sf.columns.name = None
        return fit, sf[["mean", "mean_ci_lower", "mean_ci_upper"]]
    if spec["kind"] == "ucm":
        fit = UnobservedComponents(train, level="local level", seasonal=12).fit(disp=False, cov_type="none")
        sf = fit.get_forecast(steps=steps).summary_frame(alpha=0.05)
        sf.columns.name = None
        return fit, sf[["mean", "mean_ci_lower", "mean_ci_upper"]]
    raise ValueError(model_name)


def task_backtest(args):
    model_name, window_type, origin_str = args
    origin = pd.Timestamp(origin_str)
    full_train = SERIES_1992_2025.loc[TRAIN_START:origin]
    train = full_train.iloc[-120:] if window_type == "rolling120" else full_train
    scale = mase_scale(train, 12)
    _, fc = forecast_model(train, model_name, 12)
    actuals = SERIES_1992_2025.loc[fc.index.min():fc.index.max()]
    rows = []
    for h, date in enumerate(fc.index, start=1):
        actual = actuals.get(date, np.nan)
        pred = float(fc.loc[date, "mean"])
        lower = float(fc.loc[date, "mean_ci_lower"])
        upper = float(fc.loc[date, "mean_ci_upper"])
        rows.append({
            "model": model_name,
            "window_type": window_type,
            "origin": origin,
            "horizon": h,
            "target_date": date,
            "actual": actual,
            "pred": pred,
            "abs_error": abs(actual - pred),
            "sq_error": (actual - pred) ** 2,
            "scaled_abs_error": abs(actual - pred) / scale if scale and np.isfinite(scale) else np.nan,
            "covered_95": float(lower <= actual <= upper),
            "interval_width": upper - lower,
            "lower_95": lower,
            "upper_95": upper,
        })
    return rows


def residual_variance_ratio(resid: pd.Series) -> float:
    resid = resid.dropna().to_numpy()
    third = len(resid) // 3
    first_var = np.var(resid[:third], ddof=1)
    last_var = np.var(resid[-third:], ddof=1)
    return float(last_var / first_var)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, window_type, horizon), g in df.groupby(["model", "window_type", "horizon"]):
        rows.append({
            "model": model,
            "window_type": window_type,
            "horizon": horizon,
            "n_forecasts": int(g.shape[0]),
            "mae": g["abs_error"].mean(),
            "rmse": math.sqrt(g["sq_error"].mean()),
            "mase": g["scaled_abs_error"].mean(),
            "coverage_95": g["covered_95"].mean(),
            "avg_interval_width": g["interval_width"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["window_type", "model", "horizon"]).reset_index(drop=True)


def make_plots(metrics_all: pd.DataFrame, subperiod_metrics: pd.DataFrame, final_fc_compare: pd.DataFrame, sarima_exp: pd.DataFrame):
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300

    full = SERIES_1992_2024
    d_full = full.diff().dropna()
    roll_var = d_full.rolling(24).var()

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    axes[0].plot(full.index, full.values, lw=1.2)
    axes[0].axvline(pd.Timestamp("2000-01-01"), color="tab:red", ls="--", lw=1)
    axes[0].set_title("U.S. total trade balance, 1992-2024")
    axes[0].set_ylabel("Billions of dollars")

    axes[1].plot(d_full.index, d_full.values, lw=1.0)
    axes[1].axvline(pd.Timestamp("2000-01-01"), color="tab:red", ls="--", lw=1)
    axes[1].set_title("First differences, 1992-2024")
    axes[1].set_ylabel("Differenced value")

    axes[2].plot(roll_var.index, roll_var.values, lw=1.2)
    axes[2].axvline(pd.Timestamp("2000-01-01"), color="tab:red", ls="--", lw=1)
    axes[2].set_title("24-month rolling variance of first differences")
    axes[2].set_ylabel("Variance")
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_regime_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    plot_df = metrics_all[metrics_all["window_type"] == "expanding"].copy()
    fig = plt.figure(figsize=(10, 6))
    for model in plot_df["model"].unique():
        g = plot_df[plot_df["model"] == model]
        plt.plot(g["horizon"], g["mae"], marker="o", label=model)
    plt.title("Expanding-window backtest MAE by horizon")
    plt.xlabel("Forecast horizon (months)")
    plt.ylabel("MAE (billions of dollars)")
    plt.xticks(sorted(plot_df["horizon"].unique()))
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_02_backtest_mae_by_horizon.png", bbox_inches="tight")
    plt.close(fig)

    plot_df = metrics_all[(metrics_all["window_type"] == "expanding") & (metrics_all["model"].isin(list(PARAMETRIC_MODELS)))].copy()
    fig = plt.figure(figsize=(10, 6))
    for model in plot_df["model"].unique():
        g = plot_df[plot_df["model"] == model]
        plt.plot(g["horizon"], g["coverage_95"], marker="o", label=model)
    plt.axhline(0.95, color="tab:red", ls="--", lw=1)
    plt.ylim(0, 1.05)
    plt.title("Expanding-window 95% interval coverage by horizon")
    plt.xlabel("Forecast horizon (months)")
    plt.ylabel("Coverage rate")
    plt.xticks(sorted(plot_df["horizon"].unique()))
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_03_interval_coverage_by_horizon.png", bbox_inches="tight")
    plt.close(fig)

    selected_window_metrics = metrics_all[(metrics_all["model"] == "sarima_012_001_12") & (metrics_all["window_type"].isin(["expanding", "rolling120"]))].copy()
    fig = plt.figure(figsize=(9, 5.5))
    for wt in selected_window_metrics["window_type"].unique():
        g = selected_window_metrics[selected_window_metrics["window_type"] == wt]
        plt.plot(g["horizon"], g["mae"], marker="o", label=wt)
    plt.title("Reference SARIMA: expanding vs rolling-120 MAE")
    plt.xlabel("Forecast horizon (months)")
    plt.ylabel("MAE (billions of dollars)")
    plt.xticks(sorted(selected_window_metrics["horizon"].unique()))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_04_selected_window_comparison.png", bbox_inches="tight")
    plt.close(fig)

    g = subperiod_metrics[subperiod_metrics["horizon"] == 12].copy()
    fig = plt.figure(figsize=(9, 5.5))
    plt.bar(g["subperiod"], g["mae"])
    plt.title("Reference SARIMA, 12-month horizon: subperiod MAE")
    plt.ylabel("MAE (billions of dollars)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_05_selected_subperiod_mae.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11, 6))
    hist = SERIES_1992_2024.loc["2023-01-01":"2024-12-01"]
    plt.plot(hist.index, hist.values, lw=1.2, label="Observed 2023-2024")
    plt.plot(ACTUAL_2025.index, ACTUAL_2025.values, lw=2.0, label="Realized 2025")
    for col in ["sarima_expanding_mean", "sarima_rolling120_mean", "ucm_mean", "seasonal_naive_mean"]:
        plt.plot(final_fc_compare.index, final_fc_compare[col], lw=1.4, label=col.replace("_mean", ""))
    plt.fill_between(final_fc_compare.index, sarima_exp["mean_ci_lower"], sarima_exp["mean_ci_upper"], alpha=0.15, label="SARIMA 95% PI")
    plt.title("2025 realized outcomes vs forecast benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Billions of dollars")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_06_2025_forecast_compare.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train_post = SERIES_1992_2024.loc["2000-01-01":"2024-12-01"]
    sel_fit = ARIMA(train_post, order=(0, 1, 2), seasonal_order=(0, 0, 1, 12), trend="n").fit(cov_type="approx")
    lb_df3 = acorr_ljungbox(sel_fit.resid.dropna(), lags=[12, 24, 36], model_df=3, return_df=True)

    pre = SERIES_1992_2024.loc["1992-01-01":"1999-12-01"].diff().dropna()
    post = train_post.diff().dropna()
    full_baseline = ARIMA(SERIES_1992_2024, order=(0, 1, 1), trend="n").fit(cov_type="none")
    post_baseline = ARIMA(train_post, order=(0, 1, 1), trend="n").fit(cov_type="none")

    regime_stats = pd.DataFrame({
        "metric": [
            "fd_variance_pre_1992_1999", "fd_variance_post_2000_2024", "fd_variance_ratio_post_over_pre",
            "baseline_residual_variance_ratio_full", "baseline_residual_variance_ratio_post2000",
            "levene_stat_pre_vs_post_fd", "levene_p_pre_vs_post_fd"
        ],
        "value": [
            pre.var(ddof=1), post.var(ddof=1), post.var(ddof=1) / pre.var(ddof=1),
            residual_variance_ratio(full_baseline.resid), residual_variance_ratio(post_baseline.resid),
            levene(pre, post, center="median").statistic, levene(pre, post, center="median").pvalue,
        ]
    })
    regime_stats.to_csv(TABLE_DIR / "tbl_01_regime_stats.csv", index=False)

    rerun_rows = []
    for param, val in sel_fit.params.items():
        rerun_rows.append({"section": "param", "name": param, "value": val})
    for lag, row in lb_df3.iterrows():
        rerun_rows.append({"section": "lb_model_df3", "name": f"lag_{lag}_p", "value": row["lb_pvalue"]})
    pd.DataFrame(rerun_rows).to_csv(TABLE_DIR / "tbl_08_selected_model_rerun.csv", index=False)

    origins = pd.date_range("2014-12-01", "2024-12-01", freq="MS")
    tasks = [(model, "expanding", str(origin.date())) for model in MODELS for origin in origins]
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(task_backtest, t) for t in tasks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
    backtest_expanding = pd.DataFrame(rows)

    tasks = [("sarima_012_001_12", "rolling120", str(origin.date())) for origin in origins]
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(task_backtest, t) for t in tasks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
    backtest_selected_roll = pd.DataFrame(rows)

    backtest_all = pd.concat([backtest_expanding, backtest_selected_roll], ignore_index=True)
    metrics_all = summarize_metrics(backtest_all)
    metrics_all.to_csv(TABLE_DIR / "tbl_02_backtest_metrics.csv", index=False)

    selected_window_metrics = metrics_all[(metrics_all["model"] == "sarima_012_001_12") & (metrics_all["window_type"].isin(["expanding", "rolling120"]))].copy()
    selected_window_metrics.to_csv(TABLE_DIR / "tbl_04_selected_window_metrics.csv", index=False)

    interval_metrics = metrics_all[(metrics_all["model"].isin(list(PARAMETRIC_MODELS))) & (metrics_all["window_type"] == "expanding")][["model", "horizon", "coverage_95", "avg_interval_width"]].copy()
    interval_metrics.to_csv(TABLE_DIR / "tbl_03_interval_metrics.csv", index=False)

    selected_exp = backtest_all[(backtest_all["model"] == "sarima_012_001_12") & (backtest_all["window_type"] == "expanding")].copy()
    selected_exp["subperiod"] = np.where(selected_exp["target_date"] < pd.Timestamp("2020-01-01"), "2015-2019 targets", "2020-2025 targets")
    subperiod_rows = []
    for (subperiod, horizon), g in selected_exp.groupby(["subperiod", "horizon"]):
        subperiod_rows.append({
            "subperiod": subperiod,
            "horizon": horizon,
            "n_forecasts": int(g.shape[0]),
            "mae": g["abs_error"].mean(),
            "rmse": math.sqrt(g["sq_error"].mean()),
            "mase": g["scaled_abs_error"].mean(),
            "coverage_95": g["covered_95"].mean(),
            "avg_interval_width": g["interval_width"].mean(),
        })
    subperiod_metrics = pd.DataFrame(subperiod_rows).sort_values(["horizon", "subperiod"]).reset_index(drop=True)
    subperiod_metrics.to_csv(TABLE_DIR / "tbl_05_selected_subperiod_metrics.csv", index=False)

    full_train = SERIES_1992_2024.loc["2000-01-01":"2024-12-01"]
    _, sarima_exp = forecast_model(full_train, "sarima_012_001_12", 12)
    _, sarima_roll = forecast_model(full_train.iloc[-120:], "sarima_012_001_12", 12)
    _, ucm_fc = forecast_model(full_train, "ucm_local_level_seasonal12", 12)
    _, snaive_fc = forecast_model(full_train, "seasonal_naive12", 12)

    final_fc_compare = pd.DataFrame({
        "actual_2025": ACTUAL_2025,
        "sarima_expanding_mean": sarima_exp["mean"],
        "sarima_rolling120_mean": sarima_roll["mean"],
        "ucm_mean": ucm_fc["mean"],
        "seasonal_naive_mean": snaive_fc["mean"],
    })
    final_fc_compare.to_csv(TABLE_DIR / "tbl_07_forecast_compare_2025.csv", index_label="date")

    ex_post_rows = []
    for model_col in ["sarima_expanding_mean", "sarima_rolling120_mean", "ucm_mean", "seasonal_naive_mean"]:
        errs = final_fc_compare[model_col] - final_fc_compare["actual_2025"]
        row = {
            "forecast_spec": model_col,
            "mae_2025": errs.abs().mean(),
            "rmse_2025": math.sqrt((errs ** 2).mean()),
            "mean_error_2025": errs.mean(),
        }
        if model_col == "sarima_expanding_mean":
            row["coverage_95_2025"] = ((ACTUAL_2025 >= sarima_exp["mean_ci_lower"]) & (ACTUAL_2025 <= sarima_exp["mean_ci_upper"])).mean()
            row["avg_interval_width_2025"] = (sarima_exp["mean_ci_upper"] - sarima_exp["mean_ci_lower"]).mean()
        elif model_col == "sarima_rolling120_mean":
            row["coverage_95_2025"] = ((ACTUAL_2025 >= sarima_roll["mean_ci_lower"]) & (ACTUAL_2025 <= sarima_roll["mean_ci_upper"])).mean()
            row["avg_interval_width_2025"] = (sarima_roll["mean_ci_upper"] - sarima_roll["mean_ci_lower"]).mean()
        elif model_col == "ucm_mean":
            row["coverage_95_2025"] = ((ACTUAL_2025 >= ucm_fc["mean_ci_lower"]) & (ACTUAL_2025 <= ucm_fc["mean_ci_upper"])).mean()
            row["avg_interval_width_2025"] = (ucm_fc["mean_ci_upper"] - ucm_fc["mean_ci_lower"]).mean()
        else:
            row["coverage_95_2025"] = np.nan
            row["avg_interval_width_2025"] = np.nan
        ex_post_rows.append(row)
    ex_post_metrics = pd.DataFrame(ex_post_rows).sort_values("mae_2025").reset_index(drop=True)
    ex_post_metrics.to_csv(TABLE_DIR / "tbl_06_expost_metrics_2025.csv", index=False)

    make_plots(metrics_all, subperiod_metrics, final_fc_compare, sarima_exp)
    print("Done.")


if __name__ == "__main__":
    main()
