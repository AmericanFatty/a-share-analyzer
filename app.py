from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="A股持有收益分析工具",
    page_icon="📈",
    layout="wide",
)

HORIZONS = [5, 20, 120]
RETURN_COLUMNS = ["5日收益(%)", "20日收益(%)", "120日收益(%)", "截至最新收益(%)"]
BENCHMARK_NAME = "上证指数"
BENCHMARK_SYMBOL = "sh000001"
BENCHMARK_RETURN_COLUMNS = [
    "上证5日收益(%)",
    "上证20日收益(%)",
    "上证120日收益(%)",
    "上证截至最新收益(%)",
]
EXCESS_RETURN_COLUMNS = [
    "5日超额收益(%)",
    "20日超额收益(%)",
    "120日超额收益(%)",
    "截至最新超额收益(%)",
]
WIN_COLUMNS = [
    "5日是否跑赢大盘",
    "20日是否跑赢大盘",
    "120日是否跑赢大盘",
    "截至最新是否跑赢大盘",
]
FINAL_EXPORT_COLUMNS = [
    "板块",
    "股票名称",
    "股票代码",
    "看好日期",
    "5日收益(%)",
    "上证5日收益(%)",
    "5日是否跑赢大盘",
    "30日收益(%)",
    "上证30日收益(%)",
    "30日是否跑赢大盘",
    "截至最新收益(%)",
    "上证截至最新收益(%)",
    "截至最新是否跑赢大盘",
    "备注",
]
FINAL_PERCENT_COLUMNS = [
    "5日收益(%)",
    "上证5日收益(%)",
    "30日收益(%)",
    "上证30日收益(%)",
    "截至最新收益(%)",
    "上证截至最新收益(%)",
]
STOCK_MASTER_CACHE_PATH = Path(__file__).resolve().parent / ".stock_master_cache.csv"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 8% 2%, #f5fbff 0%, #edf4ff 35%, #f8f9fb 100%);
        }
        .main .block-container {
            max-width: 1250px;
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            border: 1px solid #dfe8f5;
            background: linear-gradient(120deg, rgba(255, 255, 255, 0.94), rgba(244, 251, 255, 0.94));
            margin-bottom: 0.8rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #dbe6f4;
            border-radius: 14px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.92);
            box-shadow: 0 3px 10px rgba(26, 43, 74, 0.06);
        }
        .caption-box {
            border-left: 4px solid #2f80ed;
            padding: 0.55rem 0.75rem;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 8px;
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
            color: #1c3553;
            font-size: 0.92rem;
        }
        @media (max-width: 768px) {
            .main .block-container {
                max-width: 100%;
                padding-top: 0.8rem;
                padding-left: 0.55rem;
                padding-right: 0.55rem;
                padding-bottom: 1.2rem;
            }
            .hero {
                padding: 0.75rem 0.8rem;
            }
            .hero h2 {
                font-size: 1.2rem !important;
                margin-bottom: 0.2rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=24 * 3600)
def load_stock_master() -> pd.DataFrame:
    def _clean(master_df: pd.DataFrame) -> pd.DataFrame:
        if master_df is None or master_df.empty:
            return pd.DataFrame(columns=["code", "name"])
        master_df = master_df[["code", "name"]].copy()
        master_df["code"] = master_df["code"].astype(str).str.zfill(6)
        master_df["name"] = master_df["name"].astype(str).str.strip()
        master_df = master_df.dropna(subset=["code", "name"]).drop_duplicates("code")
        return master_df.reset_index(drop=True)

    errors: List[str] = []
    for retry in range(3):
        try:
            df = _clean(ak.stock_info_a_code_name())
            if not df.empty:
                try:
                    df.to_csv(STOCK_MASTER_CACHE_PATH, index=False, encoding="utf-8-sig")
                except Exception:
                    pass
                return df
            errors.append("在线返回空数据")
        except Exception as exc:
            errors.append(f"在线获取失败({retry + 1}/3): {type(exc).__name__}")
        time.sleep(0.35 * (retry + 1))

    if STOCK_MASTER_CACHE_PATH.exists():
        try:
            cached = pd.read_csv(STOCK_MASTER_CACHE_PATH, dtype={"code": str, "name": str})
            cached = _clean(cached)
            if not cached.empty:
                return cached
            errors.append("本地缓存为空")
        except Exception as exc:
            errors.append(f"读取本地缓存失败: {type(exc).__name__}")

    detail = "；".join(errors[-3:]) if errors else "未知错误"
    raise RuntimeError(f"在线获取与本地缓存都失败。{detail}")


def market_prefixed_code(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def normalize_history_df(
    df: pd.DataFrame,
    date_col: str,
    open_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
    volume_col: Optional[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    keep = {
        "日期": df[date_col],
        "开盘": df[open_col],
        "收盘": df[close_col],
        "最高": df[high_col],
        "最低": df[low_col],
        "成交量": df[volume_col] if volume_col and volume_col in df.columns else np.nan,
    }
    df = pd.DataFrame(keep)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).sort_values("日期").reset_index(drop=True)
    return df


def friendly_fetch_error(exc: Exception) -> str:
    error_text = f"{type(exc).__name__}: {exc}"
    lowered = error_text.lower()
    if "failed to resolve" in lowered or "nameresolutionerror" in lowered or "nodename nor servname" in lowered:
        return "DNS 解析失败（无法连接行情服务器）"
    if "connecttimeout" in lowered or "readtimeout" in lowered or "timed out" in lowered:
        return "连接超时（行情服务器响应慢或网络受限）"
    if "ssl" in lowered:
        return "SSL 握手失败（系统证书或网络代理问题）"
    return error_text[:260]


def fetch_from_eastmoney(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    return normalize_history_df(
        df=df,
        date_col="日期",
        open_col="开盘",
        close_col="收盘",
        high_col="最高",
        low_col="最低",
        volume_col="成交量",
    )


def fetch_from_tencent(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    prefixed = market_prefixed_code(code)
    df = ak.stock_zh_a_hist_tx(
        symbol=prefixed,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    return normalize_history_df(
        df=df,
        date_col="date",
        open_col="open",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="amount",
    )


def fetch_from_sina(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    prefixed = market_prefixed_code(code)
    df = ak.stock_zh_a_daily(
        symbol=prefixed,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    return normalize_history_df(
        df=df,
        date_col="date",
        open_col="open",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
    )


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_stock_history(code: str, start_date: str, end_date: str) -> Dict[str, object]:
    fetchers = [
        ("东方财富", fetch_from_eastmoney),
        ("腾讯证券", fetch_from_tencent),
        ("新浪财经", fetch_from_sina),
    ]
    errors: List[str] = []

    for source, fn in fetchers:
        try:
            hist = fn(code=code, start_date=start_date, end_date=end_date)
            if not hist.empty:
                return {"history": hist, "source": source, "errors": errors}
            errors.append(f"{source}: 返回空数据")
        except Exception as exc:
            errors.append(f"{source}: {friendly_fetch_error(exc)}")

    return {"history": pd.DataFrame(), "source": "", "errors": errors}


def filter_history_by_date(history: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    out = history.copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    if pd.notna(start_ts):
        out = out[out["日期"] >= start_ts]
    if pd.notna(end_ts):
        out = out[out["日期"] <= end_ts]
    return out.sort_values("日期").reset_index(drop=True)


def fetch_benchmark_from_eastmoney(start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily_em(
        symbol=BENCHMARK_SYMBOL,
        start_date=start_date,
        end_date=end_date,
    )
    return normalize_history_df(
        df=df,
        date_col="date",
        open_col="open",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
    )


def fetch_benchmark_from_sina(start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=BENCHMARK_SYMBOL)
    normalized = normalize_history_df(
        df=df,
        date_col="date",
        open_col="open",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
    )
    return filter_history_by_date(normalized, start_date, end_date)


def fetch_benchmark_from_tencent(start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily_tx(symbol=BENCHMARK_SYMBOL)
    normalized = normalize_history_df(
        df=df,
        date_col="date",
        open_col="open",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="amount",
    )
    return filter_history_by_date(normalized, start_date, end_date)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_benchmark_history(start_date: str, end_date: str) -> Dict[str, object]:
    fetchers = [
        ("东方财富", fetch_benchmark_from_eastmoney),
        ("新浪财经", fetch_benchmark_from_sina),
        ("腾讯证券", fetch_benchmark_from_tencent),
    ]
    errors: List[str] = []
    for source, fn in fetchers:
        try:
            hist = fn(start_date=start_date, end_date=end_date)
            if not hist.empty:
                return {"history": hist, "source": source, "errors": errors}
            errors.append(f"{source}: 返回空数据")
        except Exception as exc:
            errors.append(f"{source}: {friendly_fetch_error(exc)}")
    return {"history": pd.DataFrame(), "source": "", "errors": errors}


def normalize_code(raw: object) -> str:
    s = "".join(ch for ch in str(raw) if ch.isdigit())
    if not s:
        return ""
    if len(s) > 6:
        s = s[-6:]
    return s.zfill(6)


def resolve_stock(query: object, stock_master: pd.DataFrame) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if query is None:
        return None, "股票名称为空，请输入股票名称或6位代码。"

    query_text = str(query).strip()
    if not query_text or query_text.lower() in {"nan", "none"}:
        return None, "股票名称为空，请输入股票名称或6位代码。"

    if stock_master is None or stock_master.empty:
        code = normalize_code(query_text)
        if len(code) == 6:
            return {"code": code, "name": code}, None
        return None, "股票基础信息不可用，请输入6位股票代码。"

    code = normalize_code(query_text)
    if len(code) == 6:
        by_code = stock_master[stock_master["code"] == code]
        if not by_code.empty:
            row = by_code.iloc[0]
            return {"code": row["code"], "name": row["name"]}, None

    exact = stock_master[stock_master["name"] == query_text]
    if not exact.empty:
        row = exact.iloc[0]
        return {"code": row["code"], "name": row["name"]}, None

    fuzzy = stock_master[stock_master["name"].str.contains(query_text, na=False)]
    if len(fuzzy) == 1:
        row = fuzzy.iloc[0]
        return {"code": row["code"], "name": row["name"]}, None
    if len(fuzzy) > 1:
        candidate_text = "、".join(
            (fuzzy["name"] + "(" + fuzzy["code"] + ")").head(8).tolist()
        )
        return None, f"匹配到多只股票，请输入更精确名称或代码。候选：{candidate_text}"

    return None, f"未找到股票：{query_text}"


def analyze_with_history(
    history: pd.DataFrame,
    stock_name: str,
    stock_code: str,
    input_buy_date: pd.Timestamp,
) -> Dict[str, object]:
    if history.empty:
        return {"error": f"{stock_name}({stock_code}) 未能获取可用行情数据。"}

    input_buy_date = pd.Timestamp(input_buy_date).normalize()
    future = history[history["日期"] >= input_buy_date]
    if future.empty:
        latest_date = history["日期"].max().date()
        return {"error": f"{stock_name}({stock_code}) 买入日晚于可用数据，最新数据到 {latest_date}。"}

    buy_idx = int(future.index[0])
    buy_row = history.loc[buy_idx]
    buy_price = float(buy_row["收盘"])
    actual_buy_date = buy_row["日期"]

    result: Dict[str, object] = {
        "股票名称": stock_name,
        "股票代码": stock_code,
        "买入日期(输入)": input_buy_date.date(),
        "实际买入日": actual_buy_date.date(),
        "买入价(前复权收盘)": round(buy_price, 4),
        "可用交易日数": int(len(history) - buy_idx - 1),
    }

    for horizon in HORIZONS:
        target_idx = buy_idx + horizon
        ret_col = f"{horizon}日收益(%)"
        date_col = f"{horizon}日日期"
        if target_idx < len(history):
            target_row = history.loc[target_idx]
            target_price = float(target_row["收盘"])
            ret = (target_price / buy_price - 1.0) * 100.0
            result[ret_col] = round(ret, 3)
            result[date_col] = target_row["日期"].date()
        else:
            result[ret_col] = np.nan
            result[date_col] = pd.NaT

    latest_row = history.iloc[-1]
    latest_price = float(latest_row["收盘"])
    latest_return = (latest_price / buy_price - 1.0) * 100.0
    result["截至最新收益(%)"] = round(latest_return, 3)
    result["最新日期"] = latest_row["日期"].date()
    result["最新价(前复权收盘)"] = round(latest_price, 4)
    result["_buy_idx"] = buy_idx
    return result


def first_row_on_or_after(history: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.Series]:
    if history is None or history.empty:
        return None
    target_ts = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target_ts):
        return None
    future = history[history["日期"] >= target_ts]
    if future.empty:
        return None
    return future.iloc[0]


def calc_return_on_dates(
    history: pd.DataFrame,
    buy_date: pd.Timestamp,
    target_date: pd.Timestamp,
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    buy_row = first_row_on_or_after(history, buy_date)
    sell_row = first_row_on_or_after(history, target_date)
    if buy_row is None or sell_row is None:
        return np.nan, None, None
    buy_price = float(buy_row["收盘"])
    sell_price = float(sell_row["收盘"])
    ret = (sell_price / buy_price - 1.0) * 100.0
    return ret, pd.to_datetime(buy_row["日期"]), pd.to_datetime(sell_row["日期"])


def calc_return_by_trading_horizon(
    history: pd.DataFrame,
    input_buy_date: pd.Timestamp,
    horizon: int,
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if history is None or history.empty:
        return np.nan, None, None
    input_buy_ts = pd.to_datetime(input_buy_date, errors="coerce")
    if pd.isna(input_buy_ts):
        return np.nan, None, None
    future = history[history["日期"] >= input_buy_ts]
    if future.empty:
        return np.nan, None, None
    buy_idx = int(future.index[0])
    buy_row = history.loc[buy_idx]
    target_idx = buy_idx + horizon
    if target_idx >= len(history):
        return np.nan, pd.to_datetime(buy_row["日期"]), None
    target_row = history.loc[target_idx]
    buy_price = float(buy_row["收盘"])
    target_price = float(target_row["收盘"])
    ret = (target_price / buy_price - 1.0) * 100.0
    return ret, pd.to_datetime(buy_row["日期"]), pd.to_datetime(target_row["日期"])


def append_30day_comparison(
    result: Dict[str, object],
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> Dict[str, object]:
    out = dict(result)
    stock_30_ret, stock_buy_actual, stock_30_date = calc_return_by_trading_horizon(
        history=stock_history,
        input_buy_date=pd.to_datetime(out.get("买入日期(输入)")),
        horizon=30,
    )
    out["30日收益(%)"] = round(float(stock_30_ret), 3) if pd.notna(stock_30_ret) else np.nan
    out["30日日期"] = stock_30_date.date() if stock_30_date is not None else pd.NaT

    if (
        benchmark_history is None
        or benchmark_history.empty
        or pd.isna(stock_30_ret)
        or stock_30_date is None
    ):
        out["上证30日收益(%)"] = np.nan
        out["30日是否跑赢大盘"] = "数据不足"
        return out

    bench_buy = stock_buy_actual or pd.to_datetime(out.get("实际买入日"))
    bench_30_ret, _, _ = calc_return_on_dates(
        history=benchmark_history,
        buy_date=bench_buy,
        target_date=stock_30_date,
    )
    out["上证30日收益(%)"] = round(float(bench_30_ret), 3) if pd.notna(bench_30_ret) else np.nan
    if pd.notna(bench_30_ret):
        out["30日是否跑赢大盘"] = "是" if float(stock_30_ret) - float(bench_30_ret) > 0 else "否"
    else:
        out["30日是否跑赢大盘"] = "数据不足"
    return out


def calc_max_drawdown_pct(price_series: pd.Series) -> float:
    if price_series is None or price_series.empty:
        return np.nan
    series = pd.to_numeric(price_series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    peak = series.cummax()
    drawdown = series / peak - 1.0
    return float(drawdown.min() * 100.0)


def build_benchmark_comparison(
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    result: Dict[str, object],
) -> Dict[str, object]:
    out = dict(result)
    if benchmark_history is None or benchmark_history.empty:
        for i, horizon in enumerate(HORIZONS):
            out[BENCHMARK_RETURN_COLUMNS[i]] = np.nan
            out[EXCESS_RETURN_COLUMNS[i]] = np.nan
            out[WIN_COLUMNS[i]] = "数据不足"
        out[BENCHMARK_RETURN_COLUMNS[-1]] = np.nan
        out[EXCESS_RETURN_COLUMNS[-1]] = np.nan
        out[WIN_COLUMNS[-1]] = "数据不足"
        out["风险对齐天数"] = 0
        out["股票最大回撤(%)"] = np.nan
        out["上证最大回撤(%)"] = np.nan
        out["下跌捕获率(%)"] = np.nan
        out["Beta"] = np.nan
        out["年化Alpha(%)"] = np.nan
        return out

    stock_buy_date = pd.to_datetime(out["实际买入日"])
    for i, horizon in enumerate(HORIZONS):
        stock_ret_col = f"{horizon}日收益(%)"
        target_date_col = f"{horizon}日日期"
        bench_ret_col = BENCHMARK_RETURN_COLUMNS[i]
        excess_col = EXCESS_RETURN_COLUMNS[i]
        win_col = WIN_COLUMNS[i]

        if pd.isna(out[target_date_col]) or pd.isna(out[stock_ret_col]):
            out[bench_ret_col] = np.nan
            out[excess_col] = np.nan
            out[win_col] = "数据不足"
            continue

        target_ts = pd.to_datetime(out[target_date_col])
        bench_ret, _, _ = calc_return_on_dates(
            history=benchmark_history,
            buy_date=stock_buy_date,
            target_date=target_ts,
        )
        out[bench_ret_col] = round(float(bench_ret), 3) if pd.notna(bench_ret) else np.nan
        if pd.notna(bench_ret):
            excess = float(out[stock_ret_col]) - float(bench_ret)
            out[excess_col] = round(excess, 3)
            out[win_col] = "是" if excess > 0 else "否"
        else:
            out[excess_col] = np.nan
            out[win_col] = "数据不足"

    latest_target = pd.to_datetime(out["最新日期"])
    bench_latest_ret, bench_buy_actual, bench_latest_actual = calc_return_on_dates(
        history=benchmark_history,
        buy_date=stock_buy_date,
        target_date=latest_target,
    )
    out[BENCHMARK_RETURN_COLUMNS[-1]] = (
        round(float(bench_latest_ret), 3) if pd.notna(bench_latest_ret) else np.nan
    )
    if pd.notna(bench_latest_ret):
        latest_excess = float(out["截至最新收益(%)"]) - float(bench_latest_ret)
        out[EXCESS_RETURN_COLUMNS[-1]] = round(latest_excess, 3)
        out[WIN_COLUMNS[-1]] = "是" if latest_excess > 0 else "否"
    else:
        out[EXCESS_RETURN_COLUMNS[-1]] = np.nan
        out[WIN_COLUMNS[-1]] = "数据不足"

    out["大盘实际买入日"] = (
        bench_buy_actual.date() if bench_buy_actual is not None else pd.NaT
    )
    out["大盘最新对齐日"] = (
        bench_latest_actual.date() if bench_latest_actual is not None else pd.NaT
    )

    stock_close = stock_history[["日期", "收盘"]].copy()
    bench_close = benchmark_history[["日期", "收盘"]].copy()
    stock_close["日期"] = pd.to_datetime(stock_close["日期"], errors="coerce")
    bench_close["日期"] = pd.to_datetime(bench_close["日期"], errors="coerce")
    stock_close = stock_close[stock_close["日期"] >= stock_buy_date]
    bench_close = bench_close[bench_close["日期"] >= stock_buy_date]

    aligned = stock_close.merge(bench_close, on="日期", how="inner", suffixes=("_股", "_上证"))
    aligned = aligned.dropna(subset=["收盘_股", "收盘_上证"]).sort_values("日期")
    out["风险对齐天数"] = int(aligned.shape[0])

    if aligned.shape[0] < 5:
        out["股票最大回撤(%)"] = np.nan
        out["上证最大回撤(%)"] = np.nan
        out["下跌捕获率(%)"] = np.nan
        out["Beta"] = np.nan
        out["年化Alpha(%)"] = np.nan
        return out

    out["股票最大回撤(%)"] = round(calc_max_drawdown_pct(aligned["收盘_股"]), 3)
    out["上证最大回撤(%)"] = round(calc_max_drawdown_pct(aligned["收盘_上证"]), 3)

    aligned["股收益"] = aligned["收盘_股"].pct_change()
    aligned["上证收益"] = aligned["收盘_上证"].pct_change()
    ret_df = aligned.dropna(subset=["股收益", "上证收益"]).copy()
    if ret_df.empty:
        out["下跌捕获率(%)"] = np.nan
        out["Beta"] = np.nan
        out["年化Alpha(%)"] = np.nan
        return out

    down_mask = ret_df["上证收益"] < 0
    if down_mask.any() and abs(float(ret_df.loc[down_mask, "上证收益"].mean())) > 1e-12:
        capture = (
            float(ret_df.loc[down_mask, "股收益"].mean())
            / float(ret_df.loc[down_mask, "上证收益"].mean())
            * 100.0
        )
        out["下跌捕获率(%)"] = round(capture, 3)
    else:
        out["下跌捕获率(%)"] = np.nan

    bench_var = float(ret_df["上证收益"].var())
    if bench_var > 1e-12:
        beta = float(ret_df["股收益"].cov(ret_df["上证收益"]) / bench_var)
        alpha_daily = float(ret_df["股收益"].mean() - beta * ret_df["上证收益"].mean())
        out["Beta"] = round(beta, 4)
        if alpha_daily > -0.9999:
            out["年化Alpha(%)"] = round(((1.0 + alpha_daily) ** 250 - 1.0) * 100.0, 3)
        else:
            out["年化Alpha(%)"] = np.nan
    else:
        out["Beta"] = np.nan
        out["年化Alpha(%)"] = np.nan
    return out


def human_readable_return(value: object) -> str:
    if pd.isna(value):
        return "数据不足"
    return f"{value:.2f}%"


def make_price_figure(
    history: pd.DataFrame,
    buy_idx: int,
    result: Dict[str, object],
) -> go.Figure:
    hist = history.copy()
    hist["日期"] = pd.to_datetime(hist["日期"])
    view_start_idx = max(0, buy_idx - 25)
    hist_view = hist.iloc[view_start_idx:].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_view["日期"],
            y=hist_view["收盘"],
            mode="lines",
            line=dict(width=2.4, color="#2f80ed"),
            name="收盘价(前复权)",
            hovertemplate="%{x|%Y-%m-%d}<br>收盘:%{y:.2f}<extra></extra>",
        )
    )

    markers_x: List[pd.Timestamp] = []
    markers_y: List[float] = []
    labels: List[str] = []

    buy_date = pd.to_datetime(result["实际买入日"])
    buy_price = float(result["买入价(前复权收盘)"])
    markers_x.append(buy_date)
    markers_y.append(buy_price)
    labels.append("买入")

    for horizon in HORIZONS:
        dcol = f"{horizon}日日期"
        if pd.notna(result[dcol]):
            point_date = pd.to_datetime(result[dcol])
            point_row = hist[hist["日期"] == point_date]
            if not point_row.empty:
                markers_x.append(point_date)
                markers_y.append(float(point_row.iloc[0]["收盘"]))
                labels.append(f"{horizon}日")

    fig.add_trace(
        go.Scatter(
            x=markers_x,
            y=markers_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=10, color="#f2994a", line=dict(width=1, color="#ffffff")),
            name="关键节点",
            hovertemplate="%{x|%Y-%m-%d}<br>收盘:%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="买入后价格走势",
        xaxis_title="日期",
        yaxis_title="价格",
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=420,
    )
    return fig


def read_uploaded_file(uploaded_file) -> Dict[str, pd.DataFrame]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return {"Sheet1": pd.read_csv(uploaded_file)}
    excel_map = pd.read_excel(uploaded_file, sheet_name=None)
    return {str(k): v for k, v in excel_map.items()}


def match_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {col: str(col).strip().lower().replace(" ", "") for col in columns}
    normalized_candidates = {c.strip().lower().replace(" ", "") for c in candidates}
    for col, norm in normalized.items():
        if norm in normalized_candidates:
            return col
    return None


def build_template_excel() -> bytes:
    sample = pd.DataFrame(
        {
            "板块": ["白酒", "新能源", "银行"],
            "股票名称": ["贵州茅台", "宁德时代", "招商银行"],
            "买入日期": ["2024-01-15", "2024-06-03", "2025-02-10"],
            "备注": ["核心仓位", "成长跟踪", "防御配置"],
        }
    )
    buffer = BytesIO()
    sample.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


def summarize_batch(result_df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "5日收益(%)",
        "20日收益(%)",
        "30日收益(%)",
        "120日收益(%)",
        "截至最新收益(%)",
    ]
    rows = []
    for col in candidate_cols:
        if col not in result_df.columns:
            continue
        s = pd.to_numeric(result_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "周期": col.replace("收益(%)", ""),
                "样本数": int(s.shape[0]),
                "胜率(%)": round(float((s > 0).mean() * 100), 2),
                "平均收益(%)": round(float(s.mean()), 3),
                "中位收益(%)": round(float(s.median()), 3),
                "最佳(%)": round(float(s.max()), 3),
                "最差(%)": round(float(s.min()), 3),
            }
        )
    return pd.DataFrame(rows)


def summarize_excess_batch(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [
        ("5日", "5日超额收益(%)", "5日是否跑赢大盘"),
        ("20日", "20日超额收益(%)", "20日是否跑赢大盘"),
        ("120日", "120日超额收益(%)", "120日是否跑赢大盘"),
        ("截至最新", "截至最新超额收益(%)", "截至最新是否跑赢大盘"),
    ]
    for period, excess_col, win_col in mapping:
        if excess_col not in result_df.columns:
            continue
        s = pd.to_numeric(result_df[excess_col], errors="coerce").dropna()
        win_rate = np.nan
        if win_col in result_df.columns:
            valid_win = result_df[win_col].isin(["是", "否"])
            if valid_win.any():
                win_rate = float((result_df.loc[valid_win, win_col] == "是").mean() * 100)
        if s.empty and pd.isna(win_rate):
            continue
        rows.append(
            {
                "周期": period,
                "样本数": int(s.shape[0]),
                "跑赢率(%)": round(win_rate, 2) if pd.notna(win_rate) else np.nan,
                "平均超额收益(%)": round(float(s.mean()), 3) if not s.empty else np.nan,
                "中位超额收益(%)": round(float(s.median()), 3) if not s.empty else np.nan,
                "最佳超额(%)": round(float(s.max()), 3) if not s.empty else np.nan,
                "最差超额(%)": round(float(s.min()), 3) if not s.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def percent_to_text(value: object, ndigits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{ndigits}f}%"


def build_export_dataframe(result_df: pd.DataFrame) -> pd.DataFrame:
    temp = result_df.copy()
    if "买入日期(输入)" in temp.columns:
        temp["看好日期"] = temp["买入日期(输入)"]
    elif "看好日期" not in temp.columns:
        temp["看好日期"] = pd.NaT

    export_df = pd.DataFrame()
    for col in FINAL_EXPORT_COLUMNS:
        export_df[col] = temp[col] if col in temp.columns else ""

    for col in FINAL_PERCENT_COLUMNS:
        export_df[col] = export_df[col].apply(percent_to_text)

    if "看好日期" in export_df.columns:
        export_df["看好日期"] = pd.to_datetime(
            export_df["看好日期"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").replace({pd.NA: "", "NaT": "", "nan": ""})
    for col in ["板块", "备注", "股票名称", "股票代码"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].astype(str).replace({"nan": "", "None": ""})
    return export_df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


def safe_sheet_name(name: str) -> str:
    invalid = set('[]:*?/\\')
    cleaned = "".join(ch for ch in str(name) if ch not in invalid).strip()
    if not cleaned:
        cleaned = "Sheet"
    return cleaned[:31]


def dataframes_to_excel_bytes(sheet_df_map: Dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        used_names: Dict[str, int] = {}
        for idx, (sheet, df) in enumerate(sheet_df_map.items()):
            base = safe_sheet_name(sheet)
            if base in used_names:
                used_names[base] += 1
                suffix = f"_{used_names[base]}"
                sheet_name = (base[: 31 - len(suffix)] + suffix)[:31]
            else:
                used_names[base] = 0
                sheet_name = base
            out_df = df if df is not None else pd.DataFrame()
            out_df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.read()


def run_batch_analysis(
    df_input: pd.DataFrame,
    stock_master: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(df_input.columns)
    name_col = match_column(columns, ["股票名称", "股票", "名称", "stock_name", "name"])
    code_col = match_column(columns, ["股票代码", "代码", "stock_code", "code", "ticker"])
    buy_col = match_column(columns, ["买入日期", "买入日", "日期", "buy_date", "date"])
    sector_col = match_column(columns, ["板块", "行业", "sector", "industry"])
    note_col = match_column(columns, ["备注", "remark", "note"])

    if buy_col is None or (name_col is None and code_col is None):
        raise ValueError(
            "文件列名需包含 [买入日期]，以及 [股票名称] 或 [股票代码]。"
        )

    parsed_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []
    master_available = stock_master is not None and not stock_master.empty

    for idx, row in df_input.iterrows():
        raw_buy = row.get(buy_col)
        buy_ts = pd.to_datetime(raw_buy, errors="coerce")
        if pd.isna(buy_ts):
            failed_rows.append(
                {
                    "行号": idx + 2,
                    "原始股票": str(row.get(name_col) or row.get(code_col)),
                    "原因": "买入日期无法识别",
                }
            )
            continue

        code_query = ""
        if code_col is not None and pd.notna(row.get(code_col)):
            code_query = normalize_code(row.get(code_col))
        name_query = ""
        if name_col is not None and pd.notna(row.get(name_col)):
            name_query = str(row.get(name_col)).strip()

        stock: Optional[Dict[str, str]] = None
        error: Optional[str] = None
        if len(code_query) == 6:
            if master_available:
                code_match = stock_master[stock_master["code"] == code_query]
                if not code_match.empty:
                    stock_name = str(code_match.iloc[0]["name"])
                else:
                    stock_name = name_query if name_query else code_query
            else:
                stock_name = name_query if name_query else code_query
            stock = {"code": code_query, "name": stock_name}
        elif name_query:
            if master_available:
                stock, error = resolve_stock(name_query, stock_master)
            else:
                error = "股票名称库不可用，当前行缺少可识别的6位股票代码。"
        else:
            error = "当前行缺少股票名称和股票代码。"

        if error or stock is None:
            failed_rows.append(
                {
                    "行号": idx + 2,
                    "原始股票": name_query or code_query or "",
                    "原因": error or "无法识别股票",
                }
            )
            continue

        parsed_rows.append(
            {
                "row_no": idx + 2,
                "股票名称": stock["name"],
                "股票代码": stock["code"],
                "买入日期(输入)": buy_ts.normalize(),
                "板块": row.get(sector_col) if sector_col is not None else np.nan,
                "备注": row.get(note_col) if note_col is not None else np.nan,
            }
        )

    if not parsed_rows:
        return pd.DataFrame(), pd.DataFrame(failed_rows)

    parsed_df = pd.DataFrame(parsed_rows)
    today = pd.Timestamp(date.today())
    end_date = (today + pd.Timedelta(days=1)).strftime("%Y%m%d")

    history_map: Dict[str, Dict[str, object]] = {}
    grouped = parsed_df.groupby("股票代码")["买入日期(输入)"].min().to_dict()
    for code, min_buy_date in grouped.items():
        start_date = (pd.Timestamp(min_buy_date) - pd.Timedelta(days=45)).strftime("%Y%m%d")
        history_map[code] = fetch_stock_history(code, start_date, end_date)

    global_start_date = (
        pd.Timestamp(parsed_df["买入日期(输入)"].min()) - pd.Timedelta(days=45)
    ).strftime("%Y%m%d")
    benchmark_bundle = fetch_benchmark_history(global_start_date, end_date)
    benchmark_history = benchmark_bundle.get("history", pd.DataFrame())

    results: List[Dict[str, object]] = []
    for _, item in parsed_df.iterrows():
        fetch_bundle = history_map.get(
            str(item["股票代码"]),
            {"history": pd.DataFrame(), "source": "", "errors": []},
        )
        hist = fetch_bundle.get("history", pd.DataFrame())
        if hist.empty:
            fail_reason = "；".join(fetch_bundle.get("errors", []))
            failed_rows.append(
                {
                    "行号": int(item["row_no"]),
                    "原始股票": f"{item['股票名称']}({item['股票代码']})",
                    "原因": fail_reason or "行情源均未返回可用数据",
                }
            )
            continue

        analyzed = analyze_with_history(
            history=hist,
            stock_name=str(item["股票名称"]),
            stock_code=str(item["股票代码"]),
            input_buy_date=pd.Timestamp(item["买入日期(输入)"]),
        )
        if "error" in analyzed:
            failed_rows.append(
                {
                    "行号": int(item["row_no"]),
                    "原始股票": f"{item['股票名称']}({item['股票代码']})",
                    "原因": analyzed["error"],
                }
            )
            continue

        analyzed = build_benchmark_comparison(
            stock_history=hist,
            benchmark_history=benchmark_history,
            result=analyzed,
        )
        analyzed = append_30day_comparison(
            result=analyzed,
            stock_history=hist,
            benchmark_history=benchmark_history,
        )
        analyzed["行情来源"] = fetch_bundle.get("source", "")
        analyzed["大盘行情来源"] = benchmark_bundle.get("source", "")
        analyzed["板块"] = item.get("板块", np.nan)
        analyzed["备注"] = item.get("备注", np.nan)
        analyzed["来源行号"] = int(item["row_no"])
        results.append(analyzed)

    result_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed_rows)
    return result_df, failed_df


def render_single_stock_panel(mobile_mode: bool = False) -> None:
    st.markdown("### 单只股票分析")
    st.markdown(
        '<div class="caption-box">输入两个信息即可：股票名称（或6位代码） + 买入日期。买入日如遇休市，会自动顺延到下一个交易日，并和上证指数做同期对比。</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.8, 1.2, 1.0])
    stock_query = c1.text_input("股票名称 / 代码", placeholder="例如：贵州茅台 或 600519")
    buy_date = c2.date_input(
        "买入日期",
        value=date.today() - timedelta(days=365),
        min_value=date(1990, 1, 1),
        max_value=date.today(),
    )
    run_clicked = c3.button("计算收益", use_container_width=True)

    if not run_clicked:
        return

    stock_master: Optional[pd.DataFrame] = None
    master_error = ""
    try:
        stock_master = load_stock_master()
    except Exception as exc:
        master_error = str(exc)

    stock: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    if stock_master is not None and not stock_master.empty:
        stock, error = resolve_stock(stock_query, stock_master)
    else:
        code_only = normalize_code(stock_query)
        if len(code_only) == 6:
            stock = {"code": code_only, "name": code_only}
            st.warning("股票名称库暂时不可用，已切换为代码直算模式。")
            if master_error:
                st.caption(f"基础信息错误：{master_error}")
        else:
            detail = master_error or "网络波动导致基础信息不可用"
            st.error(f"股票基础信息加载失败：{detail}")
            st.info("请直接输入6位股票代码（例如 002371）后重试。")
            return

    if error or stock is None:
        st.error(error)
        return

    end_date = (pd.Timestamp(date.today()) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    start_date = (pd.Timestamp(buy_date) - pd.Timedelta(days=45)).strftime("%Y%m%d")

    with st.spinner("正在获取行情并计算..."):
        fetch_bundle = fetch_stock_history(stock["code"], start_date, end_date)
        history = fetch_bundle.get("history", pd.DataFrame())
        benchmark_bundle = fetch_benchmark_history(start_date, end_date)
        benchmark_history = benchmark_bundle.get("history", pd.DataFrame())
        result = analyze_with_history(
            history=history,
            stock_name=stock["name"],
            stock_code=stock["code"],
            input_buy_date=pd.Timestamp(buy_date),
        )
        if "error" not in result:
            result = build_benchmark_comparison(
                stock_history=history,
                benchmark_history=benchmark_history,
                result=result,
            )

    if "error" in result:
        st.error(result["error"])
        error_list = fetch_bundle.get("errors", [])
        if error_list:
            st.info("行情源诊断：" + "；".join(error_list))
            st.caption("通常是网络或 DNS 问题，可稍后重试，或切换网络后再试。")
        return

    source_name = fetch_bundle.get("source", "")
    bench_source = benchmark_bundle.get("source", "")
    source_parts = []
    if source_name:
        source_parts.append(f"个股：{source_name}")
    if bench_source:
        source_parts.append(f"大盘：{bench_source}")
    source_text = "，".join(source_parts)
    if source_text:
        st.success(f"分析完成：{stock['name']} ({stock['code']})，{source_text}")
    else:
        st.success(f"分析完成：{stock['name']} ({stock['code']})")
    if benchmark_history.empty and benchmark_bundle.get("errors"):
        st.warning("上证指数行情暂不可用，超额收益与跑赢判断将显示为数据不足。")
        st.caption("大盘诊断：" + "；".join(benchmark_bundle.get("errors", [])))

    if pd.Timestamp(result["实际买入日"]).date() != buy_date:
        st.warning(
            f"你输入的买入日期 {buy_date} 非交易日，已自动顺延为 {result['实际买入日']}。"
        )

    base_info = (
        f"买入价（前复权收盘）: {result['买入价(前复权收盘)']}，"
        f"最新价（前复权收盘）: {result['最新价(前复权收盘)']}，"
        f"最新交易日: {result['最新日期']}"
    )
    st.markdown(f'<div class="caption-box">{base_info}</div>', unsafe_allow_html=True)

    if mobile_mode:
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        r1c1.metric("5日收益", human_readable_return(result["5日收益(%)"]))
        r1c2.metric("20日收益", human_readable_return(result["20日收益(%)"]))
        r2c1.metric("120日收益", human_readable_return(result["120日收益(%)"]))
        r2c2.metric("截至最新收益", human_readable_return(result["截至最新收益(%)"]))
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("5日收益", human_readable_return(result["5日收益(%)"]))
        mc2.metric("20日收益", human_readable_return(result["20日收益(%)"]))
        mc3.metric("120日收益", human_readable_return(result["120日收益(%)"]))
        mc4.metric("截至最新收益", human_readable_return(result["截至最新收益(%)"]))

    compare_df = pd.DataFrame(
        {
            "周期": ["5日", "20日", "120日", "截至最新"],
            "股票收益(%)": [
                result["5日收益(%)"],
                result["20日收益(%)"],
                result["120日收益(%)"],
                result["截至最新收益(%)"],
            ],
            "上证收益(%)": [
                result.get("上证5日收益(%)"),
                result.get("上证20日收益(%)"),
                result.get("上证120日收益(%)"),
                result.get("上证截至最新收益(%)"),
            ],
            "超额收益(%)": [
                result.get("5日超额收益(%)"),
                result.get("20日超额收益(%)"),
                result.get("120日超额收益(%)"),
                result.get("截至最新超额收益(%)"),
            ],
            "是否跑赢大盘": [
                result.get("5日是否跑赢大盘"),
                result.get("20日是否跑赢大盘"),
                result.get("120日是否跑赢大盘"),
                result.get("截至最新是否跑赢大盘"),
            ],
        }
    )
    compare_show_df = compare_df.copy()
    for col in ["股票收益(%)", "上证收益(%)", "超额收益(%)"]:
        compare_show_df[col] = compare_show_df[col].apply(human_readable_return)
    st.write("与上证指数对比（同区间）")
    st.dataframe(compare_show_df, use_container_width=True, hide_index=True)

    if mobile_mode:
        d1, d2 = st.columns(2)
        d3, d4 = st.columns(2)
        d5, _ = st.columns(2)
    else:
        d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("股票最大回撤", human_readable_return(result.get("股票最大回撤(%)")))
    d2.metric("上证最大回撤", human_readable_return(result.get("上证最大回撤(%)")))
    d3.metric("下跌捕获率", human_readable_return(result.get("下跌捕获率(%)")))
    d4.metric("Beta", "数据不足" if pd.isna(result.get("Beta")) else f"{result.get('Beta'):.3f}")
    d5.metric("年化Alpha", human_readable_return(result.get("年化Alpha(%)")))
    st.caption(
        f"风险指标对齐交易日数：{int(result.get('风险对齐天数', 0))}；"
        f"大盘买入日：{result.get('大盘实际买入日', '—')}；"
        f"大盘最新对齐日：{result.get('大盘最新对齐日', '—')}"
    )

    bar_df = compare_df.melt(
        id_vars="周期",
        value_vars=["股票收益(%)", "上证收益(%)"],
        var_name="对象",
        value_name="收益率(%)",
    ).dropna(subset=["收益率(%)"])

    if mobile_mode:
        if bar_df.empty:
            st.info("可计算的周期不足，无法展示股票与上证对比图。")
        else:
            bar = px.bar(
                bar_df,
                x="周期",
                y="收益率(%)",
                color="对象",
                barmode="group",
                text_auto=".2f",
                title="股票 vs 上证 收益率对比",
                color_discrete_map={"股票收益(%)": "#2f80ed", "上证收益(%)": "#27ae60"},
            )
            bar.add_hline(y=0, line_dash="dot", line_color="#6c757d")
            bar.update_layout(margin=dict(l=12, r=12, t=50, b=20), height=320)
            st.plotly_chart(bar, use_container_width=True)
        price_fig = make_price_figure(history, int(result["_buy_idx"]), result)
        price_fig.update_layout(height=340, margin=dict(l=12, r=12, t=45, b=18))
        st.plotly_chart(price_fig, use_container_width=True)
    else:
        c_left, c_right = st.columns([1.1, 1.4])
        with c_left:
            if bar_df.empty:
                st.info("可计算的周期不足，无法展示股票与上证对比图。")
            else:
                bar = px.bar(
                    bar_df,
                    x="周期",
                    y="收益率(%)",
                    color="对象",
                    barmode="group",
                    text_auto=".2f",
                    title="股票 vs 上证 收益率对比",
                    color_discrete_map={"股票收益(%)": "#2f80ed", "上证收益(%)": "#27ae60"},
                )
                bar.add_hline(y=0, line_dash="dot", line_color="#6c757d")
                bar.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=420,
                )
                st.plotly_chart(bar, use_container_width=True)

        with c_right:
            price_fig = make_price_figure(history, int(result["_buy_idx"]), result)
            st.plotly_chart(price_fig, use_container_width=True)


def render_batch_panel(mobile_mode: bool = False) -> None:
    st.markdown("### Excel批量统计")
    st.markdown(
        '<div class="caption-box">导入你关心的股票列表后，自动计算并导出 5日/30日/截至最新收益，与上证指数比较是否跑赢大盘。</div>',
        unsafe_allow_html=True,
    )

    template_bytes = build_template_excel()
    st.download_button(
        "下载Excel模板",
        data=template_bytes,
        file_name="a_share_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded = st.file_uploader(
        "上传 Excel/CSV",
        type=["xlsx", "xls", "csv"],
        help="文件需包含：买入日期 + 股票名称(或股票代码)；可选列：板块、备注。",
    )
    if uploaded is None:
        return

    try:
        sheet_map = read_uploaded_file(uploaded)
    except Exception as exc:
        st.error(f"读取文件失败：{exc}")
        return

    if not sheet_map:
        st.warning("文件为空，请检查后重新上传。")
        return

    preview_rows: List[pd.DataFrame] = []
    for sheet_name, df_sheet in sheet_map.items():
        if df_sheet is None or df_sheet.empty:
            continue
        head = df_sheet.head(5).copy()
        head.insert(0, "工作表", sheet_name)
        preview_rows.append(head)
    if not preview_rows:
        st.warning("所有工作表都为空，请检查后重新上传。")
        return

    preview_df = pd.concat(preview_rows, ignore_index=True)
    st.write("文件预览（每个Sheet前5行）")
    st.dataframe(preview_df, use_container_width=True)

    if not st.button("开始批量计算", use_container_width=True):
        return

    stock_master: Optional[pd.DataFrame] = None
    try:
        stock_master = load_stock_master()
    except Exception as exc:
        st.warning("股票名称库加载失败，已切换为仅代码模式（需要6位股票代码列）。")
        st.caption(f"基础信息错误：{exc}")

    with st.spinner("批量计算中，请稍候..."):
        all_result_frames: List[pd.DataFrame] = []
        all_failed_frames: List[pd.DataFrame] = []
        export_sheet_map: Dict[str, pd.DataFrame] = {}
        try:
            for sheet_idx, (sheet_name, df_sheet) in enumerate(sheet_map.items()):
                if df_sheet is None or df_sheet.empty:
                    export_sheet_map[sheet_name] = pd.DataFrame(columns=FINAL_EXPORT_COLUMNS)
                    continue
                result_df_sheet, failed_df_sheet = run_batch_analysis(df_sheet, stock_master)
                if not result_df_sheet.empty:
                    result_df_sheet["__sheet_order"] = sheet_idx
                    result_df_sheet["工作表"] = sheet_name
                    all_result_frames.append(result_df_sheet)
                    ordered_sheet = result_df_sheet.sort_values("来源行号").reset_index(drop=True)
                    export_sheet_map[sheet_name] = build_export_dataframe(ordered_sheet)
                else:
                    export_sheet_map[sheet_name] = pd.DataFrame(columns=FINAL_EXPORT_COLUMNS)
                if not failed_df_sheet.empty:
                    failed_copy = failed_df_sheet.copy()
                    failed_copy.insert(0, "工作表", sheet_name)
                    all_failed_frames.append(failed_copy)
        except Exception as exc:
            st.error(f"批量分析失败：{exc}")
            return

    if not all_result_frames:
        st.error("没有成功计算的记录。")
        if all_failed_frames:
            failed_df_all = pd.concat(all_failed_frames, ignore_index=True)
            st.dataframe(failed_df_all, use_container_width=True)
        return

    result_df = pd.concat(all_result_frames, ignore_index=True)
    if all_failed_frames:
        failed_df = pd.concat(all_failed_frames, ignore_index=True)
    else:
        failed_df = pd.DataFrame()

    ordered_result_df = result_df.sort_values(["__sheet_order", "来源行号"]).reset_index(drop=True)
    st.success(f"批量分析完成：成功 {len(ordered_result_df)} 条。")

    latest_series = pd.to_numeric(ordered_result_df["截至最新收益(%)"], errors="coerce").dropna()
    win_rate_latest = float((latest_series > 0).mean() * 100) if not latest_series.empty else np.nan
    mean_latest = float(latest_series.mean()) if not latest_series.empty else np.nan
    if "截至最新超额收益(%)" in ordered_result_df.columns:
        latest_excess_series = pd.to_numeric(
            ordered_result_df["截至最新超额收益(%)"], errors="coerce"
        ).dropna()
    else:
        latest_excess_series = pd.Series(dtype=float)
    beat_col = ordered_result_df.get("截至最新是否跑赢大盘")
    beat_rate_latest = np.nan
    if beat_col is not None:
        valid = beat_col.isin(["是", "否"])
        if valid.any():
            beat_rate_latest = float((beat_col[valid] == "是").mean() * 100)
    best_row = ordered_result_df.loc[ordered_result_df["截至最新收益(%)"].idxmax()]
    worst_row = ordered_result_df.loc[ordered_result_df["截至最新收益(%)"].idxmin()]

    if mobile_mode:
        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
        k1.metric("成功样本", f"{len(result_df)}")
        k2.metric("截至最新胜率", f"{win_rate_latest:.2f}%")
        k3.metric("截至最新平均收益", f"{mean_latest:.2f}%")
        if pd.notna(beat_rate_latest):
            k4.metric("截至最新跑赢率", f"{beat_rate_latest:.2f}%")
        else:
            k4.metric("截至最新跑赢率", "数据不足")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("成功样本", f"{len(result_df)}")
        k2.metric("截至最新胜率", f"{win_rate_latest:.2f}%")
        k3.metric("截至最新平均收益", f"{mean_latest:.2f}%")
        if pd.notna(beat_rate_latest):
            k4.metric("截至最新跑赢率", f"{beat_rate_latest:.2f}%")
        else:
            k4.metric("截至最新跑赢率", "数据不足")

    summary_df = summarize_batch(ordered_result_df)
    summary_excess_df = summarize_excess_batch(ordered_result_df)
    summary_df = summary_df[summary_df["周期"].isin(["5日", "30日", "截至最新"])]
    summary_excess_df = summary_excess_df[summary_excess_df["周期"].isin(["5日", "30日", "截至最新"])]
    if mobile_mode:
        if summary_df.empty:
            st.info("汇总样本不足。")
        else:
            st.write("绝对收益汇总")
            st.dataframe(summary_df, use_container_width=True)
            fig_avg = px.bar(
                summary_df,
                x="周期",
                y="平均收益(%)",
                color="平均收益(%)",
                color_continuous_scale="RdYlGn",
                text_auto=".2f",
                title="各周期平均收益",
            )
            fig_avg.add_hline(y=0, line_dash="dot", line_color="#6c757d")
            fig_avg.update_layout(coloraxis_showscale=False, height=320, margin=dict(l=10, r=10, t=50, b=15))
            st.plotly_chart(fig_avg, use_container_width=True)

        if not summary_excess_df.empty:
            st.write("相对上证汇总")
            st.dataframe(summary_excess_df, use_container_width=True)
            fig_excess_avg = px.bar(
                summary_excess_df,
                x="周期",
                y="平均超额收益(%)",
                color="平均超额收益(%)",
                color_continuous_scale="RdYlGn",
                text_auto=".2f",
                title="各周期平均超额收益",
            )
            fig_excess_avg.add_hline(y=0, line_dash="dot", line_color="#6c757d")
            fig_excess_avg.update_layout(
                coloraxis_showscale=False,
                height=320,
                margin=dict(l=10, r=10, t=50, b=15),
            )
            st.plotly_chart(fig_excess_avg, use_container_width=True)

        hist = px.histogram(
            ordered_result_df,
            x="截至最新超额收益(%)" if "截至最新超额收益(%)" in ordered_result_df.columns else "截至最新收益(%)",
            nbins=22,
            color_discrete_sequence=["#2f80ed"],
            title="截至最新超额收益分布",
        )
        hist.add_vline(
            x=float(latest_excess_series.mean()) if not latest_excess_series.empty else float(ordered_result_df["截至最新收益(%)"].mean()),
            line_dash="dash",
            line_color="#eb5757",
            annotation_text="平均值",
            annotation_position="top",
        )
        hist.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=15))
        st.plotly_chart(hist, use_container_width=True)
        st.markdown(
            f'<div class="caption-box">最弱个股：{worst_row["股票名称"]}（{worst_row["股票代码"]}）'
            f' {worst_row["截至最新收益(%)"]:.2f}%</div>',
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([1.1, 1.2])
        with left:
            if summary_df.empty:
                st.info("汇总样本不足。")
            else:
                st.write("绝对收益汇总")
                st.dataframe(summary_df, use_container_width=True)
                fig_avg = px.bar(
                    summary_df,
                    x="周期",
                    y="平均收益(%)",
                    color="平均收益(%)",
                    color_continuous_scale="RdYlGn",
                    text_auto=".2f",
                    title="各周期平均收益",
                )
                fig_avg.add_hline(y=0, line_dash="dot", line_color="#6c757d")
                fig_avg.update_layout(coloraxis_showscale=False, height=360)
                st.plotly_chart(fig_avg, use_container_width=True)

        with right:
            if summary_excess_df.empty:
                st.info("暂无可用的相对上证统计。")
            else:
                st.write("相对上证汇总")
                st.dataframe(summary_excess_df, use_container_width=True)
                fig_excess_avg = px.bar(
                    summary_excess_df,
                    x="周期",
                    y="平均超额收益(%)",
                    color="平均超额收益(%)",
                    color_continuous_scale="RdYlGn",
                    text_auto=".2f",
                    title="各周期平均超额收益",
                )
                fig_excess_avg.add_hline(y=0, line_dash="dot", line_color="#6c757d")
                fig_excess_avg.update_layout(coloraxis_showscale=False, height=360)
                st.plotly_chart(fig_excess_avg, use_container_width=True)

        hist = px.histogram(
            ordered_result_df,
            x="截至最新超额收益(%)" if "截至最新超额收益(%)" in ordered_result_df.columns else "截至最新收益(%)",
            nbins=22,
            color_discrete_sequence=["#2f80ed"],
            title="截至最新超额收益分布",
        )
        hist.add_vline(
            x=float(latest_excess_series.mean()) if not latest_excess_series.empty else float(ordered_result_df["截至最新收益(%)"].mean()),
            line_dash="dash",
            line_color="#eb5757",
            annotation_text="平均值",
            annotation_position="top",
        )
        hist.update_layout(height=360)
        st.plotly_chart(hist, use_container_width=True)
        st.markdown(
            f'<div class="caption-box">绝对收益最佳：{best_row["股票名称"]}（{best_row["股票代码"]}）'
            f' {best_row["截至最新收益(%)"]:.2f}%；'
            f'绝对收益最弱：{worst_row["股票名称"]}（{worst_row["股票代码"]}） {worst_row["截至最新收益(%)"]:.2f}%</div>',
            unsafe_allow_html=True,
        )

    st.write("批量明细（保持原始输入顺序）")
    display_df = build_export_dataframe(ordered_result_df)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(export_sheet_map) == 1:
        single_sheet_df = next(iter(export_sheet_map.values()))
        excel_bytes = dataframe_to_excel_bytes(single_sheet_df)
    else:
        excel_bytes = dataframes_to_excel_bytes(export_sheet_map)
    st.download_button(
        "下载结果Excel",
        data=excel_bytes,
        file_name="a_share_batch_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if not failed_df.empty:
        st.warning(f"有 {len(failed_df)} 条记录处理失败（已列出）。")
        st.dataframe(failed_df, use_container_width=True)


def main() -> None:
    inject_css()
    with st.sidebar:
        st.markdown("### 显示设置")
        mobile_mode = st.toggle(
            "移动端模式（iPhone推荐）",
            value=False,
            help="启用后会使用更紧凑的单列布局，图表和排名在手机上更易读。",
        )
        st.caption("外网使用建议：部署到云端后再用 iPhone 打开链接。")

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin-bottom:0.2rem;">A股买入持有收益分析</h2>
            <div style="color:#29486d;">
                输入股票名称和买入日期，即可计算持有 5/20/120 个交易日及截至今天的收益率。<br/>
                也支持 Excel 批量导入，一次看全部关注股票的绝对收益、超额收益和跑赢大盘表现。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["单只分析", "Excel批量分析"])
    with tab1:
        render_single_stock_panel(mobile_mode=mobile_mode)
    with tab2:
        render_batch_panel(mobile_mode=mobile_mode)


if __name__ == "__main__":
    main()
