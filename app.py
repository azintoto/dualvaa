import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(page_title="국내상장 통합전략 (VAA 1/3 + 듀얼모멘텀 1/3 + BAA 1/3)", layout="wide")

st.title("📊 국내상장 통합전략")
st.markdown("**VAA 1/3 + 종합듀얼모멘텀 1/3 + BAA 1/3 (국내 상장 ETF)**")

# ── VAA 전략 자산 정의 ──
vaa_attack_assets = [
    "TIGER 미국S&P500",
    "KODEX 선진국MSCI World",
    "PLUS 신흥국MSCI(합성H)",
    "TIME 미국나스닥100액티브",
]
vaa_defense_assets = [
    "TIGER 미국달러단기채권액티브",
    "TIGER 미국채10년선물",
    "PLUS 미국장기우량회사채",
]

# ── 종합듀얼모멘텀 전략 자산 정의 ──
dm_asset_groups = {
    "주식": ["TIGER 미국S&P500", "KODEX 선진국MSCI World"],
    "채권": ["PLUS 미국장기우량회사채", "KODEX 미국하이일드액티브"],
    "부동산": ["TIGER 리츠부동산인프라", "KODEX 일본부동산리츠(H)"],
    "경제하락": ["ACE KRX금현물", "ACE 미국30년국채액티브"],
}

# ── BAA 전략 자산 정의 ──
baa_attack_assets = [
    "TIME 미국나스닥100액티브",
    "TIGER 미국S&P500",
    "KODEX 선진국MSCI World",
    "PLUS 신흥국MSCI(합성H)",
    "TIGER 일본니케이225",
    "KODEX 200",
    "ACE KRX금현물",
    "ACE 미국30년국채액티브",
    "KODEX 미국하이일드액티브",
    "PLUS 미국장기우량회사채",
]
BAA_TOP_N = 5

# ── ETF 이름 → Yahoo Finance 티커 매핑 (전체 통합) ──
NAME_TO_TICKER = {
    # VAA 공격자산
    "TIGER 미국S&P500": "360750.KS",
    "KODEX 선진국MSCI World": "251350.KS",
    "PLUS 신흥국MSCI(합성H)": "195980.KS",
    "TIME 미국나스닥100액티브": "426030.KS",
    # VAA 수비자산
    "TIGER 미국달러단기채권액티브": "329750.KS",
    "TIGER 미국채10년선물": "305080.KS",
    "PLUS 미국장기우량회사채": "332620.KS",
    # 듀얼모멘텀 추가 자산
    "KODEX 미국하이일드액티브": "468380.KS",
    "TIGER 리츠부동산인프라": "329200.KS",
    "KODEX 일본부동산리츠(H)": "352540.KS",
    "ACE KRX금현물": "411060.KS",
    "ACE 미국30년국채액티브": "453850.KS",
    # BAA 추가 자산
    "TIGER 일본니케이225": "241180.KS",
    "KODEX 200": "069500.KS",
}


# ── 공통 데이터 수집 함수 ──
@st.cache_data(ttl=3600)
def get_yearly_returns(etf_names):
    """각 ETF의 최근 1년 수익률 계산 (yfinance, 분배금 포함)"""
    returns_data = {}
    price_details = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for etf_name in etf_names:
        ticker_code = NAME_TO_TICKER.get(etf_name)
        if not ticker_code:
            returns_data[etf_name] = 0.0
            price_details[etf_name] = {"old_price": 0, "current_price": 0, "dividends": 0, "return": 0.0}
            continue

        for attempt in range(3):
            try:
                data = yf.download(ticker_code, start=start_date, end=end_date, progress=False)
                if len(data) > 1:
                    old_price = data["Close"].iloc[0].item()
                    current_price = data["Close"].iloc[-1].item()
                    old_date = data.index[0].strftime("%y.%m.%d")
                    current_date = data.index[-1].strftime("%y.%m.%d")

                    tk = yf.Ticker(ticker_code)
                    all_divs = tk.dividends
                    if not all_divs.empty:
                        all_divs.index = all_divs.index.tz_localize(None)
                        period_divs = all_divs[(all_divs.index >= start_date) & (all_divs.index <= end_date)]
                        total_dividends = float(period_divs.sum())
                    else:
                        total_dividends = 0.0

                    if old_price > 0:
                        yearly_return = ((current_price - old_price + total_dividends) / old_price) * 100
                        returns_data[etf_name] = yearly_return
                        price_details[etf_name] = {
                            "old_price": old_price,
                            "current_price": current_price,
                            "old_date": old_date,
                            "current_date": current_date,
                            "dividends": total_dividends,
                            "return": yearly_return,
                        }
                        break
                    else:
                        returns_data[etf_name] = 0.0
                        price_details[etf_name] = {"old_price": 0, "current_price": 0, "old_date": "", "current_date": "", "dividends": 0, "return": 0.0}
                        break
                else:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    returns_data[etf_name] = 0.0
                    price_details[etf_name] = {"old_price": 0, "current_price": 0, "old_date": "", "current_date": "", "dividends": 0, "return": 0.0}
            except Exception:
                if attempt < 2:
                    time.sleep(2)
                    continue
                returns_data[etf_name] = 0.0
                price_details[etf_name] = {"old_price": 0, "current_price": 0, "old_date": "", "current_date": "", "dividends": 0, "return": 0.0}

    return returns_data, price_details


@st.cache_data(ttl=3600)
def get_baa_momentum_data(etf_names):
    """BAA 전략용 1M/3M/6M/12M 수익률 및 모멘텀스코어 계산 (분배금 포함)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    PERIOD_DAYS = {"1M": 30, "3M": 91, "6M": 182, "12M": 365}
    results = {}

    for etf_name in etf_names:
        ticker_code = NAME_TO_TICKER.get(etf_name)
        if not ticker_code:
            results[etf_name] = {"1M": None, "3M": None, "6M": None, "12M": None, "momentum_score": None}
            continue

        for attempt in range(3):
            try:
                data = yf.download(ticker_code, start=start_date, end=end_date, progress=False)
                if len(data) < 2:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    results[etf_name] = {"1M": None, "3M": None, "6M": None, "12M": None, "momentum_score": None}
                    break

                data.index = pd.to_datetime(data.index)
                close_col = data["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_col = close_col.dropna()
                if len(close_col) < 2:
                    results[etf_name] = {"1M": None, "3M": None, "6M": None, "12M": None, "momentum_score": None}
                    break

                current_date = close_col.index[-1]
                current_price = float(close_col.iloc[-1])

                tk = yf.Ticker(ticker_code)
                all_divs = tk.dividends
                if not all_divs.empty:
                    all_divs.index = all_divs.index.tz_localize(None)

                returns = {}
                for period_name, days in PERIOD_DAYS.items():
                    past_target = end_date - timedelta(days=days)
                    past_series = close_col.loc[close_col.index <= past_target]
                    if len(past_series) == 0:
                        returns[period_name] = None
                        continue
                    past_price = float(past_series.iloc[-1])
                    past_date_actual = past_series.index[-1]
                    div_sum = 0.0
                    if not all_divs.empty:
                        mask = (all_divs.index >= past_date_actual) & (all_divs.index <= current_date)
                        div_sum = float(all_divs[mask].sum())
                    if past_price > 0:
                        returns[period_name] = ((current_price - past_price + div_sum) / past_price) * 100
                    else:
                        returns[period_name] = None

                r1, r3, r6, r12 = returns.get("1M"), returns.get("3M"), returns.get("6M"), returns.get("12M")
                if all(r is not None for r in [r1, r3, r6, r12]):
                    momentum_score = r1 * 12 + r3 * 4 + r6 * 2 + r12
                else:
                    momentum_score = None

                results[etf_name] = {"1M": r1, "3M": r3, "6M": r6, "12M": r12, "momentum_score": momentum_score}
                break
            except Exception:
                if attempt < 2:
                    time.sleep(2)
                    continue
                results[etf_name] = {"1M": None, "3M": None, "6M": None, "12M": None, "momentum_score": None}

    return results


# ── VAA 전략 배분 계산 ──
def calc_vaa_allocation(returns_data):
    """VAA 전략: 공격자산 모두 양수 → 최고 수익 공격자산 100%, 아니면 최고 수익 수비자산 100%"""
    attack_returns = {name: returns_data.get(name, 0.0) for name in vaa_attack_assets}
    defense_returns = {name: returns_data.get(name, 0.0) for name in vaa_defense_assets}
    attack_min = min(attack_returns.values())

    if attack_min >= 0:
        best = max(attack_returns.items(), key=lambda x: x[1])
        return {
            "selected": best[0],
            "weight": 100.0,
            "status": "공격",
            "reason": "공격자산군 모두 양수",
            "attack_returns": attack_returns,
            "defense_returns": defense_returns,
        }
    else:
        best = max(defense_returns.items(), key=lambda x: x[1])
        return {
            "selected": best[0],
            "weight": 100.0,
            "status": "수비",
            "reason": f"공격자산군 중 음수 존재 (최소: {attack_min:.2f}%)",
            "attack_returns": attack_returns,
            "defense_returns": defense_returns,
        }


# ── 종합듀얼모멘텀 전략 배분 계산 ──
def calc_dm_allocation(returns_data):
    """듀얼모멘텀: 자산군별 25%, 음수 수익률 시 현금"""
    allocation = {}
    details = {}

    for asset_class, etfs in dm_asset_groups.items():
        class_returns = {etf: returns_data.get(etf, 0.0) for etf in etfs}
        min_ret = min(class_returns.values())

        if min_ret < 0:
            details[asset_class] = {
                "invested": False,
                "reason": f"최소 수익률 {min_ret:.2f}% (음수 존재)",
                "selected_etf": None,
                "returns": class_returns,
            }
        else:
            best = max(class_returns.items(), key=lambda x: x[1])[0]
            details[asset_class] = {
                "invested": True,
                "reason": "투자 가능",
                "selected_etf": best,
                "returns": class_returns,
            }
            allocation[best] = 25.0

    return allocation, details


# ── BAA 전략 배분 계산 ──
def calc_baa_allocation(baa_momentum_data, strategy_weight):
    """BAA 전략: 모멘텀스코어 상위 BAA_TOP_N개 양수 종목에 균등 배분"""
    scored = [
        (name, data["momentum_score"])
        for name, data in baa_momentum_data.items()
        if data["momentum_score"] is not None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    positive_top = [(name, score) for name, score in scored[:BAA_TOP_N] if score > 0]
    n_selected = len(positive_top)
    weight_per_asset = strategy_weight / BAA_TOP_N
    allocation = {name: weight_per_asset for name, _ in positive_top}
    cash_pct = strategy_weight - n_selected * weight_per_asset

    details = {
        "scored": scored,
        "positive_top": positive_top,
        "n_selected": n_selected,
    }
    return allocation, cash_pct, details


# ── 통합 배분 계산 (VAA 1/3 + 듀얼모멘텀 1/3 + BAA 1/3) ──
def calc_combined_allocation(returns_data, baa_momentum_data):
    """세 전략을 1/3씨 배분"""
    STRATEGY_WEIGHT = 100.0 / 3  # ≈33.333%

    vaa = calc_vaa_allocation(returns_data)
    dm_alloc, dm_details = calc_dm_allocation(returns_data)
    baa_alloc, baa_cash_pct, baa_details = calc_baa_allocation(baa_momentum_data, STRATEGY_WEIGHT)

    combined = {}
    cash_pct = 0.0

    # VAA 1/3
    vaa_etf = vaa["selected"]
    combined[vaa_etf] = combined.get(vaa_etf, 0.0) + STRATEGY_WEIGHT

    # 듀얼모멘텀 1/3: 각 자산군 (1/3)/4 ≈8.333%
    dm_class_weight = STRATEGY_WEIGHT / len(dm_asset_groups)
    for asset_class, info in dm_details.items():
        if info["invested"]:
            etf = info["selected_etf"]
            combined[etf] = combined.get(etf, 0.0) + dm_class_weight
        else:
            cash_pct += dm_class_weight

    # BAA 1/3: 상위 BAA_TOP_N개 양수 종목에 균등 배분
    for etf, weight in baa_alloc.items():
        combined[etf] = combined.get(etf, 0.0) + weight
    cash_pct += baa_cash_pct

    return combined, cash_pct, vaa, dm_alloc, dm_details, baa_alloc, baa_details


# ── 통합 백테스트 ──
@st.cache_data(ttl=3600)
def run_combined_backtest(today_str: str = ""):
    """통합 전략 백테스트 (VAA 50% + 듀얼모멘텀 50%, 분배금 포함, 매월 첫 거래일 리밸런싱)
    today_str: 날짜 문자열 (캐시 키 일별 무효화용, 예: '2026-06-13')
    """
    all_names = list(NAME_TO_TICKER.keys())
    all_tickers = {name: NAME_TO_TICKER[name] for name in all_names}

    # 전체 가격 데이터 다운로드
    ticker_codes = list(set(all_tickers.values()))
    price_data = yf.download(ticker_codes, period="max", progress=False)["Close"]

    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()
    if hasattr(price_data.columns, "levels"):
        price_data.columns = [c[0] if isinstance(c, tuple) else c for c in price_data.columns]

    # 분배금 데이터
    div_data = {}
    for name, code in all_tickers.items():
        tk = yf.Ticker(code)
        divs = tk.dividends
        if not divs.empty:
            divs.index = divs.index.tz_localize(None)
            div_data[code] = divs
        else:
            div_data[code] = pd.Series(dtype=float)

    # ETF 시작일
    etf_start_dates = {}
    for name, code in all_tickers.items():
        if code in price_data.columns:
            valid = price_data[code].dropna()
            if len(valid) > 0:
                etf_start_dates[name] = valid.index[0]

    # 월 첫 거래일
    price_data.index = pd.to_datetime(price_data.index)
    monthly_first_dates = price_data.groupby(
        [price_data.index.year, price_data.index.month]
    ).apply(lambda x: x.index[0])

    # 백테스트 시작일: 늦게 상장한 ETF(TIGER 미국종합채권한화(H), KODEX 일본부동산리츠(H)) 제외한
    # 나머지 ETF의 데이터가 최소 1년 이상 있어야 함
    late_etfs = {"KODEX 일본부동산리츠(H)"}
    required_names = [n for n in all_names if n not in late_etfs]
    available_required = [n for n in required_names if n in etf_start_dates]
    if not available_required:
        return None, None, None, None, None
    latest_start = max(etf_start_dates[n] for n in available_required)
    backtest_earliest = latest_start + timedelta(days=365)

    rebal_dates = sorted([pd.Timestamp(d) for d in monthly_first_dates.values if d >= backtest_earliest])
    if len(rebal_dates) < 2:
        return None, None, None, None, None

    def calc_return(name, lookback_start, rebal_date):
        code = all_tickers[name]
        if name not in etf_start_dates or etf_start_dates[name] > lookback_start:
            return None
        col_prices = price_data[code]
        mask = col_prices.loc[lookback_start:rebal_date].dropna()
        if len(mask) < 2:
            return None
        old_p = float(mask.iloc[0])
        cur_p = float(mask.iloc[-1])
        divs = div_data.get(code, pd.Series(dtype=float))
        total_div = 0.0
        if not divs.empty:
            period_divs = divs[(divs.index >= mask.index[0]) & (divs.index <= mask.index[-1])]
            total_div = float(period_divs.sum())
        if old_p > 0:
            return ((cur_p - old_p + total_div) / old_p) * 100
        return 0.0

    def calc_period_return(name, start_dt, end_dt):
        code = all_tickers[name]
        col_prices = price_data[code]
        p_range = col_prices.loc[start_dt:end_dt].dropna()
        if len(p_range) < 2:
            return 0.0
        buy = float(p_range.iloc[0])
        sell = float(p_range.iloc[-1])
        divs = div_data.get(code, pd.Series(dtype=float))
        total_div = 0.0
        if not divs.empty:
            period_divs = divs[(divs.index >= p_range.index[0]) & (divs.index <= p_range.index[-1])]
            total_div = float(period_divs.sum())
        if buy > 0:
            return (sell - buy + total_div) / buy
        return 0.0

    def calc_generic_return(name, ref_date, days_back):
        """ref_date 기준 days_back일 전 대비 수익률 계산 (분배금 포함)"""
        code = all_tickers.get(name)
        if code is None or code not in price_data.columns:
            return None
        if name in etf_start_dates and etf_start_dates[name] > ref_date - timedelta(days=days_back + 30):
            return None
        col = price_data[code]
        past_target = ref_date - timedelta(days=days_back)
        past_series = col.loc[col.index <= past_target].dropna()
        if len(past_series) == 0:
            return None
        past_price = float(past_series.iloc[-1])
        past_date_actual = past_series.index[-1]
        cur_series = col.loc[col.index <= ref_date].dropna()
        if len(cur_series) == 0:
            return None
        cur_price = float(cur_series.iloc[-1])
        divs = div_data.get(code, pd.Series(dtype=float))
        div_sum = 0.0
        if not divs.empty:
            mask = (divs.index >= past_date_actual) & (divs.index <= ref_date)
            div_sum = float(divs[mask].sum())
        if past_price > 0:
            return ((cur_price - past_price + div_sum) / past_price) * 100
        return None

    def calc_baa_momentum_score_bt(name, ref_date):
        r1 = calc_generic_return(name, ref_date, 30)
        r3 = calc_generic_return(name, ref_date, 91)
        r6 = calc_generic_return(name, ref_date, 182)
        r12 = calc_generic_return(name, ref_date, 365)
        if all(r is not None for r in [r1, r3, r6, r12]):
            return r1 * 12 + r3 * 4 + r6 * 2 + r12
        return None

    # 백테스트 실행
    portfolio_value = 10000.0
    monthly_records = []

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal_date = rebal_dates[i + 1]
        lookback_start = rebal_date - timedelta(days=365)

        # ── VAA 판단 (50%) ──
        attack_returns = {}
        for name in vaa_attack_assets:
            ret = calc_return(name, lookback_start, rebal_date)
            if ret is not None:
                attack_returns[name] = ret

        defense_returns = {}
        for name in vaa_defense_assets:
            ret = calc_return(name, lookback_start, rebal_date)
            if ret is not None:
                defense_returns[name] = ret

        if attack_returns and all(r >= 0 for r in attack_returns.values()):
            vaa_selected = max(attack_returns.items(), key=lambda x: x[1])[0]
            vaa_status = "공격"
        elif defense_returns:
            vaa_selected = max(defense_returns.items(), key=lambda x: x[1])[0]
            vaa_status = "수비"
        else:
            vaa_selected = "현금"
            vaa_status = "현금"

        # ── 듀얼모멘텀 판단 (1/3) ──
        dm_selections = {}
        for asset_class, etfs in dm_asset_groups.items():
            available_etfs = []
            for name in etfs:
                if name in etf_start_dates and etf_start_dates[name] <= lookback_start:
                    available_etfs.append(name)

            if not available_etfs:
                dm_selections[asset_class] = "현금"
                continue

            class_returns = {}
            for name in available_etfs:
                ret = calc_return(name, lookback_start, rebal_date)
                if ret is not None:
                    class_returns[name] = ret
                else:
                    class_returns[name] = 0.0

            if min(class_returns.values()) < 0:
                dm_selections[asset_class] = "현금"
            else:
                dm_selections[asset_class] = max(class_returns.items(), key=lambda x: x[1])[0]

        # ── BAA 판단 (1/3) ──
        baa_scores = {}
        for name in baa_attack_assets:
            score = calc_baa_momentum_score_bt(name, rebal_date)
            if score is not None:
                baa_scores[name] = score
        sorted_baa = sorted(baa_scores.items(), key=lambda x: x[1], reverse=True)
        baa_selections = [name for name, score in sorted_baa[:BAA_TOP_N] if score > 0]

        # ── 통합 수익률 계산 ──
        # VAA 1/3
        if vaa_selected != "현금":
            vaa_return = calc_period_return(vaa_selected, rebal_date, next_rebal_date)
        else:
            vaa_return = 0.0

        # 듀얼모멘텀 1/3 (각 자산군 25% within DM)
        dm_return = 0.0
        dm_weight = 1.0 / len(dm_asset_groups)
        for asset_class, selected in dm_selections.items():
            if selected != "현금":
                dm_return += dm_weight * calc_period_return(selected, rebal_date, next_rebal_date)

        # BAA 1/3 (각 종목 20% within BAA)
        baa_return = 0.0
        baa_weight_each = 1.0 / BAA_TOP_N
        for name in baa_selections:
            baa_return += baa_weight_each * calc_period_return(name, rebal_date, next_rebal_date)

        total_return = (vaa_return + dm_return + baa_return) / 3.0
        portfolio_value *= (1.0 + total_return)

        monthly_records.append({
            "date": rebal_date,
            "next_date": next_rebal_date,
            "portfolio_value": portfolio_value,
            "monthly_return": total_return * 100,
            "vaa_selected": vaa_selected,
            "vaa_status": vaa_status,
            "dm_selections": dict(dm_selections),
            "baa_selections": list(baa_selections),
            "baa_n": len(baa_selections),
            "in_progress": False,
        })

    # ── 현재 진행 중인 기간 (마지막 리밸런싱일 ~ 오늘) ──
    if rebal_dates:
        current_rebal_date = rebal_dates[-1]
        latest_data_date = price_data.index[-1]
        if latest_data_date > current_rebal_date:
            lookback_start = current_rebal_date - timedelta(days=365)

            attack_returns = {}
            for name in vaa_attack_assets:
                ret = calc_return(name, lookback_start, current_rebal_date)
                if ret is not None:
                    attack_returns[name] = ret

            defense_returns = {}
            for name in vaa_defense_assets:
                ret = calc_return(name, lookback_start, current_rebal_date)
                if ret is not None:
                    defense_returns[name] = ret

            if attack_returns and all(r >= 0 for r in attack_returns.values()):
                cur_vaa_selected = max(attack_returns.items(), key=lambda x: x[1])[0]
                cur_vaa_status = "공격"
            elif defense_returns:
                cur_vaa_selected = max(defense_returns.items(), key=lambda x: x[1])[0]
                cur_vaa_status = "수비"
            else:
                cur_vaa_selected = "현금"
                cur_vaa_status = "현금"

            cur_dm_selections = {}
            for asset_class, etfs in dm_asset_groups.items():
                available_etfs = [n for n in etfs if n in etf_start_dates and etf_start_dates[n] <= lookback_start]
                if not available_etfs:
                    cur_dm_selections[asset_class] = "현금"
                    continue
                class_returns = {}
                for name in available_etfs:
                    ret = calc_return(name, lookback_start, current_rebal_date)
                    class_returns[name] = ret if ret is not None else 0.0
                if min(class_returns.values()) < 0:
                    cur_dm_selections[asset_class] = "현금"
                else:
                    cur_dm_selections[asset_class] = max(class_returns.items(), key=lambda x: x[1])[0]

            cur_baa_scores = {}
            for name in baa_attack_assets:
                score = calc_baa_momentum_score_bt(name, current_rebal_date)
                if score is not None:
                    cur_baa_scores[name] = score
            sorted_cur_baa = sorted(cur_baa_scores.items(), key=lambda x: x[1], reverse=True)
            cur_baa_selections = [name for name, score in sorted_cur_baa[:BAA_TOP_N] if score > 0]

            if cur_vaa_selected != "현금":
                cur_vaa_return = calc_period_return(cur_vaa_selected, current_rebal_date, latest_data_date)
            else:
                cur_vaa_return = 0.0

            cur_dm_return = 0.0
            dm_weight = 1.0 / len(dm_asset_groups)
            for asset_class, selected in cur_dm_selections.items():
                if selected != "현금":
                    cur_dm_return += dm_weight * calc_period_return(selected, current_rebal_date, latest_data_date)

            cur_baa_return = 0.0
            baa_weight_each = 1.0 / BAA_TOP_N
            for name in cur_baa_selections:
                cur_baa_return += baa_weight_each * calc_period_return(name, current_rebal_date, latest_data_date)

            cur_total_return = (cur_vaa_return + cur_dm_return + cur_baa_return) / 3.0
            cur_portfolio_value = portfolio_value * (1.0 + cur_total_return)

            monthly_records.append({
                "date": current_rebal_date,
                "next_date": latest_data_date,
                "portfolio_value": cur_portfolio_value,
                "monthly_return": cur_total_return * 100,
                "vaa_selected": cur_vaa_selected,
                "vaa_status": cur_vaa_status,
                "dm_selections": dict(cur_dm_selections),
                "baa_selections": list(cur_baa_selections),
                "baa_n": len(cur_baa_selections),
                "in_progress": True,
            })

    # 연도별 수익률 (진행 중인 기간 제외)
    yearly_records = []
    completed_records = [r for r in monthly_records if not r.get("in_progress", False)]
    if completed_records:
        df_monthly = pd.DataFrame(completed_records)
        df_monthly["year"] = df_monthly["date"].apply(lambda d: d.year)
        for year in sorted(df_monthly["year"].unique()):
            year_data = df_monthly[df_monthly["year"] == year]
            cumulative = 1.0
            for _, row in year_data.iterrows():
                cumulative *= (1.0 + row["monthly_return"] / 100)
            yearly_records.append({
                "year": year,
                "return": (cumulative - 1.0) * 100,
                "months": len(year_data),
            })

    total_cumulative = (portfolio_value / 10000.0 - 1.0) * 100
    start_date = rebal_dates[0]
    end_date = rebal_dates[-1]

    # MDD 및 Underwater Period (완료된 기간만 사용)
    values = [10000.0] + [r["portfolio_value"] for r in completed_records]
    dates = [start_date] + [r["date"] for r in completed_records]
    peak = values[0]
    mdd = 0.0
    max_underwater_days = 0
    current_underwater_start = None

    for j in range(len(values)):
        if values[j] >= peak:
            peak = values[j]
            if current_underwater_start is not None:
                uw_days = (dates[j] - current_underwater_start).days
                if uw_days > max_underwater_days:
                    max_underwater_days = uw_days
                current_underwater_start = None
        else:
            dd = (values[j] - peak) / peak * 100
            if dd < mdd:
                mdd = dd
            if current_underwater_start is None:
                current_underwater_start = dates[j]

    if current_underwater_start is not None:
        uw_days = (dates[-1] - current_underwater_start).days
        if uw_days > max_underwater_days:
            max_underwater_days = uw_days

    backtest_stats = {"mdd": mdd, "max_underwater_days": max_underwater_days}
    return monthly_records, yearly_records, total_cumulative, (start_date, end_date), backtest_stats


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

st.divider()

# ── 투자금액 입력 (추출 전) ──
st.subheader("💰 투자 금액 입력")
investment_amount = st.number_input(
    "총 투자액을 입력하세요 (원):",
    min_value=0,
    value=st.session_state.get('investment_amount_saved', 10000000),
    step=100000,
    format="%d",
    key="investment_input",
)
if investment_amount != st.session_state.get('investment_amount_saved', 10000000):
    st.session_state['investment_amount_saved'] = investment_amount

st.divider()

if st.button("🔘 추출", use_container_width=True, type="primary"):
    # 전체 ETF 목록 (중복 제거)
    all_etf_names = list(NAME_TO_TICKER.keys())

    with st.spinner("📡 국내 ETF 데이터를 불러오는 중..."):
        returns_data, price_details = get_yearly_returns(all_etf_names)
        baa_momentum_data = get_baa_momentum_data(baa_attack_assets)
        combined, cash_pct, vaa, dm_alloc, dm_details, baa_alloc, baa_details = calc_combined_allocation(returns_data, baa_momentum_data)

    st.success(f"✅ 추출 완료! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # ── 전략 규칙 설명 ──
    st.subheader("📖 통합 전략 규칙")
    st.markdown("""
**전체 자산을 VAA 1/3 + 종합듀얼모멘텀 1/3 + BAA 1/3으로 배분합니다.**

🔹 **VAA 전략 (1/3 ≈33.3%)**
1. 공격자산군 4개의 최근 1년 수익률을 확인합니다.
   - TIGER 미국S&P500 / KODEX 선진국MSCI World / PLUS 신흥국MSCI(합성H) / TIME 미국나스닥100액티브
2. **모두 양수** → 최고 수익률 공격자산에 33.3% 투자
3. **하나라도 음수** → 수비자산군 중 최고 수익률 ETF에 33.3% 투자
   - 수비자산군: TIGER 미국달러단기채권액티브 / TIGER 미국채10년선물 / PLUS 미국장기우량회사채

🔹 **종합듀얼모멘텀 전략 (1/3 ≈33.3%)**
1. 4개 자산군에 각 ≈8.3%씩 배분 (33.3% ÷ 4)
   - 주식 / 채권 / 부동산 / 경제하락
2. 각 자산군 내 2개 ETF 모두 양수 → 최고 수익률 ETF에 투자
3. 하나라도 음수 → 해당 자산군은 **현금 보유**

🔹 **BAA 전략 (1/3 ≈33.3%)**
1. 공격자산 10종목의 모멘텀스코어를 계산합니다.
   - 모멘텀스코어 = 1개월수익×12 + 3개월수익×4 + 6개월수익×2 + 12개월수익×1
2. 상위 **5종목** 선정, 모멘텀스코어 **양수**인 종목에만 각 6.67%(≈1/15)씩 투자
3. 양수 종목 5개 미만 시 나머지 비중은 **현금 보유**
""")

    st.divider()

    # ── VAA 전략 결과 ──
    st.subheader("🎯 VAA 전략 결과 (1/3 ≈33.3%)")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**상태**: {vaa['status']}자산군 매수\n\n**사유**: {vaa['reason']}")
    with col2:
        sel_ret = returns_data.get(vaa["selected"], 0.0)
        st.metric("선택 종목", vaa["selected"], delta=f"{sel_ret:.2f}%", delta_color="inverse")

    # 공격자산군 테이블
    st.markdown("**공격자산군 1년 수익률**")
    atk_rows = []
    for name, ret in vaa["attack_returns"].items():
        atk_rows.append({
            "ETF": name,
            "티커": NAME_TO_TICKER.get(name, "-"),
            "1년 수익률(%)": f"{ret:.2f}%",
            "상태": "✅ 플러스" if ret >= 0 else "❌ 마이너스",
        })
    st.dataframe(pd.DataFrame(atk_rows), use_container_width=True, hide_index=True)

    # 수비자산군 테이블
    st.markdown("**수비자산군 1년 수익률**")
    def_rows = []
    for name, ret in vaa["defense_returns"].items():
        def_rows.append({
            "ETF": name,
            "티커": NAME_TO_TICKER.get(name, "-"),
            "1년 수익률(%)": f"{ret:.2f}%",
        })
    st.dataframe(pd.DataFrame(def_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 듀얼모멘텀 전략 결과 ──
    st.subheader("📊 종합듀얼모멘텀 전략 결과 (1/3 ≈33.3%)")

    col1, col2, col3, col4 = st.columns(4)
    for i, (asset_class, info) in enumerate(dm_details.items()):
        with [col1, col2, col3, col4][i]:
            if info["invested"]:
                st.metric(asset_class, "≈8.3%", delta=f"✅ {info['selected_etf']}")
            else:
                st.metric(asset_class, "≈8.3%", delta="💰 현금 보유")

    dm_rows = []
    for asset_class, info in dm_details.items():
        for etf_name, ret in info["returns"].items():
            dm_rows.append({
                "자산군": asset_class,
                "ETF": etf_name,
                "티커": NAME_TO_TICKER.get(etf_name, "-"),
                "1년 수익률(%)": f"{ret:.2f}%",
            })
    st.dataframe(pd.DataFrame(dm_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── BAA 전략 결과 ──
    st.subheader("📈 BAA 전략 결과 (1/3 ≈33.3%)")
    baa_n = baa_details["n_selected"]
    baa_cash_in_strategy = (BAA_TOP_N - baa_n) * (100.0 / BAA_TOP_N)
    if baa_n > 0:
        st.success(f"**상위 {baa_n}종목 매수** (전략 내 현금 비중: {baa_cash_in_strategy:.0f}%)")
    else:
        st.warning("**모든 종목 모멘텀스코어 음수** → BAA 전략 전액 현금")

    baa_score_rows = []
    scored_names = {n for n, _ in baa_details["scored"]}
    rank = 0
    for name, score in baa_details["scored"]:
        rank += 1
        in_top5 = rank <= BAA_TOP_N
        is_positive = score > 0
        if in_top5 and is_positive:
            flag = "✅ 매수"
        elif in_top5:
            flag = "⬆️ 상위5위(음수)"
        else:
            flag = ""
        baa_score_rows.append({
            "순위": rank,
            "ETF": name,
            "티커": NAME_TO_TICKER.get(name, "-"),
            "모멘텀스코어": f"{score:.2f}",
            "선택": flag,
        })
    for name in baa_attack_assets:
        if name not in scored_names:
            baa_score_rows.append({
                "순위": "-",
                "ETF": name,
                "티커": NAME_TO_TICKER.get(name, "-"),
                "모멘텀스코어": "N/A",
                "선택": "⚠️ 데이터부족",
            })
    st.dataframe(pd.DataFrame(baa_score_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 통합 배분 결과 ──
    st.subheader("📋 통합 배분 결과")

    alloc_rows = []
    for etf_name, weight in sorted(combined.items(), key=lambda x: -x[1]):
        alloc_rows.append({
            "ETF": etf_name,
            "티커": NAME_TO_TICKER.get(etf_name, "-"),
            "배분비중": f"{weight:.1f}%",
            "1년 수익률(%)": f"{returns_data.get(etf_name, 0.0):.2f}%",
        })
    if cash_pct > 0:
        alloc_rows.append({
            "ETF": "💰 현금",
            "티커": "-",
            "배분비중": f"{cash_pct:.1f}%",
            "1년 수익률(%)": "-",
        })
    st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)

    st.info("💡 **데이터 출처**: Yahoo Finance (yfinance)")

    # ── 투자액 배분 및 매수 주수 ──
    if investment_amount > 0:
        st.subheader("🎯 투자액 배분 및 매수 주수")
        inv_rows = []
        total_shares_amount = 0
        total_cash = 0

        for etf_name, weight in sorted(combined.items(), key=lambda x: -x[1]):
            alloc_amount = investment_amount * weight / 100
            cur_price = price_details.get(etf_name, {}).get('current_price', 0)
            ticker = NAME_TO_TICKER.get(etf_name, "-")
            if cur_price and cur_price > 0:
                shares = int(alloc_amount / cur_price)
                actual_amount = int(shares * cur_price)
                leftover = int(alloc_amount - actual_amount)
            else:
                shares = 0
                actual_amount = 0
                leftover = int(alloc_amount)
            total_shares_amount += actual_amount
            inv_rows.append({
                "ETF": etf_name,
                "티커": ticker,
                "배분비중": f"{weight:.1f}%",
                "배분금액(원)": f"{int(alloc_amount):,}",
                "현재가(원)": f"{int(cur_price):,}" if cur_price else "-",
                "매수 주수": f"{shares:,}주" if cur_price else "-",
                "실투자액(원)": f"{actual_amount:,}" if cur_price else "-",
                "잔여현금(원)": f"{leftover:,}",
            })

        if cash_pct > 0:
            cash_amount = int(investment_amount * cash_pct / 100)
            total_cash = cash_amount
            inv_rows.append({
                "ETF": "💰 현금",
                "티커": "-",
                "배분비중": f"{cash_pct:.1f}%",
                "배분금액(원)": f"{cash_amount:,}",
                "현재가(원)": "-",
                "매수 주수": "-",
                "실투자액(원)": "-",
                "잔여현금(원)": "-",
            })

        st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)
        st.caption("※ 매수 주수는 배분금액 ÷ 현재가 소수점 버림 (1주 단위 거래 기준)")

        leftover_total = investment_amount - total_shares_amount - total_cash
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("실매수금액", f"{total_shares_amount:,}원")
        with c2:
            st.metric("현금 보유", f"{total_cash:,}원")
        with c3:
            st.metric("잔여현금(주수 미달)", f"{leftover_total:,}원")
        with c4:
            st.metric("합계", f"{investment_amount:,}원")

    # 추출 결과를 session_state에 저장
    st.session_state['alloc_combined'] = combined
    st.session_state['alloc_cash_pct'] = cash_pct
    st.session_state['alloc_price_details'] = price_details

    st.divider()

    # ── 백테스트 ──
    st.subheader("📉 통합 전략 백테스트 (분배금 포함, 매월 첫 거래일 리밸런싱)")
    with st.spinner("📡 백테스트 데이터 계산 중..."):
        bt_monthly, bt_yearly, bt_total, bt_period, bt_stats = run_combined_backtest(
            today_str=datetime.now().strftime('%Y-%m-%d')
        )

    if bt_monthly and bt_period:
        start_dt, end_dt = bt_period
        in_progress_records = [r for r in bt_monthly if r.get("in_progress", False)]
        latest_data_dt = in_progress_records[0]["next_date"] if in_progress_records else end_dt
        st.caption(
            f"백테스트 구간: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} (완료) "
            f"/ 최신 데이터: {latest_data_dt.strftime('%Y-%m-%d')} | "
            "KODEX 일본부동산리츠(H)는 상장 이후부터 반영"
        )

        n_years = (end_dt - start_dt).days / 365.25
        cagr = ((1 + bt_total / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
        mdd = bt_stats["mdd"]
        uw_days = bt_stats["max_underwater_days"]

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.metric("누적 수익률", f"{bt_total:.2f}%")
        with mc2:
            st.metric("CAGR", f"{cagr:.2f}%")
        with mc3:
            st.metric("MDD", f"{mdd:.2f}%")
        with mc4:
            st.metric("Underwater Period", f"{uw_days}일")
        with mc5:
            st.metric("백테스트 기간", f"{n_years:.1f}년")

        # 연도별 수익률
        st.markdown("**연도별 수익률**")
        yr_rows = []
        for yr in bt_yearly:
            yr_rows.append({
                "연도": str(yr["year"]),
                "수익률": f"{yr['return']:.2f}%",
                "리밸런싱 횟수": f"{yr['months']}회",
            })
        st.dataframe(pd.DataFrame(yr_rows), use_container_width=True, hide_index=True)

        # 월별 수익률
        st.markdown("**월별 수익률**")
        mo_rows = []
        for rec in bt_monthly:
            dm_text = " / ".join(f"{ac}: {s}" for ac, s in rec["dm_selections"].items())
            is_ip = rec.get("in_progress", False)
            mo_rows.append({
                "리밸런싱일": rec["date"].strftime("%Y-%m-%d"),
                "종료일": rec["next_date"].strftime("%Y-%m-%d") + (" ⏳진행중" if is_ip else ""),
                "VAA": f"{rec['vaa_status']}→{rec['vaa_selected']}",
                "듀얼모멘텀": dm_text,
                "BAA": f"{rec['baa_n']}종목" if rec["baa_n"] > 0 else "현금",
                "월 수익률": f"{rec['monthly_return']:.2f}%",
                "포트폴리오 가치": f"{rec['portfolio_value']:,.0f}",
            })
        st.dataframe(pd.DataFrame(mo_rows), use_container_width=True, hide_index=True)
        st.caption("※ 포트폴리오 가치는 초기값 10,000 기준 | ⏳진행중: 해당 월 리밸런싱 기간이 아직 진행 중")

        # 누적 수익률 차트
        st.markdown("**누적 수익률 추이**")
        chart_data = pd.DataFrame({
            "날짜": [rec["date"] for rec in bt_monthly],
            "포트폴리오 가치": [rec["portfolio_value"] for rec in bt_monthly],
        }).set_index("날짜")
        st.line_chart(chart_data)
    else:
        st.warning("백테스트 가능한 데이터가 부족합니다.")

    st.divider()

    # ── 종목별 상세 ──
    st.subheader("🔍 전체 종목 1년 수익률 상세")
    detail_rows = []
    for etf_name in all_etf_names:
        info = price_details.get(etf_name, {})
        old_p = info.get("old_price", 0)
        cur_p = info.get("current_price", 0)
        old_d = info.get("old_date", "")
        cur_d = info.get("current_date", "")
        div = info.get("dividends", 0)
        ret = info.get("return", 0.0)
        detail_rows.append({
            "ETF": etf_name,
            "티커": NAME_TO_TICKER.get(etf_name, "-"),
            "1년전 주가": f"{old_p:,.0f}원 ({old_d})",
            "현재 주가": f"{cur_p:,.0f}원 ({cur_d})",
            "분배금 합계": f"{div:,.0f}원",
            "1년 수익률": f"{ret:.2f}%",
        })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    st.caption("※ 1년 수익률 = (현재 주가 - 1년전 주가 + 분배금) / 1년전 주가 × 100")
