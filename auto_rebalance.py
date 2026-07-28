"""
한국투자증권 Open API를 이용한 통합전략 자동 리밸런싱
(VAA 1/3 + 종합듀얼모멘텀 1/3 + BAA 1/3)

사전 준비:
1. 한국투자증권 Open API 포털에서 API 키 발급
   - https://apiportal.koreainvestment.com 접속
   - 회원가입 → 마이페이지 → API 키 발급 (APP_KEY, APP_SECRET)
2. config.json에 API 키와 계좌정보 입력
   - cano: 계좌번호 앞 8자리
   - acnt_prdt_cd: 계좌 상품코드 (보통 "01")
3. 처음에는 반드시 use_mock: true (모의투자)로 테스트

사용법:
  python auto_rebalance.py              # 신호 확인만 (dry-run)
  python auto_rebalance.py --execute    # 실제 주문 실행
  python auto_rebalance.py --force      # 리밸런싱 날짜 무시하고 실행
"""

import json
import os
import re
import sys
import time
import logging
import argparse
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# ── 로깅 설정 ──
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"rebalance_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── 설정 로드 ──
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    # .env 파일이 있으면 환경변수로 로드
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # config.json 내 ${VAR_NAME} 패턴을 환경변수로 치환
    def _sub(match):
        return os.environ.get(match.group(1), match.group(0))

    content = re.sub(r"\$\{(\w+)\}", _sub, content)
    return json.loads(content)


# ── ETF 정의 ──
VAA_ATTACK = ["TIGER 미국S&P500", "KODEX 선진국MSCI World", "PLUS 신흥국MSCI(합성H)", "TIME 미국나스닥100액티브"]
VAA_DEFENSE = ["TIGER 미국달러단기채권액티브", "TIGER 미국채10년선물", "PLUS 미국장기우량회사채"]

DM_GROUPS = {
    "주식": ["TIGER 미국S&P500", "KODEX 선진국MSCI World"],
    "채권": ["PLUS 미국장기우량회사채", "KODEX 미국하이일드액티브"],
    "부동산": ["TIGER 리츠부동산인프라", "KODEX 일본부동산리츠(H)"],
    "경제하락": ["ACE KRX금현물", "ACE 미국30년국채액티브"],
}

BAA_ATTACK = [
    "TIME 미국나스닥100액티브", "TIGER 미국S&P500", "KODEX 선진국MSCI World",
    "PLUS 신흥국MSCI(합성H)", "TIGER 일본니케이225", "KODEX 200",
    "ACE KRX금현물", "ACE 미국30년국채액티브", "KODEX 미국하이일드액티브",
    "PLUS 미국장기우량회사채",
]
BAA_TOP_N = 5

# ETF 이름 → 종목코드 (6자리)
NAME_TO_CODE = {
    "TIGER 미국S&P500": "360750",
    "KODEX 선진국MSCI World": "251350",
    "PLUS 신흥국MSCI(합성H)": "195980",
    "TIME 미국나스닥100액티브": "426030",
    "TIGER 미국달러단기채권액티브": "329750",
    "TIGER 미국채10년선물": "305080",
    "PLUS 미국장기우량회사채": "332620",
    "KODEX 미국하이일드액티브": "468380",
    "TIGER 리츠부동산인프라": "329200",
    "KODEX 일본부동산리츠(H)": "352540",
    "ACE KRX금현물": "411060",
    "ACE 미국30년국채액티브": "453850",
    "TIGER 일본니케이225": "241180",
    "KODEX 200": "069500",
}

CODE_TO_NAME = {v: k for k, v in NAME_TO_CODE.items()}
CODE_TO_NAME["379810"] = "KODEX 미국나스닥100 (레거시)"
NAME_TO_YF = {name: f"{code}.KS" for name, code in NAME_TO_CODE.items()}

# 종목 교체 시 기존 보유 잔고를 강제 매도하기 위한 레거시 코드 목록
# (전략에서 제외됐지만 실제 잔고에 남아있는 종목코드 → 리밸런싱 시 전량 매도 처리)
LEGACY_CODES = {
    "379810",  # KODEX 미국나스닥100 → TIME 미국나스닥100액티브(426030)로 교체됨
}


# ════════════════════════════════════════════════════════════
# 1. 전략 신호 계산
# ════════════════════════════════════════════════════════════

def get_momentum_score(yf_ticker):
    """BAA 모멘텀 점수: r1*12 + r3*4 + r6*2 + r12*1 (분배금 미포함 근사치)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    try:
        data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
        if len(data) < 60:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        p_now = close.iloc[-1]
        def get_price_n_months_ago(n):
            target = end_date - timedelta(days=int(n * 30.5))
            idx = close.index.searchsorted(target)
            idx = min(max(idx, 0), len(close) - 1)
            return close.iloc[idx]
        p1  = get_price_n_months_ago(1)
        p3  = get_price_n_months_ago(3)
        p6  = get_price_n_months_ago(6)
        p12 = get_price_n_months_ago(12)
        if any(p <= 0 for p in [p1, p3, p6, p12]):
            return None
        r1  = (p_now / p1  - 1) * 100
        r3  = (p_now / p3  - 1) * 100
        r6  = (p_now / p6  - 1) * 100
        r12 = (p_now / p12 - 1) * 100
        return r1 * 12 + r3 * 4 + r6 * 2 + r12 * 1
    except Exception as e:
        log.warning(f"{yf_ticker} BAA 모멘텀 점수 계산 실패: {e}")
        return None


def get_baa_momentum_data():
    """BAA 공격 자산 모멘텀 점수 계산"""
    scores = {}
    for name in BAA_ATTACK:
        code = NAME_TO_CODE[name]
        yf_ticker = f"{code}.KS"
        score = get_momentum_score(yf_ticker)
        scores[name] = score
        log.info(f"[BAA] {name}: 모멘텀 점수 {score:.2f}" if score is not None else f"[BAA] {name}: 점수 계산 실패")
    return scores


def get_yearly_returns():
    """전체 ETF의 최근 1년 수익률 계산 (분배금 포함)"""
    returns_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for etf_name, yf_ticker in NAME_TO_YF.items():
        for attempt in range(3):
            try:
                data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 1:
                    old_price = data["Close"].iloc[0].item()
                    current_price = data["Close"].iloc[-1].item()

                    tk = yf.Ticker(yf_ticker)
                    all_divs = tk.dividends
                    total_dividends = 0.0
                    if not all_divs.empty:
                        all_divs.index = all_divs.index.tz_localize(None)
                        period_divs = all_divs[(all_divs.index >= start_date) & (all_divs.index <= end_date)]
                        total_dividends = float(period_divs.sum())

                    if old_price > 0:
                        returns_data[etf_name] = ((current_price - old_price + total_dividends) / old_price) * 100
                        break
                    else:
                        returns_data[etf_name] = 0.0
                        break
                else:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    returns_data[etf_name] = 0.0
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                log.warning(f"{etf_name} 수익률 조회 실패: {e}")
                returns_data[etf_name] = 0.0

    return returns_data


def calc_target_allocation(returns_data, baa_scores=None):
    """통합전략 목표 배분 비중 계산 (VAA 1/3 + DM 1/3 + BAA 1/3)"""

    # ── VAA (1/3 = 33.33%) ──
    attack_returns = {n: returns_data.get(n, 0.0) for n in VAA_ATTACK}
    defense_returns = {n: returns_data.get(n, 0.0) for n in VAA_DEFENSE}

    if min(attack_returns.values()) >= 0:
        vaa_selected = max(attack_returns.items(), key=lambda x: x[1])[0]
        vaa_status = "공격"
    else:
        vaa_selected = max(defense_returns.items(), key=lambda x: x[1])[0]
        vaa_status = "수비"

    log.info(f"[VAA] {vaa_status} → {vaa_selected} (1yr: {returns_data.get(vaa_selected, 0):.2f}%)")

    # ── 듀얼모멘텀 (1/3) ──
    dm_selections = {}
    for asset_class, etfs in DM_GROUPS.items():
        class_returns = {e: returns_data.get(e, 0.0) for e in etfs}
        if min(class_returns.values()) < 0:
            dm_selections[asset_class] = None
            log.info(f"[DM-{asset_class}] 현금 보유 (음수 존재)")
        else:
            best = max(class_returns.items(), key=lambda x: x[1])[0]
            dm_selections[asset_class] = best
            log.info(f"[DM-{asset_class}] {best} (1yr: {returns_data.get(best, 0):.2f}%)")

    # ── BAA (1/3) ──
    if baa_scores is None:
        baa_scores = get_baa_momentum_data()

    valid_scores = {n: s for n, s in baa_scores.items() if s is not None and s > 0}
    baa_selections = sorted(valid_scores.items(), key=lambda x: -x[1])[:BAA_TOP_N]
    baa_cash_slots = BAA_TOP_N - len(baa_selections)
    log.info(f"[BAA] 선택 종목: {[n for n, _ in baa_selections]}, 현금 슬롯: {baa_cash_slots}")

    # ── 통합 비중 계산 (각 전략 1/3) ──
    target = {}
    cash_pct = 0.0
    each_strategy = 100.0 / 3.0  # ~33.33%

    # VAA: 1/3 전체를 선택 종목에 배분
    target[vaa_selected] = target.get(vaa_selected, 0.0) + each_strategy

    # DM: 4개 그룹, 각 그룹 = (1/3) / 4
    dm_per_group = each_strategy / len(DM_GROUPS)
    for asset_class, selected in dm_selections.items():
        if selected:
            target[selected] = target.get(selected, 0.0) + dm_per_group
        else:
            cash_pct += dm_per_group

    # BAA: TOP_N 슬롯 균등, 점수 양수 아닌 슬롯은 현금
    baa_per_slot = each_strategy / BAA_TOP_N
    for name, _ in baa_selections:
        target[name] = target.get(name, 0.0) + baa_per_slot
    cash_pct += baa_cash_slots * baa_per_slot

    log.info(f"[통합] 목표 배분: {target}, 현금: {cash_pct:.2f}%")
    return target, cash_pct, vaa_status, vaa_selected, dm_selections, baa_selections, baa_cash_slots


# ════════════════════════════════════════════════════════════
# 2. 한국투자증권 Open API
# ════════════════════════════════════════════════════════════

class KISOpenAPI:
    """
    한국투자증권 Open API 래퍼.
    공식 문서: https://apiportal.koreainvestment.com/apiservice

    실전 도메인: https://openapi.koreainvestment.com:9443
    모의 도메인: https://openapivts.koreainvestment.com:29443
    """

    REAL_URL = "https://openapi.koreainvestment.com:9443"
    MOCK_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, config):
        api_cfg = config["api"]
        self.app_key = api_cfg["app_key"]
        self.app_secret = api_cfg["app_secret"]
        self.cano = api_cfg["cano"]                    # 계좌번호 앞 8자리
        self.acnt_prdt_cd = api_cfg["acnt_prdt_cd"]    # 상품코드 (보통 "01")
        self.use_mock = api_cfg.get("use_mock", True)

        self.base_url = self.MOCK_URL if self.use_mock else self.REAL_URL
        if self.use_mock:
            log.info("⚠️  모의투자 모드로 실행합니다.")
        else:
            log.info("🔴 실전투자 모드로 실행합니다!")

        # 실전 API 키 (시세조회용, 모의투자 서버에서는 시세 API 미지원)
        real_cfg = config.get("_real_api", {})
        self.real_app_key = real_cfg.get("app_key", self.app_key)
        self.real_app_secret = real_cfg.get("app_secret", self.app_secret)

        self.access_token = None
        self.token_expires = None
        self.real_access_token = None
        self.real_token_expires = None

    # 토큰 캐시 파일 경로 (봇 재시작 시에도 재사용)
    TOKEN_CACHE_FILE      = Path(__file__).parent / ".token_cache.json"
    REAL_TOKEN_CACHE_FILE = Path(__file__).parent / ".real_token_cache.json"

    def _load_token_cache(self, cache_file: Path):
        """디스크에서 토큰 캐시 로드"""
        try:
            if cache_file.exists():
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                expires = datetime.fromisoformat(data["expires"])
                if datetime.now() < expires:
                    return data["token"], expires
        except Exception:
            pass
        return None, None

    def _save_token_cache(self, cache_file: Path, token: str, expires: datetime):
        """토큰을 디스크에 저장"""
        try:
            cache_file.write_text(
                json.dumps({"token": token, "expires": expires.isoformat()}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning(f"토큰 캐시 저장 실패: {e}")

    def _get_token(self):
        """모의/실전 거래용 토큰 발급 (파일 캐시 우선 사용)"""
        # 1) 메모리 캐시
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return self.access_token
        # 2) 파일 캐시
        token, expires = self._load_token_cache(self.TOKEN_CACHE_FILE)
        if token:
            self.access_token = token
            self.token_expires = expires
            log.info("거래용 토큰 캐시 재사용")
            return self.access_token

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self.access_token = data["access_token"]
        expires_in = data.get("expires_in", 86400)
        self.token_expires = datetime.now() + timedelta(seconds=expires_in - 3600)
        self._save_token_cache(self.TOKEN_CACHE_FILE, self.access_token, self.token_expires)
        log.info("거래용 토큰 발급 완료")
        return self.access_token

    def _get_real_token(self):
        """실전 서버 토큰 발급 (시세조회용 — 파일 캐시 우선 사용)"""
        if not self.use_mock:
            return self._get_token()
        # 1) 메모리 캐시
        if self.real_access_token and self.real_token_expires and datetime.now() < self.real_token_expires:
            return self.real_access_token
        # 2) 파일 캐시
        token, expires = self._load_token_cache(self.REAL_TOKEN_CACHE_FILE)
        if token:
            self.real_access_token = token
            self.real_token_expires = expires
            log.info("시세조회용 토큰 캐시 재사용")
            return self.real_access_token

        url = f"{self.REAL_URL}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.real_app_key,
            "appsecret": self.real_app_secret,
        }
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self.real_access_token = data["access_token"]
        expires_in = data.get("expires_in", 86400)
        self.real_token_expires = datetime.now() + timedelta(seconds=expires_in - 3600)
        self._save_token_cache(self.REAL_TOKEN_CACHE_FILE, self.real_access_token, self.real_token_expires)
        log.info("시세조회용 토큰 발급 완료 (실전 서버)")
        return self.real_access_token

    def _headers(self, tr_id):
        """거래 API용 헤더"""
        token = self._get_token()
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _real_headers(self, tr_id):
        """시세조회 API용 헤더 (실전 서버)"""
        token = self._get_real_token()
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.real_app_key,
            "appsecret": self.real_app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def get_balance(self):
        """
        주식 잔고 조회 (GET /uapi/domestic-stock/v1/trading/inquire-balance)
        tr_id: TTTC8434R (실전) / VTTC8434R (모의)

        Returns:
          holdings: {종목코드: {"qty", "avg_price", "current_price", "eval_amount"}, ...}
          total_eval: 총 평가금액
          cash: 예수금(D+2)
        """
        tr_id = "VTTC8434R" if self.use_mock else "TTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = self._headers(tr_id)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"잔고조회 실패: {data.get('msg1', '')}")

        # 전략에서 사용하는 종목코드 집합
        strategy_codes = set(NAME_TO_CODE.values())

        holdings = {}
        for item in data.get("output1", []):
            code = item.get("pdno", "")
            qty = int(item.get("hldg_qty", 0))
            if qty > 0:
                if code not in strategy_codes:
                    if code in LEGACY_CODES:
                        log.warning(f"레거시 종목 발견 (교체 매도 예정): {item.get('prdt_name', code)} ({code}) x {qty}주")
                    else:
                        log.warning(f"전략 외 종목 무시: {item.get('prdt_name', code)} ({code}) x {qty}주")
                        continue
                # ord_psbl_qty: 실제 주문가능수량 (미체결 매수 주문 제외)
                ord_psbl_qty = int(item.get("ord_psbl_qty", qty))
                holdings[code] = {
                    "qty": ord_psbl_qty,
                    "avg_price": float(item.get("pchs_avg_pric", 0)),
                    "current_price": float(item.get("prpr", 0)),
                    "eval_amount": float(item.get("evlu_amt", 0)),
                }

        summary = data.get("output2", [{}])
        if isinstance(summary, list):
            summary = summary[0] if summary else {}
        total_eval = float(summary.get("tot_evlu_amt", 0))
        # 실제 가용 현금 = 예수금 - 당일매수금 - 전일매수금 + 당일매도금 + 전일매도금
        # (T+2 미결제: 리밸런싱 당일 및 다음날 모두 올바른 예수금 표시)
        dnca      = float(summary.get("dnca_tot_amt", 0))
        buy       = float(summary.get("thdt_buy_amt", 0))
        sell      = float(summary.get("thdt_sll_amt", 0))
        bfdy_buy  = float(summary.get("bfdy_buy_amt", 0))
        bfdy_sell = float(summary.get("bfdy_sll_amt", 0))
        cash      = dnca - buy - bfdy_buy + sell + bfdy_sell

        return holdings, total_eval, cash

    def get_returns_detail(self):
        """
        포트폴리오 수익률 상세 조회.
        Returns:
          items: [{"code", "name", "qty", "avg_price", "current_price",
                   "eval_amount", "purchase_amount", "pnl", "pnl_pct"}, ...]
          portfolio: {"purchase_amt", "eval_amt", "pnl", "pnl_pct", "cash", "total_asset"}
        """
        tr_id = "VTTC8434R" if self.use_mock else "TTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = self._headers(tr_id)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
            "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("rt_cd") != "0":
            raise RuntimeError(f"잔고조회 실패: {data.get('msg1', '')}")

        strategy_codes = set(NAME_TO_CODE.values())
        items = []
        for item in data.get("output1", []):
            code = item.get("pdno", "")
            qty  = int(item.get("hldg_qty", 0))
            if qty <= 0 or code not in strategy_codes:
                continue
            avg_price      = float(item.get("pchs_avg_pric", 0))
            current_price  = float(item.get("prpr", 0))
            purchase_amt   = float(item.get("pchs_amt", 0)) or avg_price * qty
            eval_amt       = float(item.get("evlu_amt", 0)) or current_price * qty
            pnl            = float(item.get("evlu_pfls_amt", eval_amt - purchase_amt))
            pnl_pct        = float(item.get("evlu_erng_rt", 0))
            if pnl_pct == 0 and purchase_amt > 0:
                pnl_pct = pnl / purchase_amt * 100
            items.append({
                "code": code,
                "name": CODE_TO_NAME.get(code, code),
                "qty": qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "purchase_amount": purchase_amt,
                "eval_amount": eval_amt,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

        summary = data.get("output2", [{}])
        if isinstance(summary, list):
            summary = summary[0] if summary else {}
        purchase_amt_total = float(summary.get("pchs_amt_smtl_amt", sum(i["purchase_amount"] for i in items)))
        eval_amt_total     = float(summary.get("evlu_amt_smtl_amt",  sum(i["eval_amount"] for i in items)))
        pnl_total          = float(summary.get("evlu_pfls_smtl_amt", eval_amt_total - purchase_amt_total))
        # 실제 가용 현금 = 예수금 - 당일매수금 - 전일매수금 + 당일매도금 + 전일매도금
        # (T+2 미결제: 리밸런싱 당일 및 다음날 모두 올바른 예수금 표시)
        dnca               = float(summary.get("dnca_tot_amt", 0))
        buy                = float(summary.get("thdt_buy_amt", 0))
        sell               = float(summary.get("thdt_sll_amt", 0))
        bfdy_buy           = float(summary.get("bfdy_buy_amt", 0))
        bfdy_sell          = float(summary.get("bfdy_sll_amt", 0))
        cash               = dnca - buy - bfdy_buy + sell + bfdy_sell
        pnl_pct_total      = pnl_total / purchase_amt_total * 100 if purchase_amt_total > 0 else 0

        portfolio = {
            "purchase_amt": purchase_amt_total,
            "eval_amt":     eval_amt_total,
            "pnl":          pnl_total,
            "pnl_pct":      pnl_pct_total,
            "cash":         cash,
            "total_asset":  eval_amt_total + cash,
        }
        # 수익률 내림차순 정렬
        items.sort(key=lambda x: -x["pnl_pct"])
        return items, portfolio

    def get_current_price(self, stock_code):
        """
        현재가 조회 (GET /uapi/domestic-stock/v1/quotations/inquire-price)
        tr_id: FHKST01010100
        ※ 시세조회는 항상 실전 서버 사용 (모의투자 서버에서 미지원)
        """
        url = f"{self.REAL_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._real_headers("FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            log.warning(f"현재가 조회 실패 ({stock_code}): {data.get('msg1', '')}")
            return 0

        return int(data.get("output", {}).get("stck_prpr", 0))

    def get_ask_bid_price(self, stock_code):
        """
        매도호가(ask) / 매수호가(bid) 조회 — 지정가 주문용
        매수 시: ask(매도호가 1위) 사용 → 즉시 체결 가능
        매도 시: bid(매수호가 1위) 사용 → 즉시 체결 가능
        """
        url = f"{self.REAL_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._real_headers("FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            log.warning(f"호가 조회 실패 ({stock_code}): {data.get('msg1', '')}")
            return 0, 0

        output = data.get("output", {})
        current = int(output.get("stck_prpr", 0))
        ask = int(output.get("askp", 0)) or current   # 매도호가 (매수주문용)
        bid = int(output.get("bidp", 0)) or current   # 매수호가 (매도주문용)
        log.debug(f"{stock_code} ask={ask:,} bid={bid:,} current={current:,}")
        return ask, bid

    @staticmethod
    def _round_to_tick(price: int, side: str) -> int:
        """
        KRX 호가단위에 맞게 가격 보정.
        매수(buy): 올림 → 즉시 체결 보장
        매도(sell): 내림 → 즉시 체결 보장

        KRX 주식·ETF 호가단위:
          price <   1,000 →    1원
          price <   5,000 →    5원
          price <  10,000 →   10원
          price <  50,000 →   50원
          price < 100,000 →  100원
          price < 500,000 →  500원
          price ≥ 500,000 → 1,000원
        """
        if price <= 0:
            return price
        if price < 1_000:
            tick = 1
        elif price < 5_000:
            tick = 5
        elif price < 10_000:
            tick = 10
        elif price < 50_000:
            tick = 50
        elif price < 100_000:
            tick = 100
        elif price < 500_000:
            tick = 500
        else:
            tick = 1_000

        if side == "buy":
            adjusted = ((price + tick - 1) // tick) * tick  # 올림
        else:
            adjusted = (price // tick) * tick               # 내림

        if adjusted != price:
            log.debug(f"호가단위 보정: {price:,}원 → {adjusted:,}원 (tick={tick}, side={side})")
        return adjusted

    def place_order(self, stock_code, qty, side, price=0):
        """
        주문 실행 (POST /uapi/domestic-stock/v1/trading/order-cash)
        side: "buy" 또는 "sell"
        price=0 이면 시장가.

        tr_id:
          실전 매수: TTTC0802U / 매도: TTTC0801U
          모의 매수: VTTC0802U / 매도: VTTC0801U
        """
        # 호가단위 보정 (price > 0인 지정가 주문만 적용)
        if price > 0:
            price = self._round_to_tick(price, side)

        if side == "buy":
            tr_id = "VTTC0802U" if self.use_mock else "TTTC0802U"
        else:
            tr_id = "VTTC0801U" if self.use_mock else "TTTC0801U"

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        headers = self._headers(tr_id)

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": stock_code,
            "ORD_DVSN": "01" if price == 0 else "00",   # 01=시장가, 00=지정가
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
        }

        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        rt_cd = data.get("rt_cd", "")
        msg = data.get("msg1", "")
        if rt_cd == "0":
            order_no = data.get("output", {}).get("ODNO", "")
            log.info(f"✅ 주문 성공: {side.upper()} {stock_code} x {qty}주 (주문번호: {order_no})")
            return True, order_no
        else:
            log.error(f"❌ 주문 실패: {side.upper()} {stock_code} x {qty}주 — {msg}")
            return False, msg


# ════════════════════════════════════════════════════════════
# 3. 리밸런싱 실행
# ════════════════════════════════════════════════════════════

def is_first_business_day():
    """오늘이 이번 달 첫 영업일인지 확인 (주말 제외, 공휴일은 미확인)"""
    today = datetime.now()
    day = today.replace(day=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return today.date() == day.date()


def calc_rebalance_orders(target_alloc, cash_pct, api):
    """
    목표 비중과 현재 잔고를 비교해 필요한 매수/매도 주문 목록 생성.

    ※ 주수(株數) 기준 델타 리밸런싱:
       목표 보유 주수 - 현재 보유 주수 = 거래할 주수 (차이만큼만 매매)
       → 이미 충분히 보유 중이면 거래하지 않아 수수료 최소화

    Returns: list of {"side": "buy"|"sell", "code": str, "name": str, "qty": int, "price": int}
    """
    config = load_config()
    min_trade = config["strategy"].get("min_trade_amount", 10000)

    holdings, total_eval, cash_balance = api.get_balance()

    # 전략 종목만의 평가액 합산 (API total_eval에는 전략 외 종목도 포함될 수 있음)
    strategy_eval = sum(info["eval_amount"] for info in holdings.values())
    total_asset = strategy_eval + cash_balance
    log.info(f"총 자산: {total_asset:,.0f}원 (전략종목 평가: {strategy_eval:,.0f} + 예수금: {cash_balance:,.0f})")

    # 목표 투자금액 (종목코드 → 원)
    target_amounts = {}
    for etf_name, weight in target_alloc.items():
        code = NAME_TO_CODE[etf_name]
        target_amounts[code] = total_asset * weight / 100.0

    orders = []
    all_codes = set(list(holdings.keys()) + list(target_amounts.keys()))

    for code in all_codes:
        current_qty = holdings.get(code, {}).get("qty", 0)
        target_amount = target_amounts.get(code, 0)

        # 실시간 호가 조회 (매수=매도1호가, 매도=매수1호가)
        ask, bid = api.get_ask_bid_price(code)
        if ask <= 0 and bid <= 0:
            log.warning(f"{code} 호가 조회 실패, 건너뜁니다")
            continue

        # 중간가(호가 평균)로 목표 주수 산출
        mid_price = (ask + bid) // 2 if ask > 0 and bid > 0 else (ask or bid)
        target_qty = int(target_amount / mid_price) if mid_price > 0 else 0

        delta_qty = target_qty - current_qty  # + = 매수, - = 매도

        name = CODE_TO_NAME.get(code, code)
        log.info(
            f"  [{code}] {name}: 현재 {current_qty}주 → 목표 {target_qty}주 "
            f"({'±0, 거래없음' if delta_qty == 0 else f'{delta_qty:+d}주'})"
        )

        if delta_qty == 0:
            continue

        if delta_qty > 0:
            exec_price = ask if ask > 0 else mid_price
            if delta_qty * exec_price < min_trade:
                log.info(f"    → 최소거래금액 미만 건너뜀 ({delta_qty * exec_price:,.0f}원 < {min_trade:,}원)")
                continue
            orders.append({"side": "buy", "code": code, "name": name, "qty": delta_qty, "price": exec_price})
        else:
            sell_qty = min(abs(delta_qty), current_qty)
            if sell_qty == 0:
                continue
            exec_price = bid if bid > 0 else mid_price
            if sell_qty * exec_price < min_trade:
                log.info(f"    → 최소거래금액 미만 건너뜀 ({sell_qty * exec_price:,.0f}원 < {min_trade:,}원)")
                continue
            orders.append({"side": "sell", "code": code, "name": name, "qty": sell_qty, "price": exec_price})

    # 매도 먼저, 매수 나중 (현금 확보 후 매수)
    orders.sort(key=lambda x: 0 if x["side"] == "sell" else 1)
    return orders, total_asset


def execute_rebalance(orders, api, dry_run=True):
    """주문 실행 (dry_run=True면 로그만 출력)"""
    if not orders:
        log.info("리밸런싱 필요 없음 (변경 없음)")
        return

    log.info(f"\n{'='*60}")
    log.info(f"{'[DRY-RUN] ' if dry_run else ''}리밸런싱 주문 목록:")
    log.info(f"{'='*60}")

    for o in orders:
        side_kr = "매수" if o["side"] == "buy" else "매도"
        amount = o["qty"] * o["price"]
        log.info(f"  {side_kr}: {o['name']} ({o['code']}) x {o['qty']}주 @ {o['price']:,}원 ≈ {amount:,}원")

    if dry_run:
        log.info("\n⚠️  DRY-RUN 모드입니다. 실제 주문은 실행되지 않았습니다.")
        log.info("실제 주문을 실행하려면: python auto_rebalance.py --execute")
        return

    log.info("\n🔴 실제 주문을 실행합니다... (호가 기준 지정가)")
    for o in orders:
        # 승인 시점에 실시간 호가 재조회
        ask, bid = api.get_ask_bid_price(o["code"])
        exec_price = ask if o["side"] == "buy" else bid
        if exec_price <= 0:
            log.warning(f"{o['code']} 호가 조회 실패 — 시장가로 대체")
            exec_price = 0
        side_kr = "매수" if o["side"] == "buy" else "매도"
        order_type = f"지정가 {exec_price:,}원" if exec_price else "시장가"
        log.info(f"  {side_kr}: {o['name']} ({o['code']}) x {o['qty']}주 @ {order_type}")
        success, result = api.place_order(o["code"], o["qty"], o["side"], price=exec_price)
        if not success:
            log.error(f"주문 실패로 중단합니다: {result}")
            break
        time.sleep(1)  # API 초당 호출 제한 대응


# ════════════════════════════════════════════════════════════
# 4. 텔레그램 알림 (선택)
# ════════════════════════════════════════════════════════════

def send_telegram(message, config):
    """텔레그램 알림 전송"""
    noti = config.get("notification", {})
    if not noti.get("enabled"):
        return
    token = noti.get("telegram_token", "")
    chat_id = noti.get("telegram_chat_id", "")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        log.warning(f"텔레그램 전송 실패: {e}")


# ════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="통합전략 자동 리밸런싱 (VAA 1/3 + DM 1/3 + BAA 1/3)")
    parser.add_argument("--execute", action="store_true", help="실제 주문 실행 (기본: dry-run)")
    parser.add_argument("--force", action="store_true", help="리밸런싱 날짜 체크 무시")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("통합전략 자동 리밸런싱 시작 (한국투자증권 Open API)")
    log.info(f"시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)

    if not args.force and not is_first_business_day():
        log.info("오늘은 리밸런싱 날이 아닙니다 (매월 첫 영업일). --force로 강제 실행 가능.")
        return

    config = load_config()
    if config["api"]["app_key"] == "YOUR_APP_KEY":
        log.error("❌ config.json에 API 키를 입력해주세요!")
        log.error("   한국투자증권 Open API 키 발급: https://apiportal.koreainvestment.com")
        sys.exit(1)

    # 1. 전략 신호
    log.info("\n📡 ETF 수익률 데이터 수집 중...")
    returns_data = get_yearly_returns()

    log.info("\n📊 목표 배분 비중 계산 중...")
    log.info("BAA 모멘텀 점수 계산 중...")
    baa_scores = get_baa_momentum_data()
    target, cash_pct, vaa_status, vaa_selected, dm_selections, baa_selections, baa_cash_slots = calc_target_allocation(returns_data, baa_scores)

    summary_lines = [
        f"📊 <b>통합전략 리밸런싱 신호</b> ({datetime.now():%Y-%m-%d})",
        f"\n🔹 <b>VAA (1/3)</b>: {vaa_status} → {vaa_selected}",
        f"🔹 <b>듀얼모멘텀 (1/3)</b>:",
    ]
    for ac, sel in dm_selections.items():
        summary_lines.append(f"  • {ac}: {sel or '현금'}")
    summary_lines.append(f"🔹 <b>BAA (1/3)</b>:")
    for name, score in baa_selections:
        summary_lines.append(f"  • {name} (점수: {score:.1f})")
    if baa_cash_slots > 0:
        summary_lines.append(f"  • 현금 {baa_cash_slots}슬롯")
    summary_lines.append(f"\n💰 현금 비중: {cash_pct:.1f}%")
    summary_lines.append(f"\n🎯 <b>목표 배분</b>:")
    for name, w in sorted(target.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  • {name}: {w:.1f}%")
    summary = "\n".join(summary_lines)
    log.info("\n" + summary.replace("<b>", "").replace("</b>", ""))

    # 2. API 연결 & 리밸런싱
    api = KISOpenAPI(config)

    try:
        orders, total_asset = calc_rebalance_orders(target, cash_pct, api)
    except requests.exceptions.RequestException as e:
        log.error(f"❌ API 통신 오류: {e}")
        log.error("API 키와 네트워크 연결을 확인하세요.")
        send_telegram(f"❌ 리밸런싱 실패: API 통신 오류\n{e}", config)
        sys.exit(1)

    # 3. 주문 실행
    dry_run = not args.execute
    execute_rebalance(orders, api, dry_run=dry_run)

    # 4. 알림
    if orders:
        order_text = "\n".join(
            f"  {'매수' if o['side']=='buy' else '매도'}: {o['name']} x {o['qty']}주"
            for o in orders
        )
        noti_msg = f"{summary}\n\n📝 <b>{'[DRY-RUN] ' if dry_run else ''}주문 목록</b>:\n{order_text}"
        send_telegram(noti_msg, config)

    log.info("\n✅ 완료")


if __name__ == "__main__":
    main()
