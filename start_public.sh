#!/bin/bash
# 국내상장통합전략_public — Streamlit + Telegram Bot + Cloudflare Quick Tunnel 실행 스크립트

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="/home/azin/.venv/bin/python"
VENV_STREAMLIT="/home/azin/.venv/bin/streamlit"
CLOUDFLARED="/home/azin/cloudflared"
PORT=8510
LOG_DIR="$APP_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=== 국내상장 통합전략 (공개용) 시작 ==="

# .env 로드 (텔레그램 봇 토큰 등 민감 정보)
if [ -f "$APP_DIR/.env" ]; then
    set -a
    source "$APP_DIR/.env"
    set +a
    echo "      .env 로드 완료"
else
    echo "[경고] .env 파일이 없습니다. telegram_bot.py가 실행되지 않을 수 있습니다."
fi

# 기존 포트 점유 프로세스 종료
fuser -k ${PORT}/tcp 2>/dev/null

# Streamlit 앱 실행 (백그라운드)
echo "[1/3] Streamlit 앱 실행 중 (포트 $PORT)..."
"$VENV_STREAMLIT" run "$APP_DIR/app.py" \
    --server.port $PORT \
    --server.headless true \
    --server.address 127.0.0.1 \
    > "$LOG_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!

# Streamlit 기동 대기
echo "      앱 기동 대기 중..."
sleep 4

if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "[오류] Streamlit 앱 시작 실패. 로그 확인: $LOG_DIR/streamlit.log"
    exit 1
fi
echo "      Streamlit PID: $STREAMLIT_PID"

# Cloudflare Quick Tunnel 실행
echo "[2/2] Cloudflare Quick Tunnel 연결 중..."
echo "      아래에 나타나는 https://*.trycloudflare.com 주소로 외부 접속 가능합니다."
echo "      종료하려면 Ctrl+C 를 누르세요."
echo ""

"$CLOUDFLARED" tunnel --url http://127.0.0.1:${PORT} \
    --no-autoupdate \
    2>&1 | tee "$LOG_DIR/cloudflared.log" | grep -E "trycloudflare|ERR|INF.*https"

# 터널 종료 시 Streamlit 종료
kill $STREAMLIT_PID 2>/dev/null
echo ""
echo "=== 종료됨 ==="
