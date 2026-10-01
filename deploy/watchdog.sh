#!/usr/bin/env bash
# Runs from cron on the instance every 15 minutes (installed by deploy.sh). A crashed or stuck bot cannot send its own
# alert, so this tells the owner on Telegram when a bot container is not running, or the swing bot has printed nothing
# for 75 minutes (its loop prints at least every 30: a cycle's results, or "market closed; waiting"); and once more
# when everything is back. It only reads: it never restarts anything (Docker's restart policy already does that).
set -u
cd "$HOME/ai-trading-bot" || exit 0
COMPOSE="sudo docker compose -f docker-compose.aws.yml"
STATE="$HOME/.bot-watchdog-state"
value() { grep "^$1=" .env | head -1 | cut -d= -f2- | tr -d '"\r'; }
TOKEN=$(value TELEGRAM_BOT_TOKEN)
CHAT=$(value TELEGRAM_CHAT_ID)

problem=""
for service in trading-bot intraday-bot; do
    state=$($COMPOSE ps --format '{{.State}}' "$service" 2>/dev/null | head -1)
    [ "$state" = "running" ] || problem="$problem $service is ${state:-not running};"
done
if [ -z "$problem" ] && [ -z "$($COMPOSE logs --since 75m trading-bot 2>/dev/null | grep -v '"ts"')" ]; then
    problem=" trading-bot has printed nothing for 75 minutes (stuck?);"
fi

send() {
    [ -n "$TOKEN" ] && [ -n "$CHAT" ] && curl -s -m 20 -o /dev/null "https://api.telegram.org/bot$TOKEN/sendMessage" \
        --data-urlencode "chat_id=$CHAT" --data-urlencode "text=$1"
}
previous=$(cat "$STATE" 2>/dev/null)
if [ -n "$problem" ] && [ "$previous" != "down" ]; then
    send "WATCHDOG: the trading bot server needs attention:$problem Check: docker compose -f docker-compose.aws.yml ps"
    echo down > "$STATE"
elif [ -z "$problem" ] && [ "$previous" = "down" ]; then
    send "WATCHDOG: the trading bot is running normally again."
    echo up > "$STATE"
fi
