FROM python:3.13-slim

# UTF-8 output (the rupee sign); LangGraph/LangSmith telemetry explicitly off.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PYTHONIOENCODING=utf-8 \
    LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false LOG_FORMAT=json

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py ./
COPY src ./src

RUN useradd --create-home bot && mkdir /data && chown bot /data
USER bot
ENV DATABASE_URL=sqlite:////data/trading.db
VOLUME /data

# Indian market, dry run (decisions only), every 30 minutes while NSE is open. Analysis uses completed daily bars and
# AI answers are cached, so most cycles are near-instant; the loop mainly settles stops/targets and acts on new signals.
# Add --live to place paper orders in the built-in simulator.
CMD ["python", "main.py", "--loop", "30"]
