# 家安 housafe

老人在家安全监护系统 —— 地基（Foundation）

## Tech Stack

- **Backend:** Python 3.12 / Django 5+ / DRF / Channels 4 / Pydantic 2
- **Database:** PostgreSQL 16 + TimescaleDB
- **Cache/PubSub:** Redis 7
- **App:** React Native (Expo SDK 51)

## Quick Start

```bash
# Start infrastructure
docker-compose up -d

# Backend
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python manage.py migrate
daphne housafe.asgi:application
```

## Architecture

See `docs/` for PRD, architecture, and planning documents.
