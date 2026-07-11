### Task 1: 项目脚手架与本地基础设施

**Files:**
- Create: `.gitignore`
- Create: `docker-compose.yml`
- Create: `backend/pyproject.toml`
- Create: `backend/housafe/__init__.py`, `backend/housafe/settings.py`, `backend/housafe/urls.py`, `backend/housafe/asgi.py`, `backend/manage.py`
- Create: `backend/.env.example`
- Create: `README.md`

**Interfaces:**
- Produces: 可运行的 Django ASGI 项目 `housafe`；Postgres(Timescale)/Redis 本地服务；settings 读取 `DATABASE_URL`、`REDIS_URL`。

- [ ] **Step 1: 初始化 git 与忽略文件**

```bash
cd /Users/eular/Desktop/housafe
git init
```

`.gitignore`:
```
__pycache__/
*.pyc
.env
.venv/
node_modules/
.expo/
*.sqlite3
.DS_Store
```

- [ ] **Step 2: docker-compose 起 Timescale + Redis**

`docker-compose.yml`:
```yaml
services:
  db:
    image: timescale/timescaledb:2.15.0-pg16
    environment:
      POSTGRES_DB: housafe
      POSTGRES_USER: housafe
      POSTGRES_PASSWORD: housafe
    ports: ["5432:5432"]
    volumes: ["dbdata:/var/lib/postgresql/data"]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
volumes:
  dbdata:
```

Run: `docker-compose up -d` → Expected: `db` 与 `redis` 容器 healthy。

> Note: 本机只有 `docker-compose`（带连字符），没有 `docker compose` 插件。请使用 `docker-compose up -d`。

- [ ] **Step 3: 建 Python 环境与依赖**

`backend/pyproject.toml`:
```toml
[project]
name = "housafe-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "django>=5.0",
  "djangorestframework>=3.15",
  "djangorestframework-simplejwt>=5.3",
  "channels>=4.0",
  "channels-redis>=4.2",
  "daphne>=4.1",
  "psycopg[binary]>=3.1",
  "pydantic>=2.6",
  "dj-database-url>=2.1",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-django>=4.8", "pytest-asyncio>=0.23"]
```

```bash
cd backend && python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 4: 生成 Django 项目骨架**

```bash
cd backend && django-admin startproject housafe . --name asgi.py
```

编辑 `housafe/settings.py` 关键项：
```python
import os, dj_database_url
from dotenv import load_dotenv
load_dotenv()

INSTALLED_APPS += ["rest_framework", "channels"]
ASGI_APPLICATION = "housafe.asgi.application"
DATABASES = {"default": dj_database_url.parse(
    os.environ.get("DATABASE_URL", "postgres://housafe:housafe@localhost:5432/housafe"))}
CHANNEL_LAYERS = {"default": {
    "BACKEND": "channels_redis.core.RedisChannelLayer",
    "CONFIG": {"hosts": [os.environ.get("REDIS_URL", "redis://localhost:6379/0")]}}}
```

`.env.example`:
```
DATABASE_URL=postgres://housafe:housafe@localhost:5432/housafe
REDIS_URL=redis://localhost:6379/0
```

- [ ] **Step 5: 验证启动并提交**

Run: `python manage.py migrate && python manage.py runserver`
Expected: 无报错，`http://127.0.0.1:8000/` 返回 Django 欢迎页。

```bash
git add -A && git commit -m "chore: scaffold monorepo, django asgi project, docker infra"
```

## Global Constraints (relevant to all tasks)

- Python ≥ 3.12；Django ≥ 5.0；Pydantic ≥ 2.6；Channels ≥ 4.0。
- 每个任务结束必须 commit；提交信息用 `feat:` / `test:` / `chore:` 前缀。
- 测试框架：后端用 `pytest` + `pytest-django` + `pytest-asyncio`；契约用 `pytest`。
