"""
Canonical host-side database URL defaults.

Docker Compose maps Postgres to host port 15439 (container 5432).
In-compose services still use host `db` on 5432.
"""

from __future__ import annotations

# Host machine → published compose port
DEFAULT_HOST_DATABASE_URL = "postgresql://oracle:oracle@localhost:15439/oracle"

# Inside the compose network
DEFAULT_COMPOSE_DATABASE_URL = "postgresql://oracle:oracle@db:5432/oracle"
