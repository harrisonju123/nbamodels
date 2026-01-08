# API Rate Limiting Setup

## Installing slowapi

```bash
pip install slowapi
```

## Add to requirements.txt

```
slowapi>=0.1.8
```

## Implementation

See api/dashboard_api.py - rate limiting should be added in production.

Example:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/summary")
@limiter.limit("10/minute")
def get_summary(...):
    pass
```
