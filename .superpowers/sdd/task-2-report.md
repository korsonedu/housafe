# Task 2 Report: Redis Stream 消费封装

## Summary

Created `ai/shared/redis_client.py` with:
- **`parse_pointcloud_message(data)`** — parses raw Redis Stream bytes dict into `PointCloudFrame` (includes JSON points array → `np.ndarray (N,5) float32`)
- **`parse_vital_message(data)`** — parses raw Redis Stream bytes dict into `VitalFrame` (handles empty string → `None` for `resp_rate`/`heart_rate`, `quality` defaults to `0.0`)
- **`FrameConsumer`** — blocking `consume_one()` reader that watches both `housafe:pointcloud:ingest` and `housafe:vital:ingest` streams via `XREAD`, returns `(stream_name, parsed_frame, msg_id)`

Created `ai/tests/test_redis_client.py` with 3 tests (no Redis required):
- `test_parse_pointcloud_message` — verifies field extraction, ndarray shape/dtype, point values
- `test_parse_vital_message` — verifies float parsing from bytes
- `test_parse_vital_message_none_values` — verifies empty bytes → `None` handling

## Files changed

- `ai/shared/redis_client.py` (created, 178 lines)
- `ai/tests/test_redis_client.py` (created, 80 lines)
- `.superpowers/sdd/task-2-report.md` (updated)

Test Results
```
$ python -m pytest ai/tests/test_redis_client.py -v
→ 3 passed
$ python -m pytest ai/tests/ -v
→ 10 passed (no regressions)
```

Concerns: None.
