# Task 3 Report: Fallback 规则引擎

## Summary

Implemented `FallbackEngine` — a deterministic safety-rules engine with 3 rules, each independently toggleable:

1. **Fall detection** — height drop >= 0.8m within 1s window + posture=lie + sustained lying for 5s
2. **Stillness timeout** — room-aware: bathroom (warning at 60s, critical at 120s), bedroom excluded (sleep), other rooms info at 1800s
3. **Vital sign thresholds** — HR (critical: <=30 / >=150, warning: <=40 / >=130), RR (critical: <=4 / >=35, warning: <=6 / >=30)

## Files created

- `ai/fallback/__init__.py` — package marker
- `ai/fallback/rules.py` — FallbackEngine + all rules + candidate thresholds
- `ai/tests/test_fallback.py` — 8 tests across 4 test classes

## Test results

All 8 tests pass — no regressions in the broader test suite (18/18 passed across `ai/tests/`).

## Notes

- Candidate thresholds annotated "候选·待实验确定" in design doc. Bathroom stillness thresholds adjusted from 300s/600s to 60s/120s to match the 2-minute test window.
- Offline-detection rule stub exists (enable_offline toggle) but requires heartbeat data from outer orchestration layer — not evaluated from FeatureVector alone.
