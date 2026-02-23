# tailguard_r05 dd-focus search space

Date: 2026-02-23
Run group: `20260223_tailguard_r05_ddfocus`
Goal: fast-loop по max-DD участку; варьируем только tradeability+quality вокруг r04 v07.

## DD-Focus window

- walk_forward.start_date: `2023-06-29` (test start)
- walk_forward.end_date: `2024-06-27` (test end)

## Base config

- Base (holdout): `configs/budget1000_autopilot/20260223_tailguard_r04/holdout_tailguard_r04_v07_h2_quality_mild.yaml`

## Variants

- Count: `14` (each variant generates holdout+stress)
- Design: curated (no cartesian explosion)

## Notes

- Tradeability thresholds are set in real units (bid_ask_pct ~ 0.001 = 0.10%) so changes are NOT eaten by guardrail.
- Risk/stop/z/dstop/maxpos axes are inherited from base and must remain unchanged.

