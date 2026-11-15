# Golden files

`history_writer.json`: what the history writer sends to the database for the synthetic
probe in `backend/tests/_serving_rows_probe.py`. `test_serving_rows.py` requires the shared
module (`pipeline/serving_rows.py`) and the script's old names to reproduce it.

It must come from the **old in-script writer**, never from the shared module. Regenerated
2026-11-15 (digest format changed to fixed-precision floats; the old format differed
between arm64 and x86-64, see the probe's docstring):

```sh
git worktree add --detach /tmp/old-writer de79477^      # pre-move scripts/materialize_stack_projections.py
cd /tmp/old-writer
PYTHONPATH=. python <regen script> <repo>/backend/tests/_serving_rows_probe.py out.json
```

The regen script imports `scripts.materialize_stack_projections` from the old tree, loads the
probe from the new tree by path, patches `psycopg2.extras.execute_batch` to capture batches,
and writes `run_probe(m._attach_causal_conformal_intervals,
m._attach_derived_fantasy_projection, m._upsert, ...)` as sorted, indented JSON. The SQL
text, row counts and commit count were identical to the previous golden; only the digests
changed. Remove the worktree afterwards (`git worktree remove --force /tmp/old-writer`).
