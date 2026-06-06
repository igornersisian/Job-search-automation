# tools/experiments/

One-off, **non-production** scripts: model/cost comparisons, usage probes, and
pipeline measurements used while tuning the system. They are not imported by the
bot or the pipeline and are not part of any schedule.

Run them from the repo root, e.g. `python tools/experiments/compare_models.py`.
They import the real `tools/` modules (`from tools.score_job import ...`), so the
repo root must be on `sys.path` (each script handles that itself).

Anything truly disposable (scraped fixtures, intermediate exports) belongs in
`.tmp/`, not here.
