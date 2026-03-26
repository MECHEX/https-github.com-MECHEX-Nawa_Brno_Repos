# Nawa_Brno_Repos

## Transient Refactor Plan

### Goal
- Replace the flat transient config based on `PARTS = {"part1": ...}` with a case-centric model.
- Support explicit geometry recognition, case IDs such as `uni10_003`, and per-case parts with their own time ranges.
- Migrate the current data on disk to the new config.
- Improve runtime diagnostics so failures clearly state which job, case, part, fluid, and file failed.

### Proposed Config Model
- Define `CASES` as the source of truth.
- Each case has metadata such as `geometry`, `run`, `description`, and a `parts` map.
- Each part has `source_dir`, `t_start_s`, `t_end_s`, and `dt_sim_s`.
- Parts may also declare `fixed_t_local_s` for legacy snapshots whose file names do not carry time tokens.
- Standard `PLOT_JOBS` select data through explicit case members.
- `COMPARE_JOBS` keep their current style/plot/steady blocks, but each series points to `case_id` plus selected parts.
- Each job has a required `name`, and output folders are based on that name instead of auto-generated long tags.
- Each case, plot job, or compare job may additionally use `active: False` to disable it without deleting the entry from config.

### Assumptions For Existing Data
- Current folders `uni10_003`, `uni10_005`, `uni10_007`, and `GUNI10_001` become single-part cases.
- Legacy folders `part1..part9` are mapped into explicit legacy cases using `part_desc.txt`.
- Continuation runs are modeled as multiple parts within one case only where the current metadata strongly suggests this.

### Critical Review Of The Proposal
- The strongest part of the proposal is that it separates physical identity (`geometry`, `run`, `case_id`) from simulation segments (`parts`).
- The main risk is over-interpreting legacy data. Old directories `part1..part9` do not encode geometry in their names, so grouping them into cases must stay explicit in config and not be inferred automatically.
- Another risk is making the config too clever with selectors such as `geometry=["uni10"]` plus implicit expansion. That adds ambiguity. The implementation should prefer explicit `case_id` references.
- A second risk is rewriting too much of the plotting code. The safest implementation is to normalize the new config into an internal flat part catalog so the current compute and plotting code can mostly stay intact.

### Implementation Plan
1. Add a config resolver/validator that flattens `CASES` into an internal catalog of unique part keys.
2. Rewrite `Transient_Repo/config.py` to the new model using the current datasets on disk.
3. Update `config_example.py` to show the new schema.
4. Update `main.py` to use the resolved part catalog and named jobs.
5. Update compare handling so series resolve `case_id + parts` into internal part keys.
6. Add validation and fail-fast errors for unknown cases, unknown parts, missing directories, invalid time ranges, and empty selections.
7. Keep plot styling blocks (`PLOT_DEFAULTS`, compare `fig`, `plot`, `style_map`, `steady`) stable unless code adaptation requires a narrow compatibility fix.

### Critical Review Of The Implementation Plan
- The resolver is the right first step because it isolates config churn from runtime logic.
- The migration of current data should be done manually in config, not by fragile directory-name heuristics.
- Keeping compare styling unchanged is important; otherwise the refactor mixes data-model changes with visual regressions.
- Fail-fast validation is preferable to silent skipping. For scientific postprocessing, a loud error is safer than partial output.
- The implementation should avoid broad fallback behavior. If a job references a missing case or part, the code should stop with context rather than silently continue.

### Expected Outcome
- `config.py` describes transient data as cases and parts, not as a single global part table.
- Existing datasets remain runnable.
- Plot job and compare job folders are short and stable because they use explicit `name`.
- Error messages identify exactly where the configuration or data is wrong.
