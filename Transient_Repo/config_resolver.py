from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    part_catalog: Dict[str, Dict[str, Any]]
    case_parts: Dict[str, List[str]]
    plot_jobs: List[Dict[str, Any]]
    compare_jobs: List[Dict[str, Any]]


_VALID_TIME_SOURCES = {"T", "S"}
_VALID_T_INTERPRETATION = {"local", "global", "auto"}


def _expect_dict(value: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{ctx} must be a dict, got {type(value).__name__}.")
    return value


def _expect_list(value: Any, ctx: str) -> List[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{ctx} must be a list, got {type(value).__name__}.")
    return value


def _expect_nonempty_str(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{ctx} must be a non-empty string.")
    return value.strip()


def _expect_bool(value: Any, ctx: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{ctx} must be a bool.")
    return value


def _as_float(value: Any, ctx: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{ctx} must be a number.") from exc


def _normalize_time_priority(value: Any, ctx: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ConfigError(f"{ctx} must be a string, list, or tuple.")

    out: List[str] = []
    for item in items:
        src = _expect_nonempty_str(item, ctx).upper()
        if src not in _VALID_TIME_SOURCES:
            raise ConfigError(f"{ctx} contains unsupported source '{src}'. Use only 'T' or 'S'.")
        out.append(src)

    if not out:
        raise ConfigError(f"{ctx} must not be empty.")
    return tuple(out)


def _normalize_t_interpretation(value: Any, ctx: str) -> str:
    mode = _expect_nonempty_str(value, ctx).lower()
    if mode not in _VALID_T_INTERPRETATION:
        raise ConfigError(
            f"{ctx} must be one of: {sorted(_VALID_T_INTERPRETATION)}."
        )
    return mode


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _part_key(case_id: str, part_id: str) -> str:
    return f"{case_id}::{part_id}"


def _resolve_member_part_keys(
    member: Dict[str, Any],
    *,
    case_parts: Dict[str, List[str]],
    inactive_case_ids: set[str],
    ctx: str,
) -> List[str]:
    case_id = _expect_nonempty_str(member.get("case_id"), f"{ctx}.case_id")
    if case_id not in case_parts:
        if case_id in inactive_case_ids:
            raise ConfigError(f"{ctx} references inactive case_id '{case_id}'.")
        raise ConfigError(f"{ctx} references unknown case_id '{case_id}'.")

    raw_parts = member.get("parts")
    part_ids = case_parts[case_id] if raw_parts is None else [
        _expect_nonempty_str(part_id, f"{ctx}.parts[]")
        for part_id in _expect_list(raw_parts, f"{ctx}.parts")
    ]

    if not part_ids:
        raise ConfigError(f"{ctx} resolves to no parts.")

    missing = [part_id for part_id in part_ids if part_id not in case_parts[case_id]]
    if missing:
        raise ConfigError(
            f"{ctx} references unknown parts for case '{case_id}': {missing}."
        )

    return [_part_key(case_id, part_id) for part_id in part_ids]


def _validate_unique_job_names(jobs: List[Dict[str, Any]], ctx: str) -> None:
    names = [_expect_nonempty_str(job.get("name"), f"{ctx}[].name") for job in jobs]
    dupes = sorted({name for name in names if names.count(name) > 1})
    if dupes:
        raise ConfigError(f"{ctx} contains duplicate job names: {dupes}.")


def _job_is_active(job_cfg: Dict[str, Any], ctx: str) -> bool:
    if "active" not in job_cfg:
        return True
    return _expect_bool(job_cfg.get("active"), f"{ctx}.active")


def _case_is_active(case_cfg: Dict[str, Any], ctx: str) -> bool:
    if "active" not in case_cfg:
        return True
    return _expect_bool(case_cfg.get("active"), f"{ctx}.active")


def resolve_runtime_config(
    *,
    base_data_dir: Path,
    cases: Dict[str, Any],
    plot_jobs: List[Dict[str, Any]],
    compare_jobs: List[Dict[str, Any]],
    srp_defaults: Dict[str, Any],
) -> RuntimeConfig:
    base_data_dir = Path(base_data_dir).resolve()
    if not base_data_dir.exists():
        raise ConfigError(f"Base data directory does not exist: {base_data_dir}")

    raw_cases = _expect_dict(cases, "CASES")
    raw_plot_jobs = _expect_list(plot_jobs, "PLOT_JOBS")
    raw_compare_jobs = _expect_list(compare_jobs, "COMPARE_JOBS")
    raw_srp_defaults = _expect_dict(srp_defaults, "SRP_DEFAULTS")

    default_priority = _normalize_time_priority(
        raw_srp_defaults.get("time_source_priority", ("T", "S")),
        "SRP_DEFAULTS.time_source_priority",
    )
    default_t_interpretation = _normalize_t_interpretation(
        raw_srp_defaults.get("t_interpretation", "global"),
        "SRP_DEFAULTS.t_interpretation",
    )

    part_catalog: Dict[str, Dict[str, Any]] = {}
    case_parts: Dict[str, List[str]] = {}
    inactive_case_ids = set()

    for case_id, raw_case in raw_cases.items():
        case_ctx = f"CASES['{case_id}']"
        case_id = _expect_nonempty_str(case_id, case_ctx)
        case_cfg = _expect_dict(raw_case, case_ctx)
        if not _case_is_active(case_cfg, case_ctx):
            inactive_case_ids.add(case_id)
            continue

        geometry = _expect_nonempty_str(case_cfg.get("geometry"), f"{case_ctx}.geometry")
        run_id = _expect_nonempty_str(case_cfg.get("run"), f"{case_ctx}.run")
        description = str(case_cfg.get("description", "")).strip()
        case_priority = _normalize_time_priority(
            case_cfg.get("time_source_priority", default_priority),
            f"{case_ctx}.time_source_priority",
        )
        case_t_interp = _normalize_t_interpretation(
            case_cfg.get("t_interpretation", default_t_interpretation),
            f"{case_ctx}.t_interpretation",
        )

        raw_parts = _expect_dict(case_cfg.get("parts"), f"{case_ctx}.parts")
        if not raw_parts:
            raise ConfigError(f"{case_ctx}.parts must not be empty.")

        case_parts[case_id] = []

        for part_id, raw_part in raw_parts.items():
            part_ctx = f"{case_ctx}.parts['{part_id}']"
            part_id = _expect_nonempty_str(part_id, part_ctx)
            part_cfg = _expect_dict(raw_part, part_ctx)

            source_dir_name = _expect_nonempty_str(part_cfg.get("source_dir"), f"{part_ctx}.source_dir")
            source_dir = (base_data_dir / source_dir_name).resolve()
            if not source_dir.exists():
                raise ConfigError(
                    f"{part_ctx}.source_dir points to missing directory: {source_dir}"
                )

            t_start_s = _as_float(part_cfg.get("t_start_s"), f"{part_ctx}.t_start_s")
            t_end_s = _as_float(part_cfg.get("t_end_s"), f"{part_ctx}.t_end_s")
            dt_sim_s = _as_float(part_cfg.get("dt_sim_s"), f"{part_ctx}.dt_sim_s")
            fixed_t_local_s = (
                None
                if part_cfg.get("fixed_t_local_s") is None
                else _as_float(part_cfg.get("fixed_t_local_s"), f"{part_ctx}.fixed_t_local_s")
            )
            if dt_sim_s <= 0.0:
                raise ConfigError(f"{part_ctx}.dt_sim_s must be > 0.")
            if t_end_s < t_start_s:
                raise ConfigError(
                    f"{part_ctx} has invalid time range: t_end_s < t_start_s."
                )

            part_priority = _normalize_time_priority(
                part_cfg.get("time_source_priority", case_priority),
                f"{part_ctx}.time_source_priority",
            )
            part_t_interp = _normalize_t_interpretation(
                part_cfg.get("t_interpretation", case_t_interp),
                f"{part_ctx}.t_interpretation",
            )

            part_key = _part_key(case_id, part_id)
            part_catalog[part_key] = {
                "part_key": part_key,
                "case_id": case_id,
                "part_id": part_id,
                "geometry": geometry,
                "run": run_id,
                "description": description,
                "source_dir_name": source_dir_name,
                "source_dir": source_dir,
                "t_start_s": t_start_s,
                "t_end_s": t_end_s,
                "dt_sim_s": dt_sim_s,
                "fixed_t_local_s": fixed_t_local_s,
                "time_source_priority": part_priority,
                "t_interpretation": part_t_interp,
            }
            case_parts[case_id].append(part_id)

    active_plot_jobs: List[Dict[str, Any]] = []
    for idx, raw_job in enumerate(raw_plot_jobs):
        job_ctx = f"PLOT_JOBS[{idx}]"
        job_cfg = _expect_dict(raw_job, job_ctx)
        if _job_is_active(job_cfg, job_ctx):
            active_plot_jobs.append(job_cfg)

    active_compare_jobs: List[Dict[str, Any]] = []
    for idx, raw_job in enumerate(raw_compare_jobs):
        job_ctx = f"COMPARE_JOBS[{idx}]"
        job_cfg = _expect_dict(raw_job, job_ctx)
        if _job_is_active(job_cfg, job_ctx):
            active_compare_jobs.append(job_cfg)

    _validate_unique_job_names(active_plot_jobs, "PLOT_JOBS")
    _validate_unique_job_names(active_compare_jobs, "COMPARE_JOBS")

    normalized_plot_jobs: List[Dict[str, Any]] = []
    for idx, raw_job in enumerate(active_plot_jobs):
        job_ctx = f"PLOT_JOBS[{idx}]"
        job_cfg = raw_job
        members = _expect_list(job_cfg.get("members"), f"{job_ctx}.members")
        if not members:
            raise ConfigError(f"{job_ctx}.members must not be empty.")

        member_part_keys: List[str] = []
        for member_idx, raw_member in enumerate(members):
            member_cfg = _expect_dict(raw_member, f"{job_ctx}.members[{member_idx}]")
            member_part_keys.extend(
                _resolve_member_part_keys(
                    member_cfg,
                    case_parts=case_parts,
                    inactive_case_ids=inactive_case_ids,
                    ctx=f"{job_ctx}.members[{member_idx}]",
                )
            )

        normalized_plot_jobs.append({
            "name": _expect_nonempty_str(job_cfg.get("name"), f"{job_ctx}.name"),
            "part_keys": _unique_preserve_order(member_part_keys),
            "fluids": [
                _expect_nonempty_str(fluid, f"{job_ctx}.fluids[]")
                for fluid in _expect_list(job_cfg.get("fluids"), f"{job_ctx}.fluids")
            ],
            "plots": [
                _expect_nonempty_str(plot, f"{job_ctx}.plots[]")
                for plot in _expect_list(job_cfg.get("plots"), f"{job_ctx}.plots")
            ],
            "metrics": [
                _expect_nonempty_str(metric, f"{job_ctx}.metrics[]")
                for metric in _expect_list(job_cfg.get("metrics"), f"{job_ctx}.metrics")
            ],
            "overlay_mode": str(job_cfg.get("overlay_mode", "all")).lower(),
        })

    normalized_compare_jobs: List[Dict[str, Any]] = []
    for idx, raw_job in enumerate(active_compare_jobs):
        job_ctx = f"COMPARE_JOBS[{idx}]"
        job_cfg = raw_job
        series = _expect_list(job_cfg.get("series"), f"{job_ctx}.series")
        if not series:
            raise ConfigError(f"{job_ctx}.series must not be empty.")

        normalized_job = dict(job_cfg)
        normalized_job["name"] = _expect_nonempty_str(job_cfg.get("name"), f"{job_ctx}.name")
        normalized_series: List[Dict[str, Any]] = []

        for series_idx, raw_series in enumerate(series):
            series_ctx = f"{job_ctx}.series[{series_idx}]"
            series_cfg = _expect_dict(raw_series, series_ctx)
            resolved_series = dict(series_cfg)
            resolved_series["parts"] = _resolve_member_part_keys(
                series_cfg,
                case_parts=case_parts,
                inactive_case_ids=inactive_case_ids,
                ctx=series_ctx,
            )
            resolved_series["case_id"] = _expect_nonempty_str(
                series_cfg.get("case_id"),
                f"{series_ctx}.case_id",
            )
            normalized_series.append(resolved_series)

        normalized_job["series"] = normalized_series
        normalized_compare_jobs.append(normalized_job)

    return RuntimeConfig(
        part_catalog=part_catalog,
        case_parts=case_parts,
        plot_jobs=normalized_plot_jobs,
        compare_jobs=normalized_compare_jobs,
    )
