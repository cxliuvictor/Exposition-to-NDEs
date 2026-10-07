#!/usr/bin/env python3
"""Run the frozen final Neural Local Volatility model: sabate_grid192.

This runner intentionally exposes no model-selection flag. It reproduces the final
selected NLV architecture and protocol only. Expensive research modes require CUDA
and the March-1 CPU anchor checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parent
CONFIG_NAME = "sabate_grid192"
DATA_HEADERS = {
    "one_month_spx_option_price.csv": {
        "date", "exdate", "secid", "strike_price", "best_bid", "best_offer",
        "impl_volatility", "delta", "exercise_style", "contract_size", "cfadj",
        "cp_flag", "optionid", "am_settlement",
    },
    "one_month_spx_forward_price.csv": {
        "date", "secid", "expiration", "AMSettlement", "ForwardPrice",
    },
    "one_month_spx_security_price.csv": {"date", "secid", "close"},
    "one_month_spx_zero_coupon_yield_curve.csv": {"date", "days", "rate"},
}

REPORT_SCRIPT = r"""
import sys
from pathlib import Path
import nlv_core_final as core
root = Path(sys.argv[1])
report = root / 'reports'
core.report_contracts(core.read_contracts(root), report)
print('Reports saved to', report)
"""


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def source_digest() -> str:
    digest = hashlib.sha256()
    for relative in ("run_nlv_final.py", "nlv_core_final.py"):
        digest.update(relative.encode())
        digest.update((PROJECT / relative).read_bytes())
    return digest.hexdigest()


def check_inputs(data_root: Path, checkpoint: Path, mode: str) -> None:
    problems = []
    for filename, expected in DATA_HEADERS.items():
        path = data_root / filename
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"Missing or empty file: {path}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            actual = set(next(csv.reader(handle), []))
        missing = expected - actual
        if missing:
            problems.append(f"{filename}: missing CSV columns {sorted(missing)}")
    if mode != "smoke" and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
        problems.append(f"Missing or empty CPU anchor: {checkpoint}")
    if problems:
        raise ValueError("\n".join(problems))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "development", "benchmark"], default="development")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--data-root", type=Path, default=PROJECT / "data")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--cpu-checkpoint", type=Path)
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Smoke engineering only.")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--skip-reports", action="store_true")
    return p


def load_core():
    spec = importlib.util.spec_from_file_location("nlv_core_final", PROJECT / "nlv_core_final.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nlv_core_final"] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.seed < 0 or args.seed >= 2**32:
        raise SystemExit("--seed must be in [0, 2**32).")
    if args.allow_cpu and args.mode != "smoke":
        raise SystemExit("--allow-cpu is restricted to --mode smoke.")

    data_root = args.data_root.expanduser().resolve()
    checkpoint = (
        args.cpu_checkpoint or data_root / "initial_full_otm_refinement_20240301.pt"
    ).expanduser().resolve()
    check_inputs(data_root, checkpoint, args.mode)
    if args.check_inputs:
        print("Input files and CSV headers pass. No training started.")
        return 0

    output = (
        args.output_root
        or PROJECT / "runs_final" / args.mode / CONFIG_NAME / f"seed_{args.seed}"
    ).expanduser().resolve()
    if output == data_root or output in data_root.parents or data_root in output.parents or output == PROJECT:
        raise SystemExit("--output-root must be separate from the project and data directories.")
    output.mkdir(parents=True, exist_ok=True)

    import fcntl
    with (output / ".job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another job is already writing {output}.")

        owner = {"config": CONFIG_NAME, "seed": args.seed, "mode": args.mode}
        owner_path = output / "output_owner.json"
        if owner_path.is_file() and json.loads(owner_path.read_text()) != owner:
            raise SystemExit(f"{output} belongs to a different run.")
        atomic_json(owner_path, owner)

        status = {
            **owner,
            "data_root": str(data_root),
            "cpu_checkpoint": str(checkpoint),
            "output_root": str(output),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
            "source_digest": source_digest(),
            "status": "initializing",
        }
        status_path = output / "hpc_status.json"
        atomic_json(status_path, status)

        os.environ.update({
            "NLV_CONFIG": CONFIG_NAME,
            "NLV_MODE": args.mode,
            "NLV_SEED": str(args.seed),
            "NLV_DATA_ROOT": str(data_root),
            "NLV_CPU_CHECKPOINT": str(checkpoint),
            "NLV_OUTPUT_ROOT": str(output),
            "NLV_ALLOW_CPU": str(int(args.allow_cpu)),
            "NLV_PROGRESS": str(int(args.progress)),
            "NLV_SOURCE_DIGEST": status["source_digest"],
            "MPLBACKEND": "Agg",
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        })
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "2"

        def terminate(signum, frame):
            raise InterruptedError(f"Received signal {signum}; repeat the same command to resume.")

        signal.signal(signal.SIGTERM, terminate)

        try:
            core = load_core()
            if args.mode != "smoke":
                dates = [(str(a.date()), str(b.date())) for a, b in core.development_transitions]
                expected = [
                    ("2024-03-01", "2024-03-04"),
                    ("2024-03-04", "2024-03-05"),
                    ("2024-03-05", "2024-03-06"),
                    ("2024-03-06", "2024-03-07"),
                    ("2024-03-07", "2024-03-08"),
                ]
                if len(core.development_population) != 20_001 or dates != expected:
                    raise ValueError("Inputs do not reproduce the frozen 20,001-contract March 1-8 protocol.")
                core.initial_model(core.CONFIGS[CONFIG_NAME], args.seed)

            environment = dict(core.environment)
            environment["scipy"] = core.importlib.metadata.version("scipy")
            environment["python_version"] = sys.version.split()[0]
            atomic_json(output / "hpc_environment.json", environment)
            runtime = {
                k: environment[k]
                for k in ["python_version", "torch", "numpy", "pandas", "scipy", "cuda", "gpu"]
            }
            os.environ["NLV_RUNTIME_DIGEST"] = hashlib.sha256(
                json.dumps(runtime, sort_keys=True).encode()
            ).hexdigest()

            status["status"] = "auditing"
            atomic_json(status_path, status)
            core.run_audits()
            if args.audit_only:
                status.update(status="audits_passed", finished_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(status_path, status)
                return 0

            status["status"] = "running"
            atomic_json(status_path, status)
            configured = core.CONFIGS[CONFIG_NAME]
            effective = core.smoke_config(configured) if args.mode == "smoke" else configured
            folder, run_id = core.run_identity(CONFIG_NAME, effective, args.seed)
            atomic_json(folder / "hpc_environment.json", environment)

            completion = folder / "completed.json"
            if completion.is_file():
                result = json.loads(completion.read_text())
                if result["run_id"] != run_id or result["protocol_digest"] != core.PROTOCOL_DIGEST:
                    raise ValueError(f"Completion metadata does not match this run: {completion}")
                if not (folder / "daily_summary.csv").is_file():
                    raise ValueError(f"Completed run has no daily_summary.csv: {folder}")
                print(f"Reusing completed run: {folder}", flush=True)
            else:
                result = core.execute_experiment(CONFIG_NAME, configured, args.seed)

            frame = core.pd.DataFrame([result])
            core.atomic_csv(frame, output / f"{args.mode}_experiment_summary.csv")
            print(frame.to_string(index=False), flush=True)

            # Numerical completion is final before optional reporting/plotting.
            status.update(
                status="completed",
                run_id=result["run_id"],
                folder=result["folder"],
                finished_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(status_path, status)

            if not args.skip_reports:
                root = Path(result["folder"])
                try:
                    report_child = subprocess.run(
                        [sys.executable, "-X", "faulthandler", "-c", REPORT_SCRIPT, result["folder"]],
                        capture_output=True,
                        text=True,
                        cwd=PROJECT,
                        env=os.environ.copy(),
                    )
                    (root / "report_process.log").write_text(
                        report_child.stdout + "\n" + report_child.stderr
                    )
                    if report_child.returncode:
                        print("Reports failed; numerical results are safe. See report_process.log.", flush=True)
                    else:
                        plot_child = subprocess.run(
                            [sys.executable, "-X", "faulthandler", "-c", core.PLOT_SCRIPT, result["folder"]],
                            capture_output=True,
                            text=True,
                            cwd=PROJECT,
                            env=os.environ.copy(),
                        )
                        (root / "plot_process.log").write_text(
                            plot_child.stdout + "\n" + plot_child.stderr
                        )
                        if plot_child.returncode:
                            print("Plots failed; numerical results are safe. See plot_process.log.", flush=True)
                        else:
                            print(plot_child.stdout, flush=True)
                except BaseException:
                    (root / "postprocess_exception.log").write_text(traceback.format_exc())
                    print("Post-processing failed; numerical results remain safe.", flush=True)

            print(f'Completed. Results: {result["folder"]}', flush=True)
            return 0

        except BaseException as error:
            status.update(
                status="interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                error=str(error),
                finished_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(status_path, status)
            traceback.print_exc()
            return 130 if status["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
