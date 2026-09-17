"""Deliberately weak tool descriptions for the eval ablation.

The naive variant keeps the same tools and argument schemas but strips the guidance:
terse descriptions, no "when to use", no argument constraints, and every error message
replaced by a bare "error" (see ``build_server(generic_errors=True)``).
"""

from __future__ import annotations

NAIVE_DESCRIPTIONS: dict[str, str] = {
    "list_runs": "list runs",
    "get_training_curve": "get curve",
    "compare_runs": "compare runs",
    "diagnose_run": "diagnose run",
    "launch_run": "launch run",
    "kill_run": "kill run",
}
