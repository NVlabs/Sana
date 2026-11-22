# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyrallis

from eval.libero.policy import run_episode, settle
from eval.policy import EpisodePolicy
from sana_robot.deploy.runtime.deployment import DeployConfig, Deployment

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STEP_LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
RELAXED_STEP_LIMITS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}
RESOLUTION = 256
PLUS_SEED, PRO_SEED, NOISE_SEED = 7, 42, 7
PLUS_SETTLE_STEPS, PRO_SETTLE_STEPS, MAX_SETTLE_STEPS = 10, 30, 10
PLUS_CATEGORIES = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
PRO_DIMENSIONS = {"object": "object", "position": "swap", "semantic": "lan", "task": "task"}
PRO_EPISODES_PER_TASK = 50
ARMS = ("control", "intervention")


@dataclass
class LiberoEvalConfig(DeployConfig):
    """An evaluation of SANA-Robot-AR on LIBERO-Plus, LIBERO-PRO or LIBERO-MAX; the other fields are a deployment's.

    Args:
        benchmark: ``plus``, ``pro`` or ``max``.
        libero_root: The LIBERO-plus checkout for ``plus`` and for the LIBERO-Plus cases of ``max``; the LIBERO-PRO
            checkout for ``pro`` and for the LIBERO-PRO cases of ``max``.
        pro_data: The directory of the LIBERO-PRO files (``bddl_files`` and ``init_files``), next to a LIBERO-PRO
            checkout.
        max_root: The LIBERO-MAX checkout, for ``max``.
        max_manifest: The manifest of ``max``; ``benchmark/max8000/libero_max_8000.json`` of the checkout when unset.
        max_source: The cases of the manifest to run, by the benchmark they come from: ``plus`` or ``pro``.
        output: The directory of the results; a case with a result is not run again.
        shard_index: Which of ``num_shards`` equal parts of the cases this process runs.
        num_shards: The number of parts, one process each.
        max_cases: Run only the first cases of the part.
        relaxed_step_limits: Let an episode of ``plus`` or ``pro`` run for 400 / 400 / 400 / 700 control steps
            instead of 220 / 280 / 300 / 520; the report gives the success rate within both limits.
        report: Print the success rates of the results in ``output`` instead of running episodes.
    """

    benchmark: str = "plus"
    libero_root: str = ""
    pro_data: str = ""
    max_root: str = ""
    max_manifest: Optional[str] = None
    max_source: str = "plus"
    output: str = "output/libero_plus"
    shard_index: int = 0
    num_shards: int = 1
    max_cases: Optional[int] = None
    relaxed_step_limits: bool = True
    report: bool = False


def write_json(path, value) -> None:
    """Write a JSON file in one move, so that a file that exists is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def query_seed(identity, step: int) -> int:
    """Return the seed of the noise of one request: a function of the case and of the control step alone, so that it does not depend on the order or the number of processes."""
    payload = json.dumps([identity, int(step), NOISE_SEED])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)


def link_pro_files(libero_root, pro_data, runtime_dir) -> tuple[Path, Path]:
    """Return one BDDL root and one initial-state root for LIBERO-PRO under ``runtime_dir``: links to the folders of the LIBERO-PRO files and to the clean suites of the checkout."""
    package = Path(libero_root).resolve() / "libero/libero"
    (bddl_root, init_root) = (Path(runtime_dir, "pro_files/bddl_files"), Path(runtime_dir, "pro_files/init_files"))
    folders = [(bddl_root / folder.name, folder) for folder in sorted(Path(pro_data, "bddl_files").resolve().iterdir())]
    folders += [(bddl_root / suite, package / "bddl_files" / suite) for suite in SUITES]
    folders += [
        (init_root / folder.name, folder) for folder in sorted(Path(pro_data, "init_files").resolve().iterdir())
    ]
    for link, target in folders:
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(target)
    return bddl_root.resolve(), init_root.resolve()


def configure_libero(libero_root, runtime_dir, bddl_root=None, init_root=None) -> None:
    """Point the LIBERO package of a checkout at its files and make it importable; call it before importing ``libero``. The simulator renders off screen through EGL, and LIBERO loads its initial states, which are NumPy arrays, with ``torch.load``."""
    root = Path(libero_root).resolve()
    package = root / "libero/libero"
    paths = {
        "benchmark_root": package,
        "bddl_files": bddl_root or package / "bddl_files",
        "init_states": init_root or package / "init_files",
        "assets": package / "assets",
        "datasets": root / "libero/datasets",
    }
    folder = Path(runtime_dir).resolve() / "libero_config"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text("".join(f"{name}: {path}\n" for (name, path) in paths.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(folder)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    sys.path.insert(0, str(root))


def plus_instruction(name: str, language: str) -> str:
    """Return the instruction of a LIBERO-Plus task without the perturbation tags that its file name adds to it; a language variant keeps its wording."""
    if "_language_" in name:
        return language
    tag = re.search(r"_(?:(?:view|initstate|table|tb|light|noise|add)_[-+]?\d|(?:moved_)?level_?\d)", name)
    suffix = name[tag.start() :].replace("_", " ") if tag else ""
    return language[: -len(suffix)].rstrip() if suffix and language.endswith(suffix) else language


def bddl_instruction(path) -> str:
    """Return the ``:language`` text of a BDDL file as LIBERO parses it: comments removed, lower case, single spaces."""
    text = re.sub(r";.*$", "", Path(path).read_text(), flags=re.MULTILINE).lower()
    return " ".join(re.findall(r"\(\s*:language(?=[\s()])([^()]*)\)", text)[0].split())


def plus_cases(libero_root) -> list:
    """Return the cases of LIBERO-Plus: every registered variant of the four suites once, from its first initial state, with the perturbation category the benchmark files it under."""
    from libero.libero import benchmark

    classification = json.loads(Path(libero_root, "libero/libero/benchmark/task_classification.json").read_text())
    cases = []
    for suite in SUITES:
        categories = {item["name"]: PLUS_CATEGORIES[item["category"]] for item in classification[suite]}
        registered = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        for index in range(registered.n_tasks):
            cases.append(
                {
                    "case_id": f"{suite}-{index:05d}",
                    "suite": suite,
                    "task_index": index,
                    "group": categories[registered.get_task(index).name],
                    "init_state_index": 0,
                    "seed": PLUS_SEED,
                }
            )
    return cases


def pro_cases() -> list:
    """Return the cases of the LIBERO-PRO panel: per suite and dimension (object, position, semantic, task) the ten registered tasks from fifty initial states each, ordered so that every part of the cases mixes them."""
    from libero.libero import benchmark

    cases = []
    for dimension, suffix in PRO_DIMENSIONS.items():
        for suite in SUITES:
            registered = benchmark.get_benchmark_dict()[f"{suite}_{suffix}"](task_order_index=0)
            for index in range(registered.n_tasks):
                for initial in range(PRO_EPISODES_PER_TASK):
                    cases.append(
                        {
                            "case_id": f"{dimension}-{suite}-t{index:02d}-i{initial:02d}",
                            "suite": suite,
                            "registered_suite": f"{suite}_{suffix}",
                            "task_index": index,
                            "group": dimension,
                            "init_state_index": initial,
                            "seed": PRO_SEED,
                        }
                    )
    return sorted(cases, key=lambda case: (case["init_state_index"], case["task_index"], case["group"], case["suite"]))


def max_cases(manifest_path, source: str) -> list:
    """Return the cases of a LIBERO-MAX manifest that come from LIBERO-Plus or from LIBERO-PRO, in the manifest's order."""
    from libero_max.manifest import load_manifest

    cases = load_manifest(Path(manifest_path))["cases"]
    from_pro = [(case.get("substrate_variant") or {}).get("benchmark") == "LIBERO-PRO" for case in cases]
    return [
        dict(case, suite=case["task_suite_name"]) for (case, pro) in zip(cases, from_pro) if pro == (source == "pro")
    ]


def open_environment(suite, task_index: int, seed: int, initial: int):
    """Return a new off-screen LIBERO environment of a registered task at one of its initial states, with its first observation and the task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_index)
    bddl = Path(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
    env.seed(seed)
    env.reset()
    observation = env.set_init_state(suite.get_task_init_states(task_index)[initial])
    return env, observation, task


def run_registered_case(case: dict, policy: EpisodePolicy, suites: dict, limits: dict) -> dict:
    """Run one case of LIBERO-Plus or LIBERO-PRO in a new environment and return its result."""
    from libero.libero import benchmark, get_libero_path

    name = case.get("registered_suite", case["suite"])
    if name not in suites:
        suites[name] = benchmark.get_benchmark_dict()[name](task_order_index=0)
    (env, observation, task) = open_environment(
        suites[name], case["task_index"], case["seed"], case["init_state_index"]
    )
    if "registered_suite" in case:
        bddl = Path(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        (instruction, settle_steps, identity) = (bddl_instruction(bddl), PRO_SETTLE_STEPS, case["case_id"])
    else:
        instruction = plus_instruction(task.name, task.language)
        (settle_steps, identity) = (PLUS_SETTLE_STEPS, [case["suite"], task.name, 0])
    observation = settle(env, observation, settle_steps)
    result = run_episode(
        env,
        observation,
        policy,
        instruction,
        limits[case["suite"]],
        seed=lambda step: query_seed(identity, step),
    )
    env.close()
    return dict(result, task_name=task.name, instruction=instruction)


def max_environment(case: dict, suites: dict):
    """Return the LIBERO environment of a LIBERO-MAX case as the benchmark builds it, with the initial state and the instruction."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_max.env_factory import create_libero_env_with_retry
    from libero_max.pro_runtime import wrap_case_env
    from libero_max.substrate import load_case_task
    from numpy.random import seed as reseed

    (task, states) = load_case_task(case, benchmark, suites)
    bddl = Path(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    def build():
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
        env.seed(case["policy_seed"])
        return env, task.language

    (env, instruction) = create_libero_env_with_retry(build, policy_seed=case["policy_seed"], reseed=reseed)
    if case.get("group") != "clean":
        env = wrap_case_env(env, case)
    env.seed(case["policy_seed"])
    return env, states[case["init_state_index"]], instruction


def run_max_case(case: dict, policy: EpisodePolicy, suites: dict, directory: Path) -> dict:
    """Run the two arms of a LIBERO-MAX case and return its result.

    The control arm is the plain episode. The intervention arm starts from the same state and runs the commands of
    the control arm until the benchmark applies its change, the policy only observing; from the next request on
    the policy acts again. The benchmark's wrapper writes the traces of both arms under ``directory`` and its own
    function pairs them; ``paired_summary.json`` is the file its aggregator reads.
    """
    from libero_max.cosmos_integration import CosmosInterventionEnv
    from summarize_cosmos_paired_smoke import classify_persistent_pair

    from_plus = case.get("substrate_variant") is None and case.get("substrate_category") != "Language Instructions"
    adapt = (lambda text: plus_instruction(case["task_name"], text)) if from_plus else (lambda text: text)
    outcomes = {}
    for arm in ARMS:
        (env, initial, instruction) = max_environment(case, suites)
        trace = directory / arm / "trace.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text("")
        env = CosmosInterventionEnv(
            env=env,
            task_description=instruction,
            scenario=case["scenario"],
            arm=arm,
            trace_path=trace,
            original_task_index=case["task_index"],
            init_state_index=case["init_state_index"],
            warmup_steps=MAX_SETTLE_STEPS,
        )
        env.configure_episode(
            task_suite_name=case["suite"],
            policy_seed=case["policy_seed"],
            query_interval=policy.n_action_steps,
            max_policy_steps=STEP_LIMITS[case["suite"]],
        )
        env.reset()
        observation = env.set_init_state(initial)
        if env.render_initialization_qa.get("status") != "passed":
            env.close()
            raise RuntimeError(f"{case['case_id']}: the benchmark's render check failed")
        observation = settle(env, observation, MAX_SETTLE_STEPS)
        changed = arm == "intervention"
        recorded = {row["policy_step"]: row["actions"] for row in outcomes.get("control", {}).get("policy_queries", [])}
        current = (
            (lambda env=env: env.runtime.current_instruction)
            if changed
            else (lambda instruction=instruction: instruction)
        )

        def on_query(commands, text, replayed, env=env, current=current):
            env.record_policy_query(commands, instruction=current(), source="control_replay" if replayed else "model")

        result = run_episode(
            env,
            observation,
            policy,
            lambda current=current: adapt(current()),
            STEP_LIMITS[case["suite"]],
            seed=lambda step: query_seed(case["case_id"], step),
            replay=(lambda step, env=env: None if env.runtime.applied else recorded[step]) if changed else None,
            on_query=on_query,
        )
        outcomes[arm] = env.record_outcome(result["success"])
        env.close()
    (summary, terminal) = classify_persistent_pair(outcomes["control"], outcomes["intervention"])
    if not (summary or terminal or {}).get("matched"):
        raise ValueError(f"{case['case_id']}: the benchmark could not pair the two arms")
    if summary is not None:
        write_json(directory / "paired_summary.json", summary)
    write_json(directory / "status.json", {"case_id": case["case_id"], **(terminal or {})})
    return {
        "success": bool(outcomes["intervention"]["success"]),
        "control_success": bool(outcomes["control"]["success"]),
        "policy_steps": int(outcomes["intervention"]["policy_steps"]),
        "control_policy_steps": int(outcomes["control"]["policy_steps"]),
    }


def run(config: LiberoEvalConfig) -> None:
    """Run the cases of one part of a benchmark and write one result per case under ``output/cases``."""
    output = Path(config.output)
    runtime = output / f"runtime-{config.shard_index}"
    (bddl_root, init_root) = (None, None)
    if config.pro_data:
        (bddl_root, init_root) = link_pro_files(config.libero_root, config.pro_data, runtime)
    if config.benchmark == "max":
        max_root = Path(config.max_root).resolve()
        sys.path[:0] = [
            str(max_root / "scripts/libero_source_overlay"),
            str(max_root / "src"),
            str(max_root / "scripts"),
        ]
        os.environ["LIBERO_SOURCE_PACKAGE_ROOT"] = str(Path(config.libero_root).resolve() / "libero")
    configure_libero(config.libero_root, runtime, bddl_root, init_root)
    if config.benchmark == "plus":
        cases = plus_cases(config.libero_root)
    elif config.benchmark == "pro":
        cases = pro_cases()
    else:
        manifest = config.max_manifest or Path(config.max_root, "benchmark/max8000/libero_max_8000.json")
        cases = max_cases(manifest, config.max_source)
    relaxed = config.relaxed_step_limits and config.benchmark != "max"
    limits = RELAXED_STEP_LIMITS if relaxed else STEP_LIMITS
    write_json(
        output / f"cases-{config.shard_index}.json",
        {"benchmark": config.benchmark, "step_limits": limits, "cases": cases},
    )
    policy = EpisodePolicy(Deployment(config))
    suites = {}
    for case in cases[config.shard_index :: config.num_shards][: config.max_cases]:
        directory = output / "cases" / case["case_id"]
        if (directory / "result.json").exists():
            continue
        for stale in ("error.json", "paired_summary.json", "status.json"):
            (directory / stale).unlink(missing_ok=True)
        started = time.monotonic()
        try:
            if config.benchmark == "max":
                result = run_max_case(case, policy, suites, directory)
            else:
                result = run_registered_case(case, policy, suites, limits)
        except Exception as error:
            write_json(
                directory / "error.json", {"error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc()}
            )
            print(f"{case['case_id']}: error {type(error).__name__}: {error}", flush=True)
            continue
        write_json(directory / "result.json", dict(result, case_id=case["case_id"], seconds=time.monotonic() - started))
        print(f"{case['case_id']}: success {result['success']} in {result['policy_steps']} steps", flush=True)


def report(config: LiberoEvalConfig) -> None:
    """Print the success rates of the results under ``output`` and write them to ``summary.json``: over all cases, per suite and per category, within the step limits of the run and within 220 / 280 / 300 / 520 steps."""
    output = Path(config.output)
    listing = json.loads(sorted(output.glob("cases-*.json"))[0].read_text())
    groups = defaultdict(lambda: defaultdict(int))
    for case in listing["cases"]:
        path = output / "cases" / case["case_id"] / "result.json"
        names = ["all", f"suite/{case['suite']}"] + [f"category/{case['group']}"] * ("group" in case)
        for name in names:
            groups[name]["cases"] += 1
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for name in names:
            groups[name]["finished"] += 1
            groups[name]["success"] += result["success"]
            groups[name]["success_official"] += (
                result["success"] and result["policy_steps"] <= STEP_LIMITS[case["suite"]]
            )
            groups[name]["control_success"] += result.get("control_success", 0)
    summary = {}
    for name, count in sorted(groups.items()):
        finished = max(count["finished"], 1)
        summary[name] = {
            "cases": count["cases"],
            "finished": count["finished"],
            "success_rate": 100.0 * count["success"] / finished,
            "success_rate_official_limits": 100.0 * count["success_official"] / finished,
        }
        line = f"{name}: {count['finished']} of {count['cases']} cases, success {summary[name]['success_rate']:.2f} %"
        if listing["benchmark"] == "max":
            summary[name]["control_success_rate"] = 100.0 * count["control_success"] / finished
            line += f" after the change, {summary[name]['control_success_rate']:.2f} % without it"
        else:
            line += f" ({summary[name]['success_rate_official_limits']:.2f} % within the official step limits)"
        print(line, flush=True)
    write_json(output / "summary.json", {"step_limits": listing["step_limits"], "groups": summary})


@pyrallis.wrap()
def main(config: LiberoEvalConfig) -> None:
    """Evaluate SANA-Robot-AR on a LIBERO benchmark: ``python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml --benchmark plus --libero_root <LIBERO-plus> --output <directory>``, then the same command with ``--report true``."""
    if config.report:
        report(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
