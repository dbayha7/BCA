"""BCA iql iw driver.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
import importlib
import json
import math
from numbers import Real
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import traceback
import _iql_driver as D


@dataclass(frozen=True)
class Options(D.Options):
    host_beta_multiplier: float = 1.0
    decision_gain: float = 1.0


DRIVER_REL = Path("algorithms/_iql_iw_driver.py")
SCHEMA = "corl-iql-shared-scale-iw-matched-host-v1"
CONTROL_GRID = (0.25, 0.5, 1.0, 2.0)
PUBLISHED_BETA = 3.0
COMMON_REFERENCE = (
    "online_q_params",
    "online_q_optimizer",
    "target_q_params",
    "target_q_optimizer",
    "value_params",
    "value_optimizer",
    "posterior_key",
    "fit_indices",
    "cal_residuals",
)


def validate_options(o):
    if not isinstance(o, Options):
        raise ValueError("matched host controls require the extended Options")
    D.validate_options(o)
    if o.mode != "shared":
        raise ValueError("configured scale-IW controls require shared mode")
    for name in ("host_beta_multiplier", "decision_gain"):
        value = getattr(o, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or (not math.isfinite(value))
            or (value not in CONTROL_GRID)
        ):
            raise ValueError(
                name + " must be a finite non-boolean member of " + str(CONTROL_GRID)
            )


def load_modules():
    m = D.load_modules()
    m.X = importlib.import_module("iql")
    expected = Path(__file__).resolve().parent / "iql.py"
    if Path(m.X.__file__).resolve() != expected:
        raise RuntimeError("IW core import escaped frozen source")
    return m


def require_driver_source(output, manifest):
    inventory = manifest.get("source_sha256")
    relative = DRIVER_REL.as_posix()
    if not isinstance(inventory, dict) or relative not in inventory:
        raise ValueError("missing matched host executable source binding")
    source = Path(output) / "source" / DRIVER_REL
    if (
        not source.is_file()
        or D.sha(source) != inventory[relative]
        or inventory[relative] != D.sha(__file__)
    ):
        raise ValueError("matched host executable source hash mismatch")


def resolved_arguments(o, m):
    validate_options(o)
    args, _, _ = D.resolved_arguments(o, m)
    if args.beta != PUBLISHED_BETA:
        raise ValueError("matched host requires published/fitting beta3")
    args = replace(
        args,
        algorithm="corl_iql_shared_scale_iw_matched_host",
        posterior=replace(args.posterior, decision_gain=o.decision_gain),
    )
    variants, arms = m.X.default_design(args.beta)
    host_beta = args.beta * o.host_beta_multiplier
    arms = tuple(
        (
            replace(a, beta=host_beta, gain=o.decision_gain) if a.mode != "off" else a
            for a in arms
        )
    )
    arms += (m.X.IWArm("comparison_baseline", -1, "off", host_beta, 1.0),)
    differences = []
    for arm in arms:
        actual = replace(args, beta=arm.beta)
        diff = m.H._published_config.diff_against_published("corl_iql", o.dataset, actual)
        if diff is None:
            raise ValueError("missing published configuration")
        for field, got, want in diff:
            allowed = (
                field == "num_updates"
                and got == o.updates
                or (field == "beta" and arm.name != "baseline" and (got == host_beta))
            )
            if not allowed:
                raise ValueError(
                    f"undeclared published config difference: {arm.name} {field} {got} != {want}"
                )
        check = replace(actual, **{field: want for field, got, want in diff})
        m.H._published_config.check_published_config(
            "corl_iql", o.dataset, check, allow_off_config=False
        )
        differences.append(
            dict(
                arm=arm.name,
                actual_differences=diff,
                declaration="declared horizon; actors1-7 matched host beta; actor0 published",
            )
        )
    return (args, variants, arms, differences)


def declaration(o, m):
    args, variants, arms, differences = resolved_arguments(o, m)
    declared = D.plain(
        dict(
            args=asdict(args),
            variants=[asdict(v) for v in variants],
            arms=[asdict(a) for a in arms],
            published_differences=differences,
            driver_options=asdict(o),
            evaluation_schedule=D.evaluation_schedule(o, m),
            runtime_dependencies=D.runtime_dependencies(),
            posterior_measure="unweighted held-out dataset scores; scale fitting IW only",
            iw_measure="fixed-beta capped pre-BCA AWR tilt; not policy/behavior ratio",
            configured_controls=dict(
                host_beta_multiplier=o.host_beta_multiplier,
                decision_gain=o.decision_gain,
                published_beta=args.beta,
                actor_host_beta=args.beta * o.host_beta_multiplier,
                iw_fitting_beta=args.beta,
                baseline_meaning="comparison_baseline index7 matches full/floor host beta; index0 stays published",
                gain_meaning="common full/floor actor decision coefficient; calibrated radii unchanged",
                iw_beta_meaning="fixed published pre-BCA fitting tilt, independent of actor host beta",
                selection_provenance="explicit configured values only; separate selection receipt required",
            ),
            selection_allowed=False,
        )
    )
    validate_declaration(o, declared)
    return declared


def validate_declaration(o, declared):
    validate_options(o)
    args = declared["args"]
    beta = args["beta"]
    if (
        isinstance(beta, bool)
        or not isinstance(beta, Real)
        or (not math.isfinite(beta))
        or (beta != PUBLISHED_BETA)
        or (args["algorithm"] != "corl_iql_shared_scale_iw_matched_host")
        or (args["posterior"]["mode"] != "full")
        or (args["posterior"]["decision_gain"] != o.decision_gain)
        or (declared["driver_options"] != asdict(o))
    ):
        raise ValueError("configured argument declaration mismatch")
    names = (
        "noiw_legacy_width",
        "noiw_weighted_width",
        "awr_legacy_width",
        "awr_weighted_width",
        "awr_permuted_weighted_width",
    )
    expected_arms = [dict(name="baseline", variant_index=-1, mode="off", beta=beta, gain=1.0)]
    host_beta = beta * o.host_beta_multiplier
    for index, name in enumerate(names):
        expected_arms.append(
            dict(
                name="full_" + name,
                variant_index=index,
                mode="full",
                beta=host_beta,
                gain=o.decision_gain,
            )
        )
    expected_arms.extend(
        (
            dict(
                name="floor_awr_weighted_width",
                variant_index=3,
                mode="floor",
                beta=host_beta,
                gain=o.decision_gain,
            ),
            dict(
                name="comparison_baseline",
                variant_index=-1,
                mode="off",
                beta=host_beta,
                gain=1.0,
            ),
        )
    )
    expected_variants = [
        dict(
            name=name,
            weight_width=width,
            iw=dict(
                mode=mode,
                beta=beta,
                cap=100.0,
                mixing=1.0,
                ess_floor=0.25,
                tau_min=0.05,
                iterations=32,
            ),
        )
        for name, mode, width in zip(
            names,
            ("off", "off", "awr", "awr", "awr_permuted"),
            (False, True, False, True, True),
        )
    ]
    if declared["arms"] != expected_arms or declared["variants"] != expected_variants:
        raise ValueError("configured eight-actor/five-variant declaration mismatch")
    if [d["arm"] for d in declared["published_differences"]] != [
        a["name"] for a in expected_arms
    ]:
        raise ValueError("missing actual eight-actor published recipe audit")
    controls = declared["configured_controls"]
    for field, expected in (
        ("published_beta", beta),
        ("actor_host_beta", host_beta),
        ("iw_fitting_beta", beta),
    ):
        actual = controls.get(field)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, Real)
            or (not math.isfinite(actual))
            or (actual != expected)
        ):
            raise ValueError("matched host beta metadata mismatch: " + field)
    if (
        controls["host_beta_multiplier"] != o.host_beta_multiplier
        or controls["decision_gain"] != o.decision_gain
        or declared["selection_allowed"] is not False
    ):
        raise ValueError("configured control identity mismatch")


def variant_counts(metrics, count, variants, np):
    shape = (count, variants)
    accepted = np.asarray(metrics.get("cal_variant_accepted"))
    numeric = np.asarray(metrics.get("cal_variant_numerical_valid"))
    infeasible = np.asarray(metrics.get("cal_variant_ess_infeasible"))
    for x in (accepted, numeric, infeasible):
        if x.dtype != np.bool_ or x.shape != shape:
            raise ValueError("malformed variant acceptance diagnostics")
    if np.any(accepted & (~numeric | infeasible)):
        raise ValueError("accepted calibration has invalid or infeasible evidence")
    if np.any(numeric & ~accepted & ~infeasible):
        raise ValueError("undeclared numerically valid calibration rejection")
    if not np.array_equal(np.asarray(metrics.get("calibration_valid")), numeric.all(axis=1)):
        raise ValueError("aggregate calibration validity hides a variant failure")
    return (accepted.sum(0, dtype=np.int64), infeasible.sum(0, dtype=np.int64))


def verify_common_references(records):
    if not records:
        raise ValueError("no variant references")
    first = records[0]
    for item in records[1:]:
        for name in COMMON_REFERENCE:
            if item["component_sha256"].get(name) != first["component_sha256"].get(name):
                raise ValueError("shared reference component changed across variants: " + name)
        for name in ("step", "posterior_key", "training_key", "calibration_rows", "fit_rows"):
            if item[name] != first[name]:
                raise ValueError("shared reference metadata mismatch: " + name)


def verify_reference_fit_exposure(records, scope):
    if scope == "diagnostic" and any((r["scale_updates"] < 1 for r in records)):
        raise ValueError("diagnostic reference consumed a scale with no accepted fit update")


def verify_variant_evidence(metrics, count, variants, batch_size, np):
    shape = (count, len(variants))
    flags = {}
    for name in (
        "accepted",
        "numerical_valid",
        "ess_feasible",
        "ess_infeasible",
        "iw_inputs_valid",
        "has_support",
    ):
        x = np.asarray(metrics.get("cal_variant_" + name))
        if x.dtype != np.bool_ or x.shape != shape:
            raise ValueError("missing/malformed variant flag: " + name)
        flags[name] = x
    if (
        not flags["numerical_valid"].all()
        or not flags["iw_inputs_valid"].all()
        or (not flags["has_support"].all())
    ):
        raise ValueError("invalid or unsupported calibration in accepted evidence")
    if not np.array_equal(flags["ess_feasible"], ~flags["ess_infeasible"]):
        raise ValueError("contradictory variant ESS flags")
    numeric = {}
    fields = dict(
        loss=(0.0, None),
        coverage=(0.0, 1.0),
        width_loss=(0.0, None),
        eta_mean=(0.0, None),
        resid_scale=(0.0, None),
        iw_tau=(0.0, 1.0),
        iw_raw_ess_fraction=(0.0, 1.0),
        iw_ess_fraction=(0.0, 1.0),
        prior_ess_fraction=(0.0, 1.0),
        product_ess_fraction=(0.0, 1.0),
    )
    for name, (lo, hi) in fields.items():
        x = np.asarray(metrics.get("cal_variant_" + name))
        if x.shape != shape or x.dtype.kind not in "fiu" or (not np.isfinite(x).all()):
            raise ValueError("missing/nonfinite variant numeric: " + name)
        eps = (
            8 * np.finfo(x.dtype).eps
            if x.dtype.kind == "f" and (name == "coverage" or "ess_fraction" in name)
            else 0.0
        )
        if np.any(x < lo) or (hi is not None and np.any(x > hi + eps)):
            raise ValueError("variant numeric outside range: " + name)
        if name in (
            "eta_mean",
            "resid_scale",
            "iw_ess_fraction",
            "prior_ess_fraction",
            "product_ess_fraction",
        ) and np.any(x <= 0):
            raise ValueError("variant numeric must be positive: " + name)
        numeric[name] = x
    for name in ("step", "iw_supported_count", "iw_positive_count"):
        x = np.asarray(metrics.get("cal_variant_" + name))
        if x.shape != shape or x.dtype.kind not in "iu" or np.any(x < 0):
            raise ValueError("invalid variant count: " + name)
        numeric[name] = x
    if np.any(numeric["iw_supported_count"] > batch_size) or np.any(
        numeric["iw_positive_count"] > numeric["iw_supported_count"]
    ):
        raise ValueError("invalid IW support/positive counts")
    if np.any(numeric["iw_positive_count"] < 1):
        raise ValueError("numerically unsupported IW bank")
    for i, v in enumerate(variants):
        if v["iw"]["mode"] == "off":
            if not np.all(numeric["iw_tau"][:, i] == 1) or not flags["accepted"][:, i].all():
                raise ValueError("off calibration did not update normally")
        else:
            if np.any(numeric["iw_tau"][:, i] < v["iw"]["tau_min"]):
                raise ValueError("IW temperature below declared minimum")
            ess = numeric["iw_ess_fraction"][:, i]
            tolerance = 8 * np.finfo(ess.dtype).eps if ess.dtype.kind == "f" else 0.0
            if np.any(flags["ess_feasible"][:, i] & (ess < v["iw"]["ess_floor"] - tolerance)):
                raise ValueError("feasible IW does not meet its ESS floor")
    return variant_counts(metrics, count, len(variants), np)


class IWRuntime(D.GroupRuntime):

    def __init__(self, o, m, output, args, variants, arms):
        self.o, self.m, self.output, self.args, self.arms = (o, m, Path(output), args, arms)
        self.variants = variants
        self.data = m.H.load_data(args)
        D.write_json(self.output / "data_metadata.json", self.data.metadata)
        self.env = m.H.C.gym.make(o.dataset)
        try:
            od, ad = (self.env.observation_space.shape[0], self.env.action_space.shape[0])
            state, key, actor = m.H.initialize_agent(
                args, od, ad, float(self.env.action_space.high[0])
            )
            self.fitters = m.X.make_fitters(args, state, od, ad, variants)
            self.carry = m.X.initialize_shared(args, state, key, self.fitters, arms)
            step = m.X.make_shared_train_step(
                args, self.data.train, self.fitters, variants, arms
            )
            self.scan = m.jax.jit(
                lambda carry, n: m.jax.lax.scan(step, carry, None, n), static_argnums=1
            )

            def policy(params, obs):
                normalized = (obs - self.data.obs_mean) / self.data.obs_std
                return (actor.act(params, normalized), m.jnp.all(m.jnp.isfinite(normalized)))

            self.evaluator = D.ScalarEvaluator(
                self.env,
                m.jax.jit(policy),
                lambda r: m.H.C.normalized_scores(o.dataset, r),
                lambda seed, identity: D.environment_seed(m, seed, identity),
                m.np,
                o.eval_seed,
            )
            self.actor_applied = m.np.zeros(len(arms), m.np.int64)
            self.actor_abstained = self.actor_applied.copy()
            self.actor_post_applied = self.actor_applied.copy()
            self.cal_applied = m.np.zeros(len(variants), m.np.int64)
            self.cal_ess_abstained = self.cal_applied.copy()
            self.nuisance_valid_count = 0
            self.artifacts = {"curves": {}, "references": {}}
            self.phase = "initialized"
            self.timings = dict(
                training_scan_seconds=0.0, evaluation_seconds=0.0, posterior_refresh_seconds=0.0
            )
        except BaseException:
            self.env.close()
            raise

    def refresh(self, step):
        self.phase = "refresh"
        start = time.perf_counter()
        extras, records = ([], [])
        for variant, fitter, extra in zip(self.variants, self.fitters, self.carry.extras):
            ref, row = self.m.P.refresh(
                self.args.posterior,
                fitter,
                extra.calibration,
                self.carry.nuisance,
                self.data.train,
                self.data.calibration,
                self.carry.rng,
                step,
                discount=self.args.discount,
            )
            extras.append(extra._replace(posterior=ref))
            row.update(
                variant=asdict(variant), scale_updates=int(extra.calibration.calibrator.step)
            )
            row["common_component_sha256"] = {
                k: v for k, v in row["component_sha256"].items() if not k.startswith("actor_")
            }
            records.append(row)
        verify_common_references(records)
        self.carry = self.carry._replace(extras=tuple(extras))
        checkpoint = self.checkpoint(f"reference_{step}")
        path = self.output / f"reference_{step}.json"
        D.write_json(
            path,
            dict(
                step=step,
                variants=records,
                checkpoint=checkpoint,
                experiment_scope=self.o.scope,
                training_seed=self.o.seed,
                eval_seed=self.o.eval_seed,
            ),
        )
        self.artifacts["references"][str(step)] = dict(path=path.name, sha256=D.sha(path))
        self.timings["posterior_refresh_seconds"] += time.perf_counter() - start

    def advance(self, count):
        self.phase = "training"
        start = time.perf_counter()
        self.carry, metrics = self.scan(self.carry, count)
        metrics = self.m.jax.device_get(metrics)
        applied, abstained, post = D.block_counts(metrics, count, len(self.arms), self.m.np)
        cal = self.m.np.asarray(metrics["cal_variant_accepted"]).sum(0, dtype=self.m.np.int64)
        ess = self.m.np.asarray(metrics["cal_variant_ess_infeasible"]).sum(
            0, dtype=self.m.np.int64
        )
        self.actor_applied += applied
        self.actor_abstained += abstained
        self.actor_post_applied += post
        self.cal_applied += cal
        self.cal_ess_abstained += ess
        self.nuisance_valid_count += int(self.m.np.asarray(metrics["nuisance_valid"]).sum())
        self.timings["training_scan_seconds"] += time.perf_counter() - start
        return metrics

    def state_counters(self):
        state = self.carry.nuisance
        return {
            **{k: int(getattr(state, k).step) for k in ("qf", "vf", "qf_target")},
            "calibrators": {
                v.name: int(e.calibration.calibrator.step)
                for v, e in zip(self.variants, self.carry.extras)
            },
        }

    def finish(self, steps):
        self.phase = "final_validation"
        m = self.m
        if (
            steps != self.o.updates
            or int(self.carry.step) != steps
            or self.nuisance_valid_count != steps
        ):
            raise ValueError("incomplete nuisance budget")
        counters = self.state_counters()
        if any((counters[k] != steps for k in ("qf", "vf", "qf_target"))):
            raise ValueError("actual nuisance counter mismatch")
        for v, e, count, abstained in zip(
            self.variants, self.carry.extras, self.cal_applied, self.cal_ess_abstained
        ):
            if counters["calibrators"][v.name] != int(count) or count + abstained != steps:
                raise ValueError("actual scale counter mismatch: " + v.name)
            if not bool(m.P.finite_tree(e.calibration)):
                raise ValueError("invalid final calibrator: " + v.name)
        for a, actor, count in zip(self.arms, self.actors(), self.actor_applied):
            if int(actor.step) != int(count) or not bool(m.P.finite_tree(actor)):
                raise ValueError("actual actor counter/state mismatch: " + a.name)
            if a.mode == "off" and count != steps:
                raise ValueError("native actor skipped an update")
        self.checkpoint("training_end")
        self.phase = "final_evaluation"
        start = time.perf_counter()
        ids = D.episode_ids(True, 0, self.o.final_episodes)
        rows = {
            a.name: self.evaluator.evaluate(s.params, ids)
            for a, s in zip(self.arms, self.actors())
        }
        final = {}
        for i, a in enumerate(self.arms):
            row = rows[a.name]
            path = self.output / f"final_{a.name}.npz"
            D.save_npz(
                path,
                m.np,
                final_returns=row.pop("returns"),
                final_scores=row.pop("scores"),
                **row,
                steps_completed=steps,
                eval_seed=self.o.eval_seed,
                training_seed=self.o.seed,
                experiment_scope=self.o.scope,
                actor_applied_updates=self.actor_applied[i],
            )
            final[a.name] = dict(
                path=path.name,
                sha256=D.verify_final(
                    path,
                    self.o.final_episodes,
                    steps,
                    self.o.eval_seed,
                    m.np.asarray(ids, m.np.uint32),
                    m.np,
                    training_seed=self.o.seed,
                    scope=self.o.scope,
                ),
                episodes=self.o.final_episodes,
            )
        self.artifacts["final"] = final
        self.phase = "complete"
        self.timings["evaluation_seconds"] += time.perf_counter() - start


def execute_group(o, m, output, runtime_factory=IWRuntime):
    validate_options(o)
    output = Path(output)
    start = time.perf_counter()
    runtime = None
    record = dict(
        schema=SCHEMA,
        status="running",
        started_utc=D.utc(),
        experiment_scope=o.scope,
        selection_allowed=False,
        training_seed=o.seed,
        eval_seed=o.eval_seed,
        blocks=[],
        steps_completed=0,
        first_failure=None,
    )
    try:
        declared = declaration(o, m)
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            manifest["schema"] != SCHEMA
            or manifest["options"] != asdict(o)
            or declared != manifest["declared"]
        ):
            raise ValueError("runtime arguments/design/dependencies changed after freeze")
        require_driver_source(output, manifest)
        D.write_json(output / "resolved_args.json", declared)
        args, variants, arms, _ = resolved_arguments(o, m)
        runtime = runtime_factory(o, m, output, args, variants, arms)

        def block(first, last, metrics):
            failed = D.first_failure(metrics, first, m.np)
            path = output / f"block_{first}_{last}.npz"
            digest = D.save_npz(
                path,
                m.np,
                **metrics,
                experiment_scope=o.scope,
                training_seed=o.seed,
                eval_seed=o.eval_seed,
            )
            record["blocks"].append(
                dict(
                    first_step=first,
                    last_step=last,
                    diagnostics_npz=path.name,
                    sha256=digest,
                    first_failure=failed,
                )
            )
            record["steps_completed"] = last
            D.write_json(output / "progress.json", record, progress=True)
            print(
                json.dumps(
                    dict(
                        step=last,
                        status="failed" if failed else "running",
                        actor_updates=runtime.actor_applied.tolist(),
                        scale_updates=runtime.cal_applied.tolist(),
                    )
                ),
                flush=True,
            )
            return failed

        record.update(D.run_schedule(runtime, o, block))
        if record["status"] != "complete":
            runtime.checkpoint("failed_block_end")
    except BaseException as error:
        record.update(
            status="failed",
            error=repr(error),
            phase=getattr(runtime, "phase", "setup"),
            traceback=traceback.format_exc(),
            stop_further_dispatch=True,
        )
        if runtime is not None:
            try:
                runtime.checkpoint("failed_attempt")
            except Exception as e:
                record["checkpoint_error"] = repr(e)
    finally:
        if runtime is not None:
            try:
                runtime.env.close()
                record["environment_closed"] = True
            except Exception as e:
                record.update(
                    status="failed",
                    environment_closed=False,
                    close_error=repr(e),
                    stop_further_dispatch=True,
                )
            record.update(
                artifacts=runtime.artifacts,
                state_counters=runtime.state_counters(),
                nuisance_valid_updates=runtime.nuisance_valid_count,
                actor_applied_updates=runtime.actor_applied.tolist(),
                actor_valid_abstentions=runtime.actor_abstained.tolist(),
                actor_post_ready_applied_updates=runtime.actor_post_applied.tolist(),
                scale_applied_updates=runtime.cal_applied.tolist(),
                scale_ess_abstentions=runtime.cal_ess_abstained.tolist(),
                timings=runtime.timings,
            )
        record.update(
            finished_utc=D.utc(),
            elapsed_seconds=time.perf_counter() - start,
            checkpoint_verification=D.CHECKPOINT_VERIFICATION,
        )
        D.write_json(output / "result.json", record)
    return record


def verify_result(output, result, o, np):
    validate_options(o)
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    declared = json.loads((output / "resolved_args.json").read_text())
    if (
        manifest["schema"] != SCHEMA
        or declared != manifest["declared"]
        or declared["driver_options"] != asdict(o)
    ):
        raise ValueError("frozen declaration mismatch")
    if manifest["options"] != asdict(o):
        raise ValueError("frozen options mismatch")
    require_driver_source(output, manifest)
    validate_declaration(o, declared)
    if (
        result.get("schema") != SCHEMA
        or result.get("status") != "complete"
        or result.get("steps_completed") != o.updates
    ):
        raise ValueError("incomplete result")
    if (
        not result.get("environment_closed")
        or result.get("first_failure") is not None
        or result.get("stop_further_dispatch")
    ):
        raise ValueError("failed execution or environment closure")
    if any(
        (
            result.get(k) != want
            for k, want in [
                ("training_seed", o.seed),
                ("eval_seed", o.eval_seed),
                ("experiment_scope", o.scope),
                ("selection_allowed", False),
            ]
        )
    ):
        raise ValueError("result protocol mismatch")
    variants, arms = (declared["variants"], declared["arms"])
    nv, na = (len(variants), len(arms))

    def artifact(entry, expected_name=None):
        if expected_name is not None and entry["path"] != expected_name:
            raise ValueError("artifact identity mismatch: " + expected_name)
        path = output / entry["path"]
        if path.parent.resolve() != output.resolve() or D.sha(path) != entry["sha256"]:
            raise ValueError("invalid artifact path/hash")
        return path

    aa, ca, ce, post = (np.zeros(n, np.int64) for n in (na, nv, nv, na))
    previous = 0
    cal_history = {0: ca.copy()}
    consumed_support = []
    for block in result["blocks"]:
        first, last = (block["first_step"], block["last_step"])
        if first != previous + 1 or last < first or last > o.updates:
            raise ValueError("incomplete/overlapping diagnostic blocks")
        path = artifact(
            dict(path=block["diagnostics_npz"], sha256=block["sha256"]),
            f"block_{first}_{last}.npz",
        )
        with np.load(path, allow_pickle=False) as data:
            metrics = {k: data[k] for k in data.files}
        count = last - first + 1
        D.verify_block_evidence(metrics, count, arms, "shared", np)
        if D.first_failure(metrics, first, np) is not None:
            raise ValueError("numeric failure in accepted block")
        applied, abstained, posterior = D.block_counts(metrics, count, na, np)
        cal, ess = verify_variant_evidence(
            metrics, count, variants, declared["args"]["batch_size"], np
        )
        for name, want in [
            ("training_seed", o.seed),
            ("eval_seed", o.eval_seed),
            ("experiment_scope", o.scope),
        ]:
            if np.shape(metrics.get(name)) != () or metrics[name].item() != want:
                raise ValueError("block protocol mismatch: " + name)
        expected_ready = np.arange(first, last + 1) > o.warmup
        if not np.array_equal(metrics["posterior_ready"], expected_ready):
            raise ValueError("posterior consumption schedule mismatch")
        if np.asarray(metrics["actor_step"]).dtype.kind not in "iu":
            raise ValueError("actor step evidence must be integer")
        if not np.array_equal(
            metrics["actor_step"], aa[None, :] + np.cumsum(metrics["actor_updated"], axis=0)
        ):
            raise ValueError("actor step evidence mismatch")
        if not np.array_equal(
            metrics["cal_variant_step"],
            ca[None, :] + np.cumsum(metrics["cal_variant_accepted"], axis=0),
        ):
            raise ValueError("calibrator step evidence mismatch")
        aa += applied
        ca += cal
        ce += ess
        post += posterior
        cal_history[last] = ca.copy()
        consumed_support.append((first, last, metrics["supported_fraction"]))
        previous = last
    if previous != o.updates:
        raise ValueError("missing final diagnostic block")
    for key, expected in [
        ("actor_applied_updates", aa),
        ("scale_applied_updates", ca),
        ("scale_ess_abstentions", ce),
        ("actor_post_ready_applied_updates", post),
        ("actor_valid_abstentions", o.updates - aa),
    ]:
        values = result.get(key)
        if (
            not isinstance(values, list)
            or any((type(x) is not int for x in values))
            or (not np.array_equal(values, expected))
        ):
            raise ValueError("summary counter mismatch: " + key)
    if (
        np.any(ca + ce != o.updates)
        or type(result.get("nuisance_valid_updates")) is not int
        or result.get("nuisance_valid_updates") != o.updates
    ):
        raise ValueError("missing accepted/abstained calibration or nuisance updates")
    counters = result["state_counters"]
    if any(
        (
            type(counters.get(k)) is not int or counters.get(k) != o.updates
            for k in ("qf", "vf", "qf_target")
        )
    ):
        raise ValueError("nuisance state counter mismatch")
    cal_counts = counters.get("calibrators", {})
    if (
        not isinstance(cal_counts, dict)
        or any((type(x) is not int for x in cal_counts.values()))
        or cal_counts != {v["name"]: int(n) for v, n in zip(variants, ca)}
    ):
        raise ValueError("calibrator state counter mismatch")
    if o.scope == "diagnostic" and any(
        (post[i] == 0 for i, a in enumerate(arms) if a["mode"] != "off")
    ):
        raise ValueError("diagnostic had no consumed reference for an actor")
    artifacts = result["artifacts"]
    expected_refs = list(range(o.warmup, o.updates, o.refresh_interval))
    if set(artifacts["references"]) != {str(s) for s in expected_refs}:
        raise ValueError("missing/unexpected reference artifacts")
    metadata = json.loads((output / "data_metadata.json").read_text())
    for step in expected_refs:
        row = json.loads(
            artifact(artifacts["references"][str(step)], f"reference_{step}.json").read_text()
        )
        if row["step"] != step or len(row["variants"]) != nv:
            raise ValueError("misaligned variant reference")
        if any(
            (
                row.get(k) != want
                for k, want in [
                    ("training_seed", o.seed),
                    ("eval_seed", o.eval_seed),
                    ("experiment_scope", o.scope),
                ]
            )
        ):
            raise ValueError("reference protocol identity mismatch")
        for v, reference in zip(variants, row["variants"]):
            if (
                reference["variant"] != v
                or type(reference.get("step")) is not int
                or reference["step"] != step
            ):
                raise ValueError("reference used a different scale variant")
            D.verify_reference_evidence(reference, declared["args"]["posterior"], metadata)
        if (
            step not in cal_history
            or any((type(r.get("scale_updates")) is not int for r in row["variants"]))
            or [r["scale_updates"] for r in row["variants"]] != cal_history[step].tolist()
        ):
            raise ValueError("reference calibrator counter does not match consumed updates")
        verify_reference_fit_exposure(row["variants"], o.scope)
        verify_common_references(row["variants"])
        if row["checkpoint"] != artifacts.get(f"reference_{step}"):
            raise ValueError("reference checkpoint record mismatch")
        artifact(row["checkpoint"], f"reference_{step}.msgpack")
        next_step = min(step + o.refresh_interval, o.updates)
        for first, last, support in consumed_support:
            lo, hi = (max(first, step + 1), min(last, next_step))
            if lo > hi:
                continue
            selected = support[lo - first : hi - first + 1]
            for i, arm in enumerate(arms):
                if arm["mode"] == "off":
                    expected = 1.0
                else:
                    ref = row["variants"][arm["variant_index"]]
                    radius = ref["full_radius" if arm["mode"] == "full" else "conformal_radius"]
                    expected = 0.0 if radius == "Infinity" else 1.0
                if not np.all(selected[:, i] == expected):
                    raise ValueError("actor support disagrees with consumed reference")
    schedule = declared["evaluation_schedule"]
    if set(artifacts["curves"]) != {str(x["step"]) for x in schedule["curve"]}:
        raise ValueError("incomplete curve schedule")
    for point in schedule["curve"]:
        entries = artifacts["curves"][str(point["step"])]
        if set(entries) != {a["name"] for a in arms}:
            raise ValueError("missing curve actor")
        for name, entry in entries.items():
            D.verify_evaluation(
                artifact(entry, f"curve_{point['step']}_{name}.npz"), point, o, np, final=False
            )
    if set(artifacts["final"]) != {a["name"] for a in arms}:
        raise ValueError("missing final actor")
    for i, a in enumerate(arms):
        path = artifact(artifacts["final"][a["name"]], f"final_{a['name']}.npz")
        D.verify_evaluation(path, schedule["final"], o, np, final=True)
        with np.load(path, allow_pickle=False) as archive:
            counter = archive["actor_applied_updates"]
            if counter.shape != () or counter.dtype.kind not in "iu" or counter.item() != aa[i]:
                raise ValueError("final archive actor count mismatch")
    artifact(artifacts["training_end"], "training_end.msgpack")
    D.verify_sources(output / "source", manifest["source_sha256"])
    return True
