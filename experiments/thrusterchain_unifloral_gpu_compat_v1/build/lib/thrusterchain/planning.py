"""Constrained finite-horizon references; never clipped-LQR optimality claims."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from .datasets import immutable_array, forbid_during_evaluation

from .environment import Physics, matrices, thrust


def array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value, dtype="<f8")
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class IdentifiedModel:
    A: np.ndarray
    B: np.ndarray
    receipt: dict


def identify_from_training(dataset):
    """OLS on TRAIN transitions only, with the declared known actuator map.

    The state transition matrices are learned, but state coordinates, reward,
    actuator law and horizon are public task structure. No evaluation data enter.
    """
    forbid_during_evaluation("model identification")
    train = dataset.split("train")
    x = np.asarray(train["observations"][..., :13], float)
    actions = np.asarray(train["actions"], float)
    if x.ndim != 3 or x.shape[-1] != 13 or actions.shape != (x.shape[0], x.shape[1]-1, 4):
        raise ValueError("expected complete grouped state/action trajectories")
    physics = Physics(**dataset.metadata["physics"])
    features = np.concatenate((x[:, :-1].reshape(-1, 13),
                               thrust(actions, physics).reshape(-1, 4)), axis=1)
    targets = x[:, 1:].reshape(-1, 13)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("nonfinite training transitions")
    coef, _, rank, singular = np.linalg.lstsq(features, targets, rcond=None)
    if rank != 17:
        raise ValueError(f"identification refused: training excitation rank {rank}/17")
    residual = features @ coef - targets
    A, B = immutable_array(coef[:13].T, "<f8"), immutable_array(coef[13:].T, "<f8")
    receipt = dict(kind="train_only_structure_aware_identification", dataset_hash=dataset.digest,
                   training_groups=np.asarray(train["group_ids"]).tolist(),
                   rank=int(rank), required_rank=17,
                   condition_number=float(singular[0]/singular[-1]),
                   training_rmse=float(np.sqrt(np.mean(residual**2))),
                   singular_values=singular.tolist(), regularization=0.0,
                   feature_scaling="none; raw public observation coordinates",
                   state_access="public float32 observation first13 only; exact info states not used",
                   matrix_hash=array_hash(A, B),
                   known_structure=["13 physical state coordinates", "actuator thrust map",
                                    "quadratic reward", "action bounds", "horizon", "discount"],
                   fit_uses="train split only", closed_loop_performance_guarantee=False)
    return IdentifiedModel(A, B, receipt)


def validation_residuals(dataset, model):
    """Report permitted offline validation error; never refit or select the model."""
    forbid_during_evaluation("identification validation")
    split=dataset.split("tuning_validation")
    x=np.asarray(split["observations"][..., :13],float)
    actions=np.asarray(split["actions"],float)
    physics=Physics(**dataset.metadata["physics"])
    predicted=x[:,:-1]@model.A.T+thrust(actions,physics)@model.B.T
    residual=predicted-x[:,1:]
    return dict(split="tuning_validation",used_for_model_fitting=False,used_for_model_selection=False,
        rmse=float(np.sqrt(np.mean(residual**2))),max_absolute_error=float(np.max(np.abs(residual))),
        group_ids=split["group_ids"].tolist(),model_hash=array_hash(model.A,model.B))


class BoundedPlanner:
    """Convex bounded QP in effective commands, including the canonical stall world.

    For canonical stall parameters every command outside the knee is dominated
    by its effective thrust command inside the knee: same next state, lower
    command cost. Thus restricting to [-knee,knee] loses no continuous optimum.
    The Gym float32 command lattice is a separate quantization; this solver
    certifies the real-command relaxation, not an exact lattice optimum.

    Matrix provenance is derived from the constructor path. Explicit matrices
    are caller-supplied; a requested training label is retained as a declaration,
    not independently verified evidence of how those matrices were obtained.
    """
    def __init__(self, A=None, B=None, physics=None, planning_horizon=100,
                 provenance=None, kkt_tolerance=1e-5):
        self.physics = physics or Physics()
        if (A is None) != (B is None):
            raise ValueError("provide both learned matrices or neither")
        self.declared_provenance = None if provenance is None else str(provenance)
        if A is None:
            if self.declared_provenance not in (None, "privileged_true_model"):
                raise ValueError("provenance conflicts with omitted A/B: constructor loads privileged_true_model")
            A, B = matrices(self.physics)
            self.provenance = "privileged_true_model"
            self.matrix_source = "environment.matrices"
        else:
            self.provenance = "caller_supplied_model"
            self.matrix_source = "explicit_A_B_arguments"
        self.A = immutable_array(A, "<f8")
        self.B = immutable_array(B, "<f8")
        if self.A.shape != (13, 13) or self.B.shape != (13, 4):
            raise ValueError("expected A13x13 B13x4")
        if not np.isfinite(self.A).all() or not np.isfinite(self.B).all():
            raise ValueError("nonfinite model")
        if isinstance(planning_horizon, bool) or int(planning_horizon) != planning_horizon or not 1 <= planning_horizon <= 100:
            raise ValueError("planning_horizon must be integer 1..100")
        if not 0 < self.physics.gamma <= 1 or self.physics.throttle_cost <= 0:
            raise ValueError("requires positive quadratic command cost and discount")
        if self.physics.stall and not (0 <= self.physics.stall_slope <= 2*self.physics.knee/(1-self.physics.knee)):
            raise ValueError("stall parameters outside proven dominated-command regime")
        if not np.isfinite(kkt_tolerance) or kkt_tolerance <= 0:
            raise ValueError("positive KKT tolerance required")
        self.bound = self.physics.knee if self.physics.stall else 1.0
        self.horizon = int(planning_horizon)
        self.tolerance = float(kkt_tolerance)
        self._condensed = {}
        self.last_certificate = None
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name != "last_certificate":
            raise AttributeError("planner inference configuration is frozen")
        object.__setattr__(self, name, value)

    def _qp(self, horizon):
        if horizon not in self._condensed:
            # x_t = F_t x_0 + G_t u_(0:H-1); pre-transition costs only.
            F, G = np.eye(13), np.zeros((13, 4*horizon))
            H = np.zeros((4*horizon, 4*horizon))
            C = np.zeros((4*horizon, 13))
            Q = np.diag([1.0]*10 + [0.0]*3)
            for t in range(horizon):
                weight = self.physics.gamma ** t
                H += weight * G.T @ Q @ G
                C += weight * G.T @ Q @ F
                H[t*4:t*4+4, t*4:t*4+4] += weight*self.physics.throttle_cost*np.eye(4)
                F = self.A @ F
                G = self.A @ G
                G[:, t*4:t*4+4] += self.B
            if not np.isfinite(H).all() or not np.isfinite(C).all():
                raise RuntimeError("bounded planner refused nonfinite condensed objective")
            self._condensed[horizon] = (immutable_array(2*H), immutable_array(2*C))
        return self._condensed[horizon]

    def plan(self, state, remaining_steps):
        state = np.asarray(state, float)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError("state must have 13 finite coordinates")
        if isinstance(remaining_steps, bool) or int(remaining_steps) != remaining_steps or not 1 <= remaining_steps <= 100:
            raise ValueError("remaining_steps must be integer 1..100")
        h = min(self.horizon, int(remaining_steps))
        H, C = self._qp(h)
        linear = C @ state
        def objective(u):
            Hu = H @ u
            return float(.5*u@Hu + linear@u), Hu+linear
        solved = minimize(objective, np.zeros(4*h), method="L-BFGS-B", jac=True,
                          bounds=[(-self.bound,self.bound)]*(4*h),
                          options={"ftol":1e-14,"gtol":self.tolerance*.05,
                                   "maxiter":4000,"maxls":50,"maxcor":30})
        u = solved.x
        gradient = H@u+linear
        # Unit-step projected gradient is zero exactly at the box KKT solution.
        residual = float(np.max(np.abs(u-np.clip(u-gradient,-self.bound,self.bound))))
        if not np.isfinite(u).all() or not np.isfinite(residual) or residual > self.tolerance:
            raise RuntimeError(f"bounded planner refused uncertified solve: KKT={residual:.3g}; {solved.message}")
        executed = np.asarray(u[:4], np.float32).astype(float)
        self.last_certificate = dict(provenance=self.provenance,
            declared_provenance=self.declared_provenance, matrix_source=self.matrix_source,
            training_provenance_verified=False,
            planner_horizon=h, objective_remaining_steps=int(remaining_steps),
            full_remaining_horizon=(h == remaining_steps), projected_gradient_residual=residual,
            solver_success=bool(solved.success), iterations=int(solved.nit),
            continuous_command_optimality="convex box QP to reported KKT tolerance",
            float32_lattice_optimality=False,
            first_action_quantization_linf=float(np.max(np.abs(executed-u[:4]))),
            bounded_action_limit=float(self.bound))
        return executed

    def act(self, observation):
        observation = np.asarray(observation)
        if observation.shape != (14,):
            raise ValueError("time-aware observation14 required")
        remaining = int(round(float(observation[-1])*self.physics.length))
        return self.plan(observation[:13], remaining)

    def receipt(self):
        return dict(provenance=self.provenance, declared_provenance=self.declared_provenance,
                    matrix_source=self.matrix_source, training_provenance_verified=False,
                    provenance_evidence="constructor matrix source only; explicit matrices and caller labels do not verify training history",
                    model_hash=array_hash(self.A,self.B),
                    planning_horizon=self.horizon, gamma=self.physics.gamma,
                    physics=self.physics.as_dict(), kkt_tolerance=self.tolerance,
                    optimization="constrained L-BFGS-B convex QP",
                    exact_lattice_optimum=False,
                    controller_parameters_updated_during_evaluation=False)

