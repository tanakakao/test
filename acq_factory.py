from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

from botorch.acquisition import (
    qNegIntegratedPosteriorVariance,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qPosteriorStandardDeviation
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples
from botorch.models import KroneckerMultiTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model_list_gp_regression import ModelListGP, ModelListGPyTorchModel
from botorch.models.multitask import MultiTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import draw_sobol_samples

from bayes_optimization.models.acquisitions import (
    BALDAcquisition,
    BALDMultiOutputAcquisition,
    EntropyClassifierAcquisition,
    EntropyMultiOutputAcquisition,
    JointStraddleClassifierAcquisition,
    LogDetqStraddle,
    LogDetqStraddleMultiCommon,
    StraddleClassifierAcquisition,
    make_logdetlike_variance_objective,
    qICUAcquisition,
    qJointBoundaryVariance,
    qStraddle,
    qStraddleMultiCommon,
)
from bayes_optimization.models.acquisitions.robust import (
    MCJointRobustAcquisition,
    RobustqExpectedHypervolumeImprovement,
    compute_robust_train_y,
)
from bayes_optimization.models.objectives_and_y_constraints import (
    make_constraints_y,
    y_processing_for_constraints,
)

# optional utility locations differ by BoTorch version
try:
    from botorch.acquisition.multi_objective.utils import (
        compute_sample_box_decomposition,
        random_search_optimizer,
        sample_optimal_points,
    )
except ImportError:  # pragma: no cover
    from botorch.utils.multi_objective.box_decompositions.utils import (  # type: ignore
        compute_sample_box_decomposition,
    )
    from botorch.utils.sampling import sample_optimal_points  # type: ignore
    random_search_optimizer = None

_MULTI_ONLY = {"EHI", "NEHI", "NParEGO"}
_SINGLE_ONLY = {"EI", "PI", "UCB"}
_UNSUPPORTED_WITH_Y_CONSTRAINT = {"PES", "MVE", "JES", "AL"}


def standardize_ref_point(model, ref_point_orig: Tensor) -> Tensor:
    """モデルの outcome_transform スケールに ref_point を合わせる。"""
    if isinstance(model, ModelListGP):
        return torch.stack(
            [
                (ref_point_orig[i] - sub.outcome_transform.means)
                / sub.outcome_transform.stds
                for i, sub in enumerate(model.models)
            ]
        )
    return (ref_point_orig - model.outcome_transform.means) / model.outcome_transform.stds


def get_pareto_sample(model, bounds: Tensor, n: int = 10):
    """Pareto set/front をサンプリングする。"""
    optimizer_kwargs = {"pop_size": 2000, "max_tries": 200}
    return sample_optimal_points(
        model=model,
        bounds=bounds,
        num_samples=n,
        num_points=n,
        optimizer=random_search_optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )


def _resolve_model_flags(model, train_Y: Tensor) -> tuple[str, bool, bool]:
    output_type = "multi" if train_Y.size(-1) > 1 else "single"
    is_multitask_model = isinstance(model, KroneckerMultiTaskGP)
    if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
        multi_task = isinstance(model.models[0], (MultiTaskGP, SingleTaskMultiFidelityGP))
    else:
        multi_task = False
    return output_type, is_multitask_model, multi_task


def _validate_acq_method(
    acq_method: str,
    output_type: str,
    is_multitask_model: bool,
    has_y_constraints: bool,
    model,
) -> None:
    if output_type == "single" and acq_method in _MULTI_ONLY:
        raise ValueError(f"{acq_method} はシングルアウトプットでは使用できません。")
    if output_type == "multi" and acq_method in _SINGLE_ONLY:
        raise ValueError(f"{acq_method} はマルチアウトプットでは使用できません。")
    if is_multitask_model and acq_method not in _MULTI_ONLY:
        raise ValueError(f"{acq_method} は KroneckerMultiTaskGP（多タスク）では使用できません。")
    if has_y_constraints and acq_method in _UNSUPPORTED_WITH_Y_CONSTRAINT:
        raise ValueError(f"{acq_method} では目的変数制約はサポートしていません。")
    if (not isinstance(model, ModelListGP)) and output_type == "multi" and acq_method == "KG":
        raise ValueError("多目的 Knowledge Gradient は ModelListGP のみサポートしています。")


def get_acqf(
    model,
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    acq_method: Optional[str] = "PI",
    y_constraints_idx: Optional[List[int]] = None,
    y_ops: Optional[List[Optional[str]]] = None,
    y_thresholds: Optional[List[float]] = None,
    y_weights: Optional[List[float]] = None,
    y_directions: Optional[List[str]] = None,
    h_lse: Optional[Union[List[float], Tensor]] = None,
    risk_type: str = None,
    n_cand: Optional[int] = 1,
    dtype: torch.dtype = torch.double,
):
    y_constraints_idx = y_constraints_idx or []
    y_ops = y_ops or []
    y_thresholds = y_thresholds or []
    y_weights = y_weights or []
    y_directions = y_directions or []
    h_lse = h_lse or []

    output_type, is_multitask_model, multi_task = _resolve_model_flags(model, train_Y)
    has_y_constraints = bool(y_ops) and any(op is not None for op in y_ops)

    y_constraints_idx, y_ops, y_thresholds, y_weights, y_directions = y_processing_for_constraints(
        train_Y,
        y_constraints_idx,
        y_ops,
        y_thresholds,
        y_directions,
        y_weights,
        dtype,
    )
    constraints, scalar_obj, objective = make_constraints_y(
        y_constraints_idx,
        y_ops,
        y_thresholds,
        y_weights,
        y_directions,
        risk_type,
        dtype,
    )

    _validate_acq_method(acq_method, output_type, is_multitask_model, has_y_constraints, model)

    Y_obj = scalar_obj(train_Y)
    sampler = SobolQMCNormalSampler(torch.Size([64]))

    if acq_method == "EI":
        acqf = qLogExpectedImprovement(
            model=model,
            best_f=Y_obj.max() * 0.9,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )
    elif acq_method == "PI":
        acqf = qProbabilityOfImprovement(
            model=model,
            best_f=Y_obj.max() * 0.9,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )
    elif acq_method == "UCB":
        acqf = qUpperConfidenceBound(model=model, beta=0.3, sampler=sampler, objective=objective)
    elif acq_method == "PES":
        if output_type == "single":
            optimal_inputs, _ = get_optimal_samples(model=model, bounds=bounds, num_optima=32)
            acqf = qPredictiveEntropySearch(model=model, optimal_inputs=optimal_inputs, threshold=1e-2)
        else:
            ps, _ = get_pareto_sample(model, bounds, n=10)
            acqf = qMultiObjectivePredictiveEntropySearch(model=model, pareto_sets=ps)
    elif acq_method == "MVE":
        if output_type == "single":
            candidate_set = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                1000, bounds.size(1), dtype=dtype, device=bounds.device
            )
            acqf = qLowerBoundMaxValueEntropy(model=model, candidate_set=candidate_set)
        else:
            _, pf = get_pareto_sample(model, bounds, n=10)
            acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                model=model,
                hypercell_bounds=compute_sample_box_decomposition(pf),
                estimation_type="LB",
            )
    elif acq_method == "JES":
        if output_type == "single":
            optimal_inputs, optimal_outputs = get_optimal_samples(model=model, bounds=bounds, num_optima=32)
            acqf = qJointEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
            )
        else:
            ps, pf = get_pareto_sample(model, bounds, n=10)
            acqf = qLowerBoundMultiObjectiveJointEntropySearch(
                model=model,
                pareto_sets=ps,
                pareto_fronts=pf,
                hypercell_bounds=compute_sample_box_decomposition(pf),
                estimation_type="LB",
            )
    elif acq_method == "EHI":
        ref_point = Y_obj.min(dim=0).values.to(dtype)
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_obj)
        acqf = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )
    elif acq_method == "NEHI":
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=Y_obj.min(dim=0).values.to(dtype),
            X_baseline=train_X,
            prune_baseline=True,
            cache_root=True,
            incremental_nehvi=True,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )
    elif acq_method == "NParEGO":
        acqf = qLogNParEGO(
            model=model,
            X_baseline=train_X,
            prune_baseline=True,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            scalarization_weights=torch.ones_like(y_weights),
        )
    elif acq_method == "AL":
        if output_type == "single":
            mc_samples = draw_sobol_samples(bounds=bounds, n=64, q=1).squeeze(-2)
            acqf = qNegIntegratedPosteriorVariance(model=model, mc_points=mc_samples)
        else:
            acqf = qSimpleRegret(model=model, sampler=sampler, objective=make_logdetlike_variance_objective())
    elif acq_method == "KG" and not multi_task:
        if output_type == "single":
            acqf = qKnowledgeGradient(
                model=model,
                num_fantasies=64,
                current_value=None,
                sampler=SobolQMCNormalSampler(torch.Size([64])),
                objective=objective,
            )
        else:
            ref_point = (Y_obj.min(dim=0).values - 0.1).to(dtype)
            acqf = qHypervolumeKnowledgeGradient(
                model=model,
                ref_point=ref_point,
                current_value=Hypervolume(ref_point).compute(Y_obj),
                objective=objective,
            )
    elif acq_method == "Straddle":
        acqf = qStraddle(model, 5.0, h_lse[0]) if output_type == "single" else qStraddleMultiCommon(model, 5.0, h_lse)
    elif acq_method == "LogStraddle":
        acqf = LogDetqStraddle(model, 5.0, h_lse[0]) if output_type == "single" else LogDetqStraddleMultiCommon(model, 5.0, h_lse)
    elif acq_method == "ICU":
        if output_type != "single":
            raise ValueError("ICU は多目的には対応していません。")
        acqf = qICUAcquisition(model, h_lse[0])
    elif acq_method == "JBV":
        if output_type == "single":
            raise ValueError("JBV は単目的には対応していません。")
        acqf = qJointBoundaryVariance(model, h_lse)
    elif acq_method == "BALD":
        acqf = BALDAcquisition(model) if output_type == "single" else BALDMultiOutputAcquisition(model)
    elif acq_method == "Straddle_cls":
        acqf = StraddleClassifierAcquisition(model) if output_type == "single" else JointStraddleClassifierAcquisition(model)
    elif acq_method == "Entropy":
        acqf = EntropyClassifierAcquisition(model) if output_type == "single" else EntropyMultiOutputAcquisition(model)
    elif acq_method == "STD":
        variance_obj = objective if output_type == "single" else make_logdetlike_variance_objective()
        acqf = qPosteriorStandardDeviation(
            model=model,
            sampler=sampler,
            objective=variance_obj,
            constraints=constraints,
        )
    elif acq_method == "Robust":
        if output_type == "single":
            acqf = MCJointRobustAcquisition(model, beta=2.0, noise_penalty=2.0, objective=objective)
        else:
            ref_point = Y_obj.min(dim=0).values.to(dtype)
            robust_train_y = compute_robust_train_y(model, train_X, noise_penalty=2.5)
            acqf = RobustqExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=FastNondominatedPartitioning(ref_point=ref_point, Y=robust_train_y),
                beta=2.0,
                noise_penalty=2.5,
                objective=objective,
            )
    else:
        raise ValueError(f"Unsupported acquisition function type: {acq_method}")

    if multi_task:
        acqf = FixedFeatureAcquisitionFunction(
            acq_function=acqf,
            d=train_X.shape[-1],
            columns=[train_X.shape[-1] - 1],
            values=torch.tensor([1], dtype=dtype, device=train_X.device),
        )

    return acqf
