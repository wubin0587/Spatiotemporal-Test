# -*- coding: utf-8 -*-
"""
@File    : manager.py
@Desc    : InterventionManager — the central orchestrator for the intervention subsystem.

This class implements the ``InterventionHook`` protocol defined in
``models/engine/facade.py``, making it directly injectable into a
``SimulationFacade`` via ``sim.set_intervention_manager(manager)``.

Responsibilities
----------------
1. Hold a list of (trigger, policy, optional_branch_config) triplets called
   "intervention rules".
2. At each simulation step (``evaluate_and_apply``):
      a. Evaluate each trigger against the current engine state.
      b. If a trigger fires, optionally create a checkpoint before applying
         the policy (for later counterfactual comparison).
      c. Apply the corresponding policy.
      d. Record the event in a structured log.
3. Expose the ``BranchManager`` so users can access created checkpoints.

Intervention Rule
-----------------
An "intervention rule" is the atomic unit of the system.  It binds:

    trigger  : InterventionTrigger   -- when to act
    policy   : BasePolicy            -- what to do
    auto_checkpoint : bool           -- snapshot before acting? (default True)
    label    : str                   -- human label for logs

YAML config schema
------------------
interventions:
  - label: "network_rewire_at_100"
    auto_checkpoint: true
    trigger:
      type: step
      step: 100
      max_fires: 1
    policy:
      type: network_rewire        # matches BasePolicy._REGISTRY key
      fraction: 0.1

  - label: "suppress_events_on_polarization"
    auto_checkpoint: false
    trigger:
      type: polarization
      threshold: 0.4
      cooldown: 20
      max_fires: 0
    policy:
      type: event_suppress
      source: exogenous
      duration: 10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .trigger import InterventionTrigger
from .policies.base import BasePolicy
from .branch.checkpoint import BranchManager

logger = logging.getLogger(__name__)


# =============================================================================
# InterventionRule — a named (trigger, policy) pair
# =============================================================================

@dataclass
class InterventionRule:
    """
    Binds a trigger condition to a policy action.

    Attributes
    ----------
    trigger : InterventionTrigger
    policy : BasePolicy
    label : str
        Human-readable name used in logs and the execution log.
    auto_checkpoint : bool
        If True, a BranchManager checkpoint is created *before* the policy
        is applied whenever the trigger fires.  Enables counterfactual analysis
        without manual checkpoint management.
    """

    trigger: InterventionTrigger
    policy: BasePolicy
    label: str = ""
    auto_checkpoint: bool = True

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.trigger.name}→{self.policy.name}"


# =============================================================================
# InterventionManager
# =============================================================================

class InterventionManager:
    """
    Central orchestrator that implements the ``InterventionHook`` protocol.

    Parameters
    ----------
    branch_manager : BranchManager, optional
        Shared BranchManager instance.  If None, one will be created lazily
        the first time a checkpoint is requested (requires the engine to be
        available at that point via ``evaluate_and_apply``).

    Notes
    -----
    An InterventionManager can be shared across multiple SimulationFacade
    instances only if ``branch_manager`` is set to None (each facade will
    trigger checkpoint creation against its own engine).  For multi-branch
    experiments it is cleaner to use one manager per facade.
    """

    def __init__(self, branch_manager: Optional[BranchManager] = None):
        self.rules: List[InterventionRule] = []
        self._branch_manager: Optional[BranchManager] = branch_manager

        # Execution log: one entry per firing event
        # Entry format:
        #   {
        #     'step': int,
        #     'time': float,
        #     'label': str,
        #     'checkpoint_id': str | None,
        #     'policy_result': Any,
        #   }
        self.execution_log: List[Dict[str, Any]] = []

        logger.info("InterventionManager created.")

    # ------------------------------------------------------------------
    # InterventionHook protocol implementation
    # ------------------------------------------------------------------

    def evaluate_and_apply(self, engine: Any, stats: Dict[str, Any]) -> None:
        """
        Called by ``SimulationFacade.step()`` after every engine tick.

        Iterates over all registered rules, evaluates their triggers, and
        applies policies whose triggers fire.

        Parameters
        ----------
        engine : StepExecutor
            Live engine.  Read for trigger evaluation, written by policies.
        stats : Dict[str, Any]
            Metrics dict returned by the most recent ``engine.step()`` call.
        """
        for rule in self.rules:
            # Skip exhausted triggers early (avoids log spam)
            if rule.trigger.has_exhausted:
                continue

            fires = rule.trigger.evaluate(engine, stats)

            if not fires:
                continue

            # --- Trigger fired ---
            checkpoint_id: Optional[str] = None

            if rule.auto_checkpoint:
                checkpoint_id = self._ensure_branch_manager(engine).create_checkpoint(
                    label=f"pre:{rule.label}",
                    meta={
                        "rule_label": rule.label,
                        "step": engine.time_step,
                    },
                )

            policy_entry = rule.policy.apply(engine)

            log_entry = {
                "step": engine.time_step,
                "time": engine.current_time,
                "label": rule.label,
                "checkpoint_id": checkpoint_id,
                "policy_result": policy_entry.get("result"),
            }
            self.execution_log.append(log_entry)

            logger.info(
                f"[InterventionManager] Rule '{rule.label}' executed. "
                f"checkpoint={checkpoint_id!r}"
            )

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        trigger: InterventionTrigger,
        policy: BasePolicy,
        label: str = "",
        auto_checkpoint: bool = True,
    ) -> "InterventionManager":
        """
        Register a new intervention rule.

        Returns ``self`` for method chaining.

        Parameters
        ----------
        trigger : InterventionTrigger
        policy : BasePolicy
        label : str
            Human-readable rule name.
        auto_checkpoint : bool
            Create a checkpoint before applying the policy when trigger fires.

        Example
        -------
            manager.add_rule(
                StepTrigger(step=100),
                MyPolicy(config={'param': 1}),
                label="my_intervention",
            )
        """
        rule = InterventionRule(
            trigger=trigger,
            policy=policy,
            label=label,
            auto_checkpoint=auto_checkpoint,
        )
        self.rules.append(rule)
        logger.info(f"Rule added: '{rule.label}'")
        return self

    def remove_rule(self, label: str) -> bool:
        """
        Remove the first rule with the given label.

        Returns True if a rule was found and removed.
        """
        for i, rule in enumerate(self.rules):
            if rule.label == label:
                del self.rules[i]
                logger.info(f"Rule removed: '{label}'")
                return True
        logger.warning(f"remove_rule: no rule with label '{label}'.")
        return False

    def clear_rules(self) -> None:
        """Remove all registered rules."""
        self.rules.clear()
        logger.info("All intervention rules cleared.")

    # ------------------------------------------------------------------
    # Branch manager access
    # ------------------------------------------------------------------

    @property
    def branch_manager(self) -> Optional[BranchManager]:
        """The currently attached BranchManager (may be None)."""
        return self._branch_manager

    def set_branch_manager(self, branch_manager: BranchManager) -> None:
        """Attach or replace the BranchManager."""
        self._branch_manager = branch_manager

    def _ensure_branch_manager(self, engine: Any) -> BranchManager:
        """
        Return the existing BranchManager or create one on-demand.

        Lazy creation avoids requiring the engine to be available at
        InterventionManager construction time.
        """
        if self._branch_manager is None:
            self._branch_manager = BranchManager(engine)
            logger.info("BranchManager created lazily by InterventionManager.")
        return self._branch_manager

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all triggers and policies to their initial state.

        Clears the execution log.  Does NOT clear the BranchManager's
        checkpoint registry (checkpoints persist across resets so that
        counterfactual comparisons remain valid).
        """
        for rule in self.rules:
            rule.trigger.reset()
            rule.policy.reset()
        self.execution_log.clear()
        logger.info("InterventionManager reset.")

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Return the full list of firing events."""
        return list(self.execution_log)

    def summarize(self) -> Dict[str, Any]:
        """
        Return a lightweight summary dict for logging or serialisation.

        Returns
        -------
        dict with keys:
            'num_rules', 'num_firings', 'rules', 'executions'
        """
        return {
            "num_rules": len(self.rules),
            "num_firings": len(self.execution_log),
            "rules": [
                {
                    "label": r.label,
                    "trigger": repr(r.trigger),
                    "policy": repr(r.policy),
                    "auto_checkpoint": r.auto_checkpoint,
                }
                for r in self.rules
            ],
            "executions": self.execution_log,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        sim: Any,
    ) -> "InterventionManager":
        """
        Build an InterventionManager from a YAML-derived config dict.

        Parameters
        ----------
        cfg : dict
            The ``interventions`` section of the experiment YAML.
            Expected structure::

                interventions:
                  - label: "..."
                    auto_checkpoint: true
                    trigger: { type: step, step: 100, ... }
                    policy:  { type: network_rewire, ... }

        sim : SimulationFacade
            The facade that this manager will be attached to.
            Used to initialise the BranchManager.

        Returns
        -------
        InterventionManager
        """
        branch_mgr = BranchManager(sim)
        manager = cls(branch_manager=branch_mgr)

        rules_cfg = cfg if isinstance(cfg, list) else cfg.get("interventions", [])

        for rule_cfg in rules_cfg:
            label = rule_cfg.get("label", "")
            auto_checkpoint = rule_cfg.get("auto_checkpoint", True)

            trigger = InterventionTrigger.from_config(rule_cfg["trigger"])
            policy = BasePolicy.from_config(rule_cfg["policy"])

            manager.add_rule(
                trigger=trigger,
                policy=policy,
                label=label,
                auto_checkpoint=auto_checkpoint,
            )

        logger.info(
            f"InterventionManager built from config: {len(manager.rules)} rules."
        )
        return manager

    def __repr__(self) -> str:
        return (
            f"<InterventionManager rules={len(self.rules)} "
            f"firings={len(self.execution_log)}>"
        )
