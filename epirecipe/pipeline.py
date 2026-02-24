"""Enhanced EpiRecipe Pipeline - Constraint-Based Model Generation
===================================================================

This module implements an improved pipeline for generating diverse, valid
compartmental epidemic models with:

1. Constraint-based compartment selection and topology construction
2. Hierarchical parameter sampling (epidemiologically consistent)
3. Model validation with invariant checking
4. Configurable observables
5. Population stratification as first-class citizen
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for importing from old pipeline
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

CATALOG_PATH = Path(__file__).with_name("catalog.json")

class CatalogError(RuntimeError):
    """Raised when an invalid catalog query is made."""
    pass


class ValidationError(RuntimeError):
    """Raised when model validation fails."""
    pass


@dataclass
class Catalog:
    """Enhanced catalog with compartments, dynamics, and topology rules."""
    raw: Dict
    compartments: Dict[str, Dict] = field(init=False)
    dynamics: Dict[str, Dict] = field(init=False)
    topology_rules: Dict = field(init=False)
    parameter_distributions: Dict = field(init=False)
    validation_rules: Dict = field(init=False)
    observables: Dict = field(init=False)
    stratification: Dict = field(init=False)

    def __post_init__(self) -> None:
        self.compartments = self.raw.get("compartments", {})
        self.dynamics = self.raw.get("dynamics", {})
        self.topology_rules = self.raw.get("topology_rules", {})
        self.parameter_distributions = self.raw.get("parameter_distributions", {})
        self.validation_rules = self.raw.get("validation_rules", {})
        self.observables = self.raw.get("observables", {})
        self.stratification = self.raw.get("stratification", {})

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "Catalog":
        with path.open() as f:
            raw = json.load(f)
        return cls(raw=raw)


@dataclass
class TransitionSpec:
    """Specification for a transition between compartments."""
    source: str  # Source compartment
    target: str  # Target compartment
    dynamic_type: str  # Type of dynamics (e.g., "infection", "recovery")
    variant: str  # Specific variant of the dynamic type
    equation: str  # Mathematical equation
    parameters: List[str]  # Required parameters
    
    def __str__(self):
        return f"{self.source} → {self.target} ({self.dynamic_type}:{self.variant})"


@dataclass
class ModelSelection:
    """Complete model specification."""
    compartments: List[str]
    transitions: List[TransitionSpec]
    params: Dict[str, float]
    noise_level: float
    name: str = "model"
    observables: List[str] = field(default_factory=list)
    stratification_config: Optional[Dict] = None
    
    def __post_init__(self):
        if not self.observables:
            self.observables = ["incidence", "prevalence", "R_effective"]


@dataclass
class GroupConfig:
    """Configuration for a population group in stratified models."""
    name: str
    population_fraction: float
    contact_rate_multiplier: float = 1.0
    susceptibility_multiplier: float = 1.0
    infectivity_multiplier: float = 1.0
    severity_multiplier: float = 1.0


class ConstraintSolver:
    """Constraint-based compartment selection and transition construction."""
    
    def __init__(self, catalog: Catalog, rng: random.Random):
        self.catalog = catalog
        self.rng = rng
    
    def select_compartments(self, num_compartments: int = 5) -> List[str]:
        """Select compartments ensuring all constraints are satisfied."""
        # Always include required compartments
        selected = set()
        for name, spec in self.catalog.compartments.items():
            if spec.get("required", False):
                selected.add(name)
        
        # Get optional compartments
        optional = [name for name, spec in self.catalog.compartments.items() 
                   if not spec.get("required", False)]
        
        # Sample additional compartments
        remaining = max(0, num_compartments - len(selected))
        if remaining > 0 and optional:
            k = min(remaining, len(optional))
            selected.update(self.rng.sample(optional, k))
        
        # Check and add required dependencies
        selected = self._resolve_dependencies(selected)
        
        return sorted(selected)
    
    def _resolve_dependencies(self, compartments: Set[str]) -> Set[str]:
        """Resolve compartment dependencies (e.g., A requires E)."""
        resolved = set(compartments)
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for comp in list(resolved):
                spec = self.catalog.compartments.get(comp, {})
                
                # Check if this compartment requires other compartments
                required_comps = spec.get("requires_compartment", [])
                for req in required_comps:
                    if req not in resolved:
                        resolved.add(req)
                        changed = True
        
        return resolved
    
    def construct_transitions(self, compartments: List[str]) -> List[TransitionSpec]:
        """Automatically construct valid transitions based on compartments."""
        comp_set = set(compartments)
        transitions = []
        
        # 1. Mandatory transitions
        transitions.extend(self._add_mandatory_transitions(comp_set))
        
        # 2. Conditional transitions (required if compartment present)
        transitions.extend(self._add_conditional_transitions(comp_set))
        
        # 3. Optional transitions (probabilistic)
        transitions.extend(self._add_optional_transitions(comp_set))
        
        return transitions
    
    def _add_mandatory_transitions(self, comps: Set[str]) -> List[TransitionSpec]:
        """Add mandatory transitions."""
        transitions = []
        
        # Infection: S -> E or S -> I (always required)
        infection_variant = self._select_infection_variant(comps)
        target = "E" if "E" in comps else "I"
        transitions.append(TransitionSpec(
            source="S",
            target=target,
            dynamic_type="infection",
            variant=infection_variant["name"],
            equation=infection_variant["equation"],
            parameters=infection_variant["parameters"]
        ))
        
        # E -> I if E exists
        if "E" in comps:
            latent_variant = self._select_latent_progression_variant(comps)
            if isinstance(latent_variant.get("target"), dict):
                # Branching case
                targets = latent_variant["target"]
                target_str = ",".join(targets.keys())
            else:
                target_str = latent_variant.get("target", "I")
            
            transitions.append(TransitionSpec(
                source="E",
                target=target_str,
                dynamic_type="latent_progression",
                variant=latent_variant["name"],
                equation=latent_variant["equation"],
                parameters=latent_variant["parameters"]
            ))
        
        # I -> R (or back to S if no R) - I must have outflow
        if "R" in comps:
            recovery_variant = self._select_recovery_variant(comps)
            transitions.append(TransitionSpec(
                source="I",
                target="R",
                dynamic_type="recovery",
                variant=recovery_variant["name"],
                equation=recovery_variant["equation"],
                parameters=recovery_variant["parameters"]
            ))
        else:
            # No R compartment - I must still have outflow (back to S or to D if exists)
            if "D" in comps:
                transitions.append(TransitionSpec(
                    source="I",
                    target="D",
                    dynamic_type="death",
                    variant="from_infectious",
                    equation="mu * I",
                    parameters=["mu"]
                ))
            else:
                # Last resort: recovery back to S (unrealistic but ensures no dead-end)
                transitions.append(TransitionSpec(
                    source="I",
                    target="S",
                    dynamic_type="recovery",
                    variant="constant_rate",
                    equation="gamma * I",
                    parameters=["gamma"]
                ))
        
        return transitions
    
    def _add_conditional_transitions(self, comps: Set[str]) -> List[TransitionSpec]:
        """Add transitions required by compartment presence."""
        transitions = []
        
        # A -> R (or S) if A exists (must have outflow)
        if "A" in comps:
            target = "R" if "R" in comps else "S"
            transitions.append(TransitionSpec(
                source="A",
                target=target,
                dynamic_type="recovery",
                variant="constant_rate",
                equation="gamma_A * A",
                parameters=["gamma_A"]
            ))
        
        # I -> H if H exists (H must have outflow)
        if "H" in comps:
            hosp_variant = self._select_hospitalization_variant(comps)
            transitions.append(TransitionSpec(
                source="I",
                target="H",
                dynamic_type="hospitalization",
                variant=hosp_variant["name"],
                equation=hosp_variant["equation"],
                parameters=hosp_variant["parameters"]
            ))
            
            # H -> R or H -> D (must have outflow)
            if "D" in comps and self.rng.random() < 0.5:
                # Some go to death
                transitions.append(TransitionSpec(
                    source="H",
                    target="D",
                    dynamic_type="death",
                    variant="from_hospital",
                    equation="mu_H * H",
                    parameters=["mu_H"]
                ))
            
            # Always have recovery path from H
            target = "R" if "R" in comps else "S"
            transitions.append(TransitionSpec(
                source="H",
                target=target,
                dynamic_type="recovery",
                variant="constant_rate",
                equation="gamma_H * H",
                parameters=["gamma_H"]
            ))
        
        # Death transitions if D exists
        if "D" in comps:
            death_variant = self._select_death_variant(comps)
            source = death_variant.get("source", "I")
            if source in comps:
                transitions.append(TransitionSpec(
                    source=source,
                    target="D",
                    dynamic_type="death",
                    variant=death_variant["name"],
                    equation=death_variant["equation"],
                    parameters=death_variant["parameters"]
                ))
        
        # V transitions if V exists
        if "V" in comps:
            vax_variant = self._select_vaccination_variant(comps)
            transitions.append(TransitionSpec(
                source="S",
                target="V",
                dynamic_type="vaccination",
                variant=vax_variant["name"],
                equation=vax_variant["equation"],
                parameters=vax_variant["parameters"]
            ))
        
        # Q transitions if Q exists
        if "Q" in comps:
            q_variant = self._select_quarantine_variant(comps)
            source = q_variant.get("source", "I")
            if source in comps:
                transitions.append(TransitionSpec(
                    source=source,
                    target="Q",
                    dynamic_type="quarantine",
                    variant=q_variant["name"],
                    equation=q_variant["equation"],
                    parameters=q_variant["parameters"]
                ))
            
            # Q release
            q_release_variant = self._select_quarantine_release_variant(comps)
            target = q_release_variant.get("target", "R" if "R" in comps else "S")
            transitions.append(TransitionSpec(
                source="Q",
                target=target,
                dynamic_type="quarantine_release",
                variant=q_release_variant["name"],
                equation=q_release_variant["equation"],
                parameters=q_release_variant["parameters"]
            ))
        
        # P transitions if P exists (protected, needs outflow)
        if "P" in comps:
            # P -> S (protection wears off)
            transitions.append(TransitionSpec(
                source="P",
                target="S",
                dynamic_type="waning_immunity",  # Reuse waning dynamics
                variant="constant",
                equation="delta_P * P",
                parameters=["delta_P"]
            ))
        
        # W transitions if W exists
        if "W" in comps:
            shed_variant = self._select_shedding_variant(comps)
            # Determine source based on what's available
            if "I" in shed_variant.get("equation", ""):
                source = "I"
            else:
                source = "I"  # Default
            
            transitions.append(TransitionSpec(
                source=source,
                target="W",
                dynamic_type="environmental_shedding",
                variant=shed_variant["name"],
                equation=shed_variant["equation"],
                parameters=shed_variant["parameters"]
            ))
            
            # W decay
            transitions.append(TransitionSpec(
                source="W",
                target="W",
                dynamic_type="environmental_decay",
                variant="decay",
                equation="zeta * W",
                parameters=["zeta"]
            ))
        
        # C transitions if C exists (always needs outflow)
        if "C" in comps:
            transitions.append(TransitionSpec(
                source="I",
                target="C",
                dynamic_type="chronic_progression",
                variant="constant",
                equation="chi * I",
                parameters=["chi"]
            ))
            
            # C must have outflow - prioritize R, fallback to S
            target = "R" if "R" in comps else "S"
            transitions.append(TransitionSpec(
                source="C",
                target=target,
                dynamic_type="chronic_clearance",
                variant="constant",
                equation="gamma_C * C",
                parameters=["gamma_C"]
            ))
        
        return transitions
    
    def _add_optional_transitions(self, comps: Set[str]) -> List[TransitionSpec]:
        """Add optional transitions based on probability."""
        transitions = []
        
        # Waning immunity: R -> S
        if "R" in comps and self.rng.random() < 0.3:
            transitions.append(TransitionSpec(
                source="R",
                target="S",
                dynamic_type="waning_immunity",
                variant="constant",
                equation="omega * R",
                parameters=["omega"]
            ))
        
        # Vaccine waning: V -> S (mandatory if V exists, otherwise dead-end)
        if "V" in comps:
            transitions.append(TransitionSpec(
                source="V",
                target="S",
                dynamic_type="vaccine_waning",
                variant="constant",
                equation="omega_V * V",
                parameters=["omega_V"]
            ))
        
        return transitions
    
    def _select_infection_variant(self, comps: Set[str]) -> Dict:
        """Select appropriate infection dynamics variant."""
        variants = self.catalog.dynamics.get("infection", {}).get("variants", {})
        valid_variants = []
        
        for name, spec in variants.items():
            required = spec.get("requires", [])
            if all(req in comps for req in required):
                weight = spec.get("weight", 1.0)
                valid_variants.append((name, spec, weight))
        
        if not valid_variants:
            # Fallback to mass_action
            spec = variants.get("mass_action", {})
            return {"name": "mass_action", **spec}
        
        # Weighted random selection
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        
        return {"name": selected_name, **selected_spec}
    
    def _select_latent_progression_variant(self, comps: Set[str]) -> Dict:
        """Select latent progression dynamics."""
        variants = self.catalog.dynamics.get("latent_progression", {}).get("variants", {})
        
        # If A exists, prefer branching
        if "A" in comps and self.rng.random() < 0.6:
            spec = variants.get("branching", {})
            return {"name": "branching", **spec}
        else:
            spec = variants.get("simple", {})
            return {"name": "simple", **spec}
    
    def _select_recovery_variant(self, comps: Set[str]) -> Dict:
        """Select recovery dynamics variant."""
        variants = self.catalog.dynamics.get("recovery", {}).get("variants", {})
        
        # Weighted random selection
        valid_variants = [(name, spec, spec.get("weight", 1.0)) 
                         for name, spec in variants.items()]
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        
        return {"name": selected_name, **selected_spec}
    
    def _select_hospitalization_variant(self, comps: Set[str]) -> Dict:
        """Select hospitalization dynamics variant."""
        variants = self.catalog.dynamics.get("hospitalization", {}).get("variants", {})
        valid_variants = [(name, spec, spec.get("weight", 1.0)) 
                         for name, spec in variants.items()]
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        return {"name": selected_name, **selected_spec}
    
    def _select_death_variant(self, comps: Set[str]) -> Dict:
        """Select death dynamics variant."""
        variants = self.catalog.dynamics.get("death", {}).get("variants", {})
        valid_variants = []
        
        for name, spec in variants.items():
            required = spec.get("requires", [])
            if all(req in comps for req in required):
                weight = spec.get("weight", 1.0)
                valid_variants.append((name, spec, weight))
        
        if not valid_variants:
            # Fallback to from_infectious
            spec = variants.get("from_infectious", {})
            return {"name": "from_infectious", **spec}
        
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        return {"name": selected_name, **selected_spec}
    
    def _select_vaccination_variant(self, comps: Set[str]) -> Dict:
        """Select vaccination dynamics variant."""
        variants = self.catalog.dynamics.get("vaccination", {}).get("variants", {})
        valid_variants = [(name, spec, spec.get("weight", 1.0)) 
                         for name, spec in variants.items()]
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        return {"name": selected_name, **selected_spec}
    
    def _select_quarantine_variant(self, comps: Set[str]) -> Dict:
        """Select quarantine dynamics variant."""
        variants = self.catalog.dynamics.get("quarantine", {}).get("variants", {})
        valid_variants = []
        
        for name, spec in variants.items():
            required = spec.get("requires", [])
            source = spec.get("source", "I")
            if source in comps and all(req in comps for req in required):
                weight = spec.get("weight", 1.0)
                valid_variants.append((name, spec, weight))
        
        if not valid_variants:
            spec = variants.get("from_infectious", {})
            return {"name": "from_infectious", **spec}
        
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        return {"name": selected_name, **selected_spec}
    
    def _select_quarantine_release_variant(self, comps: Set[str]) -> Dict:
        """Select quarantine release dynamics variant."""
        variants = self.catalog.dynamics.get("quarantine_release", {}).get("variants", {})
        valid_variants = []
        
        for name, spec in variants.items():
            required = spec.get("requires", [])
            if all(req in comps for req in required):
                weight = spec.get("weight", 1.0)
                valid_variants.append((name, spec, weight))
        
        if not valid_variants:
            spec = variants.get("to_susceptible", {})
            return {"name": "to_susceptible", **spec}
        
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        return {"name": selected_name, **selected_spec}
    
    def _select_shedding_variant(self, comps: Set[str]) -> Dict:
        """Select environmental shedding dynamics variant."""
        variants = self.catalog.dynamics.get("environmental_shedding", {}).get("variants", {})
        valid_variants = []
        
        for name, spec in variants.items():
            required = spec.get("requires", [])
            if all(req in comps for req in required):
                weight = spec.get("weight", 1.0)
                valid_variants.append((name, spec, weight))
        
        if not valid_variants:
            spec = variants.get("from_infectious", {})
            return {"name": "from_infectious", **spec}
        
        names, specs, weights = zip(*valid_variants)
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        selected_spec = next(s for n, s, w in valid_variants if n == selected_name)
        return {"name": selected_name, **selected_spec}


class HierarchicalParameterSampler:
    """Hierarchical parameter sampling with epidemiological constraints."""
    
    def __init__(self, catalog: Catalog, rng: random.Random):
        self.catalog = catalog
        self.rng = rng
        self.np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    
    def sample_parameters(
        self,
        compartments: List[str],
        transitions: List[TransitionSpec],
        overrides: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Sample parameters hierarchically: primary -> derived -> secondary."""
        overrides = overrides or {}
        params = {}
        
        # 1. Sample primary parameters
        primary_params = self._sample_primary_parameters(compartments)
        params.update(primary_params)
        
        # 2. Derive parameters from primary
        derived_params = self._derive_parameters(primary_params, compartments)
        params.update(derived_params)
        
        # 3. Sample secondary parameters
        secondary_params = self._sample_secondary_parameters(transitions)
        params.update(secondary_params)
        
        # 4. Sample population parameters
        pop_params = self._sample_population_parameters()
        params.update(pop_params)
        
        # 5. Simulation parameters
        sim_params = self._get_simulation_parameters()
        params.update(sim_params)
        
        # 6. Apply overrides
        params.update(overrides)
        
        return params
    
    def _sample_primary_parameters(self, compartments: List[str]) -> Dict[str, float]:
        """Sample primary epidemiological parameters."""
        params = {}
        primary = self.catalog.parameter_distributions.get("primary_parameters", {})
        
        # R0 (always sample)
        r0_spec = primary.get("R0", {})
        params["R0"] = self._sample_from_distribution(r0_spec)
        
        # Infectious period
        inf_period_spec = primary.get("infectious_period_days", {})
        params["infectious_period_days"] = self._sample_from_distribution(inf_period_spec)
        
        # Latent period (if E exists)
        if "E" in compartments:
            lat_period_spec = primary.get("latent_period_days", {})
            params["latent_period_days"] = self._sample_from_distribution(lat_period_spec)
        
        return params
    
    def _derive_parameters(self, primary: Dict[str, float], compartments: List[str]) -> Dict[str, float]:
        """Derive parameters from primary parameters."""
        params = {}
        derived = self.catalog.parameter_distributions.get("derived_parameters", {})
        
        # gamma = 1 / infectious_period
        if "infectious_period_days" in primary:
            params["gamma"] = 1.0 / max(primary["infectious_period_days"], 1.0)
        
        # sigma = 1 / latent_period
        if "latent_period_days" in primary and "E" in compartments:
            params["sigma"] = 1.0 / max(primary["latent_period_days"], 1.0)
        
        # beta = R0 * gamma
        if "R0" in primary and "gamma" in params:
            params["beta"] = primary["R0"] * params["gamma"]
        
        # === SEASONAL FORCING PARAMETERS ===
        # Can be controlled via overrides, otherwise randomly enable (50% of models)
        # Note: overrides are applied AFTER this function in sample_parameters()
        # So we always generate seasonal params, but they can be overridden
        use_seasonal = self.rng.random() < 0.5
        
        if use_seasonal:
            params["seasonal_forcing"] = True
            # Forcing amplitude: β(t) = β₀ × (1 + ε × cos(2πt/T + φ))
            # ε ∈ [0.1, 0.5] - typical for respiratory diseases
            params["seasonal_amplitude"] = self.rng.uniform(0.1, 0.5)
            # Period: 52 weeks (annual), 26 weeks (biannual), or multi-year
            period_choices = [26, 52, 104, 156]  # weeks
            period_weights = [0.15, 0.5, 0.25, 0.1]  # annual most common
            params["seasonal_period"] = self.rng.choices(period_choices, period_weights)[0]
            # Random phase offset φ ∈ [0, 2π]
            params["seasonal_phase"] = self.rng.uniform(0, 2 * math.pi)
        else:
            params["seasonal_forcing"] = False
            params["seasonal_amplitude"] = 0.0
            params["seasonal_period"] = 52
            params["seasonal_phase"] = 0.0
        
        return params
    
    def _sample_secondary_parameters(self, transitions: List[TransitionSpec]) -> Dict[str, float]:
        """Sample secondary parameters needed by transitions."""
        params = {}
        secondary = self.catalog.parameter_distributions.get("secondary_parameters", {})
        
        # Collect all required parameters
        required_params = set()
        for trans in transitions:
            required_params.update(trans.parameters)
        
        # Sample each required parameter
        for param_name in required_params:
            if param_name in secondary:
                spec = secondary[param_name]
                params[param_name] = self._sample_from_range(spec)
        
        return params
    
    def _sample_population_parameters(self) -> Dict[str, float]:
        """Sample population-related parameters."""
        params = {}
        pop_params = self.catalog.parameter_distributions.get("population_parameters", {})
        
        for param_name, spec in pop_params.items():
            params[param_name] = self._sample_from_range(spec)
        
        return params
    
    def _get_simulation_parameters(self) -> Dict[str, float]:
        """Get simulation configuration parameters."""
        params = {}
        sim_params = self.catalog.parameter_distributions.get("simulation_parameters", {})
        
        for param_name, spec in sim_params.items():
            params[param_name] = spec.get("default")
        
        return params
    
    def _sample_from_distribution(self, spec: Dict) -> float:
        """Sample from specified distribution."""
        dist_type = spec.get("distribution", "uniform")
        range_vals = spec.get("range", [0, 1])
        
        if dist_type == "log_normal":
            mean = spec.get("mean", 1.0)
            std = spec.get("std", 0.5)
            value = self.np_rng.lognormal(np.log(mean), std)
        elif dist_type == "gamma":
            shape = spec.get("shape", 2.0)
            scale = spec.get("scale", 1.0)
            value = self.np_rng.gamma(shape, scale)
        else:
            # Uniform
            value = self.rng.uniform(range_vals[0], range_vals[1])
        
        # Clip to range
        value = np.clip(value, range_vals[0], range_vals[1])
        return float(value)
    
    def _sample_from_range(self, spec: Dict) -> float:
        """Sample uniformly from a range."""
        range_vals = spec.get("range", [0, 1])
        default = spec.get("default")
        
        if default is not None and not isinstance(default, str):
            # Add noise around default
            noise_factor = 0.3
            value = default * (1 + self.rng.uniform(-noise_factor, noise_factor))
            value = np.clip(value, range_vals[0], range_vals[1])
        else:
            value = self.rng.uniform(range_vals[0], range_vals[1])
        
        return float(value)


class ModelValidator:
    """Validate model structure and parameters."""
    
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
    
    def validate(self, model: ModelSelection) -> None:
        """Run all validation checks."""
        self._check_population_conservation(model)
        self._check_no_dead_ends(model)
        self._check_infectious_pathway(model)
        self._check_R0_positive(model)
        self._check_numerical_stability(model)
    
    def _check_population_conservation(self, model: ModelSelection) -> None:
        """Check that living compartments conserve population."""
        rules = self.catalog.validation_rules.get("population_conservation", {})
        living = rules.get("living_compartments", [])
        death = rules.get("death_compartment")
        reservoir = rules.get("reservoir_compartments", [])
        
        # Check all living compartments are accounted for
        model_living = [c for c in model.compartments 
                       if c in living or (c != death and c not in reservoir)]
        
        if not model_living:
            raise ValidationError("No living compartments in model")
    
    def _check_no_dead_ends(self, model: ModelSelection) -> None:
        """Check no dead-ends except terminal compartments."""
        rules = self.catalog.validation_rules.get("no_dead_ends", {})
        terminal = set(rules.get("terminal_compartments", []))
        sink = set(rules.get("sink_compartments", []))
        
        # Build outflow map
        has_outflow = set()
        for trans in model.transitions:
            has_outflow.add(trans.source)
        
        # Check all non-terminal compartments have outflow
        for comp in model.compartments:
            if comp not in terminal and comp not in sink:
                if comp not in has_outflow:
                    # Check if it's okay (like W can decay to itself)
                    if comp != "W":
                        raise ValidationError(f"Compartment {comp} has no outflow (dead-end)")
    
    def _check_infectious_pathway(self, model: ModelSelection) -> None:
        """Check there's a path from S to infectious compartments."""
        infectious = {"I", "A", "C"}
        
        # Build adjacency
        adj = {}
        for trans in model.transitions:
            if trans.source not in adj:
                adj[trans.source] = []
            targets = trans.target.split(",")
            adj[trans.source].extend(targets)
        
        # BFS from S
        if "S" not in model.compartments:
            raise ValidationError("No S compartment")
        
        visited = set()
        queue = ["S"]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            if node in adj:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        # Check at least one infectious compartment reachable
        if not any(inf in visited for inf in infectious if inf in model.compartments):
            raise ValidationError("No path from S to infectious compartments")
    
    def _check_R0_positive(self, model: ModelSelection) -> None:
        """Check R0 is positive."""
        if model.params.get("R0", 0) <= 0:
            raise ValidationError(f"R0 must be positive, got {model.params.get('R0')}")
    
    def _check_numerical_stability(self, model: ModelSelection) -> None:
        """Check parameters are within stable ranges."""
        rules = self.catalog.validation_rules.get("numerical_stability", {})
        max_rate = rules.get("max_rate", 10.0)
        
        # Check rate parameters
        rate_params = ["gamma", "sigma", "beta", "eta", "mu", "mu_H", 
                      "gamma_H", "gamma_A", "gamma_C", "chi"]
        
        for param in rate_params:
            if param in model.params:
                if model.params[param] > max_rate:
                    raise ValidationError(
                        f"Parameter {param}={model.params[param]} exceeds max_rate={max_rate}"
                    )


class NoiseInjector:
    """Inject observation noise into simulation data."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
        seed = self.rng.randint(0, 2**32 - 1)
        self.np_rng = np.random.default_rng(seed)
    
    def sample_noise_level(self) -> float:
        """Sample noise level from reasonable range."""
        return self.rng.uniform(0.0, 0.05)
    
    def inject_noise(
        self,
        data: np.ndarray,
        noise_level: float,
        noise_type: str = "gaussian"
    ) -> np.ndarray:
        """Inject noise into simulation data."""
        if noise_level <= 0:
            return data
        
        noisy_data = data.copy()
        
        if noise_type == "gaussian":
            for col in range(data.shape[1]):
                signal_std = np.std(data[:, col])
                if signal_std > 0:
                    noise = self.np_rng.normal(0, noise_level * signal_std, size=data.shape[0])
                    noisy_data[:, col] += noise
        
        return np.maximum(noisy_data, 0)


# ============================================================================
# SIMULATION ENGINE - Self-contained (no imports from old pipeline)
# ============================================================================

# Cache for mixing matrices to avoid regeneration
_MIXING_MATRIX_CACHE = {}


class TransitionRegistry:
    """Registry of all transition handlers for ODE simulation."""
    
    def __init__(self):
        self.handlers = {
            # Infection
            "standard_mixing": self.standard_mixing,
            "environmental_cholera": self.environmental_cholera,
            "fomite_surface": self.fomite_surface,
            "vector_borne_proxy": self.vector_borne_proxy,
            "hospital_leakage": self.hospital_leakage,
            "power_law_mixing": self.power_law_mixing,
            # Progression
            "simple_linear": self.simple_linear,
            "branching_split": self.branching_split,
            "severity_trigger": self.severity_trigger,
            "overflow_trigger": self.overflow_trigger,
            "chronic_failure": self.chronic_failure,
            "hospital_death": self.hospital_death,
            "disease_induced_death": self.disease_induced_death,
            # Recovery
            "constant_recovery": self.constant_recovery,
            "resource_constraints": self.resource_constraints,
            "waning_immunity": self.waning_immunity,
            "carrier_clearance": self.carrier_clearance,
            # Intervention
            "campaign_mode": self.campaign_mode,
            "proportional_vax": self.proportional_vax,
            "reactive_quarantine": self.reactive_quarantine,
            "tracing_efficiency": self.tracing_efficiency,
            "prophylaxis_decay": self.prophylaxis_decay,
            # Reservoir
            "shedding": self.shedding,
            "asymptomatic_shedding": self.asymptomatic_shedding,
            "environmental_decay": self.environmental_decay,
            # Demography
            "birth_import": self.birth_import,
        }
    
    def derivative(
        self, 
        name: str, 
        state: Dict[str, float], 
        params: Dict[str, float], 
        delta: Dict[str, float], 
        total_pop: float
    ) -> None:
        """Apply transition rule to update derivatives."""
        handler = self.handlers.get(name)
        if handler is None:
            raise CatalogError(f"No handler for transition '{name}'")
        handler(state, params, delta, total_pop)
    
    def _infect(self, state, delta, flow):
        """Route infection flow to E or I."""
        delta["S"] -= flow
        if "E" in state:
            delta["E"] += flow
        else:
            delta["I"] += flow
    
    def _cap_flow(self, flow: float, available: float) -> float:
        """Ensure flows never exceed what is available in a compartment."""
        if available <= 0:
            return 0.0
        return min(max(flow, 0.0), available)
    
    # === Infection Rules ===
    def standard_mixing(self, state, params, delta, total_pop):
        beta = params.get("beta", 0.3)
        epsilon = params.get("epsilon", 0.5)
        delta_c = params.get("delta_C", 0.5)
        infectious = state.get("I", 0.0)
        if "A" in state:
            infectious += epsilon * state.get("A", 0.0)
        if "C" in state:
            infectious += delta_c * state.get("C", 0.0)
        flow = beta * state.get("S", 0.0) * infectious / max(total_pop, 1e-8)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        self._infect(state, delta, flow)
    
    def environmental_cholera(self, state, params, delta, total_pop):
        beta_w = params.get("beta_W", 0.1)
        K = params.get("K", 1000.0)
        W = state.get("W", 0.0)
        flow = beta_w * state.get("S", 0.0) * W / (K + W + 1e-8)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        self._infect(state, delta, flow)
    
    def fomite_surface(self, state, params, delta, total_pop):
        beta = params.get("beta", 0.3)
        rho = params.get("rho", 0.5)
        infectious = state.get("I", 0.0) + rho * state.get("W", 0.0)
        flow = beta * state.get("S", 0.0) * infectious / max(total_pop, 1e-8)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        self._infect(state, delta, flow)
    
    def vector_borne_proxy(self, state, params, delta, total_pop):
        beta = params.get("beta", 0.3)
        alpha = params.get("alpha", 0.001)
        I = state.get("I", 0.0)
        flow = beta * state.get("S", 0.0) * (I / max(total_pop, 1e-8)) * math.exp(-alpha * I)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        self._infect(state, delta, flow)
    
    def hospital_leakage(self, state, params, delta, total_pop):
        beta = params.get("beta", 0.3)
        theta = params.get("theta", 0.5)
        infectious = state.get("I", 0.0) + theta * state.get("H", 0.0)
        flow = beta * state.get("S", 0.0) * infectious / max(total_pop, 1e-8)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        self._infect(state, delta, flow)
    
    def power_law_mixing(self, state, params, delta, total_pop):
        beta = params.get("beta", 0.01)
        p = params.get("p_power", 1.0)
        q = params.get("q_power", 1.0)
        S = max(0.0, state.get("S", 0.0))
        I = max(0.0, state.get("I", 0.0))
        flow = beta * (S ** p) * (I ** q)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        self._infect(state, delta, flow)
    
    # === Progression Rules ===
    def simple_linear(self, state, params, delta, total_pop):
        sigma = params.get("sigma", 0.2)
        flow = sigma * state.get("E", 0.0)
        flow = self._cap_flow(flow, state.get("E", 0.0))
        delta["E"] -= flow
        delta["I"] += flow
    
    def branching_split(self, state, params, delta, total_pop):
        sigma = params.get("sigma", 0.2)
        p = params.get("p_branch", 0.7)
        flow = sigma * state.get("E", 0.0)
        flow = self._cap_flow(flow, state.get("E", 0.0))
        delta["E"] -= flow
        delta["I"] += p * flow
        if "A" in state:
            delta["A"] += (1 - p) * flow
    
    def severity_trigger(self, state, params, delta, total_pop):
        eta = params.get("eta", 0.1)
        flow = eta * state.get("I", 0.0)
        flow = self._cap_flow(flow, state.get("I", 0.0))
        delta["I"] -= flow
        delta["H"] += flow
    
    def overflow_trigger(self, state, params, delta, total_pop):
        eta = params.get("eta", 0.1)
        threshold = params.get("I_threshold", 500.0)
        I = state.get("I", 0.0)
        sigmoid = 1 / (1 + math.exp(-(I - threshold) / (threshold * 0.1 + 1e-8)))
        flow = eta * I * sigmoid
        flow = self._cap_flow(flow, state.get("I", 0.0))
        delta["I"] -= flow
        delta["H"] += flow
    
    def chronic_failure(self, state, params, delta, total_pop):
        chi = params.get("chi", 0.01)
        flow = chi * state.get("I", 0.0)
        flow = self._cap_flow(flow, state.get("I", 0.0))
        delta["I"] -= flow
        delta["C"] += flow
    
    def hospital_death(self, state, params, delta, total_pop):
        mu_h = params.get("mu_H", 0.05)
        flow = mu_h * state.get("H", 0.0)
        flow = self._cap_flow(flow, state.get("H", 0.0))
        delta["H"] -= flow
        delta["D"] += flow
    
    def disease_induced_death(self, state, params, delta, total_pop):
        mu = params.get("mu", 0.01)
        flow = mu * state.get("I", 0.0)
        flow = self._cap_flow(flow, state.get("I", 0.0))
        delta["I"] -= flow
        delta["D"] += flow
    
    # === Recovery Rules ===
    def constant_recovery(self, state, params, delta, total_pop):
        gamma = params.get("gamma", 0.14)
        flow = gamma * state.get("I", 0.0)
        flow = self._cap_flow(flow, state.get("I", 0.0))
        delta["I"] -= flow
        # Target R if exists, otherwise S
        target = "R" if "R" in delta else "S"
        delta[target] += flow
        if "A" in state:
            gamma_a = params.get("gamma_A", gamma)
            flow_a = gamma_a * state.get("A", 0.0)
            flow_a = self._cap_flow(flow_a, state.get("A", 0.0))
            delta["A"] -= flow_a
            delta[target] += flow_a
    
    def resource_constraints(self, state, params, delta, total_pop):
        gamma_h = params.get("gamma_H", 0.1)
        omega_h = params.get("omega_H", 0.001)
        H = state.get("H", 0.0)
        flow = gamma_h * H / (1 + omega_h * H + 1e-8)
        flow = self._cap_flow(flow, state.get("H", 0.0))
        delta["H"] -= flow
        # Target R if exists, otherwise S
        target = "R" if "R" in delta else "S"
        delta[target] += flow
    
    def waning_immunity(self, state, params, delta, total_pop):
        omega = params.get("omega", 0.01)
        # Handle both R->S (standard) and V->S (vaccine waning)
        if "R" in state and "R" in delta:
            flow = omega * state.get("R", 0.0)
            flow = self._cap_flow(flow, state.get("R", 0.0))
            delta["R"] -= flow
            delta["S"] += flow
        if "V" in state and "V" in delta:
            omega_v = params.get("omega_V", omega)
            flow_v = omega_v * state.get("V", 0.0)
            flow_v = self._cap_flow(flow_v, state.get("V", 0.0))
            delta["V"] -= flow_v
            delta["S"] += flow_v
        # Handle P->S (protection decay)
        if "P" in state and "P" in delta:
            delta_p = params.get("delta_P", omega)
            flow_p = delta_p * state.get("P", 0.0)
            flow_p = self._cap_flow(flow_p, state.get("P", 0.0))
            delta["P"] -= flow_p
            delta["S"] += flow_p
    
    def carrier_clearance(self, state, params, delta, total_pop):
        gamma_c = params.get("gamma_C", 0.001)
        flow = gamma_c * state.get("C", 0.0)
        flow = self._cap_flow(flow, state.get("C", 0.0))
        delta["C"] -= flow
        # Target R if exists, otherwise S
        target = "R" if "R" in delta else "S"
        delta[target] += flow
    
    # === Intervention Rules ===
    def campaign_mode(self, state, params, delta, total_pop):
        nu = params.get("nu_campaign", 10.0)
        flow = min(nu, state.get("S", 0.0))
        delta["S"] -= flow
        delta["V"] += flow
    
    def proportional_vax(self, state, params, delta, total_pop):
        p = params.get("p_vax", 0.001)
        flow = p * state.get("S", 0.0)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        delta["S"] -= flow
        delta["V"] += flow
    
    def reactive_quarantine(self, state, params, delta, total_pop):
        tau = params.get("tau_Q", 0.01)
        I = state.get("I", 0.0)
        flow = min(tau * I, I * 0.99)  # Linear quarantine rate
        flow = self._cap_flow(flow, state.get("I", 0.0))
        delta["I"] -= flow
        delta["Q"] += flow
        
        # Quarantine release to R (people who finish isolation)
        rho_q = params.get("rho_Q", 1.0/14.0)  # Default 14-day quarantine
        Q = state.get("Q", 0.0)
        release_flow = rho_q * Q
        release_flow = self._cap_flow(release_flow, state.get("Q", 0.0))
        delta["Q"] -= release_flow
        if "R" in delta:
            delta["R"] += release_flow
        else:
            delta["S"] += release_flow  # Back to susceptible if no R compartment
    
    def tracing_efficiency(self, state, params, delta, total_pop):
        k = params.get("k_trace", 0.5)
        tau_trace = params.get("tau_trace", 0.1)  # Base tracing rate
        E = state.get("E", 0.0)
        I = state.get("I", 0.0)
        # Trace contacts proportional to detected cases
        flow = min(tau_trace * k * (E + I), E * 0.5)  # Max 50% of E per timestep
        flow = self._cap_flow(flow, state.get("E", 0.0))
        delta["E"] -= flow
        delta["Q"] += flow
    
    def prophylaxis_decay(self, state, params, delta, total_pop):
        delta_p = params.get("delta_P", 0.05)
        flow = delta_p * state.get("P", 0.0)
        flow = self._cap_flow(flow, state.get("P", 0.0))
        delta["P"] -= flow
        delta["S"] += flow
    
    # === Reservoir Rules ===
    def shedding(self, state, params, delta, total_pop):
        xi = params.get("xi", 0.01)
        delta["W"] += xi * state.get("I", 0.0)
    
    def asymptomatic_shedding(self, state, params, delta, total_pop):
        xi = params.get("xi", 0.01)
        delta_a = params.get("delta_A", 0.5)
        delta["W"] += xi * (state.get("I", 0.0) + delta_a * state.get("A", 0.0))
    
    def environmental_decay(self, state, params, delta, total_pop):
        zeta = params.get("zeta", 0.1)
        delta["W"] -= zeta * state.get("W", 0.0)
    
    # === Demography Rules ===
    def birth_import(self, state, params, delta, total_pop):
        Lambda = params.get("Lambda", params.get("population", total_pop) * 1e-4)
        mu_nat = params.get("mu_nat", 1 / (70 * 365))
        delta["S"] += Lambda - mu_nat * state.get("S", 0.0)
        for comp in ("I", "R", "E", "A", "H", "V", "Q", "C", "P"):
            if comp in state:
                delta[comp] -= mu_nat * state.get(comp, 0.0)


class RK4Integrator:
    """Runge-Kutta 4th order ODE integrator."""
    
    def __init__(self, system):
        self.system = system
    
    def step(self, state_vec: np.ndarray, t: float, dt: float) -> np.ndarray:
        f = self.system
        k1 = f(t, state_vec)
        k2 = f(t + dt / 2, state_vec + dt * k1 / 2)
        k3 = f(t + dt / 2, state_vec + dt * k2 / 2)
        k4 = f(t + dt, state_vec + dt * k3)
        new_state = state_vec + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Clip to reasonable bounds (0 to 10x initial population as safety)
        new_state = np.clip(new_state, 0.0, 1e8)
        
        # Check for numerical issues
        if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
            raise RuntimeError("NaN or Inf detected in ODE step")
        
        return new_state
    
    def run(
        self, 
        initial: np.ndarray, 
        t_span: Tuple[float, float], 
        dt: float,
        post_step: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(math.ceil((t_span[1] - t_span[0]) / dt)) + 1
        times = np.linspace(t_span[0], t_span[0] + dt * (steps - 1), steps)
        states = np.zeros((steps, initial.size), dtype=np.float32)
        states[0] = initial
        current = initial.copy()
        
        for idx in range(1, steps):
            current = self.step(current, times[idx - 1], dt)
            if post_step:
                current = post_step(current)
            states[idx] = np.maximum(current, 0.0)
        
        return times, states


def generate_mixing_matrix(
    num_groups: int,
    mixing_type: str = "homogeneous",
    assortativity: float = 0.7,
    rng: Optional[random.Random] = None
) -> np.ndarray:
    """Generate a mixing matrix for group-stratified models.
    
    Args:
        num_groups: Number of population groups
        mixing_type: Type of mixing pattern
            - "homogeneous": Equal mixing between all groups
            - "assortative": Preferential within-group mixing
            - "hierarchical": Age-like hierarchical structure
            - "spatial": Spatial proximity-based mixing
        assortativity: Degree of within-group preference (0-1)
        rng: Random number generator
    
    Returns:
        Mixing matrix of shape (num_groups, num_groups)
    """
    if rng is None:
        rng = random.Random()
    
    if mixing_type == "homogeneous":
        # Equal mixing between all groups
        matrix = np.ones((num_groups, num_groups)) / num_groups
    
    elif mixing_type == "assortative":
        # Preferential within-group mixing
        matrix = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                if i == j:
                    matrix[i, j] = assortativity
                else:
                    matrix[i, j] = (1 - assortativity) / (num_groups - 1)
    
    elif mixing_type == "hierarchical":
        # Age-like structure with stronger nearby mixing
        matrix = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                distance = abs(i - j)
                if distance == 0:
                    matrix[i, j] = assortativity
                else:
                    # Decaying contact with distance
                    matrix[i, j] = (1 - assortativity) * np.exp(-distance / 2)
        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    
    elif mixing_type == "spatial":
        # Random spatial arrangement with distance-based mixing
        positions = np.random.rand(num_groups, 2)
        matrix = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                if i == j:
                    matrix[i, j] = assortativity
                else:
                    # Distance-based mixing
                    dist = np.linalg.norm(positions[i] - positions[j])
                    matrix[i, j] = (1 - assortativity) * np.exp(-dist * 2)
        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    
    else:
        raise ValueError(f"Unknown mixing_type: {mixing_type}")
    
    return matrix


def get_cached_mixing_matrix(
    num_groups: int,
    mixing_type: str,
    assortativity: float,
    seed: int
) -> np.ndarray:
    """Get or generate a cached mixing matrix."""
    key = (num_groups, mixing_type, round(assortativity, 2), seed)
    if key not in _MIXING_MATRIX_CACHE:
        rng = random.Random(seed)
        _MIXING_MATRIX_CACHE[key] = generate_mixing_matrix(
            num_groups, mixing_type, assortativity, rng
        )
    return _MIXING_MATRIX_CACHE[key]


@dataclass
class TransitionStructure:
    """Represents a transition between compartments (for simulator compatibility)."""
    source: str  # Source compartment(s), e.g., "E" or "I,A"
    target: str  # Target compartment(s), e.g., "I" or "R"
    category: str  # Category: infection, progression, recovery, intervention, reservoir
    rule_name: str  # Selected mathematical rule
    
    def __str__(self):
        return f"{self.source} → {self.target} ({self.rule_name})"


class Simulator:
    """Stage 6: Conduct simulations."""
    
    def __init__(self):
        self.transitions = TransitionRegistry()
    
    @staticmethod
    def _cap_flow(flow: float, available: float) -> float:
        """Cap a flow so it never exceeds what is available."""
        if available <= 0:
            return 0.0
        return min(max(flow, 0.0), available)
    
    def simulate(
        self, 
        model: ModelSelection,
        inject_noise: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run simulation and return weekly and daily data."""
        # Build ODE system
        system = self._build_system(
            model.compartments, 
            model.transitions, 
            model.params
        )
        
        # Initial conditions
        y0 = self._initial_conditions(model.compartments, model.params)
        population = model.params.get("population", 1e5)
        balancer = self._make_population_balancer(model.compartments, population)
        
        # Integrate
        integrator = RK4Integrator(system)
        max_steps = model.params.get("max_time_steps", 250)
        dt = model.params.get("dt", 1.0)
        t0, t1 = 0.0, max_steps * dt
        times, states = integrator.run(y0, (t0, t1), dt, post_step=balancer)
        
        # Check for numerical instability
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            raise RuntimeError("Numerical instability detected in simulation")
        if np.max(states) > population * 100:  # Sanity check
            raise RuntimeError(f"Unrealistic values detected: max={np.max(states):.0f}")
        
        # Inject noise if requested
        if inject_noise and model.noise_level > 0:
            noise_injector = NoiseInjector(random.Random())
            states = noise_injector.inject_noise(states, model.noise_level)
            states = np.vstack([balancer(row) for row in states])
        
        # Compute R(t)
        R_t = self._compute_R_t(model, states)
        
        # Create dataframes based on time unit
        time_unit = model.params.get("time_unit", "weeks")
        
        if time_unit == "weeks":
            # Already in weekly steps
            df_weekly = pd.DataFrame(states, columns=model.compartments)
            df_weekly.insert(0, "week", times)
            df_weekly["R_t"] = R_t
            
            # Create daily by interpolation (7 points per week)
            daily_times = np.arange(0, times[-1] * 7, 1.0)
            daily_states = np.zeros((len(daily_times), states.shape[1]))
            for i in range(states.shape[1]):
                daily_states[:, i] = np.interp(daily_times, times * 7, states[:, i])
            df_daily = pd.DataFrame(daily_states, columns=model.compartments)
            df_daily.insert(0, "day", daily_times)
            df_daily["R_t"] = np.interp(daily_times, times * 7, R_t)
        else:
            # Already in daily steps
            df_daily = pd.DataFrame(states, columns=model.compartments)
            df_daily.insert(0, "day", times)
            df_daily["R_t"] = R_t
            
            # Downsample to weekly
            week_indices = np.arange(0, len(times), 7)
            df_weekly = df_daily.iloc[week_indices].copy()
            df_weekly["week"] = df_weekly["day"] / 7.0
            df_weekly = df_weekly.drop(columns=["day"])
        
        # Reorder columns
        cols_weekly = ["week"] + model.compartments + ["R_t"]
        df_weekly = df_weekly[cols_weekly]
        
        return df_weekly, df_daily
    
    def simulate_group_stratified(
        self,
        model: ModelSelection,
        groups: List[GroupConfig],
        mixing_matrix: np.ndarray,
        inject_noise: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run group-stratified simulation.
        
        Args:
            model: Model specification (base compartments and transitions)
            groups: List of group configurations
            mixing_matrix: Contact mixing matrix between groups
            inject_noise: Whether to inject observation noise
        
        Returns:
            Tuple of (weekly_df, daily_df, metadata)
        """
        num_groups = len(groups)
        base_compartments = model.compartments
        
        # Create stratified compartment names: e.g., S_group0, I_group0, ...
        stratified_comps = []
        for comp in base_compartments:
            for i in range(num_groups):
                stratified_comps.append(f"{comp}_group{i}")
        
        # Build stratified ODE system
        system = self._build_group_stratified_system(
            base_compartments,
            model.transitions,
            model.params,
            groups,
            mixing_matrix
        )
        
        # Initial conditions (stratified by group)
        y0 = self._initial_conditions_stratified(
            base_compartments,
            model.params,
            groups
        )
        population = model.params.get("population", 1e5)
        balancer = self._make_population_balancer(stratified_comps, population)
        
        # Integrate
        integrator = RK4Integrator(system)
        max_steps = model.params.get("max_time_steps", 250)
        dt = model.params.get("dt", 1.0)
        t0, t1 = 0.0, max_steps * dt
        times, states = integrator.run(y0, (t0, t1), dt, post_step=balancer)
        
        # Check for numerical instability
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            raise RuntimeError("Numerical instability detected in group-stratified simulation")
        if np.max(states) > population * 100:
            raise RuntimeError(f"Unrealistic values detected: max={np.max(states):.0f}")
        
        # Inject noise if requested
        if inject_noise and model.noise_level > 0:
            noise_injector = NoiseInjector(random.Random())
            states = noise_injector.inject_noise(states, model.noise_level)
            states = np.vstack([balancer(row) for row in states])

        # Compute group-stratified R_t
        R_t = self._compute_R_t_stratified(model, states, base_compartments, groups)
        
        # Compute R0 dominant eigenvalue for the stratified system
        R0_matrix = self._compute_R0_matrix(model.params, groups, mixing_matrix)
        R0_dominant = np.max(np.real(np.linalg.eigvals(R0_matrix)))
        
        # Create dataframes
        time_unit = model.params.get("time_unit", "weeks")
        
        if time_unit == "weeks":
            df_weekly = pd.DataFrame(states, columns=stratified_comps)
            df_weekly.insert(0, "week", times)
            df_weekly["R_t"] = R_t
            
            # Interpolate to daily
            daily_times = np.arange(0, times[-1] * 7, 1.0)
            daily_states = np.zeros((len(daily_times), states.shape[1]))
            for i in range(states.shape[1]):
                daily_states[:, i] = np.interp(daily_times, times * 7, states[:, i])
            df_daily = pd.DataFrame(daily_states, columns=stratified_comps)
            df_daily.insert(0, "day", daily_times)
            df_daily["R_t"] = np.interp(daily_times, times * 7, R_t)
        else:
            df_daily = pd.DataFrame(states, columns=stratified_comps)
            df_daily.insert(0, "day", times)
            df_daily["R_t"] = R_t
            
            # Downsample to weekly
            week_indices = np.arange(0, len(times), 7)
            df_weekly = df_daily.iloc[week_indices].copy()
            df_weekly["week"] = df_weekly["day"] / 7.0
            df_weekly = df_weekly.drop(columns=["day"])
        
        # Aggregate stratified compartments back to base compartments
        aggregated_weekly = self._aggregate_compartments(df_weekly, base_compartments, num_groups)
        aggregated_daily = self._aggregate_compartments(df_daily, base_compartments, num_groups)
        
        # Metadata
        metadata = {
            "stratified_compartments": stratified_comps,
            "base_compartments": base_compartments,
            "num_groups": num_groups,
            "group_names": [g.name for g in groups],
            "group_fractions": [g.population_fraction for g in groups],
            "mixing_matrix": mixing_matrix.tolist(),
            "R0_dominant_eigenvalue": float(R0_dominant),
            "time_unit": time_unit,
            # Store stratified dataframes for debugging
            "stratified_weekly": df_weekly,
            "stratified_daily": df_daily
        }
        
        return aggregated_weekly, aggregated_daily, metadata
    
    def _aggregate_compartments(
        self,
        df_stratified: pd.DataFrame,
        base_compartments: List[str],
        num_groups: int
    ) -> pd.DataFrame:
        """Aggregate stratified compartments back to base compartments."""
        # Determine time column
        time_col = "week" if "week" in df_stratified.columns else "day"
        
        # Create aggregated dataframe
        aggregated_data = {time_col: df_stratified[time_col].values}
        
        # Aggregate each base compartment
        for comp in base_compartments:
            group_comps = [f"{comp}_group{i}" for i in range(num_groups)]
            # Sum across all groups
            aggregated_data[comp] = df_stratified[group_comps].sum(axis=1).values
        
        # Add R_t
        if "R_t" in df_stratified.columns:
            aggregated_data["R_t"] = df_stratified["R_t"].values
        
        df_aggregated = pd.DataFrame(aggregated_data)
        
        # Reorder columns
        cols = [time_col] + base_compartments + (["R_t"] if "R_t" in aggregated_data else [])
        return df_aggregated[cols]
    
    def _make_population_balancer(
        self,
        compartments: List[str],
        population: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a post-step function that keeps totals within the configured population."""
        living_indices = [
            i for i, c in enumerate(compartments)
            if not c.startswith("W") and not c.startswith("D")
        ]
        death_indices = [i for i, c in enumerate(compartments) if c.startswith("D")]
        cap = max(population * 2.0, population + 1.0)
        
        def _balance(state_vec: np.ndarray) -> np.ndarray:
            adjusted = np.maximum(state_vec, 0.0)
            living_total = float(adjusted[living_indices].sum()) if living_indices else 0.0
            deaths = float(adjusted[death_indices].sum()) if death_indices else 0.0
            target = max(population - deaths, 0.0)
            
            if living_total > target and living_total > 0:
                scale = target / living_total
                adjusted[living_indices] *= scale
                living_total = float(adjusted[living_indices].sum()) if living_indices else 0.0
            
            # Guard against rounding pushing totals just above the target
            if living_total > target and living_total > 0:
                scale = target / living_total
                adjusted[living_indices] *= scale
            
            # Keep a hard upper bound to avoid runaway solutions
            adjusted = np.clip(adjusted, 0.0, cap)
            return adjusted
        
        return _balance
    
    def _build_system(
        self, 
        compartments: List[str],
        transitions, 
        params: Dict[str, float]
    ):
        """Build ODE system function with optional seasonal forcing."""
        comp_list = list(compartments)
        population_comps = [c for c in comp_list if c not in {"W", "D"}]
        transition_names = [t.rule_name for t in transitions]
        
        # Extract seasonal forcing parameters
        seasonal_forcing = params.get("seasonal_forcing", False)
        seasonal_amplitude = params.get("seasonal_amplitude", 0.0)
        seasonal_period = params.get("seasonal_period", 52)
        seasonal_phase = params.get("seasonal_phase", 0.0)
        base_beta = params.get("beta", 0.3)
        
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            state = {comp: y[i] for i, comp in enumerate(comp_list)}
            total_pop = sum(state[c] for c in population_comps)
            delta = {comp: 0.0 for comp in comp_list}
            
            # Create time-varying params with seasonal forcing
            # β(t) = β₀ × (1 + ε × cos(2πt/T + φ))
            if seasonal_forcing and seasonal_amplitude > 0:
                seasonal_factor = 1.0 + seasonal_amplitude * math.cos(
                    2 * math.pi * t / seasonal_period + seasonal_phase
                )
                params_t = params.copy()
                params_t["beta"] = base_beta * seasonal_factor
            else:
                params_t = params
            
            for rule_name in transition_names:
                self.transitions.derivative(rule_name, state, params_t, delta, total_pop)
            
            return np.array([delta[c] for c in comp_list], dtype=float)
        
        return rhs
    
    def _initial_conditions(
        self, 
        compartments: List[str], 
        params: Dict[str, float]
    ) -> np.ndarray:
        """Set initial conditions."""
        pop = params.get("population", 1e5)
        init = {c: 0.0 for c in compartments}
        
        # Scale initial infections with population (but keep reasonable)
        base_infected = params.get("initial_infected", 10)
        if pop > 1e6:
            base_infected = min(base_infected * (pop / 1e5) ** 0.5, pop * 0.0001)
        
        # Ensure initial infected is reasonable (not too small, not too large)
        init["I"] = max(1.0, min(base_infected, pop * 0.001))  # 0.1% max
        
        # For SEIR models, exposed should be larger than infected initially
        if "E" in init:
            init["E"] = params.get("initial_exposed", base_infected * 2)
            # Ensure E > I for realistic dynamics
            if init["E"] < init["I"]:
                init["E"] = init["I"] * 2
        
        if "A" in init:
            init["A"] = params.get("initial_asymptomatic", base_infected * 0.5)
        
        used = init["I"] + init.get("E", 0) + init.get("A", 0)
        init["S"] = pop - used
        
        # Ensure all values are positive and realistic
        for c in init:
            if c not in {"S", "W"}:  # S and W handled separately
                init[c] = max(0.0, min(init[c], pop * 0.1))  # Cap at 10% of population
        
        # Rebalance susceptibles after capping other compartments
        used_total = sum(
            init[c] for c in init
            if c not in {"S", "W"}
        )
        if used_total > pop:
            scale = pop / used_total if used_total > 0 else 0.0
            for c in init:
                if c not in {"S", "W"}:
                    init[c] *= scale
            used_total = sum(
                init[c] for c in init
                if c not in {"S", "W"}
            )
        init["S"] = max(0.0, pop - used_total)
        
        return np.array([init[c] for c in compartments], dtype=float)
    
    def _build_group_stratified_system(
        self,
        base_compartments: List[str],
        transitions,
        params: Dict[str, float],
        groups: List[GroupConfig],
        mixing_matrix: np.ndarray
    ):
        """Build ODE system for group-stratified model."""
        num_groups = len(groups)
        
        # Create stratified compartment list
        stratified_comps = []
        for comp in base_compartments:
            for i in range(num_groups):
                stratified_comps.append(f"{comp}_group{i}")
        
        # Map base compartment to group indices
        comp_to_indices = {}
        for i, comp in enumerate(base_compartments):
            comp_to_indices[comp] = list(range(i * num_groups, (i + 1) * num_groups))
        
        transition_names = [t.rule_name for t in transitions]
        population_comps = [c for c in base_compartments if c not in {"W", "D"}]
        
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            delta = np.zeros_like(y)
            
            # Process each group separately
            for g in range(num_groups):
                group = groups[g]
                
                # Extract state for this group
                state = {}
                for comp in base_compartments:
                    idx = comp_to_indices[comp][g]
                    state[comp] = y[idx]
                
                # Total population in this group
                total_pop = sum(
                    state[c] for c in population_comps if c in state
                )
                
                # Create group-specific parameters
                group_params = params.copy()
                
                # Modify beta for infection based on mixing with other groups
                if "beta" in group_params and "I" in state:
                    # Sum infectious from all groups weighted by mixing
                    total_infectious = 0.0
                    for g2 in range(num_groups):
                        I_idx = comp_to_indices["I"][g2]
                        infectious = y[I_idx]
                        # Add contribution from other compartments if they contribute
                        if "A" in base_compartments:
                            A_idx = comp_to_indices["A"][g2]
                            infectious += group_params.get("epsilon", 0.5) * y[A_idx]
                        if "C" in base_compartments:
                            C_idx = comp_to_indices["C"][g2]
                            infectious += group_params.get("delta_C", 0.5) * y[C_idx]
                        
                        # Weight by mixing matrix and group2's infectivity
                        total_infectious += (
                            mixing_matrix[g, g2] * 
                            groups[g2].infectivity_multiplier * 
                            infectious
                        )
                    
                    # Store effective infectious for this group
                    state["_effective_infectious"] = total_infectious
                    
                    # Adjust beta by group contact rate and susceptibility
                    group_params["beta"] = (
                        params["beta"] * 
                        group.contact_rate_multiplier * 
                        group.susceptibility_multiplier
                    )
                
                # Apply transitions for this group
                group_delta = {comp: 0.0 for comp in base_compartments}
                
                for rule_name in transition_names:
                    # Special handling for infection rules in stratified models
                    if rule_name in ["standard_mixing", "fomite_surface", 
                                    "hospital_leakage", "power_law_mixing"]:
                        # Use modified infection calculation
                        self._apply_stratified_infection(
                            rule_name, state, group_params, group_delta, total_pop
                        )
                    else:
                        # Other transitions work the same
                        self.transitions.derivative(
                            rule_name, state, group_params, group_delta, total_pop
                        )
                
                # Map group deltas back to full state
                for comp in base_compartments:
                    idx = comp_to_indices[comp][g]
                    delta[idx] = group_delta[comp]
            
            return delta
        
        return rhs
    
    def _apply_stratified_infection(
        self,
        rule_name: str,
        state: Dict[str, float],
        params: Dict[str, float],
        delta: Dict[str, float],
        total_pop: float
    ):
        """Apply infection rule with stratified mixing."""
        beta = params.get("beta", 0.3)
        
        # Use pre-computed effective infectious from all groups
        if "_effective_infectious" in state:
            infectious = state["_effective_infectious"]
        else:
            # Fallback to within-group only
            infectious = state.get("I", 0.0)
        
        flow = beta * state.get("S", 0.0) * infectious / max(total_pop, 1e-8)
        flow = self._cap_flow(flow, state.get("S", 0.0))
        
        # Route to E or I
        delta["S"] -= flow
        if "E" in state:
            delta["E"] += flow
        else:
            delta["I"] += flow
    
    def _initial_conditions_stratified(
        self,
        base_compartments: List[str],
        params: Dict[str, float],
        groups: List[GroupConfig]
    ) -> np.ndarray:
        """Set initial conditions for group-stratified model."""
        num_groups = len(groups)
        pop = params.get("population", 1e5)
        
        # Get base initial conditions
        base_init = {}
        base_infected = params.get("initial_infected", 10)
        if pop > 1e6:
            base_infected = min(base_infected * (pop / 1e5) ** 0.5, pop * 0.0001)
        
        base_init["I"] = max(1.0, min(base_infected, pop * 0.001))
        
        if "E" in base_compartments:
            base_init["E"] = params.get("initial_exposed", base_infected * 2)
            if base_init["E"] < base_init["I"]:
                base_init["E"] = base_init["I"] * 2
        
        if "A" in base_compartments:
            base_init["A"] = params.get("initial_asymptomatic", base_infected * 0.5)
        
        # Initialize all other compartments to 0
        for comp in base_compartments:
            if comp not in base_init:
                base_init[comp] = 0.0
        
        # Distribute across groups based on population fractions
        stratified_init = []
        for comp in base_compartments:
            if comp == "S":
                # Susceptibles = group_pop - infected_in_group
                used = base_init.get("I", 0) + base_init.get("E", 0) + base_init.get("A", 0)
                for group in groups:
                    group_pop = pop * group.population_fraction
                    group_used = used * group.population_fraction
                    stratified_init.append(group_pop - group_used)
            else:
                # Other compartments distributed by population fraction
                for group in groups:
                    stratified_init.append(base_init[comp] * group.population_fraction)
        
        return np.array(stratified_init, dtype=float)
    
    def _compute_R0_matrix(
        self,
        params: Dict[str, float],
        groups: List[GroupConfig],
        mixing_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute next-generation matrix for group-stratified model."""
        num_groups = len(groups)
        R0_base = params.get("R0", 2.5)
        
        # Next-generation matrix: R0_ij = R0 * contact_i * suscept_i * infect_j * mix_ij
        R0_matrix = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                R0_matrix[i, j] = (
                    R0_base *
                    groups[i].contact_rate_multiplier *
                    groups[i].susceptibility_multiplier *
                    groups[j].infectivity_multiplier *
                    mixing_matrix[i, j]
                )
        
        return R0_matrix
    
    def _compute_R_t_stratified(
        self,
        model: ModelSelection,
        states: np.ndarray,
        base_compartments: List[str],
        groups: List[GroupConfig]
    ) -> np.ndarray:
        """Compute effective reproduction number for stratified model."""
        num_groups = len(groups)
        R0 = model.params.get("R0", 2.5)
        population = model.params.get("population", 1e5)
        
        # Find S compartment indices
        S_indices = []
        for g in range(num_groups):
            S_idx = base_compartments.index("S") * num_groups + g
            S_indices.append(S_idx)
        
        # Compute population-weighted R_t
        R_t = np.zeros(len(states))
        group_pops = np.array([g.population_fraction * population for g in groups])
        
        for t in range(len(states)):
            # Get S for each group
            S_values = states[t, S_indices]
            
            # Initial S (approximately group population)
            S_initial = group_pops
            
            # R_t for each group
            group_R_t = R0 * (S_values / (S_initial + 1e-10))
            
            # Population-weighted average
            R_t[t] = np.average(group_R_t, weights=group_pops)
        
        return R_t
    
    def _compute_R_t(
        self, 
        model: ModelSelection, 
        states: np.ndarray
    ) -> np.ndarray:
        """Compute effective reproduction number over time."""
        R0 = model.params.get("R0", 2.5)
        S_idx = model.compartments.index("S")
        population_comps = [c for c in model.compartments if c not in {"W", "D"}]
        
        R_t = np.zeros(len(states))
        for i in range(len(states)):
            total_pop = sum(
                states[i, j] 
                for j, c in enumerate(model.compartments) 
                if c in population_comps
            )
            if total_pop > 0:
                R_t[i] = R0 * (states[i, S_idx] / total_pop)
        
        return R_t
    
    def visualize(
        self,
        model: ModelSelection,
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> None:
        """Create visualization plots for the simulation."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Epidemic Model: {model.name}', fontsize=14, fontweight='bold')
        
        # Plot 1: All compartments (weekly)
        ax1 = axes[0, 0]
        for comp in model.compartments:
            if comp in df_weekly.columns:
                ax1.plot(df_weekly['week'], df_weekly[comp], label=comp, linewidth=2)
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Population')
        ax1.set_title('Compartment Dynamics (Weekly)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Infectious compartments only (daily)
        ax2 = axes[0, 1]
        infectious_comps = [c for c in ['I', 'E', 'A', 'H', 'Q', 'C'] if c in df_daily.columns]
        for comp in infectious_comps:
            ax2.plot(df_daily['day'], df_daily[comp], label=comp, linewidth=1.5)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Population')
        ax2.set_title('Infectious Compartments (Daily)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R_t over time
        ax3 = axes[1, 0]
        ax3.plot(df_weekly['week'], df_weekly['R_t'], color='red', linewidth=2)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='R=1 threshold')
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Effective Reproduction Number')
        ax3.set_title('R_t Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model structure info
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = f"Model Structure\n" + "="*30 + "\n\n"
        info_text += f"Compartments ({len(model.compartments)}): {', '.join(model.compartments)}\n\n"
        info_text += f"Transitions ({len(model.transitions)}): \n"
        for i, trans in enumerate(model.transitions[:8], 1):  # Show first 8
            info_text += f"  {i}. {trans}\n"
        if len(model.transitions) > 8:
            info_text += f"  ... and {len(model.transitions) - 8} more\n"
        info_text += f"\nKey Parameters:\n"
        info_text += f"  R0 = {model.params.get('R0', 'N/A'):.2f}\n"
        info_text += f"  Population = {model.params.get('population', 0):.0f}\n"
        info_text += f"  Noise level = {model.noise_level:.3f}\n"
        info_text += f"\nSimulation:\n"
        time_unit = model.params.get('time_unit', 'weeks')
        max_steps = model.params.get('max_time_steps', 250)
        info_text += f"  Steps = {max_steps} {time_unit}\n"
        info_text += f"  Peak I = {df_daily['I'].max():.0f}\n"
        info_text += f"  Final I = {df_daily['I'].iloc[-1]:.0f}\n"
        if 'R' in df_daily.columns:
            info_text += f"  Final R = {df_daily['R'].iloc[-1]:.0f}\n"
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        plt.close(fig)
    
    def visualize_stratified(
        self,
        model: ModelSelection,
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        metadata: Dict,
        save_path: Optional[Path] = None
    ) -> None:
        """Create visualization plots for group-stratified simulation."""
        num_groups = metadata.get("num_groups", 1)
        group_names = metadata.get("group_names", [f"Group {i}" for i in range(num_groups)])
        base_compartments = metadata.get("base_compartments", model.compartments)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle(f'Group-Stratified Epidemic Model: {model.name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Aggregated compartments
        ax1 = axes[0, 0]
        for comp in base_compartments:
            if comp in df_weekly.columns:
                ax1.plot(df_weekly['week'], df_weekly[comp], label=comp, linewidth=2)
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Population')
        ax1.set_title('Aggregated Compartment Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Group-stratified I compartment
        ax2 = axes[0, 1]
        stratified_df = metadata.get("stratified_weekly")
        if stratified_df is not None:
            for i, name in enumerate(group_names):
                col = f"I_group{i}"
                if col in stratified_df.columns:
                    ax2.plot(stratified_df['week'], stratified_df[col], 
                            label=f"I ({name})", linewidth=2)
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Population')
        ax2.set_title('Infectious by Group')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Group-stratified S compartment
        ax3 = axes[1, 0]
        if stratified_df is not None:
            for i, name in enumerate(group_names):
                col = f"S_group{i}"
                if col in stratified_df.columns:
                    ax3.plot(stratified_df['week'], stratified_df[col], 
                            label=f"S ({name})", linewidth=2)
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Population')
        ax3.set_title('Susceptible by Group')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: R_t over time
        ax4 = axes[1, 1]
        ax4.plot(df_weekly['week'], df_weekly['R_t'], color='red', linewidth=2)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='R=1 threshold')
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Effective Reproduction Number')
        ax4.set_title('R_t Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Mixing matrix heatmap
        ax5 = axes[2, 0]
        mixing_matrix = np.array(metadata.get("mixing_matrix", [[1]]))
        im = ax5.imshow(mixing_matrix, cmap='YlOrRd', aspect='auto')
        ax5.set_xticks(range(num_groups))
        ax5.set_yticks(range(num_groups))
        ax5.set_xticklabels(group_names, rotation=45, ha='right')
        ax5.set_yticklabels(group_names)
        ax5.set_title('Mixing Matrix')
        plt.colorbar(im, ax=ax5)
        
        # Plot 6: Model info
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        info_text = f"Model Structure\n" + "="*35 + "\n\n"
        info_text += f"Groups: {', '.join(group_names)}\n"
        info_text += f"R0 (dominant): {metadata.get('R0_dominant_eigenvalue', 0):.2f}\n\n"
        info_text += f"Compartments: {', '.join(base_compartments)}\n\n"
        info_text += f"Key Parameters:\n"
        info_text += f"  R0 (base) = {model.params.get('R0', 'N/A'):.2f}\n"
        info_text += f"  Population = {model.params.get('population', 0):.0f}\n"
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Stratified visualization saved to: {save_path}")
        
        plt.close(fig)


# ============================================================================
# ENHANCED PIPELINE CLASS
# ============================================================================


class EnhancedEpiRecipePipeline:
    """Main pipeline with constraint-based generation."""
    
    def __init__(self, catalog_path: Path | None = None, rng_seed: int | None = None):
        path = catalog_path or CATALOG_PATH
        self.catalog = Catalog.load(path)
        self.rng = random.Random(rng_seed)
        
        # Initialize components
        self.constraint_solver = ConstraintSolver(self.catalog, self.rng)
        self.parameter_sampler = HierarchicalParameterSampler(self.catalog, self.rng)
        self.validator = ModelValidator(self.catalog)
        self.noise_injector = NoiseInjector(self.rng)
        self.simulator = Simulator()
    
    def generate_model(
        self,
        num_compartments: int = 5,
        overrides: Optional[Dict] = None,
        name: str = "model",
        max_time_steps: int = 250
    ) -> ModelSelection:
        """Generate a valid compartmental model."""
        # 1. Select compartments with constraint satisfaction
        compartments = self.constraint_solver.select_compartments(num_compartments)
        
        # 2. Automatically construct valid transitions
        transitions = self.constraint_solver.construct_transitions(compartments)
        
        # 3. Sample parameters hierarchically
        params = self.parameter_sampler.sample_parameters(
            compartments,
            transitions,
            overrides
        )
        params["max_time_steps"] = max_time_steps
        
        # 4. Sample noise level
        noise_level = self.noise_injector.sample_noise_level()
        
        # 5. Select observables
        observables = self._select_observables(compartments)
        
        # Create model
        model = ModelSelection(
            compartments=compartments,
            transitions=transitions,
            params=params,
            noise_level=noise_level,
            name=name,
            observables=observables
        )
        
        # 6. Validate model
        self.validator.validate(model)
        
        return model
    
    def _select_observables(self, compartments: List[str]) -> List[str]:
        """Select which observables to track."""
        observables = ["incidence", "prevalence", "R_effective"]
        
        obs_specs = self.catalog.observables
        for obs_name, spec in obs_specs.items():
            if obs_name in observables:
                continue
            required = spec.get("requires", [])
            if all(req in compartments for req in required):
                if self.rng.random() < 0.5:  # Probabilistically add
                    observables.append(obs_name)
        
        return observables
    
    def _adapt_model_for_simulator(self, model: ModelSelection) -> ModelSelection:
        """Adapt TransitionSpec to simulator format (using local TransitionStructure)."""
        # Convert TransitionSpec to TransitionStructure
        adapted_transitions = []
        for trans in model.transitions:
            # Map dynamic_type:variant to rule names the simulator knows
            rule_name = self._map_to_old_rule_name(trans)
            
            adapted_transitions.append(TransitionStructure(
                source=trans.source,
                target=trans.target,
                category=trans.dynamic_type,
                rule_name=rule_name
            ))
        
        # Create adapted model with old format
        adapted_model = ModelSelection(
            compartments=model.compartments,
            transitions=adapted_transitions,
            params=model.params,
            noise_level=model.noise_level,
            name=model.name,
            observables=model.observables,
            stratification_config=model.stratification_config
        )
        
        return adapted_model
    
    def _map_to_old_rule_name(self, trans: TransitionSpec) -> str:
        """Map new transition spec to simulator rule names."""
        # Create mapping from new format to old simulator rule names
        mapping = {
            ("infection", "mass_action"): "standard_mixing",
            ("infection", "with_asymptomatic"): "standard_mixing",
            ("infection", "with_chronic"): "standard_mixing",
            ("infection", "environmental"): "environmental_cholera",
            ("infection", "combined_environmental"): "fomite_surface",
            ("infection", "hospital_nosocomial"): "hospital_leakage",
            ("infection", "saturating"): "vector_borne_proxy",
            ("latent_progression", "simple"): "simple_linear",
            ("latent_progression", "branching"): "branching_split",
            ("recovery", "constant_rate"): "constant_recovery",
            ("recovery", "resource_limited"): "constant_recovery",  # Use constant_recovery for I->R
            ("hospitalization", "constant_rate"): "severity_trigger",
            ("hospitalization", "severity_dependent"): "overflow_trigger",
            ("death", "from_infectious"): "disease_induced_death",
            ("death", "from_hospital"): "hospital_death",
            ("vaccination", "constant_campaign"): "campaign_mode",
            ("vaccination", "proportional"): "proportional_vax",
            ("quarantine", "from_infectious"): "reactive_quarantine",
            ("quarantine", "from_exposed"): "tracing_efficiency",
            ("quarantine_release", "to_recovered"): "reactive_quarantine",
            ("quarantine_release", "to_susceptible"): "reactive_quarantine",
            ("environmental_shedding", "from_infectious"): "shedding",
            ("environmental_shedding", "with_asymptomatic"): "asymptomatic_shedding",
            ("environmental_decay", "decay"): "environmental_decay",
            ("chronic_progression", "constant"): "chronic_failure",
            ("chronic_clearance", "constant"): "carrier_clearance",
            ("waning_immunity", "constant"): "waning_immunity",
            ("vaccine_waning", "constant"): "waning_immunity",  # Reuse same handler
        }
        
        key = (trans.dynamic_type, trans.variant)
        return mapping.get(key, "constant_recovery")  # Fallback
    
    def run_simulation(
        self,
        model: ModelSelection,
        save_path: Optional[Path] = None,
        visualize: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run simulation using the existing simulator."""
        # Convert TransitionSpec to format expected by simulator
        # This bridges the new model spec with the old simulator
        adapted_model = self._adapt_model_for_simulator(model)
        
        df_weekly, df_daily = self.simulator.simulate(adapted_model, inject_noise=True)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_weekly.to_csv(str(save_path) + "_weekly.csv", index=False)
            df_daily.to_csv(str(save_path) + "_daily.csv", index=False)
            
            if visualize:
                viz_path = Path(str(save_path) + "_plot.png")
                self.simulator.visualize(adapted_model, df_weekly, df_daily, viz_path)
        
        return df_weekly, df_daily
    
    def run_simulation_stratified(
        self,
        model: ModelSelection,
        groups: List[GroupConfig],
        mixing_type: str = "assortative",
        assortativity: float = 0.7,
        mixing_matrix: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        visualize: bool = True,
        inject_noise: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run group-stratified simulation.
        
        Args:
            model: Model specification
            groups: List of GroupConfig objects defining population groups
            mixing_type: Type of mixing matrix ('homogeneous', 'assortative', 
                        'hierarchical', 'spatial'). Ignored if mixing_matrix provided.
            assortativity: Degree of within-group preference (0-1). Ignored if 
                          mixing_matrix provided.
            mixing_matrix: Pre-computed mixing matrix. If None, generated from
                          mixing_type and assortativity.
            save_path: Path to save results
            visualize: Whether to create visualization
            inject_noise: Whether to inject observation noise
        
        Returns:
            Tuple of (aggregated_weekly_df, aggregated_daily_df, metadata)
            
            The metadata dict contains:
            - stratified_weekly: DataFrame with group-specific compartments
            - stratified_daily: DataFrame with group-specific compartments
            - group_names: List of group names
            - num_groups: Number of groups
            - mixing_matrix: The mixing matrix used
            - R0_dominant_eigenvalue: Dominant eigenvalue of R0 matrix
        
        Example:
            ```python
            groups = [
                GroupConfig('Children', 0.2, contact_rate_multiplier=1.5),
                GroupConfig('Adults', 0.5, contact_rate_multiplier=1.0),
                GroupConfig('Elderly', 0.3, contact_rate_multiplier=0.7, 
                           susceptibility_multiplier=1.5)
            ]
            
            df_weekly, df_daily, metadata = pipeline.run_simulation_stratified(
                model, groups, mixing_type='hierarchical'
            )
            
            # Access group-specific I trajectories
            strat_df = metadata['stratified_weekly']
            I_children = strat_df['I_group0']
            I_adults = strat_df['I_group1']
            I_elderly = strat_df['I_group2']
            ```
        """
        # Generate mixing matrix if not provided
        if mixing_matrix is None:
            mixing_matrix = generate_mixing_matrix(
                num_groups=len(groups),
                mixing_type=mixing_type,
                assortativity=assortativity,
                rng=self.rng
            )
        
        # Convert model to simulator format
        adapted_model = self._adapt_model_for_simulator(model)
        
        # Run stratified simulation
        df_weekly, df_daily, metadata = self.simulator.simulate_group_stratified(
            model=adapted_model,
            groups=groups,
            mixing_matrix=mixing_matrix,
            inject_noise=inject_noise
        )
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_weekly.to_csv(str(save_path) + "_weekly.csv", index=False)
            df_daily.to_csv(str(save_path) + "_daily.csv", index=False)
            
            # Also save stratified data
            metadata['stratified_weekly'].to_csv(
                str(save_path) + "_stratified_weekly.csv", index=False
            )
            metadata['stratified_daily'].to_csv(
                str(save_path) + "_stratified_daily.csv", index=False
            )
            
            if visualize:
                viz_path = Path(str(save_path) + "_stratified_plot.png")
                self.simulator.visualize_stratified(
                    adapted_model, df_weekly, df_daily, metadata, viz_path
                )
        
        return df_weekly, df_daily, metadata
    
    def generate_and_simulate(
        self,
        num_compartments: int = 5,
        overrides: Optional[Dict] = None,
        save_path: Optional[Path] = None,
        name: str = "model",
        visualize: bool = True,
        max_time_steps: int = 250
    ) -> Tuple[ModelSelection, pd.DataFrame, pd.DataFrame]:
        """Complete pipeline: generate and simulate."""
        model = self.generate_model(num_compartments, overrides, name, max_time_steps)
        df_weekly, df_daily = self.run_simulation(model, save_path, visualize=visualize)
        return model, df_weekly, df_daily
    
    def generate_and_simulate_stratified(
        self,
        groups: List[GroupConfig],
        num_compartments: int = 5,
        overrides: Optional[Dict] = None,
        mixing_type: str = "assortative",
        assortativity: float = 0.7,
        save_path: Optional[Path] = None,
        name: str = "model",
        visualize: bool = True,
        max_time_steps: int = 250
    ) -> Tuple[ModelSelection, pd.DataFrame, pd.DataFrame, Dict]:
        """Complete pipeline: generate model and run group-stratified simulation.
        
        Args:
            groups: List of GroupConfig objects defining population groups
            num_compartments: Number of compartments to include
            overrides: Parameter overrides
            mixing_type: Type of mixing matrix
            assortativity: Degree of within-group mixing preference
            save_path: Path to save results
            name: Model name
            visualize: Whether to create visualization
            max_time_steps: Maximum simulation time steps
        
        Returns:
            Tuple of (model, aggregated_weekly_df, aggregated_daily_df, metadata)
        
        Example:
            ```python
            groups = [
                GroupConfig('Young', 0.3, contact_rate_multiplier=1.5),
                GroupConfig('Middle', 0.5),
                GroupConfig('Old', 0.2, susceptibility_multiplier=1.5)
            ]
            
            model, df_weekly, df_daily, metadata = pipeline.generate_and_simulate_stratified(
                groups=groups,
                num_compartments=5,
                mixing_type='hierarchical'
            )
            ```
        """
        model = self.generate_model(num_compartments, overrides, name, max_time_steps)
        df_weekly, df_daily, metadata = self.run_simulation_stratified(
            model, groups, mixing_type, assortativity, 
            save_path=save_path, visualize=visualize
        )
        return model, df_weekly, df_daily, metadata


def demo() -> None:
    """Demonstration of enhanced pipeline."""
    pipeline = EnhancedEpiRecipePipeline(rng_seed=42)
    
    print("=" * 70)
    print("Enhanced EpiRecipe Pipeline - Constraint-Based Generation")
    print("=" * 70)
    
    # Generate model
    model = pipeline.generate_model(num_compartments=6, name="demo_enhanced")
    
    print(f"\n✓ Selected compartments: {model.compartments}")
    print(f"✓ Automatically constructed {len(model.transitions)} transitions:")
    for trans in model.transitions:
        print(f"    {trans}")
    print(f"✓ Hierarchically sampled parameters:")
    print(f"    R0 = {model.params.get('R0', 'N/A'):.2f}")
    print(f"    Infectious period = {model.params.get('infectious_period_days', 'N/A'):.1f} days")
    print(f"    gamma = {model.params.get('gamma', 'N/A'):.3f}")
    print(f"    beta = {model.params.get('beta', 'N/A'):.3f}")
    print(f"✓ Noise level = {model.noise_level:.3f}")
    print(f"✓ Observables: {', '.join(model.observables)}")
    print(f"✓ Model validated successfully!")
    
    # Run simulation
    save_dir = Path(__file__).with_name("simulations")
    df_weekly, df_daily = pipeline.run_simulation(
        model,
        save_path=save_dir / "demo_enhanced",
        visualize=True
    )
    
    print(f"\n✓ Simulation complete")
    print(f"  Weekly data shape: {df_weekly.shape}")
    print(f"  Daily data shape: {df_daily.shape}")
    print(f"\nFirst 5 weeks:")
    print(df_weekly.head())
    
    print(f"\n✓ Files saved to {save_dir}/")


if __name__ == "__main__":
    demo()
