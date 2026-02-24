import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class GroupConfig:
    """Configuration for a population group"""
    name: str
    population_fraction: float  # Fraction of total population
    contact_rate_multiplier: float = 1.0  # Relative contact rate vs baseline
    susceptibility_multiplier: float = 1.0  # Relative susceptibility
    infectivity_multiplier: float = 1.0  # Relative infectivity


def generate_mixing_matrix(
    num_groups: int,
    mixing_type: str = "homogeneous",
    assortativity: float = 0.7,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a mixing matrix for between-group contacts
    
    Args:
        num_groups: Number of population groups
        mixing_type: Type of mixing pattern
            - "homogeneous": Equal mixing between all groups
            - "assortative": Preferential within-group mixing
            - "hierarchical": Age-structured mixing (younger groups mix more)
            - "spatial": Spatial adjacency pattern
        assortativity: Strength of within-group preference (0=random, 1=only within-group)
        rng: Random number generator
    
    Returns:
        Mixing matrix [num_groups, num_groups] where M[i,j] is contact rate from group i to j
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if mixing_type == "homogeneous":
        # Equal mixing between all groups
        return np.ones((num_groups, num_groups)) / num_groups
    
    elif mixing_type == "assortative":
        # Assortative mixing: prefer contacts within own group
        M = np.ones((num_groups, num_groups)) * (1 - assortativity) / (num_groups - 1)
        np.fill_diagonal(M, assortativity)
        return M
    
    elif mixing_type == "hierarchical":
        # Age-structured: younger groups have more contacts
        M = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                # Contact rate decreases with age distance
                age_distance = abs(i - j)
                contact_rate = np.exp(-age_distance / 2)
                M[i, j] = contact_rate
        # Normalize rows to sum to 1
        M = M / M.sum(axis=1, keepdims=True)
        return M
    
    elif mixing_type == "spatial":
        # Spatial adjacency: groups mainly contact neighbors
        M = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            # Strong within-patch contact
            M[i, i] = 0.7
            # Moderate contact with neighbors
            if i > 0:
                M[i, i-1] = 0.15
            if i < num_groups - 1:
                M[i, i+1] = 0.15
        # Normalize rows
        M = M / M.sum(axis=1, keepdims=True)
        return M
    
    else:
        raise ValueError(f"Unknown mixing_type: {mixing_type}")


def create_stratified_selection(
    base_compartments: List[str],
    base_transitions: List[str],
    groups: List[GroupConfig],
    mixing_matrix: Optional[np.ndarray] = None,
    mixing_type: str = "assortative"
) -> Tuple[List[str], List[str], np.ndarray, List[GroupConfig]]:
    """
    Create group-stratified compartments and transitions
    
    Args:
        base_compartments: Base compartments (e.g., ['S', 'I', 'R'])
        base_transitions: Base transitions (e.g., ['mass_action_infection', 'recovery_linear'])
        groups: List of GroupConfig objects
        mixing_matrix: Optional mixing matrix [num_groups, num_groups]
        mixing_type: Type of mixing if mixing_matrix not provided
    
    Returns:
        stratified_compartments: List of compartment names with group suffix (e.g., 'S_group0')
        stratified_transitions: List of transition names (same as base, applied per group)
        mixing_matrix: Mixing matrix for between-group transmission
        groups: Group configurations
    """
    num_groups = len(groups)
    
    # Generate mixing matrix if not provided
    if mixing_matrix is None:
        mixing_matrix = generate_mixing_matrix(num_groups, mixing_type=mixing_type)
    
    # Create stratified compartments: compartment_groupN
    stratified_compartments = []
    for comp in base_compartments:
        for i, group in enumerate(groups):
            stratified_compartments.append(f"{comp}_{group.name}")
    
    # Transitions remain the same (will be applied per group with mixing)
    stratified_transitions = base_transitions
    
    return stratified_compartments, stratified_transitions, mixing_matrix, groups


class GroupStratifiedTransitionRegistry:
    """Extended transition registry for group-stratified models"""
    
    def __init__(
        self,
        groups: List[GroupConfig],
        mixing_matrix: np.ndarray,
        base_compartments: List[str]
    ):
        self.groups = groups
        self.mixing_matrix = mixing_matrix
        self.num_groups = len(groups)
        self.base_compartments = base_compartments
        
        # Pre-compute group-specific parameters
        self.group_params = self._compute_group_parameters()
    
    def _compute_group_parameters(self) -> List[Dict]:
        """Pre-compute group-specific epidemiological parameters"""
        group_params = []
        for group in self.groups:
            group_params.append({
                'population_fraction': group.population_fraction,
                'contact_multiplier': group.contact_rate_multiplier,
                'susceptibility': group.susceptibility_multiplier,
                'infectivity': group.infectivity_multiplier
            })
        return group_params
    
    def derivative(
        self,
        name: str,
        state: Dict[str, float],
        params: Dict[str, float],
        delta: Dict[str, float],
        total_pop: float
    ) -> None:
        """
        Compute derivatives for group-stratified transitions
        
        Handles between-group transmission using mixing matrix
        """
        if name == "mass_action_infection":
            self._group_mass_action_infection(state, params, delta, total_pop)
        
        elif name == "recovery_linear":
            self._group_recovery(state, params, delta)
        
        elif name == "incubation_progression":
            self._group_incubation(state, params, delta)
        
        elif name == "waning_immunity":
            self._group_waning(state, params, delta)
        
        elif name == "disease_induced_death":
            self._group_death(state, params, delta)
        
        # Add more transition handlers as needed
        else:
            # For other transitions, apply independently to each group
            self._apply_per_group(name, state, params, delta, total_pop)
    
    def _group_mass_action_infection(
        self,
        state: Dict[str, float],
        params: Dict[str, float],
        delta: Dict[str, float],
        total_pop: float
    ):
        """Group-stratified mass-action infection with mixing matrix"""
        beta = params.get("beta", 0.3)
        
        # Compute force of infection for each group from all groups
        for i, group_i in enumerate(self.groups):
            S_i = state.get(f"S_{group_i.name}", 0.0)
            
            # Force of infection on group i from all groups j
            foi = 0.0
            for j, group_j in enumerate(self.groups):
                I_j = state.get(f"I_{group_j.name}", 0.0)
                
                # Mixing: contact rate from group i to group j
                mixing_rate = self.mixing_matrix[i, j]
                
                # Group j's population
                pop_j = total_pop * self.group_params[j]['population_fraction']
                
                # Contribution to FOI considering:
                # - mixing_rate: how much group i contacts group j
                # - group_i susceptibility
                # - group_j infectivity
                foi += (mixing_rate * 
                       self.group_params[i]['susceptibility'] * 
                       self.group_params[j]['infectivity'] *
                       I_j / max(pop_j, 1e-8))
            
            # New infections in group i
            infections = beta * S_i * foi
            delta[f"S_{group_i.name}"] -= infections
            delta[f"I_{group_i.name}"] += infections
    
    def _group_recovery(self, state, params, delta):
        """Group-specific recovery (independent per group)"""
        gamma = params.get("gamma", 1/7)
        
        for group in self.groups:
            I_key = f"I_{group.name}"
            R_key = f"R_{group.name}"
            if I_key in state:
                recovery = gamma * state.get(I_key, 0.0)
                delta[I_key] -= recovery
                if R_key in delta:
                    delta[R_key] += recovery
    
    def _group_incubation(self, state, params, delta):
        """Group-specific incubation progression"""
        sigma = params.get("sigma", 1/5)
        
        for group in self.groups:
            E_key = f"E_{group.name}"
            I_key = f"I_{group.name}"
            if E_key in state:
                progression = sigma * state.get(E_key, 0.0)
                delta[E_key] -= progression
                delta[I_key] += progression
    
    def _group_waning(self, state, params, delta):
        """Group-specific waning immunity"""
        omega = params.get("omega", 1/180)
        
        for group in self.groups:
            R_key = f"R_{group.name}"
            S_key = f"S_{group.name}"
            if R_key in state:
                waning = omega * state.get(R_key, 0.0)
                delta[R_key] -= waning
                delta[S_key] += waning
    
    def _group_death(self, state, params, delta):
        """Group-specific disease-induced death"""
        mu = params.get("mu", 0.005)
        
        for group in self.groups:
            I_key = f"I_{group.name}"
            D_key = f"D_{group.name}"
            if I_key in state:
                deaths = mu * state.get(I_key, 0.0)
                delta[I_key] -= deaths
                delta.setdefault(D_key, 0.0)
                delta[D_key] += deaths
    
    def _apply_per_group(self, name, state, params, delta, total_pop):
        """Apply non-infection transitions independently per group"""
        # Import base transition registry for fallback
        from old.pipeline import TransitionRegistry
        base_registry = TransitionRegistry()
        
        # Apply transition to each group independently
        for group in self.groups:
            # Create group-specific state and delta
            group_state = {}
            group_delta = {}
            
            # Extract group-specific compartments
            for comp in self.base_compartments:
                key = f"{comp}_{group.name}"
                if key in state:
                    group_state[comp] = state[key]
                    group_delta[comp] = 0.0
            
            # Compute group population
            group_pop = total_pop * group.population_fraction
            
            # Apply base transition
            try:
                base_registry.derivative(name, group_state, params, group_delta, group_pop)
            except Exception:
                pass  # Skip if transition not implemented in base
            
            # Update main delta
            for comp in self.base_compartments:
                key = f"{comp}_{group.name}"
                if comp in group_delta:
                    delta[key] += group_delta[comp]


class GroupStratifiedSimulator:
    """Simulator for group-stratified epidemic models"""
    
    def __init__(
        self,
        groups: List[GroupConfig],
        mixing_matrix: np.ndarray,
        base_compartments: List[str],
        base_transitions: List[str]
    ):
        self.groups = groups
        self.mixing_matrix = mixing_matrix
        self.base_compartments = base_compartments
        self.base_transitions = base_transitions
        self.num_groups = len(groups)
        
        self.transitions = GroupStratifiedTransitionRegistry(
            groups, mixing_matrix, base_compartments
        )
    
    def _system(self, stratified_compartments: List[str], params: Dict[str, float]):
        """Create ODE system for group-stratified model"""
        total_pop = params.get("population", 1e5)
        
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            state = {comp: y[i] for i, comp in enumerate(stratified_compartments)}
            delta = {comp: 0.0 for comp in stratified_compartments}
            
            # Apply each transition rule
            for rule_name in self.base_transitions:
                self.transitions.derivative(rule_name, state, params, delta, total_pop)
            
            return np.array([delta[c] for c in stratified_compartments], dtype=float)
        
        return rhs
    
    def initial_conditions(
        self,
        stratified_compartments: List[str],
        params: Dict[str, float]
    ) -> np.ndarray:
        """Set up initial conditions for group-stratified model"""
        total_pop = params.get("population", 1e5)
        initial_infected_total = params.get("initial_infected", 10)
        
        init = {c: 0.0 for c in stratified_compartments}
        
        # Distribute initial infections and population across groups
        for i, group in enumerate(self.groups):
            group_pop = total_pop * group.population_fraction
            group_infected = initial_infected_total * group.population_fraction
            
            # Initialize compartments for this group
            init[f"S_{group.name}"] = group_pop - group_infected
            init[f"I_{group.name}"] = group_infected
            
            # Initialize other compartments if they exist
            for comp in self.base_compartments:
                if comp not in ['S', 'I']:
                    key = f"{comp}_{group.name}"
                    if key in init:
                        init[key] = 0.0
        
        return np.array([init[c] for c in stratified_compartments], dtype=float)
    
    def simulate(
        self,
        stratified_compartments: List[str],
        params: Dict[str, float]
    ) -> pd.DataFrame:
        """Run group-stratified simulation"""
        from old.pipeline import RK4Integrator
        
        system = self._system(stratified_compartments, params)
        y0 = self.initial_conditions(stratified_compartments, params)
        
        integrator = RK4Integrator(system)
        t0, t1 = 0.0, params.get("simulation_days", 180)
        dt = params.get("time_step", 1.0)
        
        times, states = integrator.run(y0, (t0, t1), dt)
        
        # Downsample to weekly data
        week_indices = np.arange(0, len(times), 7)
        weekly_times = times[week_indices] / 7.0
        weekly_states = states[week_indices]
        
        # Apply latency offset
        latency_weeks = params.get("latency_weeks", 0)
        if latency_weeks > 0 and len(weekly_states) > latency_weeks:
            latent_state = weekly_states[0:1].repeat(int(latency_weeks), axis=0)
            active_states = weekly_states[:len(weekly_states) - int(latency_weeks)]
            weekly_states = np.vstack([latent_state, active_states])
        
        df = pd.DataFrame(weekly_states, columns=stratified_compartments)
        df.insert(0, "week", weekly_times)
        
        return df
    
    def compute_group_R0_matrix(self, params: Dict[str, float]) -> np.ndarray:
        """
        Compute Next Generation Matrix for group-stratified model
        
        Returns:
            R0_matrix: [num_groups, num_groups] where R0_matrix[i,j] is the expected
                      number of secondary infections in group i caused by one infected
                      individual in group j
        """
        beta = params.get("beta", 0.3)
        gamma = params.get("gamma", 1/7)
        
        # Infectious period
        infectious_period = 1.0 / gamma
        
        # R0 matrix: R0[i,j] = beta * infectious_period * mixing[i,j] * 
        #                      susceptibility[i] * infectivity[j]
        R0_matrix = np.zeros((self.num_groups, self.num_groups))
        
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                R0_matrix[i, j] = (
                    beta * infectious_period *
                    self.mixing_matrix[i, j] *
                    self.groups[i].susceptibility_multiplier *
                    self.groups[j].infectivity_multiplier
                )
        
        return R0_matrix
    
    def compute_dominant_eigenvalue_R0(self, params: Dict[str, float]) -> float:
        """
        Compute dominant eigenvalue of NGM as overall R0
        
        This is the effective reproduction number for the entire population
        """
        R0_matrix = self.compute_group_R0_matrix(params)
        eigenvalues = np.linalg.eigvals(R0_matrix)
        return float(np.real(np.max(eigenvalues)))
