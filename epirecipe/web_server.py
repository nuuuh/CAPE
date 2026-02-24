"""
EpiRecipe Web Server (Enhanced Pipeline)
Flask web server for EpiRecipe epidemic modeling web interface.

Uses the self-contained pipeline.py for all model generation and simulation,
including group-stratified simulations with multiple mixing matrix types.

Features:
- Generate epidemiological models with constraint-based compartment selection
- Run simulations with optional group stratification (via run_simulation_stratified)
- Support for 4 mixing matrix types: homogeneous, assortative, hierarchical, spatial
- Random model generation with stratification support (via generate_and_simulate_stratified)
- Seasonal forcing: time-varying transmission rate β(t) = β₀ × (1 + ε × cos(2πt/T + φ))
- GP sample generation using epidemic-relevant kernels (Periodic, RBF, Linear, RationalQuadratic)
- Full REST API for programmatic access

Endpoints:
- GET  /                      - Web interface
- POST /api/generate          - Generate and simulate a model with specified compartments
- POST /api/generate_random   - Generate a random model using constraint-based generation
- POST /api/generate_gp       - Generate a Gaussian Process sample with epidemic-relevant kernels
- GET  /api/compartments      - Get available compartments
- GET  /api/catalog           - Get full catalog information
- GET  /api/catalog/compartments - Get detailed compartment info
- GET  /api/catalog/dynamics  - Get dynamics/transition rules info
"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys
import json
import numpy as np

# Import from enhanced pipeline (self-contained, no old pipeline imports)
from pipeline import (
    Catalog, 
    EnhancedEpiRecipePipeline, 
    ModelSelection, 
    GroupConfig,
    TransitionSpec
)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize pipeline (simulator is already part of pipeline)
pipeline = EnhancedEpiRecipePipeline()
catalog = pipeline.catalog


@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('web_interface_enhanced.html')


@app.route('/api/generate', methods=['POST'])
def generate_model():
    """Generate and simulate a model with user-specified compartments"""
    try:
        params = request.json
        
        # Extract parameters
        compartments = params.get('compartments', ['S', 'I', 'R'])
        r0 = params.get('r0', 2.5)
        infectious_period = params.get('infectious_period', 7)
        latent_period = params.get('latent_period', 5)
        population = params.get('population', 1000000)
        initial_infected = params.get('initial_infected', 100)
        max_time_steps = params.get('max_time_steps', 52)
        noise_level = params.get('noise_level', 0.02)
        time_unit = params.get('time_unit', 'weeks')
        use_groups = params.get('use_groups', False)
        
        # Seasonal forcing parameters
        seasonal_forcing = params.get('seasonal_forcing', False)
        seasonal_amplitude = params.get('seasonal_amplitude', 0.3)
        seasonal_period = params.get('seasonal_period', 52)
        seasonal_phase = params.get('seasonal_phase', 0.0)
        
        # Build parameter overrides
        if time_unit == 'days':
            gamma = 1.0 / infectious_period  # per-day
        else:
            gamma = 7.0 / infectious_period  # Convert days to weeks
        beta = r0 * gamma
        
        param_overrides = {
            'R0': r0,
            'beta': beta,
            'gamma': gamma,
            'infectious_period_days': infectious_period,
            'population': population,
            'initial_infected': initial_infected,
            'max_time_steps': max_time_steps,
            'time_unit': time_unit,
            'dt': 1.0,
            'seasonal_forcing': seasonal_forcing,
            'seasonal_amplitude': seasonal_amplitude,
            'seasonal_period': seasonal_period,
            'seasonal_phase': seasonal_phase
        }
        
        if 'E' in compartments:
            if time_unit == 'days':
                sigma = 1.0 / latent_period
            else:
                sigma = 7.0 / latent_period  # Convert days to weeks
            param_overrides['sigma'] = sigma
            param_overrides['latent_period_days'] = latent_period
        
        # Use constraint solver to build transitions for specified compartments
        transitions = pipeline.constraint_solver.construct_transitions(set(compartments))
        
        # Sample parameters with overrides
        params_dict = pipeline.parameter_sampler.sample_parameters(
            compartments, 
            transitions, 
            param_overrides
        )
        
        # Create model
        model = ModelSelection(
            compartments=compartments,
            transitions=transitions,
            params=params_dict,
            noise_level=noise_level,
            name="web_model"
        )
        
        # Run simulation - with or without groups
        if use_groups:
            # Extract group configuration
            groups_data = params.get('groups', [])
            mixing_type = params.get('mixing_type', 'assortative')
            assortativity = params.get('assortativity', 0.7)
            
            # Create GroupConfig objects (using pipeline's GroupConfig)
            groups = [
                GroupConfig(
                    name=g.get('name', f'Group {i+1}'),
                    population_fraction=g.get('pop_fraction', 1.0/len(groups_data)),
                    contact_rate_multiplier=g.get('contact', 1.0),
                    susceptibility_multiplier=g.get('suscept', 1.0),
                    infectivity_multiplier=g.get('infect', 1.0)
                )
                for i, g in enumerate(groups_data)
            ]
            
            # Use pipeline's wrapper method for stratified simulation
            df_weekly, df_daily, metadata = pipeline.run_simulation_stratified(
                model=model,
                groups=groups,
                mixing_type=mixing_type,
                assortativity=assortativity,
                visualize=False
            )
            
            # Format response with group data
            use_daily = time_unit == 'days'
            df = df_daily if use_daily else df_weekly
            time_col = 'day' if use_daily else 'week'
            strat_df = metadata.get('stratified_daily') if use_daily else metadata.get('stratified_weekly')
            
            response = {
                'time': df[time_col].tolist() if time_col in df else list(range(len(df))),
                'compartments': {},
                'R_t': df['R_t'].tolist() if 'R_t' in df else [],
                'time_unit': time_unit,
                'groups': {
                    'I': []  # Infected by group
                },
                'model_info': {
                    'compartments': compartments,
                    'num_groups': len(groups),
                    'group_names': [g.name for g in groups],
                    'mixing_type': mixing_type
                }
            }
            
            # Extract aggregated compartment data
            for comp in compartments:
                if comp in df.columns:
                    response['compartments'][comp] = df[comp].tolist()
            
            # Extract group-specific I data
            if strat_df is not None:
                for i in range(len(groups)):
                    col = f"I_group{i}"
                    if col in strat_df.columns:
                        response['groups']['I'].append(strat_df[col].tolist())
                
                # Also add S and R by group if requested
                response['groups']['S'] = []
                response['groups']['R'] = []
                for i in range(len(groups)):
                    s_col = f"S_group{i}"
                    r_col = f"R_group{i}"
                    if s_col in strat_df.columns:
                        response['groups']['S'].append(strat_df[s_col].tolist())
                    if r_col in strat_df.columns:
                        response['groups']['R'].append(strat_df[r_col].tolist())
        
        else:
            # Run standard (non-stratified) simulation
            df_weekly, df_daily = pipeline.run_simulation(
                model=model,
                visualize=False
            )
            
            # Format response
            use_daily = time_unit == 'days'
            df = df_daily if use_daily else df_weekly
            time_col = 'day' if use_daily else 'week'
            
            response = {
                'time': df[time_col].tolist() if time_col in df else list(range(len(df))),
                'compartments': {},
                'R_t': df['R_t'].tolist() if 'R_t' in df else [],
                'time_unit': time_unit,
                'model_info': {
                    'compartments': compartments,
                    'transitions': [
                        {
                            'source': t.source,
                            'target': t.target,
                            'type': t.dynamic_type,
                            'variant': t.variant
                        }
                        for t in transitions
                    ],
                    'params': {
                        'R0': params_dict.get('R0', 0),
                        'infectious_period_days': params_dict.get('infectious_period_days', 0),
                        'beta': params_dict.get('beta', 0),
                        'gamma': params_dict.get('gamma', 0)
                    },
                    'seasonal_forcing': {
                        'enabled': seasonal_forcing,
                        'amplitude': seasonal_amplitude,
                        'period': seasonal_period,
                        'phase': seasonal_phase
                    }
                }
            }
            
            # Extract compartment data
            for comp in compartments:
                if comp in df.columns:
                    response['compartments'][comp] = df[comp].tolist()
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/api/generate_gp', methods=['POST'])
def generate_gp_sample():
    """Generate a Gaussian Process sample using epidemic-relevant kernels"""
    try:
        params = request.json
        kernel_type = params.get('kernel', 'random')
        sequence_length = params.get('sequence_length', 104)
        
        # Import the GP generator from local epirecipe folder
        from gp_synthetic_generator import GPSyntheticGenerator
        
        # Create generator
        gp_gen = GPSyntheticGenerator(rng_seed=None)  # Random seed each time
        
        # Generate sample
        sample = gp_gen.generate_one(
            length=sequence_length,
            kernel_type=kernel_type if kernel_type != 'random' else None
        )
        
        response = {
            'values': sample['values'].tolist(),
            'kernel_used': sample['kernel'],
            'kernel_params': sample['kernel_params'],
            'sequence_length': sequence_length
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/api/generate_random', methods=['POST'])
def generate_random_model():
    """Generate a random model using constraint-based generation
    
    Supports optional group stratification via 'use_groups' parameter.
    """
    try:
        params = request.json
        
        num_compartments = params.get('num_compartments', 5)
        max_time_steps = params.get('max_time_steps', 52)
        time_unit = params.get('time_unit', 'weeks')
        use_groups = params.get('use_groups', False)
        
        if use_groups:
            # Extract group configuration
            groups_data = params.get('groups', [])
            mixing_type = params.get('mixing_type', 'assortative')
            assortativity = params.get('assortativity', 0.7)
            
            # Create GroupConfig objects
            groups = [
                GroupConfig(
                    name=g.get('name', f'Group {i+1}'),
                    population_fraction=g.get('pop_fraction', 1.0/max(1, len(groups_data))),
                    contact_rate_multiplier=g.get('contact', 1.0),
                    susceptibility_multiplier=g.get('suscept', 1.0),
                    infectivity_multiplier=g.get('infect', 1.0)
                )
                for i, g in enumerate(groups_data)
            ]
            
            # Use pipeline's combined method for random stratified model
            df_weekly, df_daily, metadata = pipeline.generate_and_simulate_stratified(
                groups=groups,
                num_compartments=num_compartments,
                mixing_type=mixing_type,
                assortativity=assortativity,
                max_time_steps=max_time_steps,
                visualize=False
            )
            
            # Get the generated model from metadata
            model = metadata.get('model')
            
            # Format response with group data
            use_daily = time_unit == 'days'
            df = df_daily if use_daily else df_weekly
            time_col = 'day' if use_daily else 'week'
            strat_df = metadata.get('stratified_daily') if use_daily else metadata.get('stratified_weekly')
            
            response = {
                'time': df[time_col].tolist() if time_col in df else list(range(len(df))),
                'compartments': {},
                'R_t': df['R_t'].tolist() if 'R_t' in df else [],
                'time_unit': time_unit,
                'groups': {
                    'I': [],
                    'S': [],
                    'R': []
                },
                'model_info': {
                    'compartments': model.compartments if model else [],
                    'transitions': [
                        {
                            'source': t.source,
                            'target': t.target,
                            'type': t.dynamic_type,
                            'variant': t.variant
                        }
                        for t in (model.transitions if model else [])
                    ],
                    'params': {
                        'R0': model.params.get('R0', 0) if model else 0,
                        'infectious_period_days': model.params.get('infectious_period_days', 0) if model else 0,
                        'beta': model.params.get('beta', 0) if model else 0,
                        'gamma': model.params.get('gamma', 0) if model else 0,
                        'noise_level': model.noise_level if model else 0
                    },
                    'num_groups': len(groups),
                    'group_names': [g.name for g in groups],
                    'mixing_type': mixing_type
                }
            }
            
            # Extract aggregated compartment data
            compartments = model.compartments if model else ['S', 'I', 'R']
            for comp in compartments:
                if comp in df.columns:
                    response['compartments'][comp] = df[comp].tolist()
            
            # Extract group-specific data from stratified dataframe
            if strat_df is not None:
                for i in range(len(groups)):
                    for comp in ['S', 'I', 'R']:
                        col = f"{comp}_group{i}"
                        if col in strat_df.columns:
                            response['groups'][comp].append(strat_df[col].tolist())
            
            return jsonify(response)
        
        else:
            # Standard (non-stratified) random model generation
            model = pipeline.generate_model(
                num_compartments=num_compartments,
                max_time_steps=max_time_steps
            )
            
            # Update time unit in params
            model.params['time_unit'] = time_unit
            
            # Run simulation
            df_weekly, df_daily = pipeline.run_simulation(
                model=model,
                visualize=False
            )
            
            # Format response
            use_daily = time_unit == 'days'
            df = df_daily if use_daily else df_weekly
            time_col = 'day' if use_daily else 'week'
            
            response = {
                'time': df[time_col].tolist() if time_col in df else list(range(len(df))),
                'compartments': {},
                'R_t': df['R_t'].tolist() if 'R_t' in df else [],
                'time_unit': time_unit,
                'model_info': {
                    'compartments': model.compartments,
                    'transitions': [
                        {
                            'source': t.source,
                            'target': t.target,
                            'type': t.dynamic_type,
                            'variant': t.variant
                        }
                        for t in model.transitions
                    ],
                    'params': {
                        'R0': model.params.get('R0', 0),
                        'infectious_period_days': model.params.get('infectious_period_days', 0),
                        'beta': model.params.get('beta', 0),
                        'gamma': model.params.get('gamma', 0),
                        'noise_level': model.noise_level
                    }
                }
            }
            
            # Extract compartment data
            for comp in model.compartments:
                if comp in df.columns:
                    response['compartments'][comp] = df[comp].tolist()
            
            return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/api/catalog', methods=['GET'])
def get_catalog():
    """Return catalog information"""
    return jsonify({
        'compartments': list(catalog.compartments.keys()),
        'compartment_details': {
            name: {
                'description': spec.get('description', ''),
                'type': spec.get('type', 'state'),
                'requires': spec.get('requires_compartment', []),
                'probability': spec.get('selection_probability', 0.5)
            }
            for name, spec in catalog.compartments.items()
        },
        'dynamics': list(catalog.dynamics.keys()),
        'observables': list(catalog.observables.keys())
    })


@app.route('/api/catalog/compartments', methods=['GET'])
def get_compartments():
    """Return detailed compartment information"""
    return jsonify({
        name: {
            'description': spec.get('description', ''),
            'type': spec.get('type', 'state'),
            'requires_compartment': spec.get('requires_compartment', []),
            'selection_probability': spec.get('selection_probability', 0.5),
            'mandatory': spec.get('mandatory', False)
        }
        for name, spec in catalog.compartments.items()
    })


@app.route('/api/catalog/dynamics', methods=['GET'])
def get_dynamics():
    """Return dynamics/transition rules information"""
    return jsonify({
        name: {
            'description': spec.get('description', ''),
            'variants': list(spec.get('variants', {}).keys())
        }
        for name, spec in catalog.dynamics.items()
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
