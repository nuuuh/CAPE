import numpy as np
import matplotlib.pyplot as plt
from old.pipeline import EpiRecipePipeline, Selection

def test_basic_sir():
    """Test 1: Basic SIR model - should show classic epidemic curve"""
    print("="*80)
    print("TEST 1: Basic SIR Model")
    print("="*80)
    
    pipeline = EpiRecipePipeline(rng_seed=42)
    
    # Create simple SIR model
    selection = Selection(
        compartments=['S', 'I', 'R'],
        transitions=['mass_action_infection', 'recovery_linear'],
        name='SIR'
    )
    
    config = {
        'population': 100000,
        'R0': 2.5,
        'gamma': 1/7,  # 7 day infectious period
        'initial_infected': 100,
        'simulation_weeks': 52,
        'time_step': 0.5
    }
    
    config = pipeline.configure(selection, overrides=config)
    df = pipeline.simulate(selection, config)
    
    # Verify properties
    print(f"Time points: {len(df)}")
    print(f"Final week: {df['week'].iloc[-1]:.1f}")
    print(f"Compartments: {[c for c in df.columns if c not in ['week', 'R_t']]}")
    
    # Check conservation of population (S + I + R should be constant)
    total = df['S'] + df['I'] + df['R']
    print(f"\nPopulation conservation:")
    print(f"  Initial total: {total.iloc[0]:.0f}")
    print(f"  Final total: {total.iloc[-1]:.0f}")
    print(f"  Max deviation: {(total.max() - total.min()):.2f} ({100*(total.max() - total.min())/total.iloc[0]:.4f}%)")
    
    # Check epidemic dynamics
    I_peak = df['I'].max()
    I_peak_time = df.loc[df['I'].idxmax(), 'week']
    print(f"\nEpidemic dynamics:")
    print(f"  Peak infected: {I_peak:.0f} ({100*I_peak/config['population']:.1f}%)")
    print(f"  Peak time: week {I_peak_time:.1f}")
    print(f"  Final infected: {df['I'].iloc[-1]:.0f}")
    print(f"  Final susceptible: {df['S'].iloc[-1]:.0f} ({100*df['S'].iloc[-1]/config['population']:.1f}%)")
    print(f"  Attack rate: {100*(config['population'] - df['S'].iloc[-1])/config['population']:.1f}%")
    
    # Check R_t
    print(f"\nReproduction number:")
    print(f"  R0 (configured): {config['R0']:.2f}")
    print(f"  R_t initial: {df['R_t'].iloc[0]:.2f}")
    print(f"  R_t at peak: {df.loc[df['I'].idxmax(), 'R_t']:.2f}")
    print(f"  R_t final: {df['R_t'].iloc[-1]:.2f}")
    
    # Verify R0 = beta / gamma
    beta_computed = config['beta']
    R0_from_params = beta_computed / config['gamma']
    print(f"  R0 from beta/gamma: {R0_from_params:.2f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(df['week'], df['S'], 'b-', linewidth=2, label='S (Susceptible)')
    axes[0].plot(df['week'], df['I'], 'r-', linewidth=2, label='I (Infected)')
    axes[0].plot(df['week'], df['R'], 'g-', linewidth=2, label='R (Recovered)')
    axes[0].set_xlabel('Time (weeks)', fontsize=12)
    axes[0].set_ylabel('Number of individuals', fontsize=12)
    axes[0].set_title('Basic SIR Model (R0=2.5)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['week'], df['R_t'], 'purple', linewidth=2)
    axes[1].axhline(1, color='red', linestyle='--', label='R_t = 1 (threshold)', alpha=0.7)
    axes[1].set_xlabel('Time (weeks)', fontsize=12)
    axes[1].set_ylabel('Effective reproduction number R(t)', fontsize=12)
    axes[1].set_title('Effective Reproduction Number Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, max(3, df['R_t'].max() * 1.1)])
    
    plt.tight_layout()
    plt.savefig('verify_sir_model.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved to verify_sir_model.png")
    
    # Validation checks
    issues = []
    if abs(total.max() - total.min()) > config['population'] * 0.01:
        issues.append("⚠️  Population not conserved (deviation > 1%)")
    if I_peak_time < 2 or I_peak_time > 30:
        issues.append(f"⚠️  Peak time seems unusual: {I_peak_time:.1f} weeks")
    if abs(df['R_t'].iloc[0] - config['R0']) > 0.1:
        issues.append(f"⚠️  Initial R_t ({df['R_t'].iloc[0]:.2f}) != R0 ({config['R0']:.2f})")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All checks passed!")
    
    return df


def test_seir():
    """Test 2: SEIR model with latent period"""
    print("\n" + "="*80)
    print("TEST 2: SEIR Model (with latent period)")
    print("="*80)
    
    pipeline = EpiRecipePipeline(rng_seed=43)
    
    selection = Selection(
        compartments=['S', 'E', 'I', 'R'],
        transitions=['mass_action_infection', 'incubation_progression', 'recovery_linear'],
        name='SEIR'
    )
    
    config = {
        'population': 100000,
        'R0': 3.0,
        'gamma': 1/7,  # 7 day infectious period
        'sigma': 1/5,  # 5 day incubation period
        'initial_infected': 10,
        'initial_exposed': 50,
        'simulation_weeks': 52,
        'time_step': 0.5
    }
    
    config = pipeline.configure(selection, overrides=config)
    df = pipeline.simulate(selection, config)
    
    print(f"Time points: {len(df)}")
    print(f"Compartments: {[c for c in df.columns if c not in ['week', 'R_t']]}")
    
    # Check population conservation (S + E + I + R)
    total = df['S'] + df['E'] + df['I'] + df['R']
    print(f"\nPopulation conservation:")
    print(f"  Initial: {total.iloc[0]:.0f}")
    print(f"  Final: {total.iloc[-1]:.0f}")
    print(f"  Deviation: {100*(total.max() - total.min())/total.iloc[0]:.4f}%")
    
    # Check dynamics
    E_peak = df['E'].max()
    I_peak = df['I'].max()
    E_peak_time = df.loc[df['E'].idxmax(), 'week']
    I_peak_time = df.loc[df['I'].idxmax(), 'week']
    
    print(f"\nEpidemic dynamics:")
    print(f"  Peak exposed: {E_peak:.0f} at week {E_peak_time:.1f}")
    print(f"  Peak infected: {I_peak:.0f} at week {I_peak_time:.1f}")
    print(f"  Delay (I peak - E peak): {I_peak_time - E_peak_time:.1f} weeks")
    print(f"  Expected delay (~incubation): {1/config['sigma']:.1f} days = {1/config['sigma']/7:.2f} weeks")
    
    issues = []
    if abs(total.max() - total.min()) > config['population'] * 0.01:
        issues.append("⚠️  Population not conserved")
    if I_peak_time <= E_peak_time:
        issues.append("⚠️  I peak should occur after E peak")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All checks passed!")
    
    return df


def test_complex_model():
    """Test 3: Complex model with multiple compartments"""
    print("\n" + "="*80)
    print("TEST 3: Complex Model (SEIHRD)")
    print("="*80)
    
    pipeline = EpiRecipePipeline(rng_seed=44)
    
    selection = Selection(
        compartments=['S', 'E', 'I', 'H', 'R', 'D'],
        transitions=[
            'mass_action_infection',
            'incubation_progression',
            'recovery_linear',
            'hospitalization',
            'hospital_recovery',
            'disease_induced_death'
        ],
        name='SEIHRD'
    )
    
    config = {
        'population': 100000,
        'R0': 2.0,
        'gamma': 1/10,  # 10 day infectious period
        'sigma': 1/5,   # 5 day incubation
        'eta': 0.05,    # 5% hospitalization rate
        'kappa': 1/14,  # 14 day hospital stay
        'mu': 0.01,     # 1% death rate
        'initial_infected': 50,
        'simulation_weeks': 52,
        'time_step': 0.5
    }
    
    config = pipeline.configure(selection, overrides=config)
    df = pipeline.simulate(selection, config)
    
    print(f"Time points: {len(df)}")
    
    # Check population (excluding deaths)
    living = df['S'] + df['E'] + df['I'] + df['H'] + df['R']
    total_with_deaths = living + df['D']
    
    print(f"\nPopulation tracking:")
    print(f"  Initial living: {living.iloc[0]:.0f}")
    print(f"  Final living: {living.iloc[-1]:.0f}")
    print(f"  Final deaths: {df['D'].iloc[-1]:.0f}")
    print(f"  Total (living + deaths): {total_with_deaths.iloc[-1]:.0f}")
    print(f"  Population conserved: {abs(total_with_deaths.iloc[-1] - config['population']) < 10}")
    
    # Check hospitalization
    H_peak = df['H'].max()
    I_peak = df['I'].max()
    print(f"\nHospitalization:")
    print(f"  Peak hospitalized: {H_peak:.0f}")
    print(f"  Peak infected: {I_peak:.0f}")
    print(f"  Hospital/Infected ratio: {H_peak/I_peak:.3f} (expect ~eta={config['eta']})")
    
    issues = []
    if abs(total_with_deaths.iloc[-1] - config['population']) > config['population'] * 0.01:
        issues.append("⚠️  Total population (living + dead) not conserved")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All checks passed!")
    
    return df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EPIRECIPE SIMULATION VERIFICATION")
    print("="*80)
    print("\nTesting core simulation functionality...\n")
    
    # Run tests
    sir_df = test_basic_sir()
    seir_df = test_seir()
    complex_df = test_complex_model()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nReview verify_sir_model.png to check if epidemic curves look correct.")
    print("Expected: S decreases, I peaks then declines, R increases and plateaus.")
