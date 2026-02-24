import torch
from torch import nn
import torch.nn.functional as F
from .TS_lib.SelfAttention_Family import FullAttention, AttentionLayer
from .TS_lib.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def reg_loss(z, logit, logit_0 = None):
        # import ipdb; ipdb.set_trace()
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, 1, logit.size(2))
        return torch.mean(torch.sum(torch.mul(z, log_pi), dim=-1))


def loss_proto_orth(EXO, proto_map):
    """
    Orthogonality loss for anchored compartment prototypes only.
    Forces mapped prototypes to be distinct while leaving others unconstrained.
    
    Args:
        EXO: prototype embeddings [K, D]
        proto_map: dict mapping compartment names to prototype indices
                  e.g., {'S': [0, 1], 'I': [2, 3], 'R': [4]}
    """
    idxs = sum(proto_map.values(), [])   # flatten list of anchored indices
    if len(idxs) < 2:
        return torch.tensor(0.0, device=EXO.device, dtype=EXO.dtype)
    
    G = EXO[idxs] @ EXO[idxs].t()
    I = torch.eye(len(idxs), device=EXO.device, dtype=EXO.dtype)
    return ((G - I)**2).mean()


def loss_sir_alignment(z_list, x_raw, proto_map, device, debug=False):
    """
    Compartment-specific behavioral alignment loss with strong epidemiological constraints.
    Encourages specific prototypes to align with epidemiological patterns and proper SIR dynamics.
    
    Args:
        z_list: list of attention distributions [B,1,L,K] from each layer
        x_raw: raw input data [B,L,1] for computing growth signals
        proto_map: dict mapping compartment names to prototype indices
        device: torch device
        debug: if True, print debugging information
    """
    if not z_list or len(z_list) == 0:
        return torch.tensor(0.0, device=device)
    
    # Concatenate attention distributions from all layers
    z = torch.cat([z for z in z_list if z is not None], dim=2).squeeze(1)  # [B,L,K]
    
    L_used = z.size(1)
    
    # Compute crude growth signal from raw input
    if x_raw.shape[-1] == 1:
        x_raw_squeezed = x_raw.squeeze(-1)  # [B, L]
    else:
        x_raw_squeezed = x_raw.mean(dim=-1)  # [B, L] if multiple features
    
    # Handle the case where we have patches instead of raw time series
    if x_raw_squeezed.size(1) != L_used:
        # If input is in patch format, we need to handle it differently
        if x_raw_squeezed.size(1) < L_used:
            # Repeat the last value to match length
            padding_size = L_used - x_raw_squeezed.size(1)
            padding = x_raw_squeezed[:, -1:].repeat(1, padding_size)
            x_raw_squeezed = torch.cat([x_raw_squeezed, padding], dim=1)
        else:
            # Truncate to match
            x_raw_squeezed = x_raw_squeezed[:, :L_used]
    
    # Compute growth signal with proper prepend dimension
    prepend_val = x_raw_squeezed[:, :1]  # [B, 1] - same dimensionality as each time step
    dx = torch.sign(torch.diff(x_raw_squeezed, dim=1, prepend=prepend_val))
    
    loss = torch.tensor(0.0, device=device)
    
    # Extract compartment responsibilities for validation
    s_responsibilities = []
    i_responsibilities = []
    r_responsibilities = []
    
    for s_idx in proto_map.get('S', []):
        if s_idx < z.size(-1):
            s_responsibilities.append(z[:,:,s_idx])
    
    for i_idx in proto_map.get('I', []):
        if i_idx < z.size(-1):
            i_responsibilities.append(z[:,:,i_idx])
    
    for r_idx in proto_map.get('R', []):
        if r_idx < z.size(-1):
            r_responsibilities.append(z[:,:,r_idx])
    
    # Aggregate compartment responsibilities
    if s_responsibilities:
        s_total = torch.stack(s_responsibilities, dim=-1).sum(dim=-1)  # [B, L]
    else:
        s_total = torch.zeros(z.size(0), L_used, device=device)
        
    if i_responsibilities:
        i_total = torch.stack(i_responsibilities, dim=-1).sum(dim=-1)  # [B, L]
    else:
        i_total = torch.zeros(z.size(0), L_used, device=device)
        
    if r_responsibilities:
        r_total = torch.stack(r_responsibilities, dim=-1).sum(dim=-1)  # [B, L]
    else:
        r_total = torch.zeros(z.size(0), L_used, device=device)
    
    # 1. STRONG SIR INITIAL CONDITIONS
    # S should start very high (80-95%)
    s_initial_target = 0.85
    s_initial_loss = F.mse_loss(s_total[:,0], torch.full_like(s_total[:,0], s_initial_target))
    loss += s_initial_loss * 20.0  # Very strong weight
    
    # I should start very low (0-5%)  
    i_initial_target = 0.05
    i_initial_loss = F.mse_loss(i_total[:,0], torch.full_like(i_total[:,0], i_initial_target))
    loss += i_initial_loss * 10.0  # Very strong weight
    
    # R should start near zero (0-2%)
    r_initial_target = 0.02
    r_initial_loss = F.mse_loss(r_total[:,0], torch.full_like(r_total[:,0], r_initial_target))
    loss += r_initial_loss * 25.0
    
    # 2. COMPARTMENT BUDGET CONSTRAINT (S + I + R ≈ 1)
    total_mass = s_total + i_total + r_total
    budget_loss = F.mse_loss(total_mass, torch.ones_like(total_mass))
    loss += budget_loss * 10.0
    
    # 3. PREVENT I COMPARTMENT DOMINATION
    i_domination_penalty = F.relu(i_total - 0.5).mean()
    # loss += i_domination_penalty * 8.0
    
    # Encourage I to stay within epidemic range (0.02 to 0.4)
    # i_range_penalty = F.relu(0.02 - i_total).mean() + F.relu(i_total - 0.4).mean()
    # loss += i_range_penalty * 3.0
    
    # 4. ENHANCED SUSCEPTIBLE DYNAMICS
    if L_used > 1:
        # Strong monotonic decrease for S
        for t in range(1, L_used):
            s_decrease = F.relu(s_total[:, t] - s_total[:, t-1] + 0.01)  # Allow small increases
            loss += s_decrease.mean() * 8.0
        
        # # Overall S should decrease significantly (20-60% reduction)
        # s_reduction = s_total[:, 0] - s_total[:, -1]
        # s_reduction_target = s_total[:, 0] * 0.4  # Target 40% reduction
        # s_reduction_loss = F.mse_loss(s_reduction, s_reduction_target)
        # loss += s_reduction_loss * 2.0
    
    # 5. ENHANCED INFECTIOUS DYNAMICS - Bell Curve Pattern
    if L_used >= 6:  # Need sufficient length for curve analysis
        # Divide timeline into phases for bell curve
        early_phase = L_used // 4
        mid_start = L_used // 4
        mid_end = 3 * L_used // 4
        late_phase = 3 * L_used // 4
        
        i_early = i_total[:, :early_phase].mean(dim=1)
        i_mid = i_total[:, mid_start:mid_end].mean(dim=1) 
        i_late = i_total[:, late_phase:].mean(dim=1)
        
        # Bell curve constraints: early < mid and late < mid
        bell_loss1 = F.relu(i_early - i_mid + 0.05).mean()  # Rise phase
        bell_loss2 = F.relu(i_late - i_mid + 0.05).mean()   # Fall phase
        loss += (bell_loss1 + bell_loss2) * 25.0
        
        # # Peak should be reasonable (10-40%)
        # peak_target = 0.25
        # peak_loss = F.mse_loss(i_mid, torch.full_like(i_mid, peak_target))
        # loss += peak_loss * 2.0
    
    # 6. ENHANCED RECOVERED DYNAMICS
    if L_used > 1:
        # Strict monotonic increase for R
        for t in range(1, L_used):
            r_increase = F.relu(r_total[:, t-1] - r_total[:, t] + 0.005)  # Very small tolerance
            loss += r_increase.mean() * 4.0
        
        # # R should reach significant final value (30-70%)
        # r_final_target = 0.5
        # r_final_loss = F.mse_loss(r_total[:, -1], torch.full_like(r_total[:, -1], r_final_target))
        # loss += r_final_loss * 2.0 # Penalize I when it exceeds reasonable epidemic levels (>0.5)
    #
    
    # 7. TEMPORAL CONSISTENCY AND SMOOTHNESS
    if L_used > 2:
        # Prevent abrupt changes (epidemics are gradual)
        s_smoothness = ((s_total[:, 2:] - 2*s_total[:, 1:-1] + s_total[:, :-2]) ** 2).mean()
        i_smoothness = ((i_total[:, 2:] - 2*i_total[:, 1:-1] + i_total[:, :-2]) ** 2).mean()
        r_smoothness = ((r_total[:, 2:] - 2*r_total[:, 1:-1] + r_total[:, :-2]) ** 2).mean()
        loss += (s_smoothness + i_smoothness + r_smoothness) * 1
    
    # 8. DEBUG INFORMATION
    if debug and torch.rand(1).item() < 0.01:  # Print occasionally
        print(f"\nCompartment Analysis:")
        print(f"S: initial={s_total[0,0]:.3f}, final={s_total[0,-1]:.3f}, mean={s_total[0].mean():.3f}")
        print(f"I: initial={i_total[0,0]:.3f}, peak={i_total[0].max():.3f}, final={i_total[0,-1]:.3f}")
        print(f"R: initial={r_total[0,0]:.3f}, final={r_total[0,-1]:.3f}, mean={r_total[0].mean():.3f}")
        print(f"Budget (S+I+R): {(s_total[0] + i_total[0] + r_total[0]).mean():.3f}")
    
    return loss


def loss_seir_alignment(z_list, x_raw, proto_map, device, debug=False):
    """
    Revised SEIR-specific compartment alignment loss that focuses on realistic epidemic patterns.
    
    Args:
        z_list: list of attention distributions [B,1,L,K] from each layer
        x_raw: raw input data [B,L,1] for computing growth signals
        proto_map: dict mapping compartment names to prototype indices
        device: torch device
        debug: if True, print debugging information
    """
    if not z_list or len(z_list) == 0:
        return torch.tensor(0.0, device=device)
    
    # Concatenate attention distributions from all layers
    z = torch.cat([z for z in z_list if z is not None], dim=2).squeeze(1)  # [B,L,K]
    L_used = z.size(1)
    
    loss = torch.tensor(0.0, device=device)
    
    # Extract compartment responsibilities
    s_total = torch.zeros(z.size(0), L_used, device=device)
    e_total = torch.zeros(z.size(0), L_used, device=device)
    i_total = torch.zeros(z.size(0), L_used, device=device)
    r_total = torch.zeros(z.size(0), L_used, device=device)
    
    for s_idx in proto_map.get('S', []):
        if s_idx < z.size(-1):
            s_total += z[:,:,s_idx]
    
    for e_idx in proto_map.get('E', []):
        if e_idx < z.size(-1):
            e_total += z[:,:,e_idx]
    
    for i_idx in proto_map.get('I', []):
        if i_idx < z.size(-1):
            i_total += z[:,:,i_idx]
            
    for r_idx in proto_map.get('R', []):
        if r_idx < z.size(-1):
            r_total += z[:,:,r_idx]
    
    # 1. BUDGET CONSTRAINT (S + E + I + R ≈ 1) - Most fundamental constraint
    total_mass = s_total + e_total + i_total + r_total
    budget_loss = F.mse_loss(total_mass, torch.ones_like(total_mass))
    loss += budget_loss * 10.0
    
    # 2. INITIAL CONDITIONS - Match actual ground truth patterns
    s_initial_target = 0.95  # Start with 95% susceptible (matches ground truth ~950/1000)
    e_initial_target = 0.001  # Almost zero initial exposed population  
    i_initial_target = 0.001  # Almost zero initial infectious population  
    r_initial_target = 0.001  # Almost zero recovered (epidemic hasn't started)
    
    initial_loss = (
        F.mse_loss(s_total[:,0], torch.full_like(s_total[:,0], s_initial_target)) * 50.0 +  # VERY strong S penalty
        F.mse_loss(e_total[:,0], torch.full_like(e_total[:,0], e_initial_target)) * 30.0 +  # E must start near zero
        F.mse_loss(i_total[:,0], torch.full_like(i_total[:,0], i_initial_target)) * 30.0 +  # I must start near zero
        F.mse_loss(r_total[:,0], torch.full_like(r_total[:,0], r_initial_target)) * 50.0    # R must start near zero
    )
    loss += initial_loss
    
    # 3. MONOTONICITY CONSTRAINTS - Strong enforcement
    if L_used > 2:
        # S should strictly decrease (stronger enforcement)
        s_increases = F.relu(torch.diff(s_total, dim=1))
        s_monotone_loss = s_increases.mean()
        loss += s_monotone_loss * 15.0  # Much stronger penalty
        
        # R should strictly increase (stronger enforcement)
        r_decreases = F.relu(-torch.diff(r_total, dim=1))
        r_monotone_loss = r_decreases.mean()
        loss += r_monotone_loss * 15.0  # Much stronger penalty
        
        # Additional: R should end higher than it starts
        r_growth_loss = F.relu(r_total[:, 0] - r_total[:, -1] + 0.1)  # R must grow by at least 10%
        loss += r_growth_loss.mean() * 10.0
        
        # Additional: S should end lower than it starts  
        s_depletion_loss = F.relu(s_total[:, -1] - s_total[:, 0] + 0.1)  # S must decrease by at least 10%
        loss += s_depletion_loss.mean() * 10.0
    
    # 4. EPIDEMIC CURVE DYNAMICS - Focus on realistic shapes
    if L_used >= 10:
        # Divide timeline into phases
        early_phase = L_used // 4      # First 25%
        mid_phase = L_used // 2        # Middle 50%
        late_phase = 3 * L_used // 4   # Last 25%
        
        # E should show epidemic curve: low -> peak -> decline
        e_early = e_total[:, :early_phase].mean(dim=1)
        e_mid = e_total[:, early_phase:mid_phase].mean(dim=1) 
        e_late = e_total[:, mid_phase:].mean(dim=1)
        
        # Encourage E to peak in middle phase
        e_peak_loss = F.relu(e_early - e_mid).mean() + F.relu(e_late - e_mid).mean()
        loss += e_peak_loss * 8.0
        
        # I should show epidemic curve: low -> peak -> decline (slightly delayed from E)
        i_early = i_total[:, :mid_phase//2].mean(dim=1)
        i_mid = i_total[:, mid_phase//2:late_phase].mean(dim=1)
        i_late = i_total[:, late_phase:].mean(dim=1)
        
        # STRONG I epidemic curve enforcement
        # I must start low
        i_initial_low = F.relu(i_total[:, 0] - 0.1).mean()  # I should start < 10%
        loss += i_initial_low * 20.0
        
        # I must peak in middle phase (not start high)
        i_peak_loss = F.relu(i_early - i_mid).mean() + F.relu(i_late - i_mid).mean()
        loss += i_peak_loss * 15.0  # Stronger weight
        
        # I must have clear growth in first half
        i_first_half = i_total[:, :L_used//2]
        for t in range(1, i_first_half.size(1)):
            i_growth_penalty = F.relu(i_first_half[:, t-1] - i_first_half[:, t]).mean()
            loss += i_growth_penalty * 5.0  # Penalize decreases in first half
        
        # I must have clear decline in second half  
        i_second_half = i_total[:, L_used//2:]
        for t in range(1, i_second_half.size(1)):
            i_decline_penalty = F.relu(i_second_half[:, t] - i_second_half[:, t-1]).mean()
            loss += i_decline_penalty * 5.0  # Penalize increases in second half
        
        # E should peak before I (temporal ordering)
        e_peak_time = torch.argmax(e_total, dim=1).float()
        i_peak_time = torch.argmax(i_total, dim=1).float()
        temporal_order_loss = F.relu(e_peak_time - i_peak_time + 3.0).mean()
        loss += temporal_order_loss * 6.0
    
    # 5. REASONABLE MAGNITUDE CONSTRAINTS
    # Prevent any compartment from becoming too dominant
    s_max_loss = F.relu(s_total - 0.95).mean()  # S shouldn't exceed 95%
    e_max_loss = F.relu(e_total - 0.3).mean()   # E shouldn't exceed 30%
    i_max_loss = F.relu(i_total - 0.4).mean()   # I shouldn't exceed 40%
    
    magnitude_loss = (s_max_loss + e_max_loss + i_max_loss) * 10.0
    loss += magnitude_loss
    
    # 6. ENSURE SUFFICIENT DYNAMICS - Prevent flat lines
    if L_used > 5:
        # Each compartment should have some variation
        s_variation = s_total.std(dim=1).mean()
        e_variation = e_total.std(dim=1).mean()
        i_variation = i_total.std(dim=1).mean()
        r_variation = r_total.std(dim=1).mean()
        
        # Encourage variation but not too much
        variation_loss = (
            F.relu(0.05 - s_variation) * 3.0 +  # S should vary at least 5%
            F.relu(0.02 - e_variation) * 5.0 +  # E should vary at least 2%
            F.relu(0.02 - i_variation) * 5.0 +  # I should vary at least 2%
            F.relu(0.03 - r_variation) * 3.0    # R should vary at least 3%
        )
        loss += variation_loss
    
    # 7. SMOOTH TRANSITIONS - Prevent abrupt changes
    if L_used > 3:
        # Calculate gradients and penalize large jumps
        s_grad = torch.abs(torch.diff(s_total, dim=1))
        e_grad = torch.abs(torch.diff(e_total, dim=1))
        i_grad = torch.abs(torch.diff(i_total, dim=1))
        r_grad = torch.abs(torch.diff(r_total, dim=1))
        
        # Limit step sizes
        smoothness_loss = (
            F.relu(s_grad - 0.05).mean() * 2.0 +  # S changes ≤ 5% per step
            F.relu(e_grad - 0.03).mean() * 3.0 +  # E changes ≤ 3% per step
            F.relu(i_grad - 0.03).mean() * 3.0 +  # I changes ≤ 3% per step
            F.relu(r_grad - 0.05).mean() * 2.0    # R changes ≤ 5% per step
        )
        loss += smoothness_loss*2
    
    # 8. FINAL STATE CONSTRAINTS - Much stronger enforcement to prevent inversion
    # After epidemic, expect: much lower S, near-zero E/I, much higher R
    s_final_loss = F.relu(s_total[:, -1] - s_total[:, 0] + 0.4).mean()  # S should decrease by at least 40%
    r_final_loss = F.relu(r_total[:, 0] - r_total[:, -1] + 0.4).mean()  # R should increase by at least 40%
    e_final_loss = F.relu(e_total[:, -1] - 0.05).mean()  # E should be very low at end
    i_final_loss = F.relu(i_total[:, -1] - 0.05).mean()  # I should be very low at end
    
    final_state_loss = (s_final_loss + r_final_loss + e_final_loss + i_final_loss) * 20.0  # Much stronger
    loss += final_state_loss
    
    # 9. PREVENT INVERSION - Very strong constraints to prevent R/S swap
    # R should NEVER dominate early (first 25% of timeline)
    r_early_dominance = F.relu(r_total[:, :L_used//4].mean(dim=1) - 0.1).mean()  # R < 10% early
    loss += r_early_dominance * 100.0  # Huge penalty
    
    # S should ALWAYS dominate early (first 25% of timeline)  
    s_early_minority = F.relu(0.8 - s_total[:, :L_used//4].mean(dim=1)).mean()  # S > 80% early
    loss += s_early_minority * 100.0  # Huge penalty
    
    # Additional: R should NEVER exceed S until late in epidemic
    mid_timeline = L_used // 2
    r_exceeds_s_early = F.relu(r_total[:, :mid_timeline] - s_total[:, :mid_timeline]).mean()
    loss += r_exceeds_s_early * 80.0  # Very strong penalty
    
    if debug and torch.rand(1).item() < 0.01:
        print(f"\nRevised SEIR Analysis:")
        print(f"S: init={s_total[0,0]:.3f}, final={s_total[0,-1]:.3f}, change={s_total[0,-1]-s_total[0,0]:.3f}")
        print(f"E: init={e_total[0,0]:.3f}, peak={e_total[0].max():.3f}, final={e_total[0,-1]:.3f}")
        print(f"I: init={i_total[0,0]:.3f}, peak={i_total[0].max():.3f}, final={i_total[0,-1]:.3f}")
        print(f"R: init={r_total[0,0]:.3f}, final={r_total[0,-1]:.3f}, change={r_total[0,-1]-r_total[0,0]:.3f}")
        print(f"Total mass: {(s_total[0] + e_total[0] + i_total[0] + r_total[0]).mean():.3f}")
    
    return loss


def loss_sird_alignment(z_list, x_raw, proto_map, device, debug=False):
    """
    SIRD-specific compartment alignment loss with epidemiological constraints.
    Handles Susceptible-Infectious-Recovered-Dead dynamics.
    
    Args:
        z_list: list of attention distributions [B,1,L,K] from each layer
        x_raw: raw input data [B,L,1] for computing growth signals
        proto_map: dict mapping compartment names to prototype indices
                  e.g., {'S': [0], 'I': [1], 'R': [2], 'D': [3]}
        device: torch device
        debug: if True, print debugging information
    """
    if not z_list or len(z_list) == 0:
        return torch.tensor(0.0, device=device)
    
    # Concatenate attention distributions from all layers
    z = torch.cat([z for z in z_list if z is not None], dim=2).squeeze(1)  # [B,L,K]
    L_used = z.size(1)
    
    loss = torch.tensor(0.0, device=device)
    
    # Extract compartment responsibilities
    s_total = torch.zeros(z.size(0), L_used, device=device)
    i_total = torch.zeros(z.size(0), L_used, device=device)
    r_total = torch.zeros(z.size(0), L_used, device=device)
    d_total = torch.zeros(z.size(0), L_used, device=device)
    
    for s_idx in proto_map.get('S', []):
        if s_idx < z.size(-1):
            s_total += z[:,:,s_idx]
    
    for i_idx in proto_map.get('I', []):
        if i_idx < z.size(-1):
            i_total += z[:,:,i_idx]
            
    for r_idx in proto_map.get('R', []):
        if r_idx < z.size(-1):
            r_total += z[:,:,r_idx]
            
    for d_idx in proto_map.get('D', []):
        if d_idx < z.size(-1):
            d_total += z[:,:,d_idx]
    
    # 1. SIRD INITIAL CONDITIONS
    # S starts very high (80-90%)
    s_initial_loss = F.mse_loss(s_total[:,0], torch.full_like(s_total[:,0], 0.8))  # Higher target
    loss += s_initial_loss * 20.0  # Stronger constraint
    
    # I starts very low (1-3%)
    i_initial_loss = F.mse_loss(i_total[:,0], torch.full_like(i_total[:,0], 0.02))  # Lower target
    loss += i_initial_loss * 5.0
    
    # R starts near zero
    r_initial_loss = F.mse_loss(r_total[:,0], torch.full_like(r_total[:,0], 0.01))
    loss += r_initial_loss * 20.0
    
    # D starts at zero
    d_initial_loss = F.mse_loss(d_total[:,0], torch.full_like(d_total[:,0], 0.0))
    loss += d_initial_loss * 15.0  # Very strong constraint for deaths
    
    # 2. BUDGET CONSTRAINT (S + I + R + D ≈ 1)
    total_mass = s_total + i_total + r_total + d_total
    budget_loss = F.mse_loss(total_mass, torch.ones_like(total_mass))
    loss += budget_loss * 10.0
    
    # 3. SIRD FLOW DYNAMICS
    if L_used > 1:
        # S should decrease monotonically and significantly
        s_decrease_loss = sum(F.relu(s_total[:, t] - s_total[:, t-1]).mean() 
                             for t in range(1, L_used))
        loss += s_decrease_loss * 6.0
        
        # S depletion requirement
        s_final_depletion = (s_total[:, 0] - s_total[:, -1]).mean()
        s_depletion_target = 0.4  # Should lose 40% of susceptibles
        s_depletion_loss = F.mse_loss(s_depletion_target * torch.ones_like(s_final_depletion), s_final_depletion)
        loss += s_depletion_loss * 10.0
        
        # Enhanced I epidemic curve dynamics
        if L_used >= 6:
            early_phase = L_used // 3
            mid_start = L_used // 3
            mid_end = 2 * L_used // 3
            
            i_early = i_total[:, :early_phase].mean(dim=1)
            i_mid = i_total[:, mid_start:mid_end].mean(dim=1)
            i_late = i_total[:, mid_end:].mean(dim=1)
            
            # Strong I epidemic curve requirements
            i_rise_loss = F.relu(i_early - i_mid + 0.08).mean()  # Must rise significantly
            i_fall_loss = F.relu(i_late - i_mid + 0.04).mean()   # Must fall from peak
            loss += (i_rise_loss + i_fall_loss) * 15.0
            
            # I peak magnitude
            i_peak_target = 0.25  # Target 25% peak
            i_peak_loss = F.mse_loss(i_mid, torch.full_like(i_mid, i_peak_target))
            loss += i_peak_loss * 8.0
            
            # I variation requirement
            i_variation = i_total.std(dim=1).mean()
            i_var_target = 0.08  # Target high variation
            i_var_loss = F.mse_loss(i_variation, torch.tensor(i_var_target, device=i_total.device))
            loss += i_var_loss * 12.0
        
        # R should increase monotonically and reach substantial final value
        r_increase_loss = sum(F.relu(r_total[:, t-1] - r_total[:, t]).mean() 
                             for t in range(1, L_used))
        loss += r_increase_loss * 10.0
        
        # R final target
        r_final_target = 0.4  # Target 40% recovered
        r_final_loss = F.mse_loss(r_total[:, -1], torch.full_like(r_total[:, -1], r_final_target))
        loss += r_final_loss * 10.0
        
        # D should increase monotonically (deaths are cumulative)
        d_increase_loss = sum(F.relu(d_total[:, t-1] - d_total[:, t]).mean() 
                             for t in range(1, L_used))
        loss += d_increase_loss * 10.0  # Strong constraint for death monotonicity
        
        # D should stay relatively low (case fatality rate constraint)
        d_cfr_penalty = F.relu(d_total - 0.05).mean()  # Deaths shouldn't exceed 12%
        loss += d_cfr_penalty * 25.0
    
    # 4. PREVENT INAPPROPRIATE DOMINATION
    domination_penalty = (
        F.relu(i_total - 0.5).mean() * 8.0 +   # I shouldn't dominate
        F.relu(d_total - 0.15).mean() * 15.0   # D should stay low
    )
    loss += domination_penalty
    
    # 5. ENHANCED SMOOTHNESS CONSTRAINTS
    if L_used > 2:
        # Calculate temporal derivatives (rate of change)
        s_deriv = torch.diff(s_total, dim=1)
        i_deriv = torch.diff(i_total, dim=1)
        r_deriv = torch.diff(r_total, dim=1)
        d_deriv = torch.diff(d_total, dim=1)
        
        # Second derivatives (acceleration) - should be small for smooth curves
        if L_used > 3:
            s_accel = torch.diff(s_deriv, dim=1)
            i_accel = torch.diff(i_deriv, dim=1)
            r_accel = torch.diff(r_deriv, dim=1)
            d_accel = torch.diff(d_deriv, dim=1)
            
            # Penalize large accelerations (promotes smoothness)
            smoothness_loss = (
                torch.abs(s_accel).mean() * 2.0 +
                torch.abs(i_accel).mean() * 3.0 +
                torch.abs(r_accel).mean() * 2.0 +
                torch.abs(d_accel).mean() * 3.0  # D should be very smooth
            )
            loss += smoothness_loss
        
        # Limit maximum step changes (no sudden jumps)
        max_step_s = 0.08  # S can change up to 8% per step
        max_step_i = 0.08  # I can change up to 8% per step  
        max_step_r = 0.08  # R can change up to 8% per step
        max_step_d = 0.04  # D should change more gradually
        
        step_penalty = (
            F.relu(torch.abs(s_deriv) - max_step_s).mean() * 8.0 +
            F.relu(torch.abs(i_deriv) - max_step_i).mean() * 10.0 +
            F.relu(torch.abs(r_deriv) - max_step_r).mean() * 8.0 +
            F.relu(torch.abs(d_deriv) - max_step_d).mean() * 12.0
        )
        loss += step_penalty
    
    if debug and torch.rand(1).item() < 0.01:
        print(f"\nSIRD Analysis:")
        print(f"S: init={s_total[0,0]:.3f}, final={s_total[0,-1]:.3f}")
        print(f"I: init={i_total[0,0]:.3f}, peak={i_total[0].max():.3f}")
        print(f"R: init={r_total[0,0]:.3f}, final={r_total[0,-1]:.3f}")
        print(f"D: init={d_total[0,0]:.3f}, final={d_total[0,-1]:.3f}")
    
    return loss


def loss_generic_alignment(z_list, x_raw, proto_map, device, model_type='SR', debug=False):
    """
    Generic compartment alignment loss that routes to specific model loss functions.
    
    Args:
        z_list: list of attention distributions [B,1,L,K] from each layer
        x_raw: raw input data [B,L,1] for computing growth signals
        proto_map: dict mapping compartment names to prototype indices
        device: torch device
        model_type: str, one of 'SIR', 'SEIR', 'SIRD', 'SEIRD'
        debug: if True, print debugging information
    """
    if model_type.upper() == 'SIR':
        return loss_sir_alignment(z_list, x_raw, proto_map, device, debug)
    elif model_type.upper() == 'SEIR':
        return loss_seir_alignment(z_list, x_raw, proto_map, device, debug)
    elif model_type.upper() == 'SIRD':
        return loss_sird_alignment(z_list, x_raw, proto_map, device, debug)
    elif model_type.upper() == 'SEIRD':
        # For SEIRD, combine SEIR and D dynamics
        return loss_seird_alignment(z_list, x_raw, proto_map, device, debug)
    else:
        # Fallback to SI
        return loss_compartment_alignment(z_list, x_raw, proto_map, device, debug)


def loss_compartment_alignment(z_list, x_raw, proto_map, device, debug=False):
    """
    Simplified compartment alignment loss focusing on S, I, and R compartments.
    Implements basic epidemiological constraints for three-compartment dynamics.
    
    Args:
        z_list: list of attention distributions [B,1,L,K] from each layer
        x_raw: raw input data [B,L,1] for computing growth signals
        proto_map: mapping of prototypes to compartments
        device: torch device
        debug: whether to print debug information
    
    Returns:
        dict: alignment loss components
    """
    if not z_list:
        return {'alignment_loss': torch.tensor(0.0, device=device, requires_grad=True)}
    
    # Aggregate attention across layers
    z_agg = torch.stack(z_list, dim=0).mean(dim=0)  # [B,1,L,K]
    B, _, L, K = z_agg.shape
    
    # Initialize loss components
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    loss_components = {}
    
    # Map prototypes to compartments
    S_idx = proto_map.get('S', 0)
    R_idx = proto_map.get('R', 1)
    
    # Extract compartment attention patterns
    z_S = z_agg[:, :, :, S_idx]  # [B,1,L]
    z_R = z_agg[:, :, :, R_idx]  # [B,1,L]
    
    # Squeeze to [B,L] for easier computation
    z_S = z_S.squeeze(1)
    z_R = z_R.squeeze(1)
    
    # 1. Budget constraint (S + I + R should be reasonable but flexible)
    # Instead of fixed target, use adaptive constraint based on data range
    budget_sum = z_S + z_R
    # Allow budget to vary but penalize extreme values
    budget_penalty = F.relu(budget_sum - 1.0).mean() + F.relu(0.1 - budget_sum).mean()
    total_loss = total_loss + 0.5 * budget_penalty
    loss_components['budget_penalty'] = budget_penalty.item()
    
    # 2. S compartment dynamics (monotonic decrease)
    if L > 1:
        S_diff = torch.diff(z_S, dim=1)
        # Penalize increases in S (should only decrease)
        S_increase_penalty = F.relu(S_diff).mean()
        total_loss = total_loss + 1.0 * S_increase_penalty
        loss_components['S_monotonic'] = S_increase_penalty.item()
    
    # 3. R compartment dynamics (monotonic increase)
    if L > 1:
        R_diff = torch.diff(z_R, dim=1)
        # Penalize decreases in R (should only increase)
        R_decrease_penalty = F.relu(-R_diff).mean()
        total_loss = total_loss + 1.0 * R_decrease_penalty
        loss_components['R_monotonic'] = R_decrease_penalty.item()
    
    # 4. Smoothness penalty
    if L > 1:
        S_smoothness = torch.diff(z_S, dim=1).pow(2).mean()
        R_smoothness = torch.diff(z_R, dim=1).pow(2).mean()
        smoothness_loss = S_smoothness + R_smoothness
        total_loss = total_loss + 0.5 * smoothness_loss
        loss_components['smoothness'] = smoothness_loss.item()
    
    # 5. Prevent extreme values
    extreme_penalty = F.relu(z_S - 1.0).mean() + F.relu(-z_S).mean() + \
                     F.relu(z_R - 1.0).mean() + F.relu(-z_R).mean()
    total_loss = total_loss + 2.0 * extreme_penalty
    loss_components['extreme_penalty'] = extreme_penalty.item()
    
    if debug:
        print(f"\nCompartment Alignment Loss Components:")
        for key, value in loss_components.items():
            print(f"  {key}: {value:.6f}")
        print(f"  Total: {total_loss.item():.6f}")
    
    loss_components['alignment_loss'] = total_loss
    return loss_components


def loss_seird_alignment(z_list, x_raw, proto_map, device, debug=False):
    """
    SEIRD-specific compartment alignment loss with epidemiological constraints.
    Handles Susceptible-Exposed-Infectious-Recovered-Dead dynamics.
    
    Args:
        z_list: list of attention distributions [B,1,L,K] from each layer
        x_raw: raw input data [B,L,1] for computing growth signals
        proto_map: dict mapping compartment names to prototype indices
                  e.g., {'S': [0], 'E': [1], 'I': [2], 'R': [3], 'D': [4]}
        device: torch device
        debug: if True, print debugging information
    """
    if not z_list or len(z_list) == 0:
        return torch.tensor(0.0, device=device)
    
    # Concatenate attention distributions from all layers
    z = torch.cat([z for z in z_list if z is not None], dim=2).squeeze(1)  # [B,L,K]
    L_used = z.size(1)
    
    loss = torch.tensor(0.0, device=device)
    
    # Extract compartment responsibilities
    s_total = torch.zeros(z.size(0), L_used, device=device)
    e_total = torch.zeros(z.size(0), L_used, device=device)
    i_total = torch.zeros(z.size(0), L_used, device=device)
    r_total = torch.zeros(z.size(0), L_used, device=device)
    d_total = torch.zeros(z.size(0), L_used, device=device)
    
    for s_idx in proto_map.get('S', []):
        if s_idx < z.size(-1):
            s_total += z[:,:,s_idx]
    
    for e_idx in proto_map.get('E', []):
        if e_idx < z.size(-1):
            e_total += z[:,:,e_idx]
    
    for i_idx in proto_map.get('I', []):
        if i_idx < z.size(-1):
            i_total += z[:,:,i_idx]
            
    for r_idx in proto_map.get('R', []):
        if r_idx < z.size(-1):
            r_total += z[:,:,r_idx]
            
    for d_idx in proto_map.get('D', []):
        if d_idx < z.size(-1):
            d_total += z[:,:,d_idx]
    
    # Combine SEIR and SIRD constraints
    # 1. Initial conditions
    s_initial_loss = F.mse_loss(s_total[:,0], torch.full_like(s_total[:,0], 0.85))
    e_initial_loss = F.mse_loss(e_total[:,0], torch.full_like(e_total[:,0], 0.02))
    i_initial_loss = F.mse_loss(i_total[:,0], torch.full_like(i_total[:,0], 0.01))
    r_initial_loss = F.mse_loss(r_total[:,0], torch.full_like(r_total[:,0], 0.01))
    d_initial_loss = F.mse_loss(d_total[:,0], torch.full_like(d_total[:,0], 0.0))
    
    loss += (s_initial_loss + e_initial_loss + i_initial_loss + r_initial_loss + d_initial_loss) * 8.0
    
    # 2. Budget constraint
    total_mass = s_total + e_total + i_total + r_total + d_total
    budget_loss = F.mse_loss(total_mass, torch.ones_like(total_mass))
    loss += budget_loss * 5.0
    
    # 3. Flow dynamics (combine SEIR and death processes)
    if L_used > 3:
        # Monotonic constraints
        for t in range(1, L_used):
            s_decrease = F.relu(s_total[:, t] - s_total[:, t-1] + 0.01)
            r_increase = F.relu(r_total[:, t-1] - r_total[:, t] + 0.005)
            d_increase = F.relu(d_total[:, t-1] - d_total[:, t] + 0.001)
            loss += (s_decrease.mean() * 3.0 + r_increase.mean() * 2.0 + d_increase.mean() * 5.0)
        
        # E and I epidemic curves (as in SEIR)
        if L_used >= 8:
            # E incubation dynamics
            e_third = L_used // 3
            e_early = e_total[:, :e_third].mean(dim=1)
            e_mid = e_total[:, e_third:2*e_third].mean(dim=1)
            e_late = e_total[:, 2*e_third:].mean(dim=1)
            
            e_rise_loss = F.relu(e_early - e_mid + 0.03).mean()
            e_fall_loss = F.relu(e_late - e_mid + 0.03).mean()
            loss += (e_rise_loss + e_fall_loss) * 2.0
            
            # I epidemic dynamics
            i_start = L_used // 4
            i_peak = 2 * L_used // 3
            
            i_early = i_total[:, :i_start].mean(dim=1)
            i_mid = i_total[:, i_start:i_peak].mean(dim=1)
            i_late = i_total[:, i_peak:].mean(dim=1)
            
            i_rise_loss = F.relu(i_early - i_mid + 0.05).mean()
            i_fall_loss = F.relu(i_late - i_mid + 0.05).mean()
            loss += (i_rise_loss + i_fall_loss) * 3.0
    
    # 4. Death rate constraints
    d_cfr_penalty = F.relu(d_total - 0.1).mean()  # Deaths shouldn't exceed 10%
    loss += d_cfr_penalty * 10.0
    
    if debug and torch.rand(1).item() < 0.01:
        print(f"\nSEIRD Analysis:")
        print(f"S: init={s_total[0,0]:.3f}, final={s_total[0,-1]:.3f}")
        print(f"E: init={e_total[0,0]:.3f}, peak={e_total[0].max():.3f}")
        print(f"I: init={i_total[0,0]:.3f}, peak={i_total[0].max():.3f}")
        print(f"R: init={r_total[0,0]:.3f}, final={r_total[0,-1]:.3f}")
        print(f"D: init={d_total[0,0]:.3f}, final={d_total[0,-1]:.3f}")
    
    return loss


class EncoderLayer(nn.Module):
    def __init__(self, attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu", EXO=None, d_env=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.EXO = EXO
        self.attention = attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.project_linear = nn.Linear(2*d_model, d_model)
        # self.project_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=2)


    def forward_general(self, x, attn_mask=None, tau=None, delta=None, cal_reg=True):
        # SA
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(x + self.dropout(new_x))

        loss = 0

        if self.EXO is not None:
            # import ipdb; ipdb.set_trace()
            # CA
            new_env, attn = self.cross_attention(
                x, self.EXO.unsqueeze(0).repeat(x.shape[0],1,1), self.EXO.unsqueeze(0).repeat(x.shape[0],1,1),
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )

            y = x = self.norm2(self.dropout(x*new_env))
        else:
            new_x, attn = self.cross_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
            )

            y = x = self.norm2(self.dropout(x*new_x))

        
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))


        return self.norm3(x + y), attn, loss, new_env


    def forward(self, x, attn_mask=None, tau=None, delta=None, cal_reg=True):

        return self.forward_general(x, attn_mask, tau, delta, cal_reg)
        


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, variate=None, cal_reg=True):
        # x [B, L, D]
        attns = []
        layer_envs = []
        reg_loss = 0
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, layer_reg, envs = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, cal_reg=cal_reg)
                # import ipdb; ipdb.set_trace()
                attns.append(attn)
                reg_loss += layer_reg
                layer_envs.append(envs)

        if self.norm is not None:
            x = self.norm(x)
        # import ipdb; ipdb.set_trace()
        return x, attns, reg_loss/len(self.attn_layers), layer_envs


class EXPEM(nn.Module):

    def __init__(self, patch_len, horizon, num_patches, d_model, n_layers, n_heads, stride=4, dropout=0.1, envs=8, d_env=128, proto_map=None, model_type='SR'):
        super().__init__()
        self.seq_len = num_patches*patch_len
        self.pred_len = horizon

        self.hidden = d_model
        self.n_layers = n_layers
        self.model_type = model_type  # Store model type for loss routing

        if envs !=0:
            self.EXO = nn.Parameter(torch.empty(envs, d_model))
            self.EXO = torch.nn.init.orthogonal_(self.EXO)
        else:
            print("running without EXO!")
            self.EXO=None
        
        
        # Prototype-compartment mapping for semi-supervised assignment
        # Default mapping based on model type
        if proto_map is None and envs >= 3:
            if model_type.upper() == 'SR':
                self.proto_map = {
                    'S': [0],      # Susceptible
                    'R': [1]       # Recovered
                    # Remaining prototypes are unconstrained
                }
            elif model_type.upper() == 'SIR':
                self.proto_map = {
                    'S': [0],      # Susceptible
                    'I': [1],      # Infectious  
                    'R': [2]       # Recovered
                    # Remaining prototypes are unconstrained
                }
            elif model_type.upper() == 'SEIR' and envs >= 4:
                self.proto_map = {
                    'S': [0],      # Susceptible
                    'E': [1],      # Exposed
                    'I': [2],      # Infectious  
                    'R': [3]       # Recovered
                    # Remaining prototypes are unconstrained
                }
            elif model_type.upper() == 'SIRD' and envs >= 4:
                self.proto_map = {
                    'S': [0],      # Susceptible
                    'I': [1],      # Infectious  
                    'R': [2],      # Recovered
                    'D': [3]       # Dead
                    # Remaining prototypes are unconstrained
                }
            elif model_type.upper() == 'SEIRD' and envs >= 5:
                self.proto_map = {
                    'S': [0],      # Susceptible
                    'E': [1],      # Exposed
                    'I': [2],      # Infectious  
                    'R': [3],      # Recovered
                    'D': [4]       # Dead
                    # Remaining prototypes are unconstrained
                }
            else:
                # Fallback to SIR
                self.proto_map = {
                    'S': [0],      # Susceptible
                    'R': [1]       # Recovered
                }
        else:
            self.proto_map = proto_map if proto_map is not None else {}
        
        # import ipdb; ipdb.set_trace()

        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=True), d_model, n_heads),
                    cross_attention=AttentionLayer(FullAttention(False, attention_dropout=0, output_attention=True), d_model, 1),
                    d_model=d_model,
                    d_ff=4*d_model,
                    dropout=dropout,
                    activation='gelu',
                    EXO=self.EXO,
                    d_env=d_model
                ) for l in range(n_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )
        self.cal_reg = True

        self.scale = torch.nn.Linear(1, 1, bias=True)
        torch.nn.init.constant_(self.scale.weight, 1000.0)
        torch.nn.init.constant_(self.scale.bias, 0.0)
        
        # Output projection - will be set dynamically on first forward pass
        self.out = None
        self.horizon = horizon
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_enc, time=None, dec_time=None, mask=None, return_ngm=False, layer_index=1, infected_indices=None, DFE_value=None, compute_aux_losses=True):
        # import ipdb; ipdb.set_trace()
        # variate = self.variate_projection(x_enc.flatten(1)).unsqueeze(1)

        enc_out = self.patch_embedding(x_enc)
        enc_out, attns, loss, envs = self.encoder(enc_out, cal_reg=self.cal_reg)
        # import ipdb; ipdb.set_trace()
        self.loss = loss
        
        self.estimated_envs = [env.detach().cpu().squeeze() for env in envs]
        # Store attention scores for compartment responsibility analysis
        self.stored_attns = [attn.detach().cpu() if attn is not None else None for attn in attns]
        
        # Compute auxiliary compartment losses
        self.aux_losses = {}
        # import ipdb; ipdb.set_trace()
        if compute_aux_losses and self.EXO is not None and len(self.proto_map) > 0:
            # Orthogonality loss for anchored prototypes
            self.aux_losses['proto_orth'] = loss_proto_orth(self.EXO, self.proto_map)
            
            # Compartment alignment loss using attention distributions
            if envs is not None and isinstance(envs, (list, tuple)) and len(envs) > 0:
                # Convert estimated_envs back to tensors for loss computation
                z_list = []
                for env in envs:
                    if env is not None and hasattr(env, 'shape'):
                        # Compute attention-like distribution from environment representations
                        if len(env.shape) == 3:  # [B, L, D]
                            # Compute similarities with prototypes to get attention-like weights
                            sims = torch.einsum('bld,ed->ble', env, self.EXO)  # [B, L, K]
                            z = torch.softmax(sims, dim=-1)  # [B, L, K]
                            z_list.append(z.unsqueeze(1))  # [B, 1, L, K]
                # import ipdb; ipdb.set_trace()
                # Use generic loss that routes to appropriate model
                model_type = getattr(self, 'model_type', 'SR')
                self.aux_losses['compartment_align'] = loss_generic_alignment(
                    z_list, x_enc, self.proto_map, self.EXO.device, model_type
                )
            else:
                self.aux_losses['compartment_align'] = torch.tensor(0.0, device=self.EXO.device)
        else:
            self.aux_losses['proto_orth'] = torch.tensor(0.0, device=x_enc.device)
            self.aux_losses['compartment_align'] = torch.tensor(0.0, device=x_enc.device)
        
        # Flatten and project to horizon for forecasting
        batch_size = enc_out.shape[0]
        enc_flat = enc_out.flatten(1)  # [batch, num_patches * d_model]
        
        # Initialize output layer on first forward pass
        if self.out is None or self.out.in_features != enc_flat.shape[1]:
            self.out = nn.Linear(enc_flat.shape[1], self.horizon).to(enc_out.device)
        
        output = self.out(enc_flat)  # [batch, horizon]
        output = self.dropout_layer(output)
        
        # For standard evaluation, return only the forecast
        if not return_ngm:
            return output
        
        # For detailed analysis, return additional info
        if infected_indices is None:
            infected_indices = list(range(2, self.EXO.shape[0]//2+2))
        ngm_info = self.compute_layer_R0(layer_index=layer_index, infected_indices=infected_indices, DFE_value=DFE_value)

        return output, attns, time, ngm_info
    
    def get_auxiliary_loss(self, orth_weight=1e-3, align_weight=1e-2):
        """
        Get combined auxiliary loss for compartment-specific regularization.
        
        Args:
            orth_weight: weight for orthogonality loss  
            align_weight: weight for compartment alignment loss
            
        Returns:
            Combined auxiliary loss scalar
        """
        # import ipdb; ipdb.set_trace()
        if not hasattr(self, 'aux_losses'):
            return torch.tensor(0.0)

        # import ipdb; ipdb.set_trace()
        try:
            total_aux_loss = (
                orth_weight * self.aux_losses.get('proto_orth', torch.tensor(0.0)) +
                align_weight * self.aux_losses['compartment_align']['alignment_loss']
            )
        except:
            total_aux_loss = (
                orth_weight * self.aux_losses.get('proto_orth', torch.tensor(0.0)) +
                align_weight * self.aux_losses['compartment_align'])

        return total_aux_loss
    
    def update_proto_map(self, new_proto_map):
        """
        Update the prototype-compartment mapping.
        
        Args:
            new_proto_map: dict mapping compartment names to prototype indices
        """
        self.proto_map = new_proto_map
    
    def get_compartment_responsibilities(self, layer_outputs=None):
        """
        Get the responsibility (attention weight) of each compartment over time.
        Uses actual cross-attention scores from the model.
        
        Returns:
            dict with compartment names as keys and responsibility tensors as values
        """
        if not hasattr(self, 'stored_attns') or not self.stored_attns:
            return {}
        
        compartment_resp = {}
        
        # Use the cross-attention scores from the last layer
        # stored_attns contains attention matrices from each layer
        if self.stored_attns and len(self.stored_attns) > 0:
            # Get the last layer's cross-attention scores
            last_attn = self.stored_attns[-1]  # Should be on CPU
            
            if last_attn is not None and isinstance(last_attn, torch.Tensor):
                # Convert back to device for computation
                attn_tensor = last_attn.to(self.EXO.device)
                
                # Cross-attention should have shape [B, n_heads, L, K] where K is number of prototypes
                # We need to average over heads and get [B, L, K]
                if len(attn_tensor.shape) == 4:  # [B, n_heads, L, K]
                    responsibilities = attn_tensor.mean(dim=1)  # [B, L, K] - average over heads
                elif len(attn_tensor.shape) == 3:  # [B, L, K] - already averaged
                    responsibilities = attn_tensor
                # else:
                    # Fallback to the old method if attention tensor shape is unexpected
                    # return self._get_compartment_responsibilities_fallback()
                
                # Apply temporal smoothing to encourage gradual transitions
                time_steps = responsibilities.size(1)
                if time_steps > 1:
                    smoothed_resp = responsibilities.clone()
                    # Apply mild temporal smoothing (but preserve transitions)
                    for t in range(1, time_steps):
                        alpha = 0.3  # Smoothing factor
                        smoothed_resp[:, t, :] = alpha * responsibilities[:, t, :] + (1 - alpha) * smoothed_resp[:, t-1, :]
                    responsibilities = smoothed_resp
                
                # Extract responsibilities for each compartment
                for comp_name, proto_indices in self.proto_map.items():
                    comp_resp = responsibilities[:, :, proto_indices].sum(dim=-1)  # [B, L]
                    compartment_resp[comp_name] = comp_resp.detach().cpu()
                
                # Extract the rest latent compartments not in proto_map, store each one individually
                mapped_indices = set()
                for proto_indices in self.proto_map.values():
                    mapped_indices.update(proto_indices)
                
                total_prototypes = responsibilities.size(-1)
                for proto_idx in range(total_prototypes):
                    if proto_idx not in mapped_indices:
                        latent_comp_resp = responsibilities[:, :, proto_idx]  # [B, L]
                        compartment_resp[f'Latent_{proto_idx}'] = latent_comp_resp.detach().cpu()
                
        # import ipdb; ipdb.set_trace()
        return compartment_resp
    
    
    
    def freeze_EXO(self):
        for param in self.parameters():
            # import ipdb; ipdb.set_trace()
            param.requires_grad = True

        self.EXO.requires_grad = False
        # for layer in self.encoder.attn_layers:
        #     for param in layer.cross_attention.parameters():
        #         # import ipdb; ipdb.set_trace()
        #         param.requires_grad = False
        
        self.cal_reg = False
    
    def freeze_encoder(self):
        for param in self.parameters():
            # import ipdb; ipdb.set_trace()
            param.requires_grad = False

        self.EXO.requires_grad = True
        # for layer in self.encoder.attn_layers:
        #     for param in layer.cross_attention.parameters():
        #         # import ipdb; ipdb.set_trace()
        #         param.requires_grad = True
                
        self.cal_reg = True
    
    def compute_ngm_and_R0(self, layer, infected_indices, total_mass: float = 1.0, susceptible_indices=None, baseline_m=None, use_jacobian: bool = True, eps: float = 1e-6, device=None, DFE_value=0.0):
        """
        Compute basic reproduction number R₀ using Next Generation Matrix (NGM) approach.
        
        This method computes R0 as the dominant eigenvalue of F @ V^(-1) where:
        - F matrix: rate of new infections from infected compartments
        - V matrix: rate of transitions out of infected compartments
        
        Uses DFE_value to create proper disease-free equilibrium embeddings.
        
        Args:
            layer: encoder layer to use for dynamics simulation
            infected_indices: indices of infected compartments (I, E)
            total_mass: total population mass (1.0 for normalized)
            susceptible_indices: indices of susceptible compartments (not used)
            baseline_m: optional baseline compartment state (not used)
            use_jacobian: not used in this approach
            eps: perturbation size for finite differences
            device: computation device
            DFE_value: disease-free equilibrium value for creating baseline embeddings
            
        Returns:
            dict: {'R0': float, 'NGM': tensor, 'F': tensor, 'V': tensor, 'method': str}
        """
        if self.EXO is None:
            return {'R0': 1.0, 'method': 'no_prototypes', 'components': {}}
            
        device = device or self.EXO.device
        
        # Step 1: Create DFE time series from DFE_value
        seq_length = getattr(self, 'seq_len', 64)  # Use model's sequence length
        dfe_timeseries = torch.full((1, seq_length, 4), DFE_value, 
                                    dtype=torch.float32, device=device)
        
        # import ipdb; ipdb.set_trace()
        # Step 2: Process DFE through patch embedding (same as training)
        dfe_embeddings = self.patch_embedding(dfe_timeseries)  # [1, L_patches, d_model]
        
        # Step 3: Forward through model to get DFE compartment state
        # Keep model in training mode to maintain gradient flow
        # was_training = self.training
        # self.eval()  # REMOVED: This breaks gradients!
        
        # Keep gradients flowing through layer computation
        dfe_output, dfe_attns, _, dfe_envs = layer.forward_general(dfe_embeddings, cal_reg=True)
        # print(dfe_attns)
        # import ipdb; ipdb.set_trace()
        # Get DFE compartment responsibilities with lower temperature for higher sensitivity
        dfe_compartments = self._extract_compartment_responsibilities(dfe_envs, temperature=0.5)
        # print("DFE compartments:", dfe_compartments)

        # import ipdb; ipdb.set_trace()
        # Step 4: Compute F matrix (new infections)
        F = self._compute_F_matrix_ngm(layer, dfe_embeddings, dfe_compartments, 
                                        infected_indices, eps, device)
        
        # Step 5: Compute V matrix (transitions out)
        V = self._compute_V_matrix_ngm(layer, dfe_embeddings, dfe_compartments,
                                        infected_indices, eps, device)

        
        # Mathematically rigorous two-sided bounds for R0 = ρ(F @ V^(-1))
        # Based on spectral radius inequalities and singular value bounds
        
        # SVD decomposition for both F and V matrices
        F_svd = torch.svd(F + 1e-8 * torch.eye(F.size(0), device=F.device))
        V_svd = torch.svd(V + 1e-6 * torch.eye(V.size(0), device=V.device))
        
        # Extract singular values
        F_sigma_max = F_svd.S.max()    # ||F||_2 (largest singular value)
        F_sigma_min = F_svd.S.min()    # σ_min(F) (smallest singular value)
        V_sigma_max = V_svd.S.max()    # σ_max(V) (largest singular value) 
        V_sigma_min = V_svd.S.min()    # σ_min(V) (smallest singular value)
        
        # UPPER BOUND: ρ(FV^(-1)) ≤ ||F||_2 / σ_min(V)
        # This uses operator norm of F and smallest singular value of V
        R0_upper = F_sigma_max / (V_sigma_min + 1e-8)
        
        # LOWER BOUND: ρ(FV^(-1)) ≥ σ_min(F) / σ_max(V)  
        # This uses smallest singular value of F and largest singular value of V
        # Derivation: ρ(FV^(-1)) ≥ σ_min(FV^(-1)) ≥ σ_min(F) * σ_min(V^(-1))
        #            = σ_min(F) / σ_max(V)
        R0_lower = F_sigma_min / (V_sigma_max + 1e-8)
    
        
        R0_lower = self.scale(R0_lower.unsqueeze(0)).squeeze(0)
        R0_upper = self.scale(R0_upper.unsqueeze(0)).squeeze(0)

        return {
            'R0_bounds': {
                'lower': R0_lower,
                'upper': R0_upper, 
            },
            'F': F, 
            'V': V,
            'singular_values': {
                'F_max': F_sigma_max,
                'F_min': F_sigma_min,
                'V_max': V_sigma_max,
                'V_min': V_sigma_min
            },
            'dfe_envs': dfe_envs,
            'dfe_compartments': dfe_compartments,
            'method': 'NGM_with_DFE_value_and_bounds'
        }

    
    
    
    def _extract_compartment_responsibilities(self, env, temperature=1.0, use_temporal_info=True):
        """
        Extract compartment responsibilities from environment representations.
        
        Args:
            envs: environment representations from encoder layers
            temperature: temperature for softmax (lower = more sensitive to changes)
            use_temporal_info: whether to use temporal dynamics in responsibility computation
        """
        
        # Create temporary EXO with gradients enabled for computation
        # This allows gradient computation even when self.EXO.requires_grad=False
        tmp_EXO = self.EXO.clone().detach()
        tmp_EXO.requires_grad = True
        
        # Normalize environment and prototype representations
        env_normalized = F.normalize(env, p=2, dim=-1)  # [B, L, D]
        EXO_normalized = F.normalize(tmp_EXO, p=2, dim=-1)  # [K, D]
        
        # Compute similarities and apply temperature scaling
        sims = torch.einsum('bld,ed->ble', env_normalized, EXO_normalized) / temperature  # [B, L, K]
        responsibilities = torch.softmax(sims, dim=-1)  # [B, L, K]
        
        # Apply temporal weighting if enabled
        if use_temporal_info and responsibilities.size(1) > 1:
            time_weights = torch.linspace(0.5, 1.0, responsibilities.size(1), device=responsibilities.device)
            responsibilities = responsibilities * time_weights.view(1, -1, 1)
        
        # Aggregate responsibilities across batch and time dimensions
        compartment_resp = responsibilities.mean(dim=(0, 1))  # [K]
        
        # Ensure valid probabilities
        compartment_resp = torch.clamp(compartment_resp, min=1e-6)
        compartment_resp = compartment_resp / compartment_resp.sum()
        # import ipdb; ipdb.set_trace()
        return compartment_resp
    
    def _compute_F_matrix_ngm(self, layer, dfe_embeddings, dfe_compartments, infected_indices, eps, device):
        """Compute F matrix: rate of new infections from infected compartments."""
        n_infected = len(infected_indices)
        # Create F matrix elements as a list, then stack to maintain gradients
        F_elements = []
        
        # Ensure dfe_compartments is 1D tensor
        if dfe_compartments.dim() > 1:
            dfe_compartments = dfe_compartments.flatten()
        
        max_compartment_idx = dfe_compartments.size(0) - 1
        
        # Use much larger perturbation for infection scenarios
        infection_boost = 0.5  # Increase infected compartment by 50%
        
        for j, inf_j in enumerate(infected_indices):
            # Ensure infected index is valid
            if inf_j > max_compartment_idx:
                # Add zero row for invalid indices
                F_elements.append(torch.zeros(n_infected, device=device, dtype=dfe_compartments.dtype))
                continue
                
            # Create large infection perturbation in compartment space
            perturbed_compartments = dfe_compartments.clone()
            perturbed_compartments[inf_j] += infection_boost 
            
            # Normalize to maintain total mass
            total_mass = perturbed_compartments.sum()
            if total_mass > 0:
                perturbed_compartments = perturbed_compartments / total_mass
            
            # Convert back to model embedding space
            perturbed_embeddings = self._compartments_to_embeddings(perturbed_compartments, 
                                                                   dfe_embeddings)
            
            # Forward through layer to get new compartment state
            new_output, _, _, new_envs = layer.forward_general(perturbed_embeddings, cal_reg=False)
            new_compartments = self._extract_compartment_responsibilities(new_envs, temperature=0.5)

            # Ensure new_compartments is 1D tensor
            if new_compartments.dim() > 1:
                new_compartments = new_compartments.flatten()
            
            # Compute new infections for this column
            F_row = torch.zeros(n_infected, device=device, dtype=dfe_compartments.dtype)
            for i, inf_i in enumerate(infected_indices):
                if (inf_i < new_compartments.size(0) and 
                    inf_i < dfe_compartments.size(0) and
                    inf_i <= max_compartment_idx):
                    new_infections = new_compartments[inf_i] - dfe_compartments[inf_i]

                    F_row[i] = torch.relu(new_infections) * 10000
            
            F_elements.append(F_row)
        
        # Stack all rows to create F matrix with gradients
        F = torch.stack(F_elements, dim=1)  # Stack as columns, then transpose
        return F.T  # Transpose to get correct [n_infected, n_infected] shape
    
    def _compute_V_matrix_ngm(self, layer, dfe_embeddings, dfe_compartments, infected_indices, eps, device):
        """Compute V matrix: rate of transitions out of infected compartments."""
        n_infected = len(infected_indices)
        # Create V matrix elements as a list, then stack to maintain gradients
        V_elements = []
        
        for j, inf_j in enumerate(infected_indices):
            # Create state with unit mass in infected compartment j
            infected_compartments = torch.zeros_like(dfe_compartments)
            infected_compartments[inf_j] = 1.0
            
            # Convert to embedding space
            infected_embeddings = self._compartments_to_embeddings(infected_compartments, 
                                                                   dfe_embeddings)
            
            # Forward through layer
            new_output, _, _, new_envs = layer.forward_general(infected_embeddings, cal_reg=False)
            new_compartments = self._extract_compartment_responsibilities(new_envs, temperature=0.5)
            
            # Ensure new_compartments is 1D tensor
            if new_compartments.dim() > 1:
                new_compartments = new_compartments.flatten()
            
            # Create V column for this infected compartment
            V_col = torch.zeros(n_infected, device=device, dtype=dfe_compartments.dtype)
            
            # Measure outflow from this infected compartment (diagonal)
            if inf_j < new_compartments.size(0):
                outflow = 1.0 - new_compartments[inf_j]  # Mass that left compartment j
                V_col[j] = torch.relu(outflow)  # Remove division by eps to reduce magnitude
            
            # Measure transitions to other infected compartments (off-diagonal)
            for i, inf_i in enumerate(infected_indices):
                if i != j and inf_i < new_compartments.size(0):
                    # This represents flow FROM compartment j TO compartment i
                    transition_flow = torch.relu(new_compartments[inf_i])  # Mass that appeared in compartment i
                    V_col[i] = transition_flow  # Remove division by eps
            
            # Ensure V matrix conservation: diagonal should be total outflow
            if inf_j < new_compartments.size(0):
                total_infected_inflow = sum(torch.relu(new_compartments[infected_indices[i]]) 
                                          for i in range(n_infected) if i != j)
                total_outflow_to_susceptible_recovered = 1.0 - new_compartments[inf_j] - total_infected_inflow
                
                # Adjust diagonal to represent total outflow rate (without eps division)
                V_col[j] = total_infected_inflow + torch.relu(total_outflow_to_susceptible_recovered)
            
            V_elements.append(V_col)
        
        # Stack all columns to create V matrix with gradients
        V = torch.stack(V_elements, dim=1)  # Stack as columns
        
        # Add small regularization to ensure numerical stability
        V = V + 1e-3 * torch.eye(n_infected, device=device, dtype=V.dtype)
        
        return V
    
    def _compartments_to_embeddings(self, target_compartments, base_embeddings):
        """
        Convert desired compartment state back to embedding space.
        
        This creates embeddings that would produce the target compartment responsibilities
        when processed through the model.
        """
        # Create temporary EXO with gradients enabled for computation
        tmp_EXO = self.EXO.clone().detach()
        tmp_EXO.requires_grad = True
        
        # Method 1: Use weighted combination of prototype embeddings
        # Project compartment weights onto EXO prototype space
        target_proto = torch.einsum('k,kd->d', target_compartments, tmp_EXO)  # [d_model]
        
        # Reshape to match base_embeddings dimensions [1, L, d_model]
        B, L, D = base_embeddings.shape
        target_expanded = target_proto.unsqueeze(0).unsqueeze(0).expand(B, L, D)
        
        # Use much larger perturbation to ensure detectable changes
        alpha = 0.8  # Large perturbation to create significant changes
        perturbed_embeddings = (1 - alpha) * base_embeddings + alpha * target_expanded
        
        return perturbed_embeddings
    
    
    def compute_layer_R0(self, layer_index: int, infected_indices, DFE_value=None, susceptible_indices=None, baseline_m=None, **kwargs):
        """
        Convenience wrapper: compute NGM and R0 for encoder layer at `layer_index`.
        Returns the dict from compute_ngm_and_R0 using the specified layer.
        """
        if not (0 <= layer_index < len(self.encoder.attn_layers)):
            raise IndexError(f"layer_index out of range: {layer_index}")
        layer = self.encoder.attn_layers[layer_index]
        
        # Use default DFE_value if not provided
        if DFE_value is None:
            DFE_value = 0.0  # Default disease-free equilibrium
        
        # Use larger perturbation for better numerical stability
        kwargs['eps'] = kwargs.get('eps', 0.1)  # Increase from 1e-2 to 0.1
            
        return self.compute_ngm_and_R0(layer, infected_indices=infected_indices, 
                                      DFE_value=DFE_value, susceptible_indices=susceptible_indices, 
                                      baseline_m=baseline_m, **kwargs)





