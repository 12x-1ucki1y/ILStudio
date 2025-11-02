import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F  # <-- MOVED: Fixed the import error
from torchdiffeq import odeint

# --- 1. Time Embedding ---
# Same as in Transformers and Diffusion models, used to encode scalar time t into a vector

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim: int, max_time: float = 1.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension {dim} must be even.")
        self.dim = dim
        self.max_time = max_time
        
        # Calculate log(10000) / (dim/2 - 1)
        half_dim = dim // 2
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float32) *
                             -(torch.log(torch.tensor(10000.0)) / (half_dim - 1.0)))
        self.register_buffer('div_term', div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input:
            t: [B, 1] or [B] or scalar, time steps in [0, max_time]
        Output:
            emb: [B, D]
        """
        t = t.float().view(-1, 1) / self.max_time
        pe = torch.zeros(t.shape[0], self.dim, device=t.device)
        pe[:, 0::2] = torch.sin(t * self.div_term)
        pe[:, 1::2] = torch.cos(t * self.div_term)
        return pe

# --- 2. Velocity Field Model (Velocity Model) ---
# This is our core neural network v_theta(a_t, t, s)

class VelocityModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, time_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Concatenate (action, time_emb, state) as input
        input_dim = action_dim + time_dim + state_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim) # Output velocity, same dimension as action
        )
        
    def forward(self, a_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Input:
            a_t: [B, action_dim], Action state at time t
            t: [B] or scalar, Time t
            s: [B, state_dim], Conditional state (proprioception)
        Output:
            v: [B, action_dim], Predicted velocity
        """
        if t.dim() == 0:
            # Expand scalar t to batch size
            t = t.expand(a_t.shape[0])
            
        t_emb = self.time_embed(t)
        
        # Concatenate inputs
        x = torch.cat([a_t, t_emb, s], dim=-1)
        
        return self.net(x)

# --- 3. Flow Matching Policy ---

class FlowMatchingPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, 
                 time_dim: int = 64, hidden_dim: int = 256, 
                 lr: float = 1e-4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize the velocity model
        self.model = VelocityModel(state_dim, action_dim, time_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def _get_train_loss(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Calculates the CFM training loss (L_FM)
        
        Input:
            s: [B, state_dim], Expert state
            a: [B, action_dim], Expert action
        Output:
            loss: Scalar loss value
        """
        batch_size = s.shape[0]
        device = s.device
        
        # 1. Sample t ~ U[0, 1]
        # For numerical stability, we sample t ~ U[epsilon, 1]
        epsilon = 1e-5
        t = torch.rand(batch_size, device=device) * (1.0 - epsilon) + epsilon
        
        # 2. Sample z (a_0) ~ N(0, I)
        z = torch.randn_like(a)
        
        # 3. Calculate a_t = t*a + (1-t)*z
        a_t = t.view(-1, 1) * a + (1.0 - t.view(-1, 1)) * z
        
        # 4. Calculate target velocity u_t = a - z
        u_t = a - z
        
        # 5. Predict velocity v_theta
        v_pred = self.model(a_t, t, s)
        
        # 6. Calculate MSE loss ||v_pred - u_t||^2
        loss = F.mse_loss(v_pred, u_t)
        
        return loss

    def train_step(self, batch: tuple) -> float:
        """
        Execute one training step
        """
        s, a = batch
        s = s.to(next(self.parameters()).device)
        a = a.to(next(self.parameters()).device)
        
        self.optimizer.zero_grad()
        loss = self._get_train_loss(s, a)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # --- Inference (Sampling) ---
    
    @torch.no_grad()
    def select_action(self, s: torch.Tensor) -> torch.Tensor:
        """
        Sample action using an ODE solver (recommended method)
        
        Input:
            s: [B, state_dim], Current state
        Output:
            a_1: [B, action_dim], Sampled action
        """
        self.model.eval() # Switch to evaluation mode
        device = s.device
        batch_size = s.shape[0]
        
        # 1. Define the ODE function (this is the format required by torchdiffeq)
        # We need a helper class to pass the model and state s
        class ODEFunc(nn.Module):
            def __init__(self, model, state):
                super().__init__()
                self.model = model
                self.state = state

            def forward(self, t, a_t):
                # t is the current time scalar, a_t is [B, action_dim]
                # model needs (a_t, t, s)
                return self.model(a_t, t, self.state)
        
        ode_func = ODEFunc(self.model, s)

        # 2. Sample a_0 from N(0, I)
        a_0 = torch.randn(batch_size, self.action_dim, device=device)
        
        # 3. Define integration time span [0, 1]
        t_span = torch.tensor([0.0, 1.0], device=device)
        
        # 4. Solve the ODE
        # odeint returns [T, B, action_dim]
        # We only need the solution at T=2 points (0 and 1)
        solution = odeint(
            ode_func,
            a_0,
            t_span,
            method='dopri5', # A robust adaptive solver
            atol=1e-5,
            rtol=1e-5
        )
        
        # 5. Return the solution at t=1
        a_1 = solution[1]
        
        self.model.train() # Switch back to training mode
        return a_1

    @torch.no_grad()
    def select_action_euler(self, s: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        Sample action using simple Euler method (for demonstration, fast but less precise)
        
        Input:
            s: [B, state_dim]
            num_steps: Integration steps
        Output:
            a_T: [B, action_dim]
        """
        self.model.eval()
        device = s.device
        batch_size = s.shape[0]
        
        dt = 1.0 / num_steps
        
        # Start from N(0, I)
        a_t = torch.randn(batch_size, self.action_dim, device=device)
        
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            # Euler step: a_{t+dt} = a_t + v(a_t, t, s) * dt
            v = self.model(a_t, t, s)
            a_t = a_t + v * dt
            
        self.model.train()
        return a_t


# --- 4. Training and Inference Demo ---

if __name__ == "__main__":
    
    # --- A. Setup ---
    # Define dimensions
    STATE_DIM = 10   # e.g.: 10-joint proprioceptive state
    ACTION_DIM = 4   # e.g.: 4-joint motor command
    BATCH_SIZE = 64
    TRAIN_STEPS = 1000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- B. Create Dummy Dataset (Imitation Learning) ---
    # In a real application, this is where you load your expert data
    # We create data such that action = state[:4] * 0.5 + noise
    print("Creating dummy dataset...")
    num_samples = BATCH_SIZE * TRAIN_STEPS
    expert_states = torch.randn(num_samples, STATE_DIM)
    # Simulate expert actions
    expert_actions = (expert_states[:, :ACTION_DIM] * 0.5 + 
                      torch.randn(num_samples, ACTION_DIM) * 0.1) 

    dataset = data.TensorDataset(expert_states, expert_actions)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- C. Initialize Policy Model ---
    policy = FlowMatchingPolicy(STATE_DIM, ACTION_DIM).to(device)

    # --- D. Training Loop ---
    print(f"Start training on {device}...")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            if step >= TRAIN_STEPS:
                done = True
                break
                
            loss = policy.train_step(batch)
            
            if step % 100 == 0:
                print(f"Step {step}/{TRAIN_STEPS}, Loss: {loss:.6f}")
            step += 1
    
    print("Training finished.")

    # --- E. Inference Demo ---
    print("\n--- Inference Demo ---")
    
    # Get a virtual current state
    # (In a real application, this would be the true state from your environment)
    test_state = torch.randn(1, STATE_DIM).to(device)
    print(f"Test State shape: {test_state.shape}")
    
    # 1. Using torchdiffeq (recommended)
    # Sample 1 action
    sampled_action_ode = policy.select_action(test_state)
    print(f"Sampled Action (odeint) shape: {sampled_action_ode.shape}")
    print(f"Action (odeint): {sampled_action_ode.cpu().numpy()}")

    # Sample 8 actions (demonstrating batching)
    test_states_batch = torch.randn(8, STATE_DIM).to(device)
    sampled_actions_ode_batch = policy.select_action(test_states_batch)
    print(f"Batch Action (odeint) shape: {sampled_actions_ode_batch.shape}")


    # 2. Using manual Euler (for comparison)
    sampled_action_euler = policy.select_action_euler(test_state, num_steps=100)
    print(f"\nSampled Action (Euler) shape: {sampled_action_euler.shape}")
    print(f"Action (Euler): {sampled_action_euler.cpu().numpy()}")

    # Theoretically, the two actions should be similar, 
    # as they are sampling from the same learned distribution
    diff = torch.norm(sampled_action_ode - sampled_action_euler)
    print(f"\nDifference between odeint and Euler: {diff.item():.4f}")