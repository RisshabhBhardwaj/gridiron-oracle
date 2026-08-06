"""
ml/rl_hedging_agent.py

Deep Reinforcement Learning (DRL) Agent for Dynamic Hedging.

OVERVIEW
--------
Replaces static Kelly sizing limits with a trained Soft Actor-Critic (SAC) or
Proximal Policy Optimization (PPO) agent.

The agent manages live sports betting bankroll exposed to multiple correlated 
assets. It learns to optimally size Kelly recommendations and decide when to 
place opposing hedge bets dynamically based on live score and injury context.

ENVIRONMENT (MDP Formulation):
    - State:      [Bankroll, Exposure_A, Edge_A, P(Win)_A, Game_Time_Remaining, Current_Score_Diff]
    - Action:     [Sizing_Multiplier (0.0 to 1.5)] 
                  (where 0.0 = no bet / full hedge, 1.0 = full Kelly, 1.5 = overleveraged Kelly)
    - Reward:     Log-growth of bankroll (to naturally match Kelly criterion)
"""

import logging

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    gym = None

logger = logging.getLogger(__name__)

# Constants matching engine/include/kelly_sizer.hpp bounds
MAX_BANKROLL = 1e6
MAX_BET_FRACTION = 0.05

class HedgingEnv(gym.Env if gym else object):
    """
    Gymnasium environment simulating a live betting sequence.
    """
    def __init__(self, initial_bankroll: float = 10000.0, max_steps: int = 100):
        super().__init__()
        self.initial_bankroll = initial_bankroll
        self.max_steps = max_steps
        
        # Action space: Sizing multiplier against base Kelly recommendation [0.0 to 1.5]
        self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)
        
        # State space: [Bankroll, Current_Exposure, Expected_Edge, Win_Prob, Time_Remaining, Score_Diff]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -50.0]),
            high=np.array([MAX_BANKROLL, MAX_BANKROLL, 1.0, 1.0, 60.0, 50.0]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.bankroll = initial_bankroll
        self.exposure = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.bankroll = self.initial_bankroll
        self.exposure = 0.0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Simulate incoming random opportunities (in reality, driven by live PBP stream)
        edge = np.random.uniform(0.01, 0.15)
        prob = np.random.uniform(0.40, 0.80)
        time_rem = max(0.0, 60.0 - (self.current_step / self.max_steps) * 60)
        score_diff = np.random.normal(0, 14) 
        
        return np.array([
            self.bankroll, 
            self.exposure, 
            edge, 
            prob, 
            time_rem, 
            score_diff
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Observation unpacked
        obs = self._get_obs()
        obs[2]
        prob = obs[3]
        
        # Baseline Kelly sizing
        odds = (1.0 - prob) / prob  # rough implied decimal odds offset
        b = 1.0 / odds
        pure_kelly = (b * prob - (1 - prob)) / b if b > 0 else 0
        
        # Agent scales the Kelly fraction
        base_bet_fraction = np.clip(pure_kelly * action[0], 0.0, MAX_BET_FRACTION)
        bet_size = self.bankroll * base_bet_fraction
        
        # Simulate binomial outcome of the bet
        won = np.random.rand() < prob
        
        prev_bankroll = self.bankroll
        if won:
            self.bankroll += bet_size * b
        else:
            self.bankroll -= bet_size
            
        # Reward is log difference (Kelly natural log growth objective)
        # Add epsilon to prevent log(0)
        reward = np.log((self.bankroll + 1e-6) / (prev_bankroll + 1e-6))
        
        terminated = self.bankroll <= 0 or self.current_step >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}


class HedgingAgent:
    """Wrapper to train and query the PPO/SAC reinforcement learning model."""
    def __init__(self, algo: str = "PPO"):
        if gym is None:
            raise ImportError("gymnasium and stable_baselines3 are required for RL.")
            
        self.env = HedgingEnv()
        self.algo = algo
        self.model = None
        
    def train(self, timesteps: int = 50_000):
        """Train the policy gradient agent to maximize geometric growth."""
        logger.info(f"Training {self.algo} hedging agent for {timesteps} steps...")
        
        eval_callback = EvalCallback(
            self.env, 
            best_model_save_path="./ml/checkpoints/",
            log_path="./ml/oof/",
            eval_freq=10000,
            deterministic=True, 
            render=False
        )
        
        if self.algo == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=0)
        elif self.algo == "SAC":
            self.model = SAC("MlpPolicy", self.env, verbose=0)
        else:
            raise ValueError("Unsupported algorithm. Use 'PPO' or 'SAC'.")
            
        self.model.learn(total_timesteps=timesteps, callback=eval_callback)
        logger.info("Training complete.")
        
    def get_sizing_multiplier(self, bankroll_state: dict) -> float:
        """Query trained model for live inference."""
        if not self.model:
            logger.warning("Agent not trained. Returning neutral 1.0 multiplier.")
            return 1.0
            
        obs = np.array([
            bankroll_state.get("bankroll", 10000.0),
            bankroll_state.get("exposure", 0.0),
            bankroll_state.get("edge", 0.05),
            bankroll_state.get("prob", 0.55),
            bankroll_state.get("time_rem", 30.0),
            bankroll_state.get("score_diff", 0.0)
        ], dtype=np.float32)
        
        action, _states = self.model.predict(obs, deterministic=True)
        # action is an array of shape (1,)
        return float(action[0])
