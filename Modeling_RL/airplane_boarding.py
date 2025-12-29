import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

# Register this module as gym environment. Once registered, this id is usable in gym.make()
# When running this code, you can ignore this warning: "UseWarning: WARN: Overriding environment airplane-boarding-v0 already in registry"
register(
    id='airplane-boarding-v0',
    entry_point='airplane_boarding:AirplaneEnv', # module_name:class_name
)

class AirplaneEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, render_mode=None):

        # Reset the environment
        self.reset()

        # Define the Action space.
        # self.action_space = ...

        # Define the Observation space.
        # The observation space is used to validate the observation returned by reset() and step().
        # self.observation_space = ...

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options) # gym requires this call to control randomness and reproduce scenarios.

        # return observation , info
    
    def step(self, action):
        pass
        
        # return observation, reward, terminated, truncated, info
    
    def render(self):
        return super().render()
    
# Check validity of the environment
def my_check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('airplane-boarding-v0', render_mode=None)
    check_env(env.unwrapped)

if __name__ == "__main__":
    my_check_env()