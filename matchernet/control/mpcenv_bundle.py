# -*- coding: utf-8 -*-
import numpy as np
from matchernet import Bundle, Matcher


class MPCEnvBundle(Bundle):
    """
    MPCEnv bundle class that communicates with matcher.
    """
    def __init__(self, env):
        """
        Arguments:
          env
            MPCEnv instance
        """
        super(MPCEnvBundle, self).__init__("mpcenv_bundle")
        self.env = env
        self.timestamp = 0.0
        
        self.update_component()

    def __call__(self, inputs):
        if "mpcenv_matcher" in inputs and inputs["mpcenv_matcher"] is not None:
            feedback = inputs["mpcenv_matcher"]
            
            # Receive action from Matcher
            u = feedback["u"]

            # Step environment with received action
            x, reward = self.env.step(u)
            
            # Increment timestamp
            self.timestamp += self.env.dt

            # Send state to matcher
            state = {
                "x" : x,
                "reward" : reward,
                "timestamp" : self.timestamp
            }
            return {
                "state" : state
            }
        else:
            return {}


class MPCEnvDebugMatcher(Matcher):
    """
    Debug Matcher implementation for MPCEnv bundle debugging.
    """
    def __init__(self, *bundles):
        super(MPCEnvDebugMatcher, self).__init__("mpcenv_matcher", *bundles)
        self.timestamp = 0.0

    def __call__(self, inputs):
        if "mpcenv_bundle" in inputs and inputs["mpcenv_bundle"] is not None:
            # Receive sate from bundle
            mpcenv_state = inputs["mpcenv_bundle"]
            x         = mpcenv_state["x"]
            reward    = mpcenv_state["reward"]
            timestamp = mpcenv_state["timestamp"]

            print("x received from bundle: {}".format(x))
            
        # Send action to Bundle
        u = np.array([0.1, 0.2], dtype=np.float32)
        results = {
            "u" : u
        }
        return {
            "mpcenv_bundle" : results
        }
