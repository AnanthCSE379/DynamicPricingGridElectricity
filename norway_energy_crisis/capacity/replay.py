"""CPU replay with shared immutable arrays for repeated household observations."""

import hashlib

import numpy as np
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from norway_energy_crisis.clean_data import require


class CompactReplayBuffer(BaseBuffer):
    """Keep every transition, sharing identical arrays rather than dropping fields.

    Household descriptors repeat throughout training; reference schedules repeat
    within a day; a transition's next observation often is the next transition's
    current observation. Reference counts bound this pool to the active replay.
    Samples use ordinary uniform transition sampling and float32 MPS tensors.
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    ):
        require(
            n_envs == 1 and not optimize_memory_usage,
            "One active capacity fit required.",
        )
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.handle_timeout_termination = handle_timeout_termination
        self.pool = {}
        self.slots = [None] * self.buffer_size
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.timeouts = np.zeros((buffer_size, 1), dtype=np.float32)

    def _intern(self, values):
        result = {}
        for name, array in values.items():
            value = np.ascontiguousarray(array[0], dtype=np.float32)
            key = (name, value.shape, hashlib.sha256(value.tobytes()).digest())
            if key in self.pool:
                require(
                    np.array_equal(value, self.pool[key][0]), "Replay hash collision."
                )
                self.pool[key][1] += 1
            else:
                self.pool[key] = [value.copy(), 1]
            result[name] = key
        return result

    def add(self, obs, next_obs, action, reward, done, infos):
        old = self.slots[self.pos]
        if old is not None:
            for observation in old:
                for key in observation.values():
                    self.pool[key][1] -= 1
                    if self.pool[key][1] == 0:
                        del self.pool[key]
        self.slots[self.pos] = (self._intern(obs), self._intern(next_obs))
        self.actions[self.pos] = np.asarray(action).reshape(1)
        self.rewards[self.pos] = np.asarray(reward).reshape(1)
        self.dones[self.pos] = np.asarray(done).reshape(1)
        self.timeouts[self.pos] = [float(infos[0].get("TimeLimit.truncated", False))]
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full, self.pos = True, 0

    def _get_samples(self, batch_inds, env=None):
        def observations(side):
            values = {
                name: np.stack(
                    [
                        self.pool[self.slots[index][side][name]][0]
                        for index in batch_inds
                    ]
                )
                for name in self.observation_space.spaces
            }
            return {
                name: self.to_torch(value)
                for name, value in self._normalize_obs(values, env).items()
            }

        dones = self.dones[batch_inds]
        if self.handle_timeout_termination:
            dones = dones * (1 - self.timeouts[batch_inds])
        return DictReplayBufferSamples(
            observations(0),
            self.to_torch(self.actions[batch_inds]),
            observations(1),
            self.to_torch(dones),
            self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
        )

    def reset(self):
        super().reset()
        self.pool.clear()
        self.slots = [None] * self.buffer_size

    @property
    def bytes_stored(self):
        return sum(value.nbytes for value, _ in self.pool.values())
