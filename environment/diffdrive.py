from __future__ import print_function
import sys
from numpy import array, linalg, arctan2, pi, inf, float32, sin, cos, finfo
from numpy.random import randn
from gym import spaces
from utils import normalize_angle, RK4

EPS32 = finfo(float32).eps

__all__ = ['DifferentialDriveKinematic']



class DifferentialDriveKinematic:

  def __init__(self, dt=0.05, verbose=False, tol=1e-3, timeout=1000,
    observation_noise=array([1e-2, 1e-2, 1e-2], dtype=float32),
    transition_noise=array([1e-2, 1e-2, 1e-2], dtype=float32)
  ):

    metadata = {'render.modes': ['None']}
    reward_range = (-float32(inf), float32(inf))
    spec = None
    self.verbose = verbose

    self.wheel_diameter = float32(0.2)
    self.wheel_radius = float32(0.1)
    self.wheel_separation = float32(0.628)
    self.w_max = float32(2.0)
    v_max = self.w_max * self.wheel_radius
    omega_max = v_max * float32(2) / self.wheel_separation
    self.vel_max = array([v_max, v_max, omega_max])

    self.dt = float32(dt)
    self.tol = float32(tol)
    self.timeout = float32(timeout)
    self.action_space = spaces.Box(low=-float32(inf), high=float32(inf), shape=(2,), dtype=float32)
    self.observation_space = spaces.Box(low=-float32(inf), high=float32(inf), shape=(3,), dtype=float32)
    self.observation_noise = observation_noise
    self.transition_noise = transition_noise

    self.pose = None

    self.prev_state = None


  def enter(self):

    return self


  def __exit__(self, *args):

    #self.close()

    return False


  def render(self, mode='None'):

    return None


  def seed(self, seed=None):

    return None

    
  def reset(self, state):

    self.prev_state = None
    self.count = 0
    self.reward_sum = float32(0.0)

    if self.verbose:
      sys.stderr.write('\n')

    observation, done = self.start(state)

    return observation

    
  def start(self, state):

    self.pose = array(
      (
        state[0] * cos(pi + state[1] - state[2]),
        state[0] * sin(pi + state[1] - state[2]),
        -state[2]
      ),
      dtype=float32
    )

    return self.get_observation(), False

  @staticmethod
  def __dx(pose, v, omega):
    return array(
      (
        v * cos(pose[2]),
        v * sin(pose[2]),
        omega
      ),
      dtype=float32
    )

    
  def step(self, action):
    v = float32((action[0] + action[1]) * self.wheel_radius / 2)
    omega = float32((action[1] - action[0]) * self.wheel_radius / self.wheel_separation)

    self.count += 1
    noise = self.transition_noise * self.__dx(self.pose, v, omega) / self.vel_max * randn(3)
    dtheta = omega * self.dt
    dtheta2 = dtheta / 2
    ds = v * self.dt
    if dtheta2 > EPS32:
      _lambda = sin(dtheta2) / dtheta2
      ds *= _lambda
    self.pose[0] += ds * cos(self.pose[2] + dtheta2) + float32(noise[0])
    self.pose[1] += ds * sin(self.pose[2] + dtheta2) + float32(noise[1])
    self.pose[2] = float32(normalize_angle(self.pose[2] + dtheta + noise[2]))
    #self.pose = float32(RK4(self.__dx, self.dt, self.pose, v, omega)) + noise

    observation = self.get_observation()
    timestamp = self.dt * self.count

    if self.verbose:
      sys.stderr.write(
        '\rstep:%d| pose:% 2.1f, % 2.1f, % 2.1f | vel:% 2.1f, % 2.1f '
        %(
          self.count,
          self.pose[0], self.pose[1], self.pose[2],
          v, omega
        )
      )
    if linalg.norm(self.pose) < self.tol:
      done = True
      reward = float32(0)
      success = True
      if self.verbose:
        sys.stderr.write(' | Success\n')
    else:
      done = False
      reward = float32(-1)
      success = None
    if timestamp > self.timeout:
      done = True
      success = False
      if self.verbose:
        sys.stderr.write(' | Fail\n')

    return observation, reward, done, {
      'pose':self.pose,
      'action':action,
      'timestamp':timestamp,
      'done':done,
      'reward':reward,
      'success':success
    }


  def get_observation(self):
    noise = self.observation_noise * randn(3)
    observation = noise + self.pose
    return array(
      [
        linalg.norm(observation[:2]),
        normalize_angle(
          arctan2(observation[1], observation[0]) - pi - observation[2]
        ),
        normalize_angle(-observation[2])
      ],
      dtype=float32
    )


  def get_state(self):
    return array(
      [
        linalg.norm(self.pose[:2]),
        normalize_angle(
          arctan2(self.pose[1], self.pose[0]) - pi - self.pose[2]
        ),
        normalize_angle(-self.pose[2])
      ],
      dtype=float32
    )


if __name__=="__main__":

  env = DifferentialDriveKinematic(dt=0.1, verbose=True)
  env.reset([1, 1, 1])

  for t in range(1000):
    res = env.step([0.1, 0.1])
