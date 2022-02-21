import os
import numpy as np
import argparse
from distutils.util import strtobool
from multiprocessing import Pool, Manager

from numpy.core.numeric import normalize_axis_tuple
from mdp import (
  States,
  Actions,
  Rewards,
  Policy,
  StateTransitionProbability,
  MarkovDecisionProcess,
  PolicyIteration,
  ValueIteration
)
from environment import DifferentialDriveKinematic
from planner import NmpcCartesian, NmpcPolar
from utils import Verbose, print_test, normalize_angle



parser = argparse.ArgumentParser()
env_group = parser.add_argument_group('env')
env_group.add_argument('--env-dt', type=float, default=0.05)
env_group.add_argument('--env-pose_tol', type=float, default=1e-2)
env_group.add_argument('--env-control_tol', type=float, default=1e-2)
env_group.add_argument('--env-timeout', type=float, default=1000)
mpc_group = parser.add_argument_group('mpc')
mpc_group.add_argument('--mpc-dt', type=float, default=0.05)
mpc_group.add_argument('--mpc-time_horizon', type=int, default=10)
mpc_group.add_argument('--mpc-regularization_control', type=float, default=1e-3)
mpc_group.add_argument('--mpc-regularization_v', type=float, default=0)
mpc_group.add_argument('--mpc-regularization_omega', type=float, default=0)
mpc_group.add_argument('--mpc-r_tol', type=float, default=5e-3)
mdp_group = parser.add_argument_group('mdp')
mdp_group.add_argument('--n_r', type=int, default=40)
mdp_group.add_argument('--n_alpha', type=int, default=24)
mdp_group.add_argument('--n_psi', type=int, default=25)
mdp_group.add_argument('--n_theta', type=int, default=7)
mdp_group.add_argument('--repeats_per_sample_point', type=int, default=1)
mdp_group.add_argument('--mdp-discount', type=float, default=0.99)
mdp_group.add_argument('--mdp-max_iter', type=int, default=100000)
mdp_group.add_argument('--mdp-tol', type=float, default=1e-6)
mdp_group.add_argument('--mdp-solver', type=str, default='ValueIteration',
  choices=['PolicyIteration', 'ValueIteration']
)
parser.add_argument('--data_dir', type=str, default='data/mdp')
parser.add_argument('--sample', type=strtobool, default=True)
parser.add_argument('--solve', type=strtobool, default=True)
parser.add_argument('--test', type=strtobool, default=True)

args = parser.parse_args()

if not os.path.isdir(args.data_dir):
  print('{} does not exist. Create target directory...'.format(args.data_dir))
  os.mkdir(args.data_dir)
file_mdp = os.path.join(args.data_dir, 'mdp.npz')
file_result = os.path.join(args.data_dir, 'result.npz')
file_test = os.path.join(args.data_dir, 'test.npy')
if not args.sample:
  if not os.path.exists(file_mdp):
    raise FileNotFoundError('{} does not exist.'.format(file_mdp))
if not args.solve:
  if not os.path.exists(file_result):
    raise FileNotFoundError('{} does not exist.'.format(file_result))

r_list = np.abs(-np.log(np.linspace(1, 0, args.n_r + 1, dtype=np.float32)[:-1]))
alpha_list =  np.linspace(-np.pi+np.pi/args.n_alpha, np.pi-np.pi/args.n_alpha, args.n_alpha, dtype=np.float32)
psi_list = np.linspace(-np.pi+np.pi/args.n_psi, np.pi-np.pi/args.n_psi, args.n_psi, dtype=np.float32)
states = States(
  r_list,
  alpha_list,
  psi_list,
  cycles=[None, np.pi * 2, np.pi * 2],
  terminal_states=[np.array([0., alpha, 0.], dtype=np.float32) for alpha in alpha_list]
)

action_list = []
for theta1 in np.cos(np.linspace(0, np.pi/2, args.n_theta)):
  for theta2 in np.cos(np.linspace(0, np.pi/2, args.n_theta)):
    for theta3 in np.cos(np.linspace(0, np.pi/2, args.n_theta)):
      action_list.append(
        np.array([2 - 2*theta1, 2 - 2*theta2, 2 - 2*theta3], dtype=np.float32) + np.finfo(np.float32).eps
      )
actions = Actions(
  action_list
)

env = DifferentialDriveKinematic(
  dt=args.env_dt, tol=np.finfo(np.float32).eps, timeout=args.env_timeout,
  observation_noise=np.zeros((3), dtype=np.float32),
  transition_noise=np.zeros((3), dtype=np.float32)
)
nmpc = NmpcCartesian(
  T=args.mpc_dt,
  N=args.mpc_time_horizon,
  R_u=args.mpc_regularization_control*np.eye(2),
  R_vel=np.diag(
    np.array([args.mpc_regularization_v, args.mpc_regularization_omega])
  ),
  w_max=2,
  a_max=1,
  v_max=0.2,
  omega_max=0.2 * 0.8 * 2 / 0.653,
  r_tol=args.mpc_r_tol
)

def simulate(state, action):
  reward = -1
  state = env.reset(state)
  u = nmpc.solve(state, action, warm_start=0)
  env.step(u)
  state = env.get_state()
  return state, reward

state_transition_prob = StateTransitionProbability(
  states = states, actions = actions,
)

rewards = Rewards(states, actions)

policy = Policy(states = states, actions = actions)

mdp = MarkovDecisionProcess(
  states = states,
  actions = actions,
  rewards = rewards,
  state_transition_probability = state_transition_prob,
  policy = policy,
  discount = args.mdp_discount
)

if args.sample:
  mdp.sample(simulate, n_repeat=args.repeats_per_sample_point, sample_reward=True)
  mdp.save(file_mdp)
  print('The MDP is recorded in {}'.format(file_mdp))
else:
  mdp.load(args.data_dir+'/mdp.npz')
  mdp.discount = args.mdp_discount
if args.solve:
  solver = eval(args.mdp_solver)(mdp)
  solver.solve(max_iteration=args.mdp_max_iter, tolerance=args.mdp_tol)

  solver.save(file_result)
  print('The results are recorded in {}'.format(file_result))
else:
  res = np.load(file_result)
  mdp.policy.update(res['policy'])

if args.test:
  policy = mdp.policy

  env.tol = args.env_pose_tol
  nmpc.r_tol = args.mpc_r_tol

  def simulate(queue, state):

    state = env.reset(state)
    done = False
    traj = []
    traj.append(env.get_state())
    u_prev = np.zeros((2))
    info = {'success':None}
    while not done:
      theta = policy.get_action(state)
      u = nmpc.solve(state, theta, u_prev=u_prev, warm_start=0)
      state, _, done, info = env.step(u)
      if np.linalg.norm(u) + np.linalg.norm(u - u_prev) < args.env_control_tol:
        done = True
      u_prev = u
      traj.append(env.get_state())
    queue.put(1)
    return traj, info['success']

  data = []
  for r in np.linspace(0.5, 1.5, 3):
    for alpha in np.linspace(-np.pi, np.pi, 19)[1:]:
      for psi in np.linspace(-np.pi, np.pi, 19)[1:]:
        data.append([r, alpha, psi])

  verbose = Verbose(True)
  queue = Manager().Queue()
  with Pool(os.cpu_count()) as p:
    trajs = p.starmap_async(simulate, [(queue, state) for state in data])
    counter = 0
    while counter < len(data):
      counter += queue.get()
      progress = counter / len(data)
      progress *= 100
      verbose('Simulation progress: %5.1f %%...'%(progress))
    verbose('Simulation is done.\n')
    trajs = trajs.get()

  err = []
  arrival_time = []
  for target, (traj, success) in zip(data, trajs):
    target.append(traj)
    target.append(success)

  np.save(
    file_test,
    np.array(data, dtype=np.object),
    allow_pickle=True
  )

  print_test(data, 3)
