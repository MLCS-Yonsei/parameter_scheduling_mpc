# parameter_scheduling_mpc
Finds the optimal parameter scheduing policy of MPC for differential-drive wheeled mobile robot by solving MDP via dynamic programming

## Dependencies
- CasADi: https://web.casadi.org/

## Usage
```
python3 solve_mdp.py
```

## Optional arguments
- env-dt: type=float, default=0.05
- env-pose_tol: type=float, default=1e-2
- env-control_tol: type=float, default=1e-2
- env-timeout: type=float, default=1000
- mpc-dt: type=float, default=0.05
- mpc-time_horizon: type=int, default=10
- mpc-regularization_control: type=float, default=1e-3
- mpc-regularization_v: type=float, default=0
- mpc-regularization_omega: type=float, default=0
- mpc-r_tol: type=float, default=5e-3
- n_r: type=int, default=40
- n_alpha: type=int, default=24
- n_psi: type=int, default=25
- n_theta: type=int, default=7
- repeats_per_sample_point: type=int, default=1
- mdp-discount: type=float, default=0.99
- mdp-max_iter: type=int, default=100000
- mdp-tol: type=float, default=1e-6
- mdp-solver: type=str, default='ValueIteration', choices=['PolicyIteration', 'ValueIteration']
- data_dir: type=str, default='data/mdp'
- sample: type=bool, default=True
- solve: type=bool, default=True
- test: type=bool, default=True
