import numpy as np

from utils import *
from curriculum import random, update_backward_reachable_set, sample_from_backward_reachable_set, backward_reachable
from problem import Problem
from rl import ppo
from collections import defaultdict
from data_logger import DataLogger
from time import strftime
from random_utils import fixed_random_seed
from plotting_with_theta_sections import visualize_starts, visualize_rollouts, plot_performance

import os
import argparse

import myenv
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("type", help="what kind of curriculum to employ",
                    type=str)
parser.add_argument("--debug", help="whether to print loads of information and create plots or not",
                    action="store_true")
#parser.add_argument("--brs_sample", help="how to sample from a backreachable set",
#                    type=str, default='contour_edges')

parser.add_argument("--brs_sample", help="how to sample from a backreachable set",type=str, default='contour_edges')
parser.add_argument("--run_name", help="name of run that determines where outputs are saved",
                    type=str, default=None)
parser.add_argument("--zero_goal_v", help="whether to make the goal have zero initial velocity",
                    action="store_true")
parser.add_argument("--finish_threshold", help="what fraction of starts must be successful to finish training.",
                    type=float, default=0.95)
parser.add_argument("--seed", help="what seed to use for the random number generators.",
                    type=int, default=2018)
parser.add_argument("--disturbance", help="what disturbance to use in the gym environment.",
                    type=str, default=None)
parser.add_argument("--gym_env", help="which gym environment to use.",
                    type=str, default='DrivingOrigin-v0')
parser.add_argument("--hover_at_end", help="whether to null velocity and rates at the end",
                    action="store_true")
parser.add_argument("--variation", help="what variation to use",
                    type=int, default=None)

args = parser.parse_args()

if args.type in ['backreach', 'random', 'ppo_only']:
    if args.run_name is not None:
        run_dir = args.run_name
    else:
        run_dir = args.type
else:
    parser.error('"%s" is not in ["backreach", "random", "ppo_only"]!' % args.type);

RUN_DIR = os.path.join(os.getcwd(), 'runs', 'PlanarQuad-v0_backreach_04-Nov-2018_23-10-58')
DATA_DIR = os.path.join(RUN_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
EVAL_DIR = os.path.join(RUN_DIR,'eval')

if __name__ == "__main__":
    with fixed_random_seed(args.seed):
        maybe_mkdir(RUN_DIR)
        maybe_mkdir(DATA_DIR)
        maybe_mkdir(MODEL_DIR)
        maybe_mkdir(EVAL_DIR)

        with fixed_random_seed(2018):
            problem = Problem(args.gym_env,
                              zero_goal_v=args.zero_goal_v,
                              disturbance=args.disturbance)
            if args.gym_env == 'DrivingOrigin-v0':
                num_iters = 100
                print('Goal State:', problem.goal_state)
                full_starts = problem.sample_behind_goal(problem.goal_state,
                                                         num_states=100,
                                                         zero_idxs=[3, 4])
            elif args.gym_env == 'PlanarQuad-v0':
                # NOTE: HERE I increase the iteration nums to see if the final performance is stable.
                num_iters = 40
                full_starts = [problem.env.unwrapped.start_state]
                problem.env.unwrapped.set_hovering_goal(args.hover_at_end)

        #  start state and its' distribution
        full_start_dist = list(zip(full_starts, uniform(full_starts)))

        ppo.create_session()
        eval_policy = ppo.create_policy('pi', problem)
        ppo.initialize()



        global_perf_metric = []
        stop_iter = 10
        for i in range(num_iters):
            if i < stop_iter:
                eval_policy.load_model(MODEL_DIR,iteration=i)
                perf_metric, rollout_num = evaluate(eval_policy,
                               full_start_dist,
                               problem,
                               debug=args.debug,
                               figfile=None)
            else:
                eval_policy.load_model(MODEL_DIR, iteration=stop_iter)
                perf_metric, rollout_num = evaluate(eval_policy,
                                                    full_start_dist,
                                                    problem,
                                                    debug=args.debug,
                                                    figfile=None)
            global_perf_metric.append(perf_metric)

        print("perf_metric",perf_metric)

        plot_performance(range(num_iters),global_perf_metric,xlabel='Algorithm iteration',ylabel='Eval reward from start states',
                         figfile=os.path.join(EVAL_DIR, 'eval_perf'))
