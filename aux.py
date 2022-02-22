#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:33:38 2022

@author: marius
"""

import argparse
import numpy as np

# Parse and returns arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default='CDL-A')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--lmbda', type=float, default=0.3)
    parser.add_argument('--lifting', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--num_steps', type=int, default=1000)
    args = parser.parse_args()
    
    # Signal-to-noise ratio and pilot density (alpha)
    snr_range   = np.asarray(np.arange(-10, 17.5, 2.5))
    alpha       = float(args.alpha)
    channel     = args.channel
    # Optimization parameters
    num_steps   = int(args.num_steps)
    lifting     = int(args.lifting)
    lmbda       = float(args.lmbda)
    lr          = float(args.lr)
    
    return snr_range, alpha, num_steps, lifting, lmbda, lr, channel

# Textbook normalized MSE
def normalized_mse(est, truth):
    top    = np.sum(np.square(np.abs(est - truth)))
    bottom = np.sum(np.square(np.abs(truth)))
    
    return top / bottom