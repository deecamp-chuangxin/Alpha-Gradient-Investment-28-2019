# coding=utf-8

import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # Training arg 
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='train', help='train | test | backtest')
    # Data arg
    # TODO: If not use 
    parser.add_argument('--tushare_token', type=str)
    # TODO: This arg can be combine into one?
    parser.add_argument('--data_mode', type=str, default='baseline', help="baseline| , make origin baseline data or other stocks infomations.")
    parser.add_argument('--data_source', type=str, default='sz50', help='zz500 | sz50 | hs300, choose your stocks poll: 中证500, 上证50 or 沪深300')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    
    # TODO: we can add more args here

    # Model arg
    parser.add_argument('--version', type=str, default="TODO:")
    parser.add_argument('--model_dir', type=str, default='./model_zoo')


    # Strategy arg
    # TODO:
    


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    opt = parse_opt()
    print(opt.train_mode)
