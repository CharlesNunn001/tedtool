#!/usr/bin/env python3

import argparse

from te_reader.tethinker import Examine
from te_reader.tethinker_prep import Assessment

def main_runner(args):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--library', required=True, help='The library to learn from')
parser.add_argument('-m', '--metadata', required=False, default=None, help='Metadata guidance file for library')
args = parser.parse_args()
main_runner(args)

