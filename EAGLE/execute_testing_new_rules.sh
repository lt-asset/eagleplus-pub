#!/bin/bash

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="./$now.log"

exec &>$LOG_FILE

# the last argument sets the the numer of model to test
python execute_testing_new_rules.py 1
