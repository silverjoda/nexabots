#!/bin/bash
#finds the pycache directories and delets them

find -name "__pycache__" -exec rm -r {} \;
find -name "*.pyc" -exec rm -r {} \;
