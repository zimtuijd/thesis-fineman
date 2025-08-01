import sys
import os

# Program for generating stdin from a .mtx file
# Run using: python3 mtx.py [path to .mtx file]
# Before inputting a .mtx file, remove the comments in % manually

if len(sys.argv) == 2:
  fn = sys.argv[1]

if os.path.isfile(fn):
  with open(fn, 'r') as file:

    line = file.readline().rstrip().split(' ', 1)[1]
    print(line, end=' ')

    while line := file.readline():
      line = line.split(' ')
      for i in line:
        print(int(i) - 1, end=' ')