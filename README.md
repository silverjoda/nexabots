# Nexabots
To run an example environment with given terrain type:

### Add src and project directory to pythonpath
```
export PYTHONPATH="${PYTHONPATH}:/absolute_dir_to_project/nexabots/"
export PYTHONPATH="${PYTHONPATH}:/absolute_dir_to_project/nexabots/src"
```

### Navigate to environment
```
cd src/envs/locom_benchmarks/hex_locomotion  (or quad_locomotion, or snake_locomotion)
```

### Run demo example.
```
python hex_blind.py --terrain perlin
```

### Gives the list of available terrains for generation methods #1 and #2.
```
python quad_blind.py --help
```

### To run a reinforcement learning example with any evironment navigate to and run policy gradient algorithm on environment of your choice (you can set it in the script). Any custom environments should work if they implement the step and reset methods similarly as to OpenAI Gym

```
cd nexabots/src/algos/PG
```

```
python pg.py
```

