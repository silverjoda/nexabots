# nexabots
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
