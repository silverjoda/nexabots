import os
import numpy as np

class Gen:
    def __init__(self):
        self.template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "template.xml")
        self.test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "afgm_env.xml")

    def generate(self, N, goal):
        xg, yg = goal

        # Generate N cube positions
        range = 6
        cubes = []
        for i in np.linspace(-range,range, N):
            for j in np.linspace(-range,range, N):
                cubes.append(np.array([i, j]) + np.random.rand(2) * 3 - 1.5)

        idx = 0
        for pos in cubes[:]:
            x,y = pos
            if (abs(x) < 1.0 and abs(y) < 1.0) or (abs(x - xg) < 0.5 and abs(y - yg) < 0.5):
                del cubes[idx]
            else:
                idx += 1

        with open(self.template_path, "r") as in_file:
            buf = in_file.readlines()

        with open(self.test_path, "w") as out_file:
            for line in buf:
                if line == "<!-- DEFINE CUBES AND GOAL HERE -->\n":
                    for i, (x,y) in enumerate(cubes):
                        cube_str = "    <body> \n" \
                                   "        <geom conaffinity='1' condim='3' name='cube{}_geom' pos='{} {} 0.6' size='{} {} 0.6' rgba='1 1 1 1' mass='70' type='box'/>\n" \
                                   "     </body> \n".format(i,x,y, np.random.rand() / 3 + 0.3, np.random.rand() / 3 + 0.3)
                        line = line + cube_str
                    goal_str = "    <body> \n" \
                               "        <geom conaffinity='1' condim='3' name='goal_geom' pos='{} {} 0.6' size='0.3' rgba='1 0 0 1' mass='1' type='sphere'/>\n" \
                               "     </body> \n".format(xg, yg)
                    line = line + goal_str
                out_file.write(line)


if __name__ == "__main__":
    ant = Gen()
    ant.generate(3, np.random.rand(2) * 8 - 4)
