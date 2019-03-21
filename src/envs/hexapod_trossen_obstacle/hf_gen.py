import numpy as np
import cv2
import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import cma

class EvoGen():
    def __init__(self, noise_dim):
        super(EvoGen, self).__init__()
        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     "assets/hf_gen.png")

        self.sdir = "terrain_evogen_es.p".format()

        self.noise_dim = noise_dim
        self.pop_size = 24
        self.weight_decay = 0.01

        self.convnet = ConvGen(self.noise_dim)

        self.w = parameters_to_vector(self.convnet.parameters()).detach().numpy()
        print("N_conv params: {}".format(len(self.w)))
        self.es = cma.CMAEvolutionStrategy(self.w, 0.5)

        self.candidates = self.es.ask(self.pop_size)
#        self.es.tell(self.candidates, [0.] * self.pop_size)
        self.candidate_scores = []
        self.candidate_idx = 0


    def save(self):
        vector_to_parameters(torch.from_numpy(self.es.result.xbest).float(), self.convnet.parameters())
        T.save(self.convnet, self.sdir)
        print("Saved checkpoint, {}".format(self.sdir))


    def load(self):
        self.convnet = T.load(self.sdir)
        print("Loaded checkpoint, {}".format(self.sdir))


    def test_generate(self):
        print("Generating from loaded net")
        seed_noise = T.randn(1, self.noise_dim)
        with T.no_grad():
            mat = self.convnet(seed_noise)[0].numpy()
        mat = self.normalize_map(mat)

        mat[0, :] = 255
        mat[:, 0] = 255
        mat[-1, :] = 255
        mat[:, -1] = 255

        cv2.imwrite(self.filename, mat)


    def generate(self):
        if self.candidate_idx == self.pop_size:
            # Tell candidate scores
            self.es.tell(self.candidates, self.candidate_scores)
            self.candidates = self.es.ask(self.pop_size)
            if self.weight_decay > 0:
                self.candidates = [self.decay(c, self.weight_decay) for c in self.candidates]
            self.candidate_scores = []
            self.candidate_idx = 0
            self.es.disp()

        candidate = self.candidates[self.candidate_idx]
        self.candidate_idx += 1
        vector_to_parameters(torch.from_numpy(candidate).float(), self.convnet.parameters())
        seed_noise = T.randn(1, self.noise_dim)
        with T.no_grad():
            mat = self.convnet(seed_noise)[0].numpy()
        mat = self.normalize_map(mat)

        mat[0, :] = 255
        mat[:, 0] = 255
        mat[-1, :] = 255
        mat[:, -1] = 255

        cv2.imwrite(self.filename, mat)


    def feedback(self, r):
        if r is None: return
        self.candidate_scores.append(r + np.abs(cv2.imread(self.filename)).sum() * 0.0001)


    def normalize_map(self, X):
        return X


    def get_best(self):
        with torch.no_grad():
            vector_to_parameters(torch.from_numpy(self.es.result.xbest).float(), self.convnet.parameters())
            sol = self.convnet(np.random.randn(self.noise_dim)).squeeze(0).numpy()
            return self.normalize_map(sol)


    def decay(self, w, l):
        wpen = np.square(w) * l
        return w - wpen * (w > 0) + wpen * (w < 0)


class ConvGen(nn.Module):
    def __init__(self, noise_dim):
        super(ConvGen, self).__init__()
        self.noise_dim = noise_dim

        self.c1 = nn.ConvTranspose2d(self.noise_dim, 4, kernel_size=(3,3), stride=(1,1))
        self.c2 = nn.ConvTranspose2d(4, 3, kernel_size=(3,3), stride=(1,1))
        self.c3 = nn.ConvTranspose2d(3, 4, kernel_size=(3,3), stride=(1,1))

        T.nn.init.xavier_normal_(self.c1.weight)
        self.c1.bias.data.fill_(0.01)
        T.nn.init.xavier_normal_(self.c2.weight)
        self.c2.bias.data.fill_(0.01)
        T.nn.init.xavier_normal_(self.c3.weight)
        self.c3.bias.data.fill_(0.01)


    def forward(self, z):
        z = z.view(-1,self.noise_dim,1,1)

        out = F.leaky_relu(self.c1(z))
        out = F.interpolate(out, scale_factor=2)

        out = F.leaky_relu(self.c2(out))
        out = F.interpolate(out, scale_factor=2)

        out = T.tanh(self.c3(out))
        out = F.interpolate(out, size=(30, 30))

        return out.view(-1, 30, 120) * 50 + 50


class ManualGen:
    def __init__(self, noise_dim):
        self.N = 120
        self.M = 30
        self.div = 5

        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "assets/hf_gen.png")

        self.noise_dim = noise_dim
        self.convgen = ConvGen(self.noise_dim)

        self.genfun = self.gen_conv


    def generate(self):
        self.genfun()


    def gen_manual(self):
        mat = np.random.randint(0, 70, size=(self.M // self.div, self.N // self.div), dtype=np.uint8).repeat(self.div, axis=0).repeat(self.div,
                                                                                                             axis=1)
        mat[0, :] = 255
        mat[:, 0] = 255
        mat[-1, :] = 255
        mat[:, -1] = 255
        cv2.imwrite(self.filename, mat)


    def gen_conv(self):
        mat = self.convgen(T.randn(1, self.noise_dim)).detach().squeeze(0).numpy()

        mat[0, :] = 255
        mat[:, 0] = 255
        mat[-1, :] = 255
        mat[:, -1] = 255
        cv2.imwrite(self.filename, mat)


class HMGen:
    def __init__(self):
        self.N = 30
        self.M = 30

        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "assets/hf_gen.png")


    def generate(self):
        mat = cv2.imread(self.filename)
        mat -= mat.min()

        cv2.imwrite(self.filename, mat)


    def test_generate(self):
        mat = cv2.imread(self.filename)
        mat -= mat.min()
        cv2.imwrite(self.filename, mat)


    def load(self):
        pass


    def feedback(self, _):
        pass


def genstairs():
    N = 150
    M = 30

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/stairs.png")

    # Generate stairs
    mat = np.zeros((M, N))

    stair_height = 20
    stair_width = 3
    current_height = 0

    for i in range(6):
        mat[:, 10 + i * stair_width : 10 + i * stair_width + stair_width] = current_height
        current_height += stair_height

    for i in range(3):
        mat[:, 28 + i * stair_width :  28 + i * stair_width + stair_width] = current_height

    for i in range(4):
        mat[:, 37 + i * stair_width : 37 + i * stair_width + stair_width] = current_height
        current_height -= stair_height

    for i in range(2):
        mat[:, 49 + i * stair_width :  49 + i * stair_width + stair_width] = current_height

    for i in range(3):
        mat[:, 55 + i * stair_width: 55 + i * stair_width + stair_width] = current_height
        current_height -= stair_height

    #---
    for i in range(12):
        mat[:, 55 + 10 + i * stair_width : 55 + 10 + i * stair_width + stair_width] = current_height
        current_height += stair_height

    for i in range(15):
        mat[:, 70 + 28 + i * stair_width : 70 +  28 + i * stair_width + stair_width] = current_height


    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


def genflat():
    N = 180
    M = 30

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/flat.png")

    mat = np.zeros((M,N))

    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


def gentiles():
    N = 180
    M = 30
    div = 5

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/tiles.png")

    # Generate stairs
    mat = np.random.randint(0, 70, size=(M // div, N // div), dtype=np.uint8).repeat(div, axis=0).repeat(div,axis=1)

    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


def genflattiles():
    N = 90
    M = 30
    div = 5

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/flattiles.png")

    # Generate stairs
    mat = np.random.randint(0, 70, size=(M // div, N // div), dtype=np.uint8).repeat(div, axis=0).repeat(div,axis=1)

    mat[:, :30] = 0
    mat[:, 60:] = 0

    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


def genflat_tile_hybrid():
    scale = 15
    N = 24 * scale
    M = 2 * scale
    div = 5

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/flat_tile_hybrid.png")

    # Generate stairs
    mat = np.random.randint(0, 70, size=(M // div, N // div), dtype=np.uint8).repeat(div, axis=0).repeat(div, axis=1)

    indeces = [1, 4, 7, 14, 20]
    gaps = [10, 15, 20, 10, 15]
    for idx, gap in zip(indeces, gaps):
        mat[:, idx * scale : idx * scale + gap] = 0

    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


def gen_flat_pipe():
    N = 90
    M = 30

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/flatpipe.png")

    # Generate stairs
    mat = np.zeros((M, N))

    pipe = np.ones((M, 30))
    pipe *= np.square(np.linspace(-16, 16, M))

    #mat[:,  0: 30] = np.transpose(pipe)
    mat[:, 30: 60] = np.transpose(pipe)
    #mat[:, 60: 90] = np.transpose(pipe)

    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


if __name__ == "__main__":
    gen_flat_pipe()
    #gen = HMGen(12)
    #gen.generate()