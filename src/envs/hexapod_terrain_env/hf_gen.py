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

        self.noise_dim = noise_dim
        self.pop_size = 24
        self.weight_decay = 0.0

        self.convnet = ConvGen(self.noise_dim)

        self.w = parameters_to_vector(self.convnet.parameters()).detach().numpy()
        print("N_conv params: {}".format(len(self.w)))
        self.es = cma.CMAEvolutionStrategy(self.w, 0.5)

        self.candidates = self.es.ask(self.pop_size)
#        self.es.tell(self.candidates, [0.] * self.pop_size)
        self.candidate_scores = []
        self.candidate_idx = 0


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

        return out.view(-1, 30, 120) * 70 + 70


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

if __name__ == "__main__":
    gen = EvoGen(12)
    gen.generate()