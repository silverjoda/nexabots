��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq csrc.policies
CYC_HEX_BS
qX9   /home/silverjoda/PycharmProjects/nexabots/src/policies.pyqX  class CYC_HEX_BS(nn.Module):
    def __init__(self):
        super(CYC_HEX_BS, self).__init__()

        self.phase_stepsize = 0.3
        self.phase_global = 0.0

        self.phase_scale_global = T.nn.Parameter(T.ones(1))
        #self.phase_offset_R = T.nn.Parameter(T.ones(1))
        self.phase_offset_joints = T.nn.Parameter(T.zeros(9))
        self.amplitude_offset_joints = T.nn.Parameter(T.zeros(9))
        self.amplitude_scale_joints = T.nn.Parameter(T.ones(9))


    def forward(self, _):
        phase_L = self.phase_global
        phase_R = (self.phase_global + np.pi) % (2 * np.pi)
        phase_LR_vec = T.tensor([phase_L, phase_L, phase_L, phase_R, phase_R, phase_R]).repeat(3)
        phase_offset_joints_expanded = T.cat([self.phase_offset_joints[0:3].repeat(2),
                                              self.phase_offset_joints[3:6].repeat(2),
                                              self.phase_offset_joints[6:].repeat(2)])
        amplitude_offset_joints_expanded = T.cat([self.amplitude_offset_joints[0:3].repeat(2),
                                                  self.amplitude_offset_joints[3:6].repeat(2),
                                                  self.amplitude_offset_joints[6:].repeat(2)])
        amplitude_scale_joints_expanded = T.cat([self.amplitude_scale_joints[0:3].repeat(2),
                                                  self.amplitude_scale_joints[3:6].repeat(2),
                                                  self.amplitude_scale_joints[6:].repeat(2)])

        act = amplitude_offset_joints_expanded + amplitude_scale_joints_expanded * T.sin(phase_LR_vec + phase_offset_joints_expanded).unsqueeze(0)
        self.phase_global = (self.phase_global + self.phase_stepsize * self.phase_scale_global) % (2 * np.pi)
        return act
qtqQ)�q}q(X   trainingq�X   _parametersqccollections
OrderedDict
q	)Rq
(X   phase_scale_globalqctorch._utils
_rebuild_parameter
qctorch._utils
_rebuild_tensor_v2
q((X   storageqctorch
FloatStorage
qX   94641683299040qX   cpuqKNtqQK K�qK�q�h	)RqtqRq�h	)Rq�qRqX   phase_offset_jointsqhh((hhX   94641683299040qhKNtqQKK	�qK�q�h	)Rq tq!Rq"�h	)Rq#�q$Rq%X   amplitude_offset_jointsq&hh((hhX   94641683299040q'hKNtq(QK
K	�q)K�q*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   amplitude_scale_jointsq1hh((hhX   94641683299040q2hKNtq3QKK	�q4K�q5�h	)Rq6tq7Rq8�h	)Rq9�q:Rq;uX   _buffersq<h	)Rq=X   _backward_hooksq>h	)Rq?X   _forward_hooksq@h	)RqAX   _forward_pre_hooksqBh	)RqCX   _state_dict_hooksqDh	)RqEX   _load_state_dict_pre_hooksqFh	)RqGX   _modulesqHh	)RqIX   phase_stepsizeqJG?�333333X   phase_globalqKh((hhX   94641730588128qLhKNtqMQK K�qNK�qO�h	)RqPtqQRqRub.�]q (X   94641683299040qX   94641730588128qe.       ��#>ՙ�����>�D��^?�y���F�텼?�\�?�?��=Xؗ?!��=��@��>)G:�% D�rڐ?����D��?y�@S9k?K�}?�s@�*"?'��?&��?
��       i�K@