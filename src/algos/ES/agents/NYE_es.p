��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq csrc.policies
CYC_HEX
qX9   /home/silverjoda/PycharmProjects/nexabots/src/policies.pyqXQ  class CYC_HEX(nn.Module):
    def __init__(self):
        super(CYC_HEX, self).__init__()

        self.phase_stepsize = 0.3
        self.phase_global = 0

        self.phase_scale_global = T.nn.Parameter(T.ones(1))
        self.phase_offset_joints = T.nn.Parameter(T.zeros(18))
        self.amplitude_offset_joints = T.nn.Parameter(T.zeros(2))
        self.amplitude_scale_joints = T.nn.Parameter(T.ones(3))


    def forward(self, _):
        #act = 0.6 * T.sin(self.phase_global + self.phase_offset_joints).unsqueeze(0)  # Original
        act = T.cat((T.tensor([0.]), self.amplitude_offset_joints)).repeat(6) + self.amplitude_scale_joints.repeat(6) * T.sin(self.phase_global + self.phase_offset_joints).unsqueeze(0)
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
qX   94257298844896qX   cpuqKNtqQK K�qK�q�h	)RqtqRq�h	)Rq�qRqX   phase_offset_jointsqhh((hhX   94257298844896qhKNtqQKK�qK�q�h	)Rq tq!Rq"�h	)Rq#�q$Rq%X   amplitude_offset_jointsq&hh((hhX   94257298844896q'hKNtq(QKK�q)K�q*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   amplitude_scale_jointsq1hh((hhX   94257298844896q2hKNtq3QKK�q4K�q5�h	)Rq6tq7Rq8�h	)Rq9�q:Rq;uX   _buffersq<h	)Rq=X   _backward_hooksq>h	)Rq?X   _forward_hooksq@h	)RqAX   _forward_pre_hooksqBh	)RqCX   _state_dict_hooksqDh	)RqEX   _load_state_dict_pre_hooksqFh	)RqGX   _modulesqHh	)RqIX   phase_stepsizeqJG?�333333X   phase_globalqKh((hhX   94257284932368qLhKNtqMQK K�qNK�qO�h	)RqPtqQRqRub.�]q (X   94257284932368qX   94257298844896qe.       C�@       ȼ�?Qj�
ݽ��%�I��s���7꾮����@�?+��>����ou>u��?ݼ(����`O?�ܾ���?.~��d���K,>���?`�@?�/G?