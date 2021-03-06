��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq csrc.policies
CYC_HEX_NN
qX9   /home/silverjoda/PycharmProjects/nexabots/src/policies.pyqX�  class CYC_HEX_NN(nn.Module):
    def __init__(self, obs_dim):
        super(CYC_HEX_NN, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = 18 + 1
        self.hidden_dim = 8

        self.phase_stepsize = 0.3
        self.phase_global = 0

        self.f1 = nn.Linear(self.obs_dim, self.act_dim)
        #self.f2 = nn.Linear(self.hidden_dim, self.act_dim)

    def forward(self, x):
        x = T.ones((1,4))
        #x1 = T.tanh(self.f1(x))
        out = self.f1(x)

        act = T.sin(self.phase_global + out[:, :18])
        self.phase_global = (self.phase_global + self.phase_stepsize * (out[:, 18] + 1)) % (2 * np.pi)
        return act
qtqQ)�q}q(X   trainingq�X   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _state_dict_hooksqh	)RqX   _load_state_dict_pre_hooksqh	)RqX   _modulesqh	)RqX   f1q(h ctorch.nn.modules.linear
Linear
qXU   /home/silverjoda/SW/miniconda3/lib/python3.7/site-packages/torch/nn/modules/linear.pyqX�	  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
qtqQ)�q}q(h�hh	)Rq (X   weightq!ctorch._utils
_rebuild_parameter
q"ctorch._utils
_rebuild_tensor_v2
q#((X   storageq$ctorch
FloatStorage
q%X   94205670042240q&X   cpuq'K_Ntq(QK KK�q)KK�q*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   94205670042240q2h'K_Ntq3QKLK�q4K�q5�h	)Rq6tq7Rq8�h	)Rq9�q:Rq;uhh	)Rq<hh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBX   in_featuresqCKX   out_featuresqDKubsX   obs_dimqEKX   act_dimqFKX
   hidden_dimqGKX   phase_stepsizeqHG?�333333X   phase_globalqIh#((h$h%X   94205671146368qJh'KNtqKQK K�qLK�qM�h	)RqNtqORqPub.�]q (X   94205670042240qX   94205671146368qe._       �/�>��'�ޥl?h�?�����?��B?�?TU��i�H��S@�{@���$�Q�wت�3l��1�=�8�>4W˿��0���H���
��2��ө>�F�PS�?�����/������1��X;@�	K?f�?��_v��'	����?�@E7����z��@��n���G�?cu�N�p�C1$��?I�c>>%����%�DȢ?L]�t���\D��zz�N��?��z�6�w>f#!����>>DN@����-�?���s��@�:z��3���?]�1@�ſ:;@wCF?�>4�߾���#I'��<@v72����>���JZS������4=S������?o ����#@-����+�?��=@d��>$�w?q�?&(/�6 �?��=       y�8@