import tensorflow.contrib.slim as slim
import tflearn

bn_mom = 0.9
#bn_mom = 0.9997

def Act(data, act_type, name):
    #ignore param act_type, set it in this function
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)
    slim.batchnorm(inputs,scope='%s%s_batchnorm' %(name, suffix), decay=bn_mom )
    act = tf.learn()
    return act

def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    slim.con2d
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    slim.con2d(inputs=data,)
    return conv


def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv=slim.conv2d(inputs=data,num_group)
    conv = slim.conv2d(inputs=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = slim.seperable_convolution2d(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj

def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity


def get_symbol(num_classes, **kwargs):
    global bn_mom
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd_mult = kwargs.get('wd_mult', 1.)
    data = mx.symbol.Variable(name="data")
    data = data-127.5
    data = data*0.0078125
    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
    conv_23 = DResidual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23")
    conv_3 = Residual(conv_23, num_block=4, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, name="res_3")
    conv_34 = DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
    conv_4 = Residual(conv_34, num_block=6, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_4")
    conv_45 = DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
    conv_5 = Residual(conv_45, num_block=2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_5")

    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
    conv_6_dw = Linear(conv_6_sep, num_filter=512, num_group=512, kernel=(7,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")
    #conv_6_dw = mx.symbol.Dropout(data=conv_6_dw, p=0.4)
    _weight = mx.symbol.Variable("fc1_weight", shape=(num_classes, 512), lr_mult=1.0, wd_mult=wd_mult)

    conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, weight=_weight, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    return fc1
