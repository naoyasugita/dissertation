import keras
import numpy as np
from keras.layers import (Activation, Add, Conv2D, Conv2DTranspose,
                          Convolution2D, Input, InputLayer, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D
from keras.models import Model
from keras.utils import plot_model


class Encoder:
    def __init__(self,
                 data,
                 filters,
                 kernel_size,
                 strides,
                 input_flg=False):
        self.data = data
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.input_flg = input_flg
        self.data_format = 'channels_first'

    def forward(self):
        if self.input_flg:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='same',
                       input_shape=(3, 256, 256),
                       kernel_initializer='he_normal',
                       data_format=self.data_format)(self.data)
        else:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       data_format=self.data_format)(self.data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # def input_layer(self, shape=(3, 256, 256)):
    #     input_ = Input(shape=shape)
    #     return input_


class Residual:
    def __init__(self,
                 data,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 upsampling_flg=False,
                 connect_flg=False,
                 activate=True):
        self.data = data
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsampling_flg = upsampling_flg
        self.connect_flg = connect_flg
        self.activate = activate
        self.data_format = 'channels_first'

    def forward(self):
        if self.upsampling_flg:
            x = UpSampling2D(data_format=self.data_format)(self.data)
            return x

        if self.connect_flg:
            x = Add()([self.data[0], self.data[1]])
            return x

        else:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       data_format=self.data_format)(self.data)
            x = BatchNormalization()(x)
            if self.activate:
                x = Activation('relu')(x)
            return x


class Decoder:
    def __init__(self,
                 data,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 upsampling_flg=False,
                 output=False):
        self.data = data
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsampling_flg = upsampling_flg
        self.output = output
        self.data_format = 'channels_first'

    def forward(self):
        if self.output:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='same',
                       kernel_initializer='glorot_normal',
                       data_format=self.data_format)(self.data)
            output = Activation('sigmoid')(x)
            return output

        if self.upsampling_flg:
            x = UpSampling2D(data_format=self.data_format)(self.data)
            return x

        else:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       data_format=self.data_format)(self.data)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x


class FCN:
    def __init__(self):
        # encoder
        self.input_ = Input(shape=(3, 256, 256))
        # オリジナル
        self.encoder = Encoder(self.input_, 64, (5, 5),
                               (2, 2), input_flg=True).forward()
        self.encoder = Encoder(self.encoder, 64, (3, 3), (1, 1)).forward()
        self.encoder = Encoder(self.encoder, 128, (3, 3), (2, 2)).forward()
        self.add_input = Encoder(self.encoder, 128, (3, 3), (1, 1)).forward()

        # =================== GCN追加 =================
        # self.conv1 = BASE_GCN(self.input_, 7, 64, strides=(2, 2)).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 64, order='one').forward()
        # self.conv2 = BASE_GCN(self.input_, 7, 64,
        #                       order='one', strides=(2, 2)).forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 64).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (5, 5), filters=64, connect_flg=True, activate=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 64).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 64, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv, 7, 64, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 64).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=64, connect_flg=True, activate=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 128, strides=(2, 2)).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 128, order='one').forward()
        # self.conv2 = BASE_GCN(
        #     self.conv, 7, 128, order='one', strides=(2, 2)).forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 128).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=128, connect_flg=True, activate=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 128).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 128, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv, 7, 128, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 128).forward()
        # self.add_input = BASE_GCN([self.conv1, self.conv2],
        #                           (3, 3), filters=128, connect_flg=True, activate=True).forward()

        # recurrent netwoek
        self.res = Residual(self.add_input, 128).forward()
        self.res = Residual(self.res, 128, strides=(2, 2)).forward()
        self.res = Residual(self.res, 256).forward()
        self.res = Residual(self.res, 256).forward()
        self.res = Residual(self.res, 256, upsampling_flg=True).forward()
        self.res = Residual(self.res, 128).forward()
        self.res = Residual(self.res, 128).forward()
        self.add_input = Residual([self.res, self.add_input],
                                  128, connect_flg=True).forward()

        self.res = Residual(self.add_input, 128).forward()
        self.res = Residual(self.res, 128, strides=(2, 2)).forward()
        self.res = Residual(self.res, 256).forward()
        self.res = Residual(self.res, 256).forward()
        self.res = Residual(self.res, 256, upsampling_flg=True).forward()
        self.res = Residual(self.res, 128).forward()
        self.res = Residual(self.res, 128).forward()
        self.add_input = Residual([self.res, self.add_input],
                                  128, connect_flg=True).forward()

        self.res = Residual(self.add_input, 128).forward()
        self.res = Residual(self.res, 128, strides=(2, 2)).forward()
        self.res = Residual(self.res, 256).forward()
        self.res = Residual(self.res, 256).forward()
        self.res = Residual(self.res, 256, upsampling_flg=True).forward()
        self.res = Residual(self.res, 128).forward()
        self.res = Residual(self.res, 128).forward()
        self.add_input = Residual([self.res, self.add_input],
                                  128, connect_flg=True).forward()

        # decoder
        # オリジナル
        self.decode = Decoder(self.add_input, 128).forward()
        self.decode = Decoder(
            self.decode, 128, upsampling_flg=True).forward()
        self.decode = Decoder(self.decode, 64).forward()
        self.decode = Decoder(self.decode, 64).forward()
        self.decode = Decoder(self.decode, 64, upsampling_flg=True).forward()
        self.decode = Decoder(self.decode, 32).forward()
        self.decode = Decoder(self.decode, 32).forward()
        self.output = Decoder(self.decode, 3).forward()

        self.model = Model(inputs=self.input_, outputs=self.output)

        # =================== GCN追加 =================
        # self.conv1 = BASE_GCN(self.add_input, 7, 128).forward()
        # self.conv1 = BASE_GCN(self.conv1, 5, 128, order='one').forward()
        # self.conv2 = BASE_GCN(self.add_input, 7, 128, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 5, 128).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=128, connect_flg=True, activate=True).forward()

        # self.conv = Decoder(
        #     self.conv, 128, upsampling_flg=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 64).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 64, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv, 7, 64,
        #                       order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 64).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=64, connect_flg=True, activate=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 64).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 64, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv, 7, 64, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 64).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=64, connect_flg=True, activate=True).forward()

        # self.conv = Decoder(
        #     self.conv, 64, upsampling_flg=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 32).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 32, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv, 7, 32, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 32).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=64, connect_flg=True, activate=True).forward()

        # self.conv1 = BASE_GCN(self.conv, 7, 32).forward()
        # self.conv1 = BASE_GCN(self.conv1, 7, 32, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv, 7, 32, order='one').forward()
        # self.conv2 = BASE_GCN(self.conv2, 7, 32).forward()
        # self.conv = BASE_GCN([self.conv1, self.conv2],
        #                      (3, 3), filters=32, connect_flg=True, activate=True).forward()

        # self.output = Decoder(self.conv, 3).forward()

        # self.model = Model(inputs=self.input_, outputs=self.output)

    def build(self):
        return self.model


class BASE_GCN:
    def __init__(self,
                 data,
                 kernel_size,
                 filters,
                 strides=(1, 1),
                 connect_flg=False,
                 order='k',
                 data_format="channels_first",
                 activate=False):
        self.data = data
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        # self.input_channel = input_channel
        # self.output_channel = output_channel
        self.connect_flg = connect_flg
        self.order = order
        self.data_format = data_format
        self.activate = activate

    def forward(self):
        if self.connect_flg:
            x = Add()([self.data[0], self.data[1]])
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='same',
                       data_format=self.data_format)(x)
            if self.activate:
                x = Activation('relu')(x)
            return x

        if self.order == 'k':
            x = Conv2D(filters=self.filters,
                       kernel_size=(self.kernel_size, 1),
                       strides=self.strides,
                       padding='same',
                       data_format=self.data_format)(self.data)
            x = BatchNormalization()(x)
            if self.activate:
                x = Activation('relu')(x)
            return x

        if self.order == 'one':
            x = Conv2D(filters=self.filters,
                       kernel_size=(1, self.kernel_size),
                       strides=self.strides,
                       padding='same',
                       data_format=self.data_format)(self.data)
            x = BatchNormalization()(x)
            if self.activate:
                x = Activation('relu')(x)
            return x


class PretrainedGCN:
    def __init__(self, data):
        self.data = data
        self.conv1 = BASE_GCN(self.data, 11, (1, 1), activate=True).forward()
        self.conv1 = BASE_GCN(self.conv1, 11, (1, 1), order='one',
                              activate=True).forward()
        self.conv2 = BASE_GCN(self.data, 11, (1, 1), order='one',
                              activate=True).forward()
        self.conv2 = BASE_GCN(self.conv2, 11, (1, 1), activate=True).forward()
        self.add = BASE_GCN([self.conv1, self.conv2], 1, (1, 1),
                            connect_flg=True).forward()
        self.gcn = BASE_GCN(self.add, 1, (1, 1)).forward()
        self.gcn = BASE_GCN([self.data, self.gcn], 1, (1, 1),
                            connect_flg=True).forward()

    def forward(self):
        return self.gcn


class GCN:
    def __init__(self, data, filters=21):
        self.data = data
        self.filters = filters
        self.conv1 = BASE_GCN(self.data, 3, 21).forward()
        self.conv1 = BASE_GCN(self.conv1, 3, 21, order='one').forward()
        self.conv2 = BASE_GCN(self.data, 3, 21, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 3, 21).forward()
        self.gcn = BASE_GCN([self.conv1, self.conv2],
                            3, 21, connect_flg=True).forward()

    def forward(self):
        return self.gcn


class BASE_BR:
    def __init__(self,
                 data,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 filters=21,
                 connect_flg=False,
                 order="k",
                 activate=False,
                 data_format="channels_first",
                 output_flg=False):
        self.data = data
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        # self.input_channel = input_channel
        # self.output_channel = output_channel
        self.connect_flg = connect_flg
        self.order = order
        self.activate = activate
        self.data_format = data_format
        self.output_flg = output_flg

    def forward(self):
        if self.output_flg:
            x = Conv2D(filters=3,
                       kernel_size=self.kernel_size,
                       strides=(1, 1),
                       padding='same',
                       kernel_initializer='glorot_normal',
                       data_format=self.data_format)(self.data)
            output = Activation('sigmoid')(x)
            return output

        if self.connect_flg:
            x = Add()([self.data[0], self.data[1]])
            return x

        else:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=(1, 1),
                       padding='same',
                       kernel_initializer='he_normal',
                       data_format=self.data_format)(self.data)
            if self.activate:
                x = Activation('relu')(x)
            return x


class BR:
    def __init__(self, data):
        self.data = data
        self.conv1 = BASE_BR(self.data, activate=True).forward()
        self.conv2 = BASE_BR(self.conv1).forward()
        self.br = BASE_BR([self.data, self.conv2], connect_flg=True).forward()

    def forward(self):
        return self.br


class BASE_RES:
    def __init__(self,
                 data,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 data_format="channels_first",
                 connect_flg=False,
                 input_flg=False,
                 output_flg=False):
        self.data = data
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.connect_flg = connect_flg
        self.input_flg = input_flg
        self.output_flg = output_flg

    def forward(self):
        if self.input_flg:
            x = MaxPool2D(pool_size=(3, 3),
                          strides=(2, 2),
                          padding='same',
                          data_format=self.data_format)(self.data)
            return x
        if self.connect_flg:
            x = Add()([self.data[0], self.data[1]])
            return x
        if self.output_flg:
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       strides=(2, 2),
                       padding='same',
                       data_format=self.data_format)(self.data)
            return x
        x = Conv2D(filters=self.filters,
                   kernel_size=self.kernel_size,
                   strides=self.strides,
                   padding='same',
                   data_format=self.data_format)(self.data)
        return x


class RES_GCN:
    def __init__(self, data):
        self.data = data

        self.res2 = BASE_RES(self.data, 1, 1, input_flg=True).forward()
        # self.res2 = BASE_RES(self.res2, 1, 1).forward()
        self.res2 = BASE_RES(self.res2, 64, (1, 1)).forward()
        self.res2 = BASE_RES(self.res2, 64, (3, 3)).forward()
        # self.res2 = BASE_RES(self.res2, 256, (1, 1), strides=(2, 2)).forward()
        self.res2 = BASE_RES(self.res2, 256, (1, 1), output_flg=True).forward()
        self.res2 = BASE_RES([self.data, self.res2],
                             256, (1, 1), connect_flg=True).forward()

        self.res3 = BASE_RES(self.res2, 128, (1, 1)).forward()
        self.res3 = BASE_RES(self.res3, 128, (3, 3)).forward()
        self.res3 = BASE_RES(self.res3, 512, (1, 1)).forward()
        self.res3 = BASE_RES([self.res2, self.res3], 512,
                             (1, 1), connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.res3, 5, filters=85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, filters=85, order='one').forward()
        self.conv2 = BASE_GCN(self.data, 5, filters=85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, filters=85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             (1, 1), filters=1024, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.res4, 128, filters=128).forward()
        self.conv1 = BASE_GCN(
            self.conv1, 128, filters=128, order='one').forward()
        self.conv2 = BASE_GCN(self.data, 128, filters=128,
                              order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 128, filters=128).forward()
        self.res5 = BASE_GCN([self.conv1, self.conv2],
                             (1, 1), filters=2048, connect_flg=True).forward()

    def forward(self, layer_name):
        pass

    def forward2(self):
        return self.res2

    def forward3(self):
        return self.res3

    def forward4(self):
        return self.res4

    def forward5(self):
        return self.res5


class RES:
    def __init__(self, data):
        self.data = data

        self.res2 = BASE_RES(self.data, 1, 1, input_flg=True).forward()
        self.res2 = BASE_RES(self.res2, 64, (1, 1)).forward()
        self.res2 = BASE_RES(self.res2, 64, (3, 3)).forward()
        self.res2 = BASE_RES(self.res2, 256, (1, 1)).forward()
        self.res2 = BASE_RES([self.data, self.res2],
                             256, (1, 1), connect_flg=True)

        self.res3 = BASE_RES(self.res2, 128, (1, 1)).forward()
        self.res3 = BASE_RES(self.res3, 128, (3, 3)).forward()
        self.res3 = BASE_RES(self.res3, 512, (1, 1)).forward()
        self.res3 = BASE_RES([self.res2, self.res3], 512,
                             (1, 1), connect_flg=True)

        self.res4 = BASE_RES(self.res3, 85, (1, 5)).forward()
        self.res4 = BASE_RES(self.res4, 85, (5, 1)).forward()
        self.res4 = BASE_RES(self.res4, 1024, (1, 1)).forward()

        self.res5 = BASE_RES(self.res4, 128, (7, 1)).forward()
        self.res5 = BASE_RES(self.res5, 128, (1, 7)).forward()
        self.res5 = BASE_RES(self.res5, 2048, (1, 1)).forward()

    def forward2(self):
        return self.res2

    def forward3(self):
        return self.res3

    def forward4(self):
        return self.res4

    def forward5(self):
        return self.res5
        # if layer_name == 'res2':
        #     return self.res2
        # if layer_name == 'res3':
        #     return self.res3
        # if layer_name == 'res4':
        #     return self.res4
        # if layer_name == 'res5':
        #     return self.res5


class Deconv:
    def __init__(self, data, kernel_size=(3, 3), strides=(2, 2), filters=21, data_format="channels_first"):
        self.data = data
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format

    def forward(self):
        x = Conv2DTranspose(filters=self.filters,
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding='same',
                            data_format=self.data_format)(self.data)
        return x


class LKM:
    def __init__(self):
        self.input_ = Input(shape=(3, 256, 256))
        self.conv1 = BASE_RES(self.input_, 64, (7, 7),
                              strides=(2, 2)).forward()
###################res2#########################
        # self.pool = BASE_RES(self.conv1, 64, (3, 3), input_flg=True).forward()
        self.res2 = BASE_RES(self.conv1, 64, (1, 1), strides=(2, 2)).forward()
        self.res2 = BASE_RES(self.res2, 64, (3, 3)).forward()
        self.out = BASE_RES(self.res2, 256, (1, 1)).forward()
        # self.out = BASE_RES([self.pool, self.res2],
        #                     1, (1, 1), connect_flg=True).forward()

        self.res2 = BASE_RES(self.out, 64, (1, 1)).forward()
        self.res2 = BASE_RES(self.res2, 64, (3, 3)).forward()
        self.res2 = BASE_RES(self.res2, 256, (1, 1)).forward()
        self.out = BASE_RES([self.out, self.res2],
                            1, (1, 1), connect_flg=True).forward()

        self.res2 = BASE_RES(self.out, 64, (1, 1)).forward()
        self.res2 = BASE_RES(self.res2, 64, (3, 3)).forward()
        self.res2 = BASE_RES(self.res2, 256, (1, 1)).forward()
        self.res2 = BASE_RES([self.out, self.res2],
                             1, (1, 1), connect_flg=True).forward()

###################res3#########################
        self.res3 = BASE_RES(self.res2, 128, (1, 1), strides=(2, 2)).forward()
        self.res3 = BASE_RES(self.res3, 128, (3, 3)).forward()
        self.out = BASE_RES(self.res3, 512, (1, 1)).forward()
        # self.out = BASE_RES([self.res2, self.res3],
        #                     1, (1, 1), connect_flg=True).forward()

        self.res3 = BASE_RES(self.out, 128, (1, 1)).forward()
        self.res3 = BASE_RES(self.res3, 128, (3, 3)).forward()
        self.res3 = BASE_RES(self.res3, 512, (1, 1)).forward()
        self.out = BASE_RES([self.out, self.res3],
                            1, (1, 1), connect_flg=True).forward()

        self.res3 = BASE_RES(self.out, 128, (1, 1)).forward()
        self.res3 = BASE_RES(self.res3, 128, (3, 3)).forward()
        self.res3 = BASE_RES(self.res3, 512, (1, 1)).forward()
        self.out = BASE_RES([self.out, self.res3],
                            1, (1, 1), connect_flg=True).forward()

        self.res3 = BASE_RES(self.out, 64, (1, 1)).forward()
        self.res3 = BASE_RES(self.res3, 64, (3, 3)).forward()
        self.res3 = BASE_RES(self.res3, 512, (1, 1)).forward()
        self.res3 = BASE_RES([self.out, self.res3],
                             1, (1, 1), connect_flg=True).forward()

###################res4#########################
        self.conv1 = BASE_GCN(self.res3, 5, 85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.res3, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, 85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             1, 1024, connect_flg=True).forward()
        self.out = BASE_GCN(self.res4, 1, 1024, strides=(2, 2)).forward()
        # self.out = BASE_GCN([self.res3, self.res4],
        #                     1, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 5, 85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.out, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, 85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             1, 1024, connect_flg=True).forward()
        self.res4 = BASE_GCN(self.res4, 1, 1024).forward()
        self.out = BASE_GCN([self.out, self.res4],
                            1, 1024, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 5, 85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.out, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, 85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             1, 1024, connect_flg=True).forward()
        self.res4 = BASE_GCN(self.res4, 1, 1024).forward()
        self.out = BASE_GCN([self.out, self.res4],
                            1, 1024, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 5, 85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.out, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, 85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             1, 1024, connect_flg=True).forward()
        self.res4 = BASE_GCN(self.res4, 1, 1024).forward()
        self.out = BASE_GCN([self.out, self.res4],
                            1, 1024, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 5, 85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.out, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, 85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             1, 1024, connect_flg=True).forward()
        self.res4 = BASE_GCN(self.res4, 1, 1024).forward()
        self.out = BASE_GCN([self.out, self.res4],
                            1, 1024, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 5, 85).forward()
        self.conv1 = BASE_GCN(self.conv1, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.out, 5, 85, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 5, 85).forward()
        self.res4 = BASE_GCN([self.conv1, self.conv2],
                             1, 1024, connect_flg=True).forward()
        self.res4 = BASE_GCN(self.res4, 1, 1024).forward()
        self.res4 = BASE_GCN([self.out, self.res4],
                             1, 1024, connect_flg=True).forward()

###################res5#########################
        self.conv1 = BASE_GCN(self.res4, 7, 128).forward()
        self.conv1 = BASE_GCN(self.conv1, 7, 128,
                              order='one').forward()
        self.conv2 = BASE_GCN(self.res4, 7, 128, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 7, 128).forward()
        self.res5 = BASE_GCN([self.conv1, self.conv2],
                             1, 2048, connect_flg=True).forward()
        self.out = BASE_GCN(self.res5, 1, 2048, strides=(2, 2)).forward()
        # self.out = BASE_GCN([self.res4, self.res5],
        #                     1, 1, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 7, 128).forward()
        self.conv1 = BASE_GCN(self.conv1, 7, 128,
                              order='one').forward()
        self.conv2 = BASE_GCN(self.out, 7, 128, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 7, 128).forward()
        self.res5 = BASE_GCN([self.conv1, self.conv2],
                             1, 2048, connect_flg=True).forward()
        self.res5 = BASE_GCN(self.res5, 1, 2048).forward()
        self.out = BASE_GCN([self.out, self.res5],
                            1, 2048, connect_flg=True).forward()

        self.conv1 = BASE_GCN(self.out, 7, 128).forward()
        self.conv1 = BASE_GCN(self.conv1, 7, 128,
                              order='one').forward()
        self.conv2 = BASE_GCN(self.out, 7, 128, order='one').forward()
        self.conv2 = BASE_GCN(self.conv2, 7, 128).forward()
        self.res5 = BASE_GCN([self.conv1, self.conv2],
                             1, 2048, connect_flg=True).forward()
        self.res5 = BASE_GCN(self.res5, 1, 2048).forward()
        self.res5 = BASE_GCN([self.out, self.res5],
                             1, 2048, connect_flg=True).forward()

        # self.res_gcn = RES_GCN(self.conv1)
        # self.res2 = self.res_gcn.forward2()
        # self.res3 = self.res_gcn.forward3()
        # self.res4 = self.res_gcn.forward4()
        # self.res5 = self.res_gcn.forward5()
        # self.res3 = RES(self.res2).forward('res3')
        # self.res4 = RES_GCN(self.res3).forward('res4')
        # self.res5 = RES_GCN(self.res4).forward('res5')

        self.gcn5 = GCN(self.res5).forward()
        self.br5 = BR(self.gcn5).forward()
        self.deconv5 = Deconv(self.br5).forward()

        self.gcn4 = GCN(self.res4).forward()
        self.br4 = BR(self.gcn4).forward()
        self.add4 = Add()([self.deconv5, self.br4])
        self.br4 = BR(self.add4).forward()
        self.deconv4 = Deconv(self.br4).forward()

        self.gcn3 = GCN(self.res3).forward()
        self.br3 = BR(self.gcn3).forward()
        self.add3 = Add()([self.deconv4, self.br3])
        self.br3 = BR(self.add3).forward()
        self.deconv3 = Deconv(self.br3).forward()

        self.gcn2 = GCN(self.res2).forward()
        self.br2 = BR(self.gcn2).forward()
        self.add2 = Add()([self.deconv3, self.br2])
        self.br2 = BR(self.add2).forward()
        self.deconv2 = Deconv(self.br2).forward()

        self.br = BR(self.deconv2).forward()
        self.deconv = Deconv(self.br).forward()
        self.output = BASE_BR(self.deconv, output_flg=True).forward()

        self.model = Model(inputs=self.input_, outputs=self.output)

    def build(self):
        return self.model
