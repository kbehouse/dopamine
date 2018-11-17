


import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
# import keras
# from tf.keras.utils.data_utils import get_file
import os

VGG_16_WEIGHTS_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

def VGGSegnet( tf_img_input_tensor, n_classes , input_height=224, input_width=224 , vgg_level=3): #input_height=416, input_width=608 , vgg_level=3):

	# img_input_tensor = Input(shape=(input_height,input_width, 3))
	# img_input_tensor= InputLayer(input_tensor=img_input_tensor,
    #                  input_shape=(None, input_height,input_width, 3))

	# if not K.is_keras_tensor(tf_img_input_tensor):
	# 	print('no keras layer')
	# 	# img_input_tensor = Input(tensor=tf_img_input_tensor, shape=(tf_img_input_tensor.shape[0], input_height,input_width, 3))
	# 	img_input_tensor = Input(tensor=tf_img_input_tensor)
	# else:
	# 	img_input_tensor = tf_img_input_tensor

	# print('VGGSegnet img_input_tensor.shape = ', img_input_tensor)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(tf_img_input_tensor)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1000 , activation='softmax', name='predictions')(x)

	# vgg  = Model(  img_input_tensor , x  )
	# vgg.load_weights(VGG_Weights_path)
	# weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 VGG_16_WEIGHTS_URL,
    #                                 cache_subdir='models')
	# print('weights_path = ', weights_path)
	# vgg.load_weights(weights_path)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = levels[ vgg_level ]
	
	o = ( ZeroPadding2D( (1,1)  ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid'))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2)))(o)
	o = ( ZeroPadding2D( (1,1)))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid'))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D((2,2)   ) )(o)
	o = ( ZeroPadding2D((1,1)  ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid'  ))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D((2,2)   ))(o)
	o = ( ZeroPadding2D((1,1)   ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'   ))(o)
	o = ( BatchNormalization())(o)


	o =  Conv2D( n_classes , (3, 3) , padding='same' )( o )
	# o_shape = Model(img_input_tensor , o ).output_shape
	# m =  Model(img_input_tensor , o )
	# outputHeight = o_shape[1]
	# outputWidth = o_shape[2]
	# print('o_shape.shape = ', o_shape)
	# print('outputHeight =', outputHeight,', outputWidth=', outputWidth)
	# print('before resahpe o -> ',o )
	# o = keras.backend.argmax(o, axis=-1)
	# self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
	# o = (Permute((3, 1, 2)))(o)
	# print('after first permute o -> ', o)
	# o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
	# print('before permute o -> ', o)
	# o = (Permute((2, 1)))(o)
	# print('after permute o -> ', o)
	# o = (Activation('softmax'))(o)

	# for l in model.layers:
	# 	print('{} -> trainable={}'.format(l,l.trainable))

	# print('o -> ', o )
	# model = Model( img_input_tensor , o )
	# model.outputWidth = outputWidth
	# model.outputHeight = outputHeight

	

	# return m
	return o




if __name__ == '__main__':
	m = VGGSegnet( 101 )
	# m.summary()
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')

