import tensorflow as tf
import numpy as np
import math
from net.generator import G_net, Upsample, Upsample2
from net.discriminator import D_net


def batch_norm(x, phase_train, scope):
    n_out = x.get_shape().as_list()[-1]
    #with tf.variable_scope(scope):
    if 1:
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def leaky_relu(x, leak=0.1, name='leaky_relu'):
        return tf.maximum(x, x * leak, name=name)

class build_network():
    def __init__(self, learning_rate, size, l2_weight):
        self.size = size
        self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='x')
        self.z = tf.placeholder(tf.float32, shape=[None, 128], name='z')
        #self.x = tf.placeholder(tf.float32, shape=[None, size//4, size//4, 3], name='x')
        self.label = tf.placeholder(tf.float32, shape=[None, size, size, 3], name="label")

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.ch = 64
        self.n_dis = 4
        self.sn = 2 #stride num
        
        feature = self.mapping_network(self.z)

        self.logits = sgenerator(self.x)
        self.logit2 = sgenerator(feature, reuse=True)
        #generator = G_net(self.x)
        #self.logits = generator.fake 

        scope = "discriminator"
        self.D_fake = self.discriminator(self.logit2)
        self.D_fake2 = self.discriminator(self.logits, reuse=True)
        self.D_real = self.discriminator(self.label, reuse=True, scope=scope)

        #self.D_real = dis(self.label)
        #self.D_fake = dis(self.logits, reuse=True)
        #self.reconst_loss = tf.reduce_mean(tf.squared_difference(self.x, self.x_))
        #self.reconst_loss = tf.reduce_mean(tf.squared_difference(self.x, self.x_))
        #mean, variance = tf.nn.moments(self.feature, axes=0)
        #self.reconst_loss +=  tf.reduce_mean(tf.square(variance -1) + tf.square(mean)) 
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.zeros_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        #self.G_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake2, labels=tf.zeros_like(self.D_fake2)))


        discriminator_vars =  [var for var in tf.global_variables() if  "discriminator" in var.name]
        generator_vars =  [var for var in tf.global_variables() if  "generator" in var.name]

        #self.l2_loss = tf.nn.sigmoid_cross_entropy(labels = self.labels, logits=self.logits, weights=1) 
        self.l2_loss = tf.reduce_mean(tf.squared_difference(self.label, self.logits))
        #self.prob = tf.nn.softmax(self.logits)[:,1]
        self.all_loss = self.G_loss + l2_weight * self.l2_loss

        self.D_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.D_loss, var_list=discriminator_vars)
        self.G_opt = tf.train.AdamOptimizer(learning_rate=0.1*learning_rate).minimize(self.G_loss, var_list=generator_vars)
        self.R_opt = tf.train.AdamOptimizer(learning_rate=0.1*learning_rate).minimize(self.l2_loss, var_list=generator_vars)

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D = D_net(x_init, self.ch, self.n_dis, self.sn, reuse=reuse, scope=scope)
        return D

    def mapping_network(self, z):
        with tf.variable_scope("generator") as scope:
            x = tf.layers.dense(z, 64)
            x = tf.layers.dense(x, 128)
            x = tf.layers.dense(x, 128)
            x = tf.layers.dense(x, 256)
            x = tf.layers.dense(x, 256)

            x = tf.reshape(x, [-1, 16, 16, 1])
            x = tf.layers.dense(x, 32)
            x = tf.contrib.layers.conv2d_transpose(x, 32, 3, stride=2, padding='same', activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
            x = tf.contrib.layers.conv2d_transpose(x, 3, 3, stride=2, padding='same', activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)

        return x

    def conv2d(self, x, filters, kernel_size, s):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=(s, s), padding='same', activation=leaky_relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dropout(x, 0.2)
        return x

    def encoder(self, x):
        with tf.variable_scope("generator") as scope:
            x = self.conv2d(x, 32, (3,3), 2)
            x = self.conv2d(x, 64, (3,3), 1)
            x = self.conv2d(x, 128, (3,3), 1)
            x = self.conv2d(x, 128, (3,3), 1)
            x = self.conv2d(x, 3, (3,3), 1)

        return encoder

def sgenerator(x, reuse = False): 
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        #x = tf.reshape(x, [-1, 128, 128, 3])
        #net = tf.map_fn(lambda im: tf.image.random_flip_left_right(im), x)
        #x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
        #x = tf.layers.batch_normalization(x)
        #x = tf.layers.dropout(x, 0.2)
        x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(1, 1), padding='same', activation=leaky_relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dropout(x, 0.2)
        
        #x = tf.contrib.layers.conv2d_transpose(x, 32, 3, stride=2, padding='same', 
        #                activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
        x = Upsample2(x, 128)
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation=leaky_relu)
        x = tf.layers.dropout(x, 0.2)

        #x = tf.contrib.layers.conv2d_transpose(x, 32, 3, stride=2, padding='same', 
        #                activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
        #x = Upsample(x, 128)
        #x = tf.layers.dropout(x, 0.2)
        x = Upsample(x, 128)
        #x = tf.contrib.layers.conv2d_transpose(x, 64, 3, stride=2, padding='same', activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
        x = tf.layers.dropout(x, 0.2)

        x = tf.layers.conv2d(x, filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', activation=leaky_relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation=leaky_relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dropout(x, 0.2)
        x = tf.layers.conv2d(x, filters=3, kernel_size=(3,3), strides=(1, 1), padding='same', activation=leaky_relu)
        x = tf.nn.tanh(x)
        return x

def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = generator(inputs)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)


def dis(x, reuse = False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        #x = tf.reshape(x, [-1, size, size, 3])
        x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
        x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dropout(x, 0.2)
        x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
        x = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), strides=(2, 2), padding='same', activation=leaky_relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dropout(x, 0.2)
        
        #x = tf.layers.flatten(x)
        #x = tf.reshape(x, (-1, 256*256*32//(2**4)))
        #x = tf.reshape(x, (-1, 32*32*32))
        x = tf.reshape(x, (-1, 16*16*32))
        x = tf.layers.dense(x, 32)
        x = tf.layers.dense(x, 1)
    return x





