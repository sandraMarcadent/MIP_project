"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
from DTN_layers import *
from parameters import *
from flip_gradient import flip_gradient

ngf = NGF
ndf = NDF

def get_outputs(inputs, network="tensorflow", skip=False):
    
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_b = inputs['fake_pool_b']
    encode_pool_b = inputs['encode_pool_b']
    encode_pool_a = inputs['encode_pool_a']

    with tf.variable_scope("Model") as scope:

        if network == "pytorch":
            current_discriminator = discriminator
            current_generator = build_generator_resnet_9blocks
        elif network == "tensorflow":
            current_discriminator = discriminator_tf
            current_encoder = build_encoder_resnet_5blocks_tf
            current_decoder = build_decoder_resnet_4blocks_tf
            current_do_discriminator = do_discriminator_tf
        else:
            raise ValueError(
                'network must be either pytorch or tensorflow'
            )

        
        
        # FIRST VARIABLES INITIALIZATION
        # Encode images a
        
        encode_images_a = current_encoder(images_a, name="e_ML")
        
        # Recognize encoding of images a
        
        prob_class_a = current_do_discriminator(encode_images_a, name="do")
        
        # Generate images b from encoded images a
        
        fake_images_b = current_decoder(encode_images_a, name="gen_B")
        
        # Recognize generated images b from true b
        
        prob_fake_b_is_real = current_discriminator(fake_images_b, "d_B")
        
        
        # REUSING VARIABLES
        # discriminate true images b
        
        scope.reuse_variables()
        
        prob_real_b_is_real = current_discriminator(images_b, "d_B")
        
        
        # encode images b
        scope.reuse_variables()
        
        encode_images_b = current_encoder(images_b, name="e_ML")
        
        # encode images a from generated b (semantic cycle)
        scope.reuse_variables()
        
        encode_cycle_a = current_encoder(fake_images_b, name='e_ML')
        
        # recognize encoding b
        scope.reuse_variables()
        
        prob_class_b = current_do_discriminator(encode_images_b, name="do")
        
        # generate images b from encoded images b (cycle)
        scope.reuse_variables()
        
        cycle_images_b = current_decoder(encode_images_b, name="gen_B")
 

        # training of do from pool
        scope.reuse_variables()
        
        prob_class_pool_a = current_do_discriminator(encode_pool_a, name="do")
        
        scope.reuse_variables()
        
        prob_class_pool_b = current_do_discriminator(encode_pool_b, name="do")
        

        # training generator from pool
        scope.reuse_variables()
        
        fake_images_b_from_pool = current_decoder(encode_pool_a, name="gen_B")
        
        scope.reuse_variables()
        
        cycle_images_b_from_pool = current_decoder(encode_pool_b, name="gen_B")
        
        
        # training discriminator from pool
        scope.reuse_variables()
        
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, "d_B")
        
        scope.reuse_variables()
        
        prob_fake_b_is_real_from_pool = current_discriminator(fake_images_b_from_pool, "d_B")
        
        
        ########

    return {
        'encode_images_a' : encode_images_a,
        'encode_images_b' : encode_images_b,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'fake_images_b': fake_images_b,
        'cycle_images_b': cycle_images_b,
        'fake_images_b_from_pool': fake_images_b_from_pool,
        'cycle_images_b_from_pool': cycle_images_b_from_pool,
        'prob_class_a':prob_class_a,
        'prob_class_b':prob_class_b,
        'prob_class_pool_a':prob_class_pool_a,
        'prob_class_pool_b':prob_class_pool_b,
        'prob_fake_b_is_real_from_pool': prob_fake_b_is_real_from_pool,
        'encode_cycle_a':encode_cycle_a
    }


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_encoder_resnet_5blocks_tf(inputenc, name="encoder", skip=False):
    
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputenc, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        
        
    return o_r5
        

def build_decoder_resnet_4blocks_tf(out_enc, name="decoder", skip=False):
    
    with tf.variable_scope(name):    
        f = 7
        ks = 3
        padding = "REFLECT"
        
        o_r1 = build_resnet_block(out_enc, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)

        o_c1 = general_deconv2d(
            o_r4, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")
        o_c2 = general_deconv2d(
            o_c1, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
            "SAME", "c5")
        o_c3 = general_conv2d(o_c2, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        out_gen = tf.nn.tanh(o_c3, "t1")

        return out_gen


def discriminator_tf(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm=False,
                                     relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02,
            "SAME", "c5", do_norm=False, do_relu=False
        )

        return o_c5

    
def do_discriminator_tf(inputdo, name="do_discriminator"):
    with tf.variable_scope(name):
        f = 4
        
        flip_1 = flip_gradient(inputdo)
        
        o_c1 = general_conv2d(flip_1, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm=False,
                                     relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02,
            "SAME", "c5", do_norm=False, do_relu=False
        )

        return o_c5

    
    
