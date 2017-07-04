import tensorflow as tf


def weight_variable(shape, name=None):
    units = shape[0]
    if len(shape) == 4:
        units *= shape[1]*shape[2]
    else:
        assert("Invalid shape")

    return tf.get_variable(
        name+"/weight",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=tf.sqrt(1.0/units)),  # initialization for S-ELU
        dtype=tf.float32
    )


def bias_variable(shape, name=None):
    return tf.get_variable(
        name+"/bias",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=.0),  # initialization for S-ELU
        dtype=tf.float32
    )


def elu(input_t, alpha=1.0, name="elu"):
    with tf.variable_scope(name):
        return tf.where(input_t >= 0, input_t, alpha*(tf.exp(input_t) - 1.0))


def selu(input_t, name="selu"):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*elu(input_t, alpha)


def vgg_block(input_tensor, output_channel, layer_name):
    input_channel = input_tensor.get_shape().as_list()[-1]
    output_tensor = input_tensor

    with tf.variable_scope(layer_name):
        w_conv = weight_variable([3, 3, input_channel, output_channel], name=layer_name)
        b_conv = bias_variable([output_channel], name=layer_name)
        output_tensor = tf.nn.conv2d(output_tensor, w_conv, strides=[1, 1, 1, 1], padding='SAME')
        output_tensor += b_conv
        # output_tensor = tf.nn.relu(output_tensor)
        output_tensor = selu(output_tensor)
        tf.summary.histogram(layer_name, output_tensor)

    return output_tensor


class Model:
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')

        self.input_image_placeholder = tf.placeholder(
            dtype=tf.uint8,
            shape=[None, 32, 32, 3],
            name='input_image_placeholder')

        self.label_placeholder = tf.placeholder(
            dtype=tf.int64,
            shape=[None],
            name='label_placeholder')
        self.output = self.build_model_vgg()
        self.each_loss, self.accum_loss = self.build_loss()
        self.pred_label = tf.arg_max(tf.nn.softmax(self.output), 1)

        optimizer = tf.train.GradientDescentOptimizer
        # optimizer = tf.train.AdamOptimizer
        self.train_op = optimizer(self.lr_placeholder).minimize(
            self.accum_loss,
            global_step=self.global_step,
        )

        self.conf_matrix = tf.confusion_matrix(self.label_placeholder, self.pred_label, num_classes=10)
        self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred_label, self.label_placeholder)), axis=0)
        print(self.each_loss, self.accum_loss)
        print(self.pred_label)

        return

    def build_model_vgg(self, name="vgg"):
        layers_size = [1, 1, 2, 2, 2]  #vgg11
        # layers_size = [2, 2, 3, 3, 3]  #vgg16

        output_tensor = self.input_image_placeholder
        output_tensor = tf.div(tf.to_float(output_tensor), 255.0, name="input_image_float")
        print(output_tensor)
        with tf.variable_scope(name):

            # input size 32
            for idx in range(layers_size[0]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=64,
                    layer_name="layer1_"+str(idx+1),
                )
            with tf.variable_scope("layer1_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

            # input size 16
            for idx in range(layers_size[1]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=128,
                    layer_name="layer2_"+str(idx+1),
                )
            with tf.variable_scope("layer2_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

            # input size 8
            for idx in range(layers_size[2]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=256,
                    layer_name="layer3_"+str(idx+1),
                )
            with tf.variable_scope("layer3_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

            # input size 4
            for idx in range(layers_size[3]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=512,
                    layer_name="layer4_"+str(idx+1),
                )
            with tf.variable_scope("layer4_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

            # input size 2
            for idx in range(layers_size[4]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=512,
                    layer_name="layer5_"+str(idx+1),
                )
            with tf.variable_scope("layer5_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

            # input size 1 w/ channel 512 => fc layer
            output_tensor = tf.squeeze(output_tensor, axis=[1, 2])

            with tf.variable_scope("fc_1"):
                w_fc1 = weight_variable([512, 512], name="fc_1")
                b_fc1 = bias_variable([512], name="fc_1")
                output_tensor = tf.matmul(output_tensor, w_fc1) + b_fc1
                # output_tensor = tf.nn.relu(output_tensor)
                output_tensor = selu(output_tensor)
                tf.summary.histogram("fc_1", output_tensor)

            with tf.variable_scope("fc_2"):
                w_fc2 = weight_variable([512, 10], name="fc_2")
                b_fc2 = bias_variable([10], name="fc_2")
                output_tensor = tf.matmul(output_tensor, w_fc2) + b_fc2
                tf.summary.histogram("fc_2", output_tensor)

            print("last layer of the model =", output_tensor)
        return output_tensor

    def build_loss(self):
        each_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.output)
        accum_loss = tf.reduce_mean(each_loss, axis=[0])
        return each_loss, accum_loss


