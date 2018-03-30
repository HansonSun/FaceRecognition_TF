

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])

    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers_op = tf.scatter_sub(centers, label, diff)

    loss = tf.reduce_mean(tf.square(features - centers_batch))

    return loss, centers,centers_op


'''
def center_loss(features, labels, alpha, num_classes):

    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    loss = tf.nn.l2_loss(features - centers_batch)

    return loss, centers, centers_update_op
'''
