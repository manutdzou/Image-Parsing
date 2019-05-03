import numpy as np
import tensorflow as tf
import six
from model import PSPNet

class PSPnet_backbone(object):
    """Densenet model."""
    def __init__(self, num_class, images, is_training):
        self.num_classes = num_class
        self._images = images
        self.training  = is_training
    
    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
    
    def _build_model(self):
        net = PSPNet({'data': self._images}, is_training=self.training, num_classes=self.num_classes)

        self.logits = tf.squeeze(net.layers['conv6'],[1,2])
