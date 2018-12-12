#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

1. Use a framework of your choice (preferably Pytorch) to implement a simple neural network and the corresponding loss
function for image classification task. You are only required to implement one class for the network definition and one
function for the loss (docstrings are welcome). Main function and training loop are not required.
2. Modify the network just created to perform pixel-level semantic segmentation.
3. Assuming a batch size B and input images shape (H, W, 3), what is the shape of the input tensor of the two networks you
implemented? and the corresponding output tensors shape?
4. Given a network with an image as input and an embedding vector as output, and another network with a word as input
and an embedding vector as output, how would you train those two networks so that images and words representing the
same concept would have similar embedding vectors? [open-ended question]
# # answer to 4-1
# answer to 4-1
### Net1 : image -> Model -> vector1
### Net2:  word -> Model - > vector2

Honestly this question first remind me the following paper:
"See, Hear, and Read: Deep Aligned Representations"
However, in this question it seems we can not combine these two embeddings something like image captioning approach or the idea in this paper. 

After some research, I found this paper: 
"Learning Deep Structure-Preserving Image-Text Embeddings
" : http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf
And the idea of Bi-directional ranking constraints in this paper gave me some hints to develope the idea. 

Given a training image x1, training word x2, let Y+ and Y− denote its sets of matching (positive) and non-matching (negative) concepts, respectively.
We want the distance between x1 and x2 embedding vector with positive concepts y+ to be smaller than the distance between x1 and x2 embedding vector with negative concept y- by some enforced margin m.

dP(x1Em, x2Em) + m < dN(x1Em, x2Em)

x1Em : Output of the first neural network
x2Em : Output of the second neural network
dP: Distance for samples with positive concepts
dN: Distance for samples with negative concepts

This is the constraint that I have to convert to an equation. 

In the following equations for simplicity I will define Y. Y is 1 if the samples are in the set of y+ and Y is 0 if the samples are in the set of y- or negative concepts. 

So in order to define a loss function I will use Hinge/Contrastive Loss as following:
L(x1,x2) = Y *0.5* {dP}^2 + (1-Y)*0.5* {max(0,m-dN)}^2

As you can see, in this loss funciton we will decrease the distance between two embeddings for possitive concepts and we will increase the distance between two embeddings for negative concepts. 

The only problem is providing the dataset. I think, if we want to train networks like that we have to define a training dataset with positive and negative samples similar to Dr. LIM approach ("Dimensionality Reduction by Learning an Invariant Mapping
"). 

Please notice that, the whole idea is very similar to the siamese architectures. 

Later we can also use techniques like Structure-preserving as mentioned in that paper to make sure the embedding for similar samples are also similar. 
# In[2]:


# implement a simple nn : image classification


# # answer to 3-1

# In[ ]:


# answer to 3-1 : Consider H and W equal 32 and you can find shape of the input and output tensor in the next blocks for two networks


# # answer to 2-1

# In[3]:


# answer to 2-1
class CNetwork:
    def __init__(self, x, y):
        self.inputShape = [32,32,3]
        self.classNumber = 10 # number of class in the data-set
        self.x = x # features
        self.y = y # ground truth
        
    # x is input
    def arch(self):
        #TODO: add dropout and bn later
        self.inputLayer = tf.reshape(self.x,[-1,self.inputLayer]) # b,32,32,3
        
        # layer 1
        layer = tf.layers.conv2d(inputs=self.inputLayer, filters=40, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        # b,32,32,40
        layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2) # pooling layer
        # b,16,16,40
        
        # layer 2
        layer = tf.layers.conv2d(inputs=layer, filters=80, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        # b,16,16,80
        layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2) # pooling layer
        # b,8,8,80
        
        # dense to classify
        layer = tf.reshape(layer, [-1,8*8*80]) # flatten the conv layer 
        layer = tf.layers.dense(inputs=layer, units=800, activation=tf.nn.relu) # Dense Layer
        self.output = tf.layers.dense(inputs=layer, units=self.classNumber) # logits pass to loss funciton
        
    def loss(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.output)
    


# # answer to 2-2

# In[ ]:


# answer to 2-2
# convert the CNN to FCN - Fully Convolutional Neural NEtwork
class FCNetwork:
    def __init__(self, x, y):
        self.inputShape = [32,32,3]
        self.classNumber = 10 # number of class in the data-set
        self.x = x # features
        self.y = y # ground truth
        
    # x is input
    def arch(self):
        #TODO: add dropout, bn, regularization and skip-connection later
        self.inputLayer = tf.reshape(self.x,[-1,self.inputLayer]) # b,32,32,3
        
        # layer 1
        layer = tf.layers.conv2d(inputs=self.inputLayer, filters=40, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        # b,32,32,40
        layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2) # pooling layer
        # b,16,16,40
        
        # layer 2
        layer = tf.layers.conv2d(inputs=layer, filters=80, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        # b,16,16,80
        layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2) # pooling layer
        # b,8,8,80
        
        # conv 1x1 --> same as fully connected, spatial version
        layer = tf.layers.conv2d(inputs=layer, filters=self.classNumber, kernel_size=[1,1], padding="same")
        # b,8,8,classNumber
        
        # use the reverse function to build up the image again : upsampling
        # layer 2 reverse
        layer = tf.layers.conv2d_transpose(inputs=layer, filters=80, strides=[2,2],output_shape= [-1,16,16,80],kernel_size=[5,5], padding="same")
        # b,16,16,80
        
        # layer 1 reverese
        layer = tf.layers.conv2d_transpose(inputs=layer, filters=40, strides=[2,2],output_shape= [-1,32,32,40],kernel_size=[5,5], padding="same")
        # b,32,32,40
        
        self.output = tf.layers.conv2d_transpose(inputs=layer, filters=self.classNumber, strides=[1,1],output_shape= [-1,32,32,self.classNumber],kernel_size=[5,5], padding="same")# back to input, the cchannels are the class outputs
        # b,32,32,classNumber
        
    def loss(self):
        yResahepd = tf.reshape(self.y,[-1,self.classNumber])
        outputReshaped = rf.reshape(self.outputReshaped,[-1,self.classNumber])
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy_with_logits(labels=yReshaped, logits=outputReshaped))
    

