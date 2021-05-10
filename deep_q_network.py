#Using Deep Reinforcement Learning on How To Play Flappy Bird.
from __future__ import print_function

import creatingnetwork
import training_network
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque



def playGame():
    sess = tf.compat.v1.InteractiveSession()
    s, readout, h_fc1 = creatingnetwork.createNetwork()
    training_network.trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
