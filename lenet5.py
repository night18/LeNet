from mnist import MNIST
from numpy import *
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import operator
import math

#initial the kernel with random number
def randomKernel(size):
	kernel = zeros(size)
	for x in xrange(size):
		kernel[x] = random.random()   # Random float x, 0.0 <= x < 1.0
	return kernel


#use one kernel to whole image
def applyKernel(source, kernel):
	k_len = int(sqrt(len(kernel)))	#5 #5
	s_len = int(sqrt(len(source)))	#28#12
	run_len = int(s_len - k_len + k_len%2 )	#24#8
	maps = zeros(run_len**2)

	kernel_sum = np.sum(kernel)

	#go through the source image
	for x in xrange(run_len):		
		for y in xrange(run_len):
			#go through the kernel
			for i in xrange(k_len):
				for j in xrange(k_len):
					maps[ x * run_len + y ] += (source[ (x + i)*s_len + (y+j) ] * kernel[ i* k_len + j]) / kernel_sum

	return maps



#convolution
def conv(source, kernels, source_size, kernel_size):
	feature_map = zeros(( len(kernels) , (source_size- kernel_size + kernel_size%2)**2 ))
	for x in xrange(len(kernels)):
		feature_map[x] = applyKernel(source, kernels[x])
	return feature_map




#C3 where source is a list of feature maps
def convcube(source, kernels, source_size, kernel_size, kernel_depth):
	feature_map = zeros((len(kernels), (source_size - kernel_size + kernel_size%2)**2 ))
	for x in xrange(len(kernels)):
		for y in xrange(kernel_depth):
			#TODO 1030 get the subarray or the cube kernel
			feature_map[x] += applyKernel[]
		
	return feature_map
		
#use a 2*2 kernel to max pooling the source and jump by 2		
def maxPooling(source):
	s_len = int(sqrt(len(source)))
	m_len = int(s_len/2)
	maps = zeros(m_len ** 2)

	for x in xrange(m_len):
		for y in xrange(m_len):
			start_index =  2 * x * s_len + 2 * y 

			max_pixel = max(source[start_index], source[ start_index + 1 ], source[start_index + s_len], source[start_index + s_len + 1])
			maps[x * m_len + y] = max_pixel

	return maps

def pool(source, s_len):
	feature_map = zeros(( len(source), int(s_len/2) ** 2 ))
	for x in xrange(len(source)):
		feature_map[x] = maxPooling(source[x])

	return feature_map

#parameters
source_size = 28
kernel_size = 5	#recommend use odd number
fk_size = 6		#first kernel size
sk_size = 16	#second kernel size

#input data
mndata = MNIST('sample')
images, labels = mndata.load_training()
# test_image, test_label = mndata.load_testing()



#ndarray
trainingMats0 = zeros((60000, source_size ** 2))
trainingLabels = []

for x in xrange(60000):
	trainingMats0[x] = images[x]
	trainingLabels.append(labels[x])

#Filters: initial series of kernels

first_kernel = zeros((fk_size, kernel_size ** 2))  
for x in xrange(fk_size):
	first_kernel[x] = randomKernel(kernel_size ** 2)

C1_size = int(source_size- kernel_size + kernel_size%2) #24
C1 = zeros((fk_size, C1_size ** 2 ))

S2_size = int(C1_size/2) #12
S2 = zeros((fk_size, S2_size ** 2)) 

second_kernel = zeros((sk_size, fk_size * kernel_size ** 2))
for x in xrange(sk_size):
	second_kernel[x] = randomKernel(fk_size * kernel_size ** 2 ) #the kernel are cubes 

C3_size = int(S2_size - kernel_size + kernel_size%2) #8
C3 = zeros((sk_size, C3_size ** 2))



for x in xrange(5,6):
	#C1 Convolutional Layer

	C1 = conv(trainingMats0[x],first_kernel, source_size, kernel_size)
	S2 = pool(C1, C1_size)
	# for y in xrange()
	C3 = convcube(S2, second_kernel, S2_size, kernel_size, fk_size)

	# pprint(C1)

# 	for y in xrange(6):
# 		plt.subplot(1,6,y+1)
# 		plt.imshow(C3[y].reshape(8,8), cmap='gray')
# 		plt.axis('off')
# plt.show()