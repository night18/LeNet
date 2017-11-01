from mnist import MNIST
from numpy import *
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import operator
import math

#initial the kernel with random number
def randomKernel(size):
	kernel = np.random.uniform(-1,1, size)
	return kernel


#use one kernel to whole image
def applyKernel(source, kernel):
	k_len = int(sqrt(len(kernel)))	#5 #5 #4
	s_len = int(sqrt(len(source)))	#28#12#4
	run_len = max(int(s_len - k_len + k_len%2 ), 1)	#24#8#1
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
	feature_map = zeros((len(kernels), (max((source_size - kernel_size + kernel_size%2), 1))**2 ))
	for x in xrange(len(kernels)):
		for y in xrange(kernel_depth):
			kernel = kernels[x]
			layer_kernel = kernel[y*(kernel_size**2): (y+1)*(kernel_size**2)]
			feature_map[x] += applyKernel(source[y], layer_kernel)
		
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

def fileSaver(data, txt_name):
	file = open(txt_name+".txt", "w+")
	file.write(data)
	file.close()

#parameters
source_size = 28
kernel_size = 5	#recommend use odd number
fk_size = 6		#first kernel size
sk_size = 16	#second kernel size
trd_size = 120 	#third kernel size for C5
second_nodes_size = 84 #amount of second full connection layer nodes
class_size = 10 #classes number

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
	second_kernel[x] = randomKernel(fk_size * kernel_size ** 2 ) #the kernel are cubes  5*5*6

C3_size = int(S2_size - kernel_size + kernel_size%2) #8
C3 = zeros((sk_size, C3_size ** 2))

C5_kernel_size = int(C3_size/2) #4 in Lenet paper, it is using 5*5, it is also S4_size
third_kernel = zeros(( trd_size, sk_size * C5_kernel_size ** 2))
for x in xrange(trd_size):
	third_kernel[x] = randomKernel(sk_size * C5_kernel_size ** 2) #the kernel are cubes 4*4*16

for x in xrange(5,6):
	
	#C1 Convolutional Layer 6@24*24
	C1 = conv(trainingMats0[x],first_kernel, source_size, kernel_size)
	#S2 subsampling Layer 6@12*12
	S2 = pool(C1, C1_size)
	
	#C3 Convolutional Layer 16@8*8
	C3 = convcube(S2, second_kernel, S2_size, kernel_size, fk_size)
	#S4 subsampling Layer 16@4*4
	S4 = pool(C3, C3_size)
	
	#C5 Actually it is a convolutional layer 120@1*1, the kernel is 120@5*5*16
	C5 = convcube(S4, third_kernel, C5_kernel_size, C5_kernel_size, sk_size)

	pprint(C5.shape)

# 	for y in xrange(16):
# 		plt.subplot(1,16,y+1)
# 		plt.imshow(C3[y].reshape(8,8), cmap='gray')
# 		plt.axis('off')
# plt.show()

# fileSaver(first_kernel,"first_kernel")
# fileSaver(second_kernel,"second_kernel")
# fileSaver(third_kernel,"third_kernel")
# fileSaver(C1,"C1")
# fileSaver(C3,"C3")
# fileSaver(C5,"C5")
