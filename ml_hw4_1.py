import  numpy as np
from scipy import misc, random
import matplotlib.pyplot as plt
import math
import sys
from time import gmtime, strftime

g_TEST = "BSDS300/images/train/8143.jpg"
g_DATASET_FILE = "BSDS300/iids_test.txt"
g_OUTPUT_FILE = strftime("output_%Y%m%d%H%M%S", gmtime())
g_MAX_ITERATION = 100
g_CLUSTER_COUNT = 4
g_RED_INDEX = 0
g_GREEN_INDEX = 1
g_BLUE_INDEX = 2

false = False
true = True

file_output = open(g_OUTPUT_FILE, 'w')
def printl(obj):
    orig_stdout = sys.stdout
    sys.stdout = file_output
    print obj
    sys.stdout = orig_stdout
    print obj
def print_(obj):
    orig_stdout = sys.stdout
    sys.stdout = file_output
    print obj,
    sys.stdout = orig_stdout
    print obj,

# <editor-fold desc="Image declaration">
class Image:
    image = np.ndarray(shape=(0, 0))

    def __init__(self):
        self.image = np.ndarray(shape=(0, 0))
        printl("Image initialized : 0x0")

    def __init__(self, img):
        self.image = img
        row_size, col_size = self.size()
        printl("Image initialized : %dx%d" % (row_size, col_size))

    def get_rgb(self, row, col):
        return self.image[row, col]

    def count(self):
        return self.image.size / 3

    def size(self):
        c = self.count()
        row_count = len(self.image)
        col_count = c / row_count
        return row_count, col_count

    def instance(self, row, col):
        sel_rgb = self.image[row, col]
        return [ sel_rgb[g_RED_INDEX], sel_rgb[g_GREEN_INDEX], sel_rgb[g_BLUE_INDEX] ]
# </editor-fold>

def get_labels(obj, clusters):
    labels = [[]]
    row_index = -1
    for row in obj.image:
        row_index += 1
        labels.append([])
        # print_('[')
        for cell in row:
            min_d_index = -1
            min_d = 256
            index = -1
            for c in clusters:
                index += 1
                d = math.sqrt(sum(cell - c) ^ 2)
                if d < min_d:
                    min_d = d
                    min_d_index = index

            labels[row_index].append(min_d_index)
            # Update each cell (or pixel) with center of nearest cluster
            cell[g_RED_INDEX] = clusters[min_d_index][g_RED_INDEX]
            cell[g_GREEN_INDEX] = clusters[min_d_index][g_GREEN_INDEX]
            cell[g_BLUE_INDEX] = clusters[min_d_index][g_BLUE_INDEX]
    # printl(labels)
    #         print_(min_d_index)
    #     printl('] ')
    return labels

def k_means(obj):
    print_("k-means started...")

    clusters = []
    # initialize center of clusters
    for i in range(g_CLUSTER_COUNT):
        r, c = obj.size()
        rand_r = random.randint(0, r)
        rand_c = random.randint(0, c)
        clusters.append( obj.instance(rand_r, rand_c) )

    # Converging loop
    iter = 0
    old_clusters = None
    should_stop = false
    while not should_stop:
        old_clusters = clusters
        iter += 1

        get_labels(obj, clusters)

        if (iter > g_MAX_ITERATION) or (old_clusters == clusters):
            should_stop = true
    print_("...finished")

ds = Image(misc.imread(g_TEST))

plt.imshow(ds.image)
plt.show()

k_means(ds)

plt.imshow(ds.image)
plt.show()

file_output.close()
