from scipy import misc, random
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import math
import platform
import multiprocessing

if platform.system() == 'Linux':
    print '<platform , Linux>\n'

    # <editor-fold desc="Necessary Standards">
    import sys
    from time import gmtime, strftime

    false = False
    true = True

    g_OUTPUT_FILE = strftime("output_%Y%m%d%H%M%S", gmtime())
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


    # </editor-fold>

    # <editor-fold desc="Parameters">
    g_TEST01 = "BSDS300/images/train/8049.jpg"
    g_TEST02 = "BSDS300/images/train/8143.jpg"
    g_TEST03 = "BSDS300/images/train/12003.jpg"
    g_TEST04 = "BSDS300/images/train/12074.jpg"
    g_TEST05 = "BSDS300/images/train/22013.jpg"
    g_TEST06 = "BSDS300/images/train/26031.jpg"
    g_TEST07 = "BSDS300/images/train/28075.jpg"
    g_TEST08 = "BSDS300/images/train/35008.jpg"
    g_TEST09 = "BSDS300/images/train/181018.jpg"
    g_TEST10 = "test.jpg"
    g_FILES_ADDRESS = [g_TEST01, g_TEST02, g_TEST03, g_TEST04, g_TEST05, g_TEST06, g_TEST07, g_TEST08, g_TEST09,
                       g_TEST10]

    g_DATASET_FILE = "BSDS300/iids_test.txt"
    g_MAX_ITERATION = 100
    g_CLUSTER_COUNT = 2
    g_RED_INDEX = 0
    g_GREEN_INDEX = 1
    g_BLUE_INDEX = 2
    # </editor-fold>

    # <editor-fold desc="Image declaration">
    class Image:
        image = np.ndarray(shape=(0, 0))

        def __init__(self):
            self.image = np.ndarray(shape=(0, 0))
            self.image.setflags(write=1, align=1)
            printl("Image initialized : 0x0")

        def __init__(self, img):
            self.image = img
            row_size, col_size = self.size()
            self.image.setflags(write=1, align=1)
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
            return [sel_rgb[g_RED_INDEX], sel_rgb[g_GREEN_INDEX], sel_rgb[g_BLUE_INDEX]]

        def update_mask(self, colors, mask):
            row_count, col_count = self.size()
            for i in range(row_count):
                for j in range(col_count):
                    color_index = mask[i][j]
                    color = colors[color_index]
                    self.image[i, j][g_RED_INDEX] = color[g_RED_INDEX]
                    self.image[i, j][g_GREEN_INDEX] = color[g_GREEN_INDEX]
                    self.image[i, j][g_BLUE_INDEX] = color[g_BLUE_INDEX]
    # </editor-fold>

    # <editor-fold desc="Make a mask of labels for any cell (pixel of image)">
    def set_labels(obj, clusters):
        labels = []
        row_index = -1
        for row in obj.image:
            row_index += 1
            labels.append([])
            for cell in row:
                min_d_index = -1
                min_d = 256
                index = -1
                for c in clusters:
                    index += 1
                    d = math.sqrt(sum((cell - c) ** 2))
                    if d < min_d:
                        min_d = d
                        min_d_index = index

                labels[row_index].append(min_d_index)
        return labels
    # </editor-fold>


    def update_centers(obj, labels):
        new_clusters = []
        clusters_count = []
        for i in range(g_CLUSTER_COUNT):
            new_clusters.append([0, 0, 0])
            clusters_count.append(0)

        row_index = -1
        for row in obj.image:
            row_index += 1
            cell_index = -1
            for cell in row:
                cell_index += 1
                c = labels[row_index][cell_index]
                new_clusters[c][g_RED_INDEX] += cell[g_RED_INDEX]
                new_clusters[c][g_GREEN_INDEX] += cell[g_GREEN_INDEX]
                new_clusters[c][g_BLUE_INDEX] += cell[g_BLUE_INDEX]
                clusters_count[c] += 1

        r, c = obj.size()
        cluster_index = -1
        for cluster in new_clusters:
            cluster_index += 1
            if clusters_count[cluster_index] == 0:
                rand_r = random.randint(0, r)
                rand_c = random.randint(0, c)
                new_clusters[cluster_index] = obj.instance(rand_r, rand_c)
            else:
                cluster[g_RED_INDEX] /= clusters_count[cluster_index]
                cluster[g_GREEN_INDEX] /= clusters_count[cluster_index]
                cluster[g_BLUE_INDEX] /= clusters_count[cluster_index]
        # print new_clusters
        return new_clusters


    def k_means(obj):
        print_("k-means started...")

        clusters = []
        # initialize center of clusters
        for i in range(g_CLUSTER_COUNT - 1):
            r, c = obj.size()
            rand_r = random.randint(0, r)
            rand_c = random.randint(0, c)
            clusters.append(obj.instance(rand_r, rand_c))
        clusters.append(obj.instance(270, 447))

        # Converging loop
        iter = 0
        labels = None
        should_stop = false
        while not should_stop:
            old_clusters = [[row[g_RED_INDEX], row[g_GREEN_INDEX], row[g_BLUE_INDEX]] for row in clusters]
            iter += 1

            # Set label for every point
            labels = set_labels(obj, clusters)

            # Update center of clusters
            clusters = update_centers(obj, labels)

            is_converged = true
            for i in range(g_CLUSTER_COUNT):
                if (old_clusters[i][g_RED_INDEX] != clusters[i][g_RED_INDEX])\
                        or (old_clusters[i][g_GREEN_INDEX] != clusters[i][g_GREEN_INDEX])\
                        or (old_clusters[i][g_BLUE_INDEX] != clusters[i][g_BLUE_INDEX]):
                    is_converged = false
                    break
            if (iter > g_MAX_ITERATION) or is_converged:
                should_stop = true
        print_("...finished")
        return clusters, labels


    imgs = [Image(misc.imread(file)) for file in g_FILES_ADDRESS]

    # plt.imshow(img.image)
    # plt.show()

    def start(img):
        centers, mask = k_means(img)
        img.update_mask(centers, mask)
        plt.imshow(img.image)
        plt.show()
        plt.imsave(strftime("%Y%m%d%H%M%S.jpg", gmtime()))
        return img

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(start)(img) for img in imgs)

    file_output.close()
elif platform.system() == 'Windows':
    print 'Windows:'
else:
    print 'Something else...'
