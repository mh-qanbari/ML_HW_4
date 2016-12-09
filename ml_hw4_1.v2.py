# from scipy import misc, random
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import math
import platform
import multiprocessing
from PIL import Image
import random

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
    g_TEST10 = "BSDS300/images/train/65019.jpg"
    # g_TEST10 = "test.jpg"
    g_FILES_ADDRESS = []
    g_FILES_ADDRESS.append(g_TEST01)
    g_FILES_ADDRESS.append(g_TEST02)
    g_FILES_ADDRESS.append(g_TEST03)
    g_FILES_ADDRESS.append(g_TEST04)
    g_FILES_ADDRESS.append(g_TEST05)
    g_FILES_ADDRESS.append(g_TEST06)
    g_FILES_ADDRESS.append(g_TEST07)
    g_FILES_ADDRESS.append(g_TEST08)
    g_FILES_ADDRESS.append(g_TEST09)
    g_FILES_ADDRESS.append(g_TEST10)

    g_DATASET_FILE = "BSDS300/iids_test.txt"
    g_MAX_ITERATION = 100
    g_CLUSTER_COUNT = 3
    g_RED_INDEX = 0
    g_GREEN_INDEX = 1
    g_BLUE_INDEX = 2
    g_MODE_IMG_RGB = "rgb"
    g_MODE_IMG_BW = "bw"
    # </editor-fold>

    # <editor-fold desc="ImageData declaration">
    class ImageData:
        image = Image
        id = None
        __mode = None

        # def __init__(self):
        #     self.pixels = np.ndarray(shape=(0, 0))
        #     self.pixels.setflags(write=true)
        #     printl("Image initialized : 0x0")

        def __init__(self, img):
            self.image = img
            if isinstance(img.getpixel((0, 0)), int):
                self.__mode = g_MODE_IMG_BW
            else:
                self.__mode = g_MODE_IMG_RGB
            printl("Image initialized : %dx%d" % (img.size[0], img.size[1]))

        def count(self):
            return self.size[0] * self.size[1]

        def pixel(self, row, col):
            return self.image.getpixel((row, col))

        def setpixel(self, row, col, pixel):
            self.image.putpixel((row, col), pixel)

        def update_mask(self, colors, mask):
            row_count = self.image.size[0]
            col_count = self.image.size[1]
            for i in range(row_count):
                for j in range(col_count):
                    color_index = mask[i][j]
                    color = colors[color_index]
                    if self.__mode == g_MODE_IMG_RGB:
                        self.setpixel(i, j, (color[g_RED_INDEX], color[g_GREEN_INDEX], color[g_BLUE_INDEX]))
                    else:
                        self.setpixel(i, j, color)

        def is_color(self):
            if self.__mode == g_MODE_IMG_RGB:
                return true
            else:
                return false

        def is_gray(self):
            if self.__mode == g_MODE_IMG_BW:
                return true
            else:
                return false
    # </editor-fold>

    # <editor-fold desc="Make a mask of labels for any cell (pixel of image)">
    def set_labels(obj, clusters):
        labels = []
        row_count = obj.image.size[0]
        col_count = obj.image.size[1]
        for row_index in range(row_count):
            labels.append([])
            for cell_index in range(col_count):
                pixel = obj.pixel(row_index, cell_index)
                min_d_index = -1
                min_d = 256
                index = -1
                for c in clusters:
                    index += 1
                    d = 0
                    if obj.is_color():
                        d = math.sqrt(sum((np.array(pixel) - np.array(c)) ** 2))
                    else:
                        d = math.fabs(pixel - c)
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
            if obj.is_color():
                new_clusters.append([0, 0, 0])
            else:
                new_clusters.append(0)
            clusters_count.append(0)

        row_count = obj.image.size[0]
        col_count = obj.image.size[1]
        for row_index in range(row_count):
            for cell_index in range(col_count):
                pixel = obj.pixel(row_index, cell_index)
                c = labels[row_index][cell_index]
                if obj.is_color():
                    new_clusters[c][g_RED_INDEX] += pixel[g_RED_INDEX]
                    new_clusters[c][g_GREEN_INDEX] += pixel[g_GREEN_INDEX]
                    new_clusters[c][g_BLUE_INDEX] += pixel[g_BLUE_INDEX]
                else:
                    new_clusters[c] += pixel
                clusters_count[c] += 1

        r, c = obj.image.size
        cluster_index = -1
        for cluster in new_clusters:
            cluster_index += 1
            if clusters_count[cluster_index] == 0:
                if obj.is_color():
                    rand_r = random.randint(0, r)
                    rand_c = random.randint(0, c)
                    new_clusters[cluster_index] = obj.pixel(rand_r, rand_c)
                else:
                    new_clusters[cluster_index] = random.random() * 255
            else:
                if obj.is_color():
                    cluster[g_RED_INDEX] /= clusters_count[cluster_index]
                    cluster[g_GREEN_INDEX] /= clusters_count[cluster_index]
                    cluster[g_BLUE_INDEX] /= clusters_count[cluster_index]
                else:
                    cluster /= clusters_count[cluster_index]
        # print new_clusters
        return new_clusters


    def k_means(obj):
        print_("k-means started...")

        clusters = []
        # initialize center of clusters
        for i in range(g_CLUSTER_COUNT):
            r, c = obj.image.size
            if obj.is_color():
                rand_r = random.randint(0, r)
                rand_c = random.randint(0, c)
                clusters.append(obj.pixel(rand_r, rand_c))
            else:
                clusters.append(random.random() * 255)
        # clusters.append(obj.instance(270, 447))

        # Converging loop
        iter = 0
        labels = None
        should_stop = false
        while not should_stop:
            if obj.is_color():
                old_clusters = [[row[g_RED_INDEX], row[g_GREEN_INDEX], row[g_BLUE_INDEX]] for row in clusters]
            elif obj.is_gray():
                old_clusters = [row for row in clusters]
            iter += 1

            # Set label for every point
            labels = set_labels(obj, clusters)

            # Update center of clusters
            clusters = update_centers(obj, labels)

            is_converged = true
            for i in range(g_CLUSTER_COUNT):
                if obj.is_color():
                    if (old_clusters[i][g_RED_INDEX] != clusters[i][g_RED_INDEX])\
                            or (old_clusters[i][g_GREEN_INDEX] != clusters[i][g_GREEN_INDEX])\
                            or (old_clusters[i][g_BLUE_INDEX] != clusters[i][g_BLUE_INDEX]):
                        is_converged = false
                        break
                else:
                    if old_clusters[i] != clusters[i]:
                        is_converged = false
                        break
            if (iter > g_MAX_ITERATION) or is_converged:
                should_stop = true
        print_("...finished")
        return clusters, labels


    def start(img):
        printl("image["+str(img.id)+"] started")
        centers, mask = k_means(img)
        img.update_mask(centers, mask)
        # plt.imshow(img.image)
        # plt.show()
        output_img_file = strftime("%d.jpg" % img.id)
        output_img = img.image
        plt.imshow(output_img)
        plt.show()
        output_img.save(output_img_file)
        # printl("Final image <" + output_img_file + "> saved.")
        printl("image[" + str(img.id) + "] finished")
        return img

    imgs = []
    for i in range(len(g_FILES_ADDRESS)):
        filename = g_FILES_ADDRESS[i]
        img_file = Image.open(filename)
        img_file = img_file.convert('1')
        img_dt = ImageData(img_file)
        iid = filename.split('/')[-1].split('.')[0]
        img_dt.id = int(iid)
        imgs.append(img_dt)
        # plt.imshow(img_file)
        # plt.show()
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(start)(img) for img in imgs[:])
    # results1 = Parallel(n_jobs=num_cores)(delayed(start)(img) for img in imgs[:num_cores])
    # results2 = Parallel(n_jobs=num_cores)(delayed(start)(img) for img in imgs[num_cores:])
    # start(imgs[0])

    file_output.close()

elif platform.system() == 'Windows':
    print 'Windows:'
else:
    print 'Something else...'
