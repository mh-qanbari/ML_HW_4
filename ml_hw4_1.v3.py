from joblib import Parallel, delayed
import numpy as np
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
    # g_TEST10 = "1.IMG_0033.jpg"
    # g_TEST10 = "1.test.jpg"
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
    g_MAX_ITERATION = 10
    g_CLUSTER_COUNT = 7
    g_RGB_RED_INDEX = 0
    g_RGB_GREEN_INDEX = 1
    g_RGB_BLUE_INDEX = 2
    g_GRAY_INDEX = 0
    g_MODE_IMG_RGB = "rgb"
    g_MODE_IMG_BW = "bw"
    g_MODE_IMG_GRAY = "gray"
    g_MODE_IMG_SELECTED = g_MODE_IMG_RGB
    # </editor-fold>

    # <editor-fold desc="ImageData declaration">
    class ImageData:
        image = None
        id = None
        __mode = None

        def __init__(self, img, mode=g_MODE_IMG_RGB):
            self.image = img
            self.__mode = mode
            printl("Image initialized : %dx%d" % (img.size[0], img.size[1]))

        def count(self):
            return self.size[0] * self.size[1]

        def pixel(self, row, col):
            return self.image.getpixel((row, col))

        def update_mask(self, colors, mask):
            # self.show()
            row_count = self.image.size[0]
            col_count = self.image.size[1]
            if self.__mode == g_MODE_IMG_RGB:
                pixels = np.asarray(self.image.getdata(), dtype=np.int).reshape((col_count, row_count, 3))
            elif self.__mode == g_MODE_IMG_BW or self.__mode == g_MODE_IMG_GRAY:
                pixels = np.asarray(self.image.getdata(), dtype=np.float64).reshape((col_count, row_count))
            for i in range(row_count):
                for j in range(col_count):
                    color_index = mask[i][j]
                    color = colors[color_index]
                    if self.__mode == g_MODE_IMG_RGB:
                        pixels[j, i] = (color[g_RGB_RED_INDEX], color[g_RGB_GREEN_INDEX], color[g_RGB_BLUE_INDEX])
                        # print pixels[i, j]
                        # pixels[i, j][g_RGB_RED_INDEX] = color[g_RGB_RED_INDEX]
                        # pixels[i, j][g_RGB_GREEN_INDEX] = color[g_RGB_GREEN_INDEX]
                        # pixels[i, j][g_RGB_BLUE_INDEX] = color[g_RGB_BLUE_INDEX]
                    else:
                        pixels[j, i] = color
            pixels = np.asarray(pixels, dtype=np.uint8)
            if self.__mode == g_MODE_IMG_RGB:
                self.image = Image.fromarray(pixels, mode='RGB')
            elif self.__mode == g_MODE_IMG_BW:
                self.image = Image.fromarray(pixels, mode='1')
            elif self.__mode == g_MODE_IMG_GRAY:
                self.image = Image.fromarray(pixels, mode='L')
            # self.show()

        def is_color(self):
            if self.__mode == g_MODE_IMG_RGB:
                return true
            else:
                return false

        def is_black_white(self):
            if self.__mode == g_MODE_IMG_BW:
                return true
            else:
                return false

        def is_gray(self):
            if self.__mode == g_MODE_IMG_GRAY:
                return true
            else:
                return false

        def show(self):
            self.image.show()
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
                min_d = 196608 + obj.image.width ** 2 + obj.image.height ** 2
                index = -1
                for c in clusters:
                    index += 1
                    d = 0
                    if obj.is_color():
                        d = (pixel[g_RGB_RED_INDEX] - c[g_RGB_GREEN_INDEX]) ** 2 +\
                            (pixel[g_RGB_GREEN_INDEX] - c[g_RGB_GREEN_INDEX]) ** 2 +\
                            (pixel[g_RGB_BLUE_INDEX] - c[g_RGB_BLUE_INDEX]) ** 2 +\
                            (row_index - c[3]) ** 2 + (cell_index - c[4]) ** 2  # considering line and column
                    elif obj.is_black_white() or obj.is_gray():
                        d = (pixel - c[0]) ** 2 +\
                            (row_index - c[1]) ** 2 + (cell_index - c[2]) ** 2  # considering line and column
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
            elif obj.is_black_white() or obj.is_gray():
                new_clusters.append(0)
            clusters_count.append(0)

        row_count = obj.image.size[0]
        col_count = obj.image.size[1]
        for row_index in range(row_count):
            for cell_index in range(col_count):
                pixel = obj.pixel(row_index, cell_index)
                c = labels[row_index][cell_index]
                if obj.is_color():
                    new_clusters[c][g_RGB_RED_INDEX] += pixel[g_RGB_RED_INDEX]
                    new_clusters[c][g_RGB_GREEN_INDEX] += pixel[g_RGB_GREEN_INDEX]
                    new_clusters[c][g_RGB_BLUE_INDEX] += pixel[g_RGB_BLUE_INDEX]
                elif obj.is_black_white() or obj.is_gray():
                    new_clusters[c] += pixel
                clusters_count[c] += 1

        r, c = obj.image.size
        cluster_index = -1
        for cluster in new_clusters:
            cluster_index += 1
            if clusters_count[cluster_index] == 0:
                # if obj.is_color():
                    rand_r = random.randint(0, r)
                    rand_c = random.randint(0, c)
                    new_clusters[cluster_index] = obj.pixel(rand_r, rand_c)
                # elif obj.is_black_white() or obj.is_gray():
                #     new_clusters[cluster_index] = random.random() * 255
            else:
                count = clusters_count[cluster_index]
                if obj.is_color():
                    cluster[g_RGB_RED_INDEX] /= count
                    cluster[g_RGB_GREEN_INDEX] /= count
                    cluster[g_RGB_BLUE_INDEX] /= count
                elif obj.is_black_white or obj.is_gray():
                    cluster /= count
                new_clusters[cluster_index] = cluster
        # print new_clusters
        return new_clusters


    def k_means(obj):
        print_("k-means started...")

        clusters = []
        # initialize center of clusters
        r, c = obj.image.size
        for i in range(g_CLUSTER_COUNT):
            rand_r = random.randint(0, r)
            rand_c = random.randint(0, c)
            pixel = obj.pixel(rand_r, rand_c)
            if obj.is_color():
                clusters.append([pixel[g_RGB_RED_INDEX], pixel[g_RGB_GREEN_INDEX], pixel[g_RGB_BLUE_INDEX],
                                 rand_r, rand_c])  # considering line and column
            elif obj.is_black_white() or obj.is_gray():
                clusters.append([pixel, rand_r, rand_c])  # considering line and column
        del c, r

        # Converging loop
        iter = 0
        labels = None
        should_stop = false
        while not should_stop:
            iter += 1

            # Set label for every point
            labels = set_labels(obj, clusters)

            # Update center of clusters
            new_clusters = update_centers(obj, labels)

            is_converged = true
            for i in range(g_CLUSTER_COUNT):
                if obj.is_color():
                    if (clusters[i][g_RGB_RED_INDEX] != new_clusters[i][g_RGB_RED_INDEX]) \
                            or (clusters[i][g_RGB_GREEN_INDEX] != new_clusters[i][g_RGB_GREEN_INDEX]) \
                            or (clusters[i][g_RGB_BLUE_INDEX] != new_clusters[i][g_RGB_BLUE_INDEX]):
                        is_converged = false
                        break
                elif obj.is_black_white() or obj.is_gray():
                    if clusters[i] != new_clusters[i]:
                        is_converged = false
                        break
            if (iter > g_MAX_ITERATION) or is_converged:
                should_stop = true
        print_("...finished")
        return new_clusters, labels


    def start(img):
        printl("image["+str(img.id)+"] started")
        centers, mask = k_means(img)
        img.update_mask(centers, mask)
        output_img_file = strftime("%d.jpg" % img.id)
        output_img = img.image
        output_img.save(output_img_file)
        printl("image[" + str(img.id) + "] finished")
        return img


    import time
    start_time = time.time()

    # <editor-fold desc="Main">
    images = []
    for i in range(len(g_FILES_ADDRESS)):
        filename = g_FILES_ADDRESS[i]
        img_file = Image.open(filename, 'r')

        if g_MODE_IMG_SELECTED == g_MODE_IMG_RGB:
            img_file = img_file.convert('RGB')
            img_dt = ImageData(img_file, mode=g_MODE_IMG_RGB)
        elif g_MODE_IMG_SELECTED == g_MODE_IMG_BW:
            img_file = img_file.convert('1')
            img_dt = ImageData(img_file, mode=g_MODE_IMG_BW)
        elif g_MODE_IMG_SELECTED == g_MODE_IMG_GRAY:
            img_file = img_file.convert('L')
            img_dt = ImageData(img_file, mode=g_MODE_IMG_GRAY)

        iid = filename.split('/')[-1].split('.')[0]
        img_dt.id = int(iid)
        images.append(img_dt)
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(start)(img) for img in images[:])
    # start(images[0])

    printl("--- %s seconds ---" % (time.time() - start_time))

    file_output.close()
    # </editor-fold>

elif platform.system() == 'Windows':
    print 'Windows:'
else:
    print 'Something else...'
