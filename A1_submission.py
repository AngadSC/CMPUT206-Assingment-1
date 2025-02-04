"""Include your imports here
Some example imports are below"""

import numpy as np 
from skimage import io, img_as_ubyte, exposure
import matplotlib.pyplot as plt


def part1_histogram_compute():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)


    """add your code here"""
    n = 128

    hist = np.zeros(n,dtype=int)
    bin_width= 256/n 
    for i in img.flatten():
        bin_index = int(i//bin_width)
        hist[bin_index] +=1 

    hist_np, _ = np.histogram(img, bins=n, range = (0,256))


    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    assert np.all(hist_np == hist), "mismatch between hist_np (Numpy) and hist"

    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.plot(hist), plt.title('My Histogram')
    plt.xlim([0, n])
    plt.subplot(122), plt.plot(hist_np), plt.title('Numpy Histogram')
    plt.xlim([0, n])

    plt.show()


def part2_histogram_equalization():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)
    
    """add your code here"""
    n_bins = 128

    # 128-bin Histogram computed by your code (cannot use in-built functions!)
    hist = np.zeros(n_bins, dtype=int)
    bin_width = 256/n_bins

    for i in img.flatten():
        bin_index = int(i // bin_width)
        hist[bin_index] +=1

    # cdf computation
    cdf = np.zeros(n_bins)
    cdf[0] = hist[0]
    for n in range(1, n_bins):
        cdf[n] = cdf[n-1] + hist[n]

    cdf_min = cdf[0]
    total_pixels = img.size
    cdf_normal = (cdf - cdf_min) / (total_pixels - cdf_min) *255


    #Initialize another image img_eq (you can use np.zeros) and update the pixel intensities in every location

    img_eq = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j]
            bin_index = int(pixel//bin_width)
            img_eq[i,j] = cdf_normal[bin_index]
    

    # Histogram of equalized image
    hist_eq = np.zeros(n_bins, dtype=int)
    for pixel in img_eq.flatten():
        bin_index = int(pixel//bin_width)
        hist_eq[bin_index] +=1

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(10, 10))
    plt.subplot(221), plt.imshow(image, 'gray'), plt.title('Original Image')
    plt.subplot(222), plt.plot(hist), plt.title('Histogram')
    plt.xlim([0, n_bins])
    plt.subplot(223), plt.imshow(img_eq, 'gray'), plt.title('New Image')
    plt.subplot(224), plt.plot(hist_eq), plt.title('Histogram After Equalization')
    plt.xlim([0, n_bins])
    
    plt.show()   


def part3_histogram_comparing():

    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    # Read in the image
    img1 = img_as_ubyte(io.imread(filename1, as_gray=True))
    # Read in another image
    img2 = img_as_ubyte(io.imread(filename2, as_gray=True))

    n_bins_all = [4, 8, 16, 32, 64, 128, 256]

    #Append the Bhattacharyya coefficient for each bin to this list
    bc_all = []

    for n_bins in n_bins_all:
        hist1, _ = np.histogram(img1, bins=n_bins, range=(0,256), density=True)
        hist2, _ = np.histogram(img2, bins=n_bins, range=(0,256), density=True)
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        bc = np.sum(np.sqrt(hist1*hist2))
        bc_all.append(bc)

        print(f"bins: {n_bins} Bhattacharyya Coefficient: {bc:.4f}")

    plt.plot(n_bins_all, bc_all, marker='o')
    plt.xlabel('Number of bins')
    plt.ylabel('Bhattacharyya Coefficient')
    plt.show()

def ecdf(x):
    
    nbins = np.sort(x.ravel())
    cdf = np.arange(1, len(nbins) +1) / len(nbins)

    return nbins, cdf

#Function for plotting cdfs - you do not need to modify this
def show_with_cdf(source_gs, template_gs, matched_gs, name):

    x1, y1 = ecdf(source_gs)
    x2, y2 = ecdf(template_gs)
    x3, y3 = ecdf(matched_gs)

    fig_gs = plt.figure()
    gs = plt.GridSpec(2, 3)
    ax1_gs = fig_gs.add_subplot(gs[0, 0])
    ax2_gs = fig_gs.add_subplot(gs[0, 1], sharex=ax1_gs, sharey=ax1_gs)
    ax3_gs = fig_gs.add_subplot(gs[0, 2], sharex=ax1_gs, sharey=ax1_gs)
    ax4 = fig_gs.add_subplot(gs[1, :])
    for aa in (ax1_gs, ax2_gs, ax3_gs):
        aa.set_axis_off()

    ax1_gs.imshow(source_gs, cmap=plt.cm.gray)
    ax1_gs.set_title(f'Source {name}')
    ax2_gs.imshow(template_gs, cmap=plt.cm.gray)
    ax2_gs.set_title(f'Template {name}')
    ax3_gs.imshow(matched_gs, cmap=plt.cm.gray)
    ax3_gs.set_title(f'Matched {name}')

    ax4.plot(x1, y1 * 100, '-r', lw=3, label=f'Source {name}')
    ax4.plot(x2, y2 * 100, '-k', lw=3, label=f'Template {name}')
    ax4.plot(x3, y3 * 100, '--r', lw=3, label=f'Matched {name}')
    ax4.set_xlim(x1[0], x1[-1])
    ax4.set_xlabel('Pixel value')
    ax4.set_ylabel('Cumulative %')
    ax4.legend(loc=5)

    plt.show()

def match_histograms(input_image, reference_image):
    
    input_hist, _ = np.histogram(input_image.flatten(), bins=256, range=(0, 256))
    reference_hist, _ = np.histogram(reference_image.flatten(), bins=256, range=(0, 256))

    input_cdf = np.cumsum(input_hist).astype(float)
    input_cdf /= input_cdf[-1]

    reference_cdf = np.cumsum(reference_hist).astype(float)
    reference_cdf /= reference_cdf[-1]

    mapping = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for inp_idx in range(256):
        while ref_idx < 255 and reference_cdf[ref_idx] < input_cdf[inp_idx]:
            ref_idx += 1
        mapping[inp_idx] = ref_idx

    matched_image = mapping[input_image]
    return matched_image

def part4_histogram_matching():
    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    # ============ Grayscale ============
    source_gs = io.imread(filename1, as_gray=True)
    source_gs = img_as_ubyte(source_gs)

    template_gs = io.imread(filename2, as_gray=True)
    template_gs = img_as_ubyte(template_gs)

    matched_gs = match_histograms(source_gs, template_gs)
    show_with_cdf(source_gs, template_gs, matched_gs, "Grayscale")

    # ============ RGB ============
    source_rgb = io.imread(filename1)
    template_rgb = io.imread(filename2)

    matched_rgb = np.zeros_like(source_rgb)

    for channel in range(3):
        matched_rgb[:, :, channel] = match_histograms(source_rgb[:, :, channel], template_rgb[:, :, channel])

    show_with_cdf(source_rgb, template_rgb, matched_rgb, "RGB")



if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
