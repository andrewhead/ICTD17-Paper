import argparse
import re
import os.path
import math
from tqdm import tqdm

# These two lines allow us to use headless matplotlib (e.g., without Tkinter or
# some other display front-end)
import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def visualize_activations(activations_filename, image_dir, output_filename):

    # Open file first time just to get the number of filters
    activations_file = open(activations_filename)
    num_filters = sum(1 for line in activations_file)
    activations_file.close()

    with open(activations_filename) as activations_file:

        # We use PdfPages as it will let us make a multi-page PDF.  If there
        # are a lot of images in it, then PDF readers should be able to load
        # just the content on one page at a time, instead of taking a very
        # long time initializing to load one huge page of figures.  This
        # use of PdfPages is based on the code example at:
        # https://matplotlib.org/examples/pylab_examples/multipage_pdf.html
        with PdfPages(output_filename) as output_file:

            for line in tqdm(activations_file, total=num_filters):

                # For each line, find the filter and exemplar images
                match = re.match(r"(\d+): \[(.*)\]$", line.strip())
                filter_index = int(match.group(1))
                image_indexes = [int(n) for n in match.group(2).split()]

                # Make array of subplots for showing exemplars                
                rows = 4
                cols = math.ceil(len(image_indexes) / rows)
                f, axarr = plt.subplots(rows, cols, figsize=(24, 18))
                f.suptitle("Images activating filter %d" % (filter_index))

                # Hide axes, make it look prettier
                for ax in axarr.flatten():
                    ax.axis('off')

                # Show each image in each cell
                for i, image_index in enumerate(image_indexes):
                    image = mpimg.imread(os.path.join(
                        image_dir, str(image_index) + ".jpg"))
                    row = int(i / cols)
                    col = i % cols
                    axarr[row, col].imshow(image)

                # These two lines save the figure to a page of the PDF
                output_file.savefig()
                plt.close()
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a PDF with the images " +
        "that maximize the activation of each filter")
    parser.add_argument("activations", help="File mapping filter indexes " +
        "to indexes of images that maximize their activation.  Produced by the " +
        "`get_activations` script")
    parser.add_argument("image_dir", help="Directory that contains all of the " +
        "images for whcih activations were previously measured")
    parser.add_argument("output", help="Title of a PDF file to which to output " +
        "the resulting visualizations")
    args = parser.parse_args()

    visualize_activations(
        activations_filename=args.activations,
        image_dir=args.image_dir,
        output_filename=args.output,
    )
