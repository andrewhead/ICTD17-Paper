from argparse import ArgumentParser
import csv
import shutil
import os.path
import glob
import re


def make_cell_to_index_lookup(csv_filename):
    lookup = {}
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the first row (headers)
        first_row = True
        for row in reader:
            if first_row:
                first_row = False
                continue
            index = int(row[0])
            i = int(row[2])
            j = int(row[3])
            lookup[(i, j)] = index
    return lookup


def make_cell_to_file_lookup(input_dir):
    lookup = {}
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            match = re.match(r"(\d+)_(\d+).jpg", filename)
            if match:
                i = int(match.group(1))
                j = int(match.group(2))
                lookup[(i, j)] = os.path.join(dirpath, filename)
    return lookup


def move_files(cell_to_index_dict, cell_to_file_dict, output_dir, move):

    # Create the output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    copy_count = 0
    for (i, j), index in cell_to_index_dict.items():

        # Only transfer a file if there's a record for it in the
        # nightlights data, and the image actually exists
        if not (i, j) in cell_to_file_dict:
            continue
        source_filename = cell_to_file_dict[(i, j)]
        if not os.path.exists(source_filename):
            continue

        # Copy or move the file
        copy_function = shutil.copyfile if not move else shutil.move
        copy_function(
            source_filename,
            os.path.join(output_dir, str(index) + ".jpg")
        )
        copy_count += 1
        print(".", end="")
        if copy_count % 100 == 0:
            print()


if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Rename image files")
    argument_parser.add_argument("input_dir")
    argument_parser.add_argument("csvfile", help="Name of file with image indexes")
    argument_parser.add_argument("--output-dir", default="images")
    argument_parser.add_argument("--move", help="Move images instead of copying", action="store_true")
    args = argument_parser.parse_args()
    cell_to_index_dict = make_cell_to_index_lookup(args.csvfile)
    cell_to_file_dict = make_cell_to_file_lookup(args.input_dir)
    move_files(cell_to_index_dict, cell_to_file_dict, args.output_dir, args.move)

