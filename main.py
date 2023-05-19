import argparse
import math
import os
import random
import secrets

from gimpformats.gimpXcfDocument import GimpDocument
from PIL import Image

"""
#################
Parsing functions
#################
"""


def process_parse(args):
    """
    Adjusts terminal input for later interaction, e.g. fitting path to input filename.
    :param args: (Object) Generic object containing terminal options as attributes.
    :return [args, img]: (Array) Array containing updated args and input image objects.
    """
    if args.out_file is None or args.out_file == "":
        args.out_file = "output.jpg"
    args.in_file = os.path.join(os.getcwd(), args.in_file)
    img = Image.open(args.in_file)
    img = img.convert("RGB")

    return [args, img]


def str_to_bin(message):
    """
    Turns a string input into a string of 7-bit binary values.
    :param message: (String) A payload string to embed in the cover.
    :return payload_arr: (Array) String values to be written as payload.
    """
    payload = []
    for ch in message:  # For character in message
        x = format(ord(ch), '7b')  # chr -> int -> 7b binary
        for b in x:  # for bit
            payload.append(str(b))

    return payload


"""
#################
Mask functions
#################
"""


def generate_mask():
    bin_string = ""
    mask_value = list(("0" * 32) + ("1" * 32))  # 64 binary values in a 50:50 split
    random.shuffle(mask_value)  # Shuffle array

    while mask_value:
        bin_string += mask_value[0]
        mask_value = mask_value[1:]

    return int(bin_string, 2)


def value_to_mask(mask_key):
    """
    Turn an integer value into a 2D binary array.
    :param mask_key: (int) Integer equivalent of 64-bit binary mask.
    :return mask_arr: (Array) Returns generated 2D array mask.
    """
    mask_arr = []
    str_value = format(mask_key, 'b').zfill(64)

    while str_value:
        sub_arr = []
        for bit in str_value[0:8]:
            sub_arr.append(int(bit))
        mask_arr.append(sub_arr)
        str_value = str_value[8:]

    return mask_arr


def mask_to_value(mask_arr):
    """
    Turn a 2D binary mask array into an integer mask_key.
    :param mask_arr: (Array) Mask to be reduced to an int.
    :return mask_key: (int) Integer value for the mask.
    """
    mask_key = ""
    for r in mask_arr:
        for c in r:
            mask_key += str(c)
    mask_key = int(mask_key, 2)

    return mask_key


def generate_cover(mode, w, h):
    """
    Creates a cover layer of generated noise.
    :param mode: (Mixed) Whether "--fast" option was entered; None if not, otherwise int.
    :param w: (int) Layer width.
    :param h: (int) Layer height.
    :return: (Object) Created image.
    """
    cover_layer = Image.new("RGB", (w, h))
    per = 0
    print(f'{per}%', end='')

    if not mode:
        for x in range(w):
            for y in range(h):
                i = secrets.randbelow(256)
                j = secrets.randbelow(256)
                k = secrets.randbelow(256)
                cover_layer.putpixel([x, y], (i, j, k))
            if 100 * x // w == per + 10:
                per += 10
                print(f'\r{per}%', end='', flush=True)
    else:
        for x in range(w):
            for y in range(h):
                i = random.randrange(256)
                j = random.randrange(256)
                k = random.randrange(256)
                cover_layer.putpixel([x, y], (i, j, k))
            if 100 * x // w == per + 10:
                per += 10
                print(f'\r{per}%', end='', flush=True)

    print('\r100%', end='\n', flush=True)
    cover_layer.save("./noise.png", interlace=False, optimize=True, subsampling=0, quality=100)
    return cover_layer


"""
#################
Mathematical functions
#################
"""


def stddev_iterator(img, mask_arr):
    """
    Iterates through all 8x8 grids to calculate q value for each region
    :param img: (Object) Image being examined.
    :param mask_arr: (Array) 2D array of binary values representing a mask.
    :return q_outputs: (Array) Array of calculated q values.
    """
    h = img.height - (img.height % 8)
    w = img.width - (img.width % 8)
    q_outputs = []

    for py in range(0, h, 8):  # For page y, prev: for py in range(0, h, 8):
        for px in range(0, w, 8):  # For page x, prev: for px in range(0, w, 8):
            means = mean_calc(img, mask_arr, px, py)  # Calculate mean values
            zero_mean, one_mean = means[0], means[1]  # Assign means according to 0/1 mask value
            standard_dev = stddev_calc(img, mask_arr, px, py)  # Calculate standard deviation
            if standard_dev == 0.0:  # Handles excess cycles that cause Div0 errors
                break
            q = (one_mean - zero_mean) / standard_dev  # Calculates q value
            q_outputs.append(q)  # Appends q value to an index for each 8x8 grid

    return q_outputs


def q_sd_calc(q_output):
    """
    Uses q values to determine
    Returned values used to identify statistical significance.
    :param q_output: (Array) Index of each 8x8 q value.
    :return q_mean, q_variance: (Array) Returns the mean and sqrt(variance) of q values.
    """
    q_count = len(q_output)
    q_mean = 0
    q_variance = 0

    for q in q_output:
        q_mean += q
    q_mean /= q_count  # mean: sum / count

    for q in q_output:
        q_variance += (q - q_mean) ** 2  # difference
    q_variance /= q_count  # variance

    return [q_mean, math.sqrt(q_variance)]  # [mean, standard deviation]


def mean_calc(img, mask_arr, px, py):
    """
    Collects values of pixels according to alignment with mask 0 and 1 values.
    Returns the mean values and sum values of both groups.
    :param img: (Object) Image being examined.
    :param mask_arr: (Array) 2D array of binary values representing a mask.
    :param px: (int) Grid coordinate, used to iterate 8x8 grids of (x, y) pixels.
    :param py: (int) Grid coordinate.
    :return [multi]: (Array) Returns the independent mean and sum values of 0 and 1 pixels.
    """
    zero_sum = 0
    zero_values = []
    one_sum = 0
    one_values = []

    for y in range(0, 8):  # for y pixel
        for x in range(0, 8):  # for x pixel
            a, b = (px + x), (py + y)  # Co-ords offset by pages
            pixel = img.getpixel((a, b))  # Get pixel values
            px_val = pixel[0] + pixel[1] + pixel[2]  # Sum values

            if mask_arr[y][x] == 1:  # If this pixel on the input mask is 1
                one_sum += px_val
                one_values.append(px_val)
            else:  # Otherwise it is 0
                zero_sum += px_val
                zero_values.append(px_val)

    # mean: sum(values) / count(values)
    zero_mean = zero_sum / 32
    one_mean = one_sum / 32

    return [zero_mean, one_mean, zero_values, one_values]


def stddev_calc(img, mask_arr, px, py):
    """
    Calculates the standard deviation within each 8x8 grid
    :param img: (Object) Image for pixel value examination.
    :param mask_arr: (Array) Pixel mask for set grouping.
    :param px: (int) Grid coordinate, used to iterate 8x8 grids of (x, y) pixels.
    :param py: (int) Grid coordinate.
    :return std_dev: (float) Standard deviation calculated from inputs.
    """
    means = mean_calc(img, mask_arr, px, py)
    zero_mean, one_mean = means[0], means[1]  # Mean values
    zero_values, one_values = means[2], means[3]  # Array of separated values

    # difference: value - mean
    zero_diff_sum = 0
    one_diff_sum = 0
    for z in zero_values:
        zero_diff_sum += (z - zero_mean) ** 2  # ^2 here to reduce redundancy
    for o in one_values:
        one_diff_sum += (o - one_mean) ** 2

    # variance: sum(difference^2)
    zero_var = zero_diff_sum / 32
    one_var = one_diff_sum / 32

    # std dev: sqrt((variance + variance)/(count/2))
    std_dev = (zero_var + one_var) / 32
    std_dev = math.sqrt(std_dev)

    return std_dev


"""
#################
Codec functions
#################
"""


def encode_image(cover_layer, mask_key, modifier, payload):
    """
    Encodes a payload into a provided cover layer.
    :param cover_layer: (Image) Image file written to.
    :param mask_key: (int) Used to calculate binary mask.
    :param modifier: (int) Offset value input.
    :param payload: (String) Payload message to embed.
    :return: None.
    """
    print("CONVERTING PAYLOAD...")
    payload_arr = str_to_bin(payload)  # CHANGE TO TERMINAL INPUT
    print("CONVERTING TO MASK...")
    mask_arr = value_to_mask(mask_key)

    w = cover_layer.width - (cover_layer.width % 8)  # Do not use for final boundary testing
    h = cover_layer.height - (cover_layer.height % 8)  # Use for testing acceptable range

    print("PAGINATING IMAGES...")
    for py in range(0, h, 8):  # For page y [prev: for py in range(0, h, 8):]
        for px in range(0, w, 8):  # For page x [prev: for px in range(0, w, 8):]
            if payload_arr and payload_arr[0] == "1":  # If next value is 1
                standard_dev = stddev_calc(cover_layer, mask_arr, px, py)  # Standard deviation
                offset = standard_dev + modifier  # Further offset colour values by custom input

                for y in range(0, 8):  # for y pixel
                    for x in range(0, 8):  # for x pixel
                        if mask_arr[y][x] == 1:  # Mask = 1
                            a, b = (px + x), (py + y)  # Co-ords offset by pages
                            pixel = cover_layer.getpixel((a, b))  # Get pixel values
                            cover_layer.putpixel([a, b], (
                                int(pixel[0] + offset),
                                int(pixel[1] + offset),
                                int(pixel[2] + offset)
                            ))
                        elif mask_arr[y][x] == 0:  # Mask = 0
                            a, b = (px + x), (py + y)  # Co-ords offset by pages
                            pixel = cover_layer.getpixel((a, b))  # Get pixel values
                            cover_layer.putpixel([a, b], (
                                int(pixel[0] - offset),
                                int(pixel[1] - offset),
                                int(pixel[2] - offset)
                            ))
            payload_arr = payload_arr[1:]
    cover_layer.save("/home/doro/Desktop/StegoProject/noise.png",
                     interlace=False, optimize=True,
                     subsampling=0, quality=100)


def decode_image(file_input, mask_key):
    """
    Decodes a passed image using a mask-derived integer value as a key.
    :param file_input: (Object) Image to be examined.
    :param mask_key: (int) Used to calculate binary mask.
    :return: None.
    """
    bin_stream = ""
    final_str = ""

    mask_arr = value_to_mask(mask_key)  # Uses input integer to generate binary mask.
    review_out = stddev_iterator(file_input, mask_arr)  # Calculates and compares means of 0 vs 1 mask pixels
    q_mean, q_sd = q_sd_calc(review_out)

    for q in review_out:
        x = q - (q_mean + q_sd)
        if x >= 1:
            bin_stream += "1"
        else:
            bin_stream += "0"

    while bin_stream:
        byte = bin_stream[0:7]
        final_str += chr(int(byte, 2))
        if bin_stream[0:28] == ("0" * 28) or bin_stream[0:28] == ("1" * 28):  # If the next 4 * 7b are blank/filled
            break
        bin_stream = bin_stream[7:]

    print(f'OUTPUT:{final_str}')


"""
#################
GIMP functions
#################
"""


def xcf_generate(bg_dir, noise_layer, xcf_dir):
    """

    :param bg_dir:
    :param noise_layer:
    :param xcf_dir:
    :return:
    """
    if bg_dir[-4:] == ".png":
        import_func = "image = pdb.file_png_load("
    else:
        import_func = "image = pdb.file_jpeg_load("

    bg_dir = "\"" + bg_dir + "\""
    noise_layer = "\"" + noise_layer + "\""
    xcf_dir = "\"" + xcf_dir + "\""

    bash = "gimp -dfi --batch-interpreter python-fu-eval -b " \
           "'from gimpfu import *; " + \
           import_func + bg_dir + "," + bg_dir + "); " \
           "mask = pdb.gimp_file_load_layer(image, " + noise_layer + "); " \
           "pdb.gimp_image_insert_layer(image, mask, None, 0); " \
           "pdb.gimp_layer_add_alpha(mask);" \
           "pdb.gimp_layer_set_mode(mask, 23); " \
           "pdb.gimp_xcf_save(0, image, None, " + xcf_dir + ", " + xcf_dir + ");'"

    os.system(bash)



"""
#################
Terminal functions
#################
"""


def terminal_parse():
    parser = argparse.ArgumentParser(prog="stexcf.py", description='Embed or retrieve a payload from an xcf file.')
    subparsers = parser.add_subparsers(help='Three modes: [encode], [decode], [keygen]', dest='subp_type')

    # Parse encoding instructions
    parser_en = subparsers.add_parser('encode', help='Encode a message into an xcf.')
    parser_en.add_argument('in_file', help='Cover file. Should be .jpg or .png.', type=str)
    parser_en.add_argument('payload', help='Message to conceal.', type=str)
    parser_en.add_argument('-k', dest='key', help='Provide key from [keygen].', type=int, required=False)
    parser_en.add_argument('-o', dest='offset', help='Data offset; higher values are more robust. Default = 5.',
                           type=int, required=False)
    parser_en.add_argument('--fast', dest='fast_mode',
                           help='Run faster with weaker randomness. Off by default, recommended only for offset tests.',
                           action='count', required=False)
    parser_en.set_defaults(func=main_encode)

    # Parse decoding instructions
    parser_de = subparsers.add_parser('decode', help='Decode an xcf embedded message.')
    parser_de.add_argument('in_file', help='Examined file. Must be .xcf format.', type=str)
    parser_de.add_argument('target', help='Target layer name.', type=str)
    parser_de.add_argument('key', help='Extraction key.', type=int)
    parser_de.set_defaults(func=main_decode)

    # Parse key generation instructions
    parser_ge = subparsers.add_parser('keygen', help='Generate a key for encoding.')
    parser_ge.set_defaults(func=generate_mask())

    args = parser.parse_args()

    if args.subp_type == "encode":
        main_encode(args.in_file, args.offset, args.payload, args.key, args.fast_mode)
    elif args.subp_type == "decode":
        main_decode(args.in_file, args.target, args.key)
    elif args.subp_type == "keygen":
        print(f'Decryption key: [{generate_mask()}] (without brackets)')
    else:
        print("Please select encode, decode or keygen mode.")
    exit()


"""
#################
Main functions
#################
"""


def main_encode(in_file, offset, payload, mask_key, fast_mode):
    cwd = os.getcwd()
    bg_dir = cwd + "/" + in_file
    bg = Image.open(bg_dir)
    project_specs = [bg.width, bg.height]

    if not offset:
        offset = 5

    if in_file[-4:] not in [".png", ".jpg"] and in_file[-5:] != ".jpeg":
        print("MAIN IMAGE MUST BE .PNG OR .JPG!\nEXITING...")
        exit()

    if not mask_key:
        print("NO KEY. GENERATING...")
        mask_key = generate_mask()

    print("GENERATING COVER...")
    cover_layer = generate_cover(fast_mode, project_specs[0], project_specs[1])

    print("ENCODING IMAGE...")
    encode_image(cover_layer, mask_key, offset, payload)

    cover_dir = cwd + "/noise.png"
    xcf_dir = cwd + "/output.xcf"
    print("GENERATING XCF...\n>>>Please use SIGINT (Ctrl+C) once 'batch command executed successfully' displays.")
    xcf_generate(bg_dir, cover_dir, xcf_dir)
    print(f'Decryption key: [{mask_key}] (without brackets)')
    print("XCF ENCODED. EXITING...")


def main_decode(in_file, target, mask_key):
    in_file = os.getcwd() + "/" + in_file
    project = GimpDocument(in_file)

    for layer in project.layers:
        if layer.name == target:
            print("ATTEMPTING EXTRACTION...")
            decode_image(layer.image, mask_key)
            print("XCF DECODED. EXITING...")
            exit()
    print("LAYER NOT FOUND. EXITING...")


terminal_parse()
