import re, numpy as npy
import sys, os


def extract_iso_pdb(grid_file, output_filename, iso_value):
    """Using mdpocket's functionality to extract a PDB file of the wanted isovalue"""
    f = open(grid_file, "r")
    # get the axis that shows the most variation during the trajectory, this will be the leading axis
    # read the header - here is an example
    header = ""
    tmp = f.readline()
    while tmp[0] != "o":
        header = header + tmp
        tmp = f.readline()
    # print header
    # read the grid size
    r = re.compile('\w+')
    gsize = r.findall(tmp)
    gsize = [int(gsize[-3]), int(gsize[-2]), int(gsize[-1])]
    # print gsize
    # read the origin of the system
    line = f.readline().split()
    origin = [float(line[-3]), float(line[-2]), float(line[-1])]
    # print origin
    # read grid space
    line = f.readline().split()
    deltax = [float(line[-3]), float(line[-2]), float(line[-1])]
    line = f.readline().split()
    deltay = [float(line[-3]), float(line[-2]), float(line[-1])]
    line = f.readline().split()
    deltaz = [float(line[-3]), float(line[-2]), float(line[-1])]

    # pay attention here, this assumes always orthogonal normalized space, but normally it should be ok
    delta = npy.array([deltax[0], deltay[1], deltaz[2]])

    # read the number of data
    f.readline()
    r = re.compile('\d+')
    n_entries = int(r.findall(f.readline())[2])

    if (n_entries != gsize[0] * gsize[1] * gsize[2]): sys.exit(
        "Error reading the file. The number of expected data points does not correspond to the number of labeled data points in the header.")

    # create a 3D numpy array filled up with 0
    # initiate xyz counter for reading the grid data
    z = 0
    y = 0
    x = 0

    print("Reading the grid. Depending on the number of data points you have this might take a while....")
    path = open(output_filename, "w")

    counter = 1
    for count in range(n_entries // 3):
        c = f.readline().split()
        if (len(c) != 3):
            print("error reading grid data")
            sys.exit("exiting the program")
        for i in range(3):
            if (iso_value < 0 and float(c[i]) < iso_value) or (iso_value > 0 and float(c[i]) > iso_value):
                path.write('ATOM  %5d  C   PTH     1    %8.3f%8.3f%8.3f%6.2f%6.2f\n' % (
                counter, origin[0] + float(x) * delta[0], origin[1] + float(y) * delta[1],
                origin[2] + float(z) * delta[2], 0.0, 0.0))
                counter += 1
            z += 1
            if z >= gsize[2]:
                z = 0
                y += 1
                if y >= gsize[1]:
                    y = 0
                    x += 1
    path.close()
    f.close()
    print("finished writing %s" % (output_filename))