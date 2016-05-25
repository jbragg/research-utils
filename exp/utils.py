"""Random utils useful for experiments."""
import random


def get_random_sample(filename, sample_size,  outfile):
    """Dump a random sample of the given csv data."""
    with open(filename, 'rb') as infile:
        header = infile.readline();
        rows = []
        for row in infile:
            rows.append(row)
        random_sample = random.sample(rows, sample_size)
        with open(outfile, 'wb') as ofile:
            ofile.writelines([header] + random_sample)

