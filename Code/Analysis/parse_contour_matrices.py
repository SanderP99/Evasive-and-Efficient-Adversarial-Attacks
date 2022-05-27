import csv

import numpy as np


def parse_contour_matrices():
    for i in range(25):
        with open(f'contour_files/{i}.csv', 'w') as file:
            writer = csv.writer(file)
            x = np.load(f'contour_files/{i}_x.npy')
            y = np.load(f'contour_files/{i}_y.npy')
            z = np.load(f'contour_files/{i}_z.npy')
            for x_i, x_v in enumerate(x):
                for y_i, y_v in enumerate(y):
                    writer.writerow([x_v, y_v, z[y_i][x_i]/2])
                writer.writerow([])



if __name__ == '__main__':
    parse_contour_matrices()
