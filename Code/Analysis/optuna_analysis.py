import csv
import sqlite3
from sqlite3 import Error

import fiona
import matplotlib.pyplot as plt
import numpy as np


def create_connection(db):
    c = None
    try:
        c = sqlite3.connect(db)
    except Error as e:
        print(e)
    return c


def get_all_results(c: sqlite3.Connection):
    cur = c.cursor()
    cur.execute(
        'SELECT trial_id, GROUP_CONCAT(value, "," ) AS result FROM trial_values GROUP BY trial_id')
    rows = cur.fetchall()
    results = np.zeros((len(rows), 8))
    n_trials = len(rows)
    for i in range(n_trials):
        detections, distance = rows[i][1].split(',')
        results[i][:3] = int(rows[i][0]), int(float(detections)), float(distance)

    cur.execute('SELECT trial_id, GROUP_CONCAT(param_value, ",") AS result FROM trial_params GROUP BY trial_id')
    values = np.zeros((n_trials, 6))
    rows = cur.fetchall()
    for i in range(n_trials):
        c1, c2, n_particles, source_step, w_start = rows[i][1].split(',')
        results[i][3:] = c1, c2, n_particles, source_step, w_start

    sc = plt.scatter(results[:, 1]/2, results[:, 2]/2)
    with open(f'../Analysis/distance_v_detections.csv', 'w') as file:
        writer = csv.writer(file)
        for i in range(len(results)):
            writer.writerow([results[i, 1]/2, results[i, 2]/2])
    plt.show()


if __name__ == '__main__':
    dataset = 'cifar'
    database = f'../Experiments/DistributedBBA/results/{dataset}_optuna_log_log.db'
    conn = create_connection(database)
    get_all_results(conn)
