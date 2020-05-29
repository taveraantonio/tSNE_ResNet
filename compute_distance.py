import os
import sys
import re
import numpy as np
from math import *

FILE_PATH = '/home/idg-ad1/PoliTo/Networks/tSNE_ResNet/result/105_classes_resnet.txt'
FROM = None
TO = None

scenarios = []
x_distances = []
y_distances = []
max_dist = 0.0
min_dist = sys.maxsize
max_scenario = None
min_scenario = None
print_all = False


def compute_distance():
    scenarios = []
    x_distances = []
    y_distances = []

    with open(FILE_PATH, "r") as fp:
        lines = fp.readlines()
        fp.close()

    for line in lines:
        split = line.split(" ")
        scenarios.append(str(split[0].rstrip()))
        x_distances.append(float(split[1].rstrip()))
        y_distances.append(float(split[2].rstrip()))

    return scenarios, x_distances, y_distances

def compute_distance_among_scenarios(from_s, to_s):
    if from_s is not None and to_s is not None:
        idx_a = scenarios.index(from_s)
        idx_b = scenarios.index(to_s)
        x_a = x_distances[idx_a]
        y_a = y_distances[idx_a]
        x_b = x_distances[idx_b]
        y_b = y_distances[idx_b]

        print("Distance between " + from_s + " and " + to_s + ": " + str(distance(x_a, y_a, x_b, y_b)))


def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)




# MAIN
scenarios, x_distances, y_distances = compute_distance()

for i, scenario_a in enumerate(scenarios):
    x_a = x_distances[i]
    y_a = y_distances[i]
    for j, scenario_b in enumerate(scenarios):
        if i != j:
            x_b = x_distances[j]
            y_b = y_distances[j]
            dist = distance(x_a, y_a, x_b, y_b)
            if print_all:
                print("Distance between " + scenario_a + " and " + scenario_b + ": " + str(dist))
            if dist > max_dist:
                max_dist = dist
                max_scenario = str(scenario_a + " vs " + scenario_b)
            if dist < min_dist:
                min_dist = dist
                min_scenario = str(scenario_a + " vs " + scenario_b)

print()
print("Max Distance Scenario: " + max_scenario)
print("with distance: " + str(max_dist))
print()
print("Min Distance Scenario: " + str(min_scenario))
print("with distance: " + str(min_dist))
print()


if FROM is not None and TO is not None:
    compute_distance_among_scenarios(FROM, TO)
