#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:46:37 2019

@author: andreamourelo
"""

from IPython.display import display, Markdown, Latex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import csv
import numpy as np
import sys, getopt

def gvrp(filename):
    print('Start processing file :', instance)

    #filename = 'GVRP/A-n32-k5-C11-V2.gvrp'
    instance = (filename[:-5])
    delimiter = ' '
    #print(instance)
    
    # No final, vamos ter:
    # nodes = [node1, node2, ..., nodenNodes] = [[x_pos_node1, y_pos_node1, group_node_1], ....]
    # com group_node entre [1, nGroups]
    
    # e demands = [demand_group_1, demand_group_2, ..., demand_group_nGroups]
    with open(filename,'r') as file:
        csv_reader = csv.reader(file,
                               delimiter = delimiter)
        line_count = 0
        node_count = 0
        group_count = 0
        nNodes, nGroups, nVehicles, capacity = 0,0,0,0
        for row in csv_reader:
            if line_count == 2 :  # DIMENSION : nNodes
                nNodes = int(row[2])
                nodes = np.zeros((nNodes,3), dtype=np.int)
            elif line_count == 3 : # VEHICLES : nVehicles
                nVehicles = int(row[2])
            elif line_count == 4 : # GVRP_SETS : nGroups
                nGroups = int(row[2])
                demands = np.zeros((nGroups,1), dtype=np.int)
            elif line_count == 5 :
                capacity = int(row[2]) # CAPACITY : capacity
            elif line_count >= 8 and line_count <= 8 + nNodes +  - 1: # Informacoes sobre as posicoes dos nos => num x_pos y_pos
                nodes[node_count] = [int(row[1]),int(row[2]), -1] # Ainda nao temos info sobre o grupo do nó
                node_count += 1
            elif line_count >= 8 + nNodes + 1 and line_count <= 8 + nNodes + nGroups: # Informacoes sobre os grupos => num_grupo elem elem ... elem -1
                nNodesInGroup = len(row)
                for numNode in range(1,nNodesInGroup - 1 ): # Para evitar o "-1" que mostra quando pular linha
                    nodeOfGroup = int(row[numNode]) - 1
                    nodes[nodeOfGroup][2]  = group_count + 1
                group_count += 1
                if group_count == nGroups:
                    group_count = 0  # Reseteando no final para usar na leitura seguinte
            elif line_count >= 8 + nNodes + nGroups + 2 and line_count <= 8 + nNodes + 2*(nGroups + 1) - 1:
                demands[group_count] = int(row[1])
                group_count += 1
            line_count += 1
    
    #print(nNodes,nVehicles, nGroups, capacity)
    #print(nodes)
    #print(demands)
    
    
    # Uso da simetria da matriz diagonal no calculo 
    distances = np.zeros((nNodes,nNodes))
    for node in range(0,nNodes):
        for second_note in range(node+1,nNodes):
            x1, x2 = nodes[node][0], nodes[second_note][0]
            y1, y2 = nodes[node][1], nodes[second_note][1]
            distance2 = (x2 - x1)**2 + (y2 - y1)**2
            distances[node][second_note] = distance2
            distances[second_note][node] = distance2
        
    distances = np.sqrt(distances)
    #print(distances)
    #print(pd.DataFrame(distances).head())
    
    
    print('Ending processing file :', instance)
    
if __name__ == "__main__":
    gvrp(sys.argv[1])