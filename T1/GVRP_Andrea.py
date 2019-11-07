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
import math
import random
import timeit
import os
import networkx as nx

instance = ''
nNodes = 0
nVehicles = 0
nGroups = 0
capacity = 0
nodes = []
groups = []
globalDemands = []
distances = []

######## FUNCTIONS

### Functions to process and print in log
def processFile(filename):
    '''No final, vamos ter:
    nodes = [node1, node2, ..., nodenNodes] = [[x_pos_node1, y_pos_node1, group_node_1], ....]
    com group_node entre [1, nGroups]
    e globalDemands = [demand_group_1, demand_group_2, ..., demand_group_nGroups] 
    
    Esta funcao calcula tambem a matriz de distâncias
    '''
    
    #print('Start processing file :', filename)
    instance = filename[:-5]
    delimiter = ' '
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
                groups = [[0]] # Para guardar uma lista com os grupos
                globalDemands = np.zeros((nGroups), dtype=np.int)
            elif line_count == 5 :
                capacity = int(row[2]) # CAPACITY : capacity
            elif line_count >= 8 and line_count <= 8 + nNodes +  - 1: # Informacoes sobre as posicoes dos nos => num x_pos y_pos
                nodes[node_count] = [int(row[1]),int(row[2]), 0] # Ainda nao temos info sobre o grupo do nó, no depot vai ter grupo 0 
                node_count += 1
            elif line_count >= 8 + nNodes + 1 and line_count <= 8 + nNodes + nGroups: # Informacoes sobre os grupos => num_grupo elem elem ... elem -1
                nNodesInGroup = len(row)
                group = []
                for numNode in range(1,nNodesInGroup - 1 ): # Para evitar o "-1" que mostra quando pular linha
                    nodeOfGroup = int(row[numNode]) - 1
                    nodes[nodeOfGroup][2]  = group_count + 1
                    group.append(nodeOfGroup)
                group_count += 1
                groups.append(group)
                if group_count == nGroups:
                    group_count = 0  # Reseteando no final para usar na leitura seguinte
            elif line_count >= 8 + nNodes + nGroups + 2 and line_count <= 8 + nNodes + 2*(nGroups + 1) - 1:
                globalDemands[group_count] = int(row[1])
                group_count += 1
            line_count += 1

    '''print("nNodes, nVehicles, nGroups, capacity : ", nNodes, nVehicles, nGroups, capacity)
    print("nodes : ",nodes)
    print("groups : ",groups)
    print("globalDemands : ", globalDemands)  
    print("total Demand :", sum(globalDemands))'''
    
    distances = np.full((nNodes,nNodes), 100000000000)

    for node in range(0,nNodes):
        for second_node in range(node+1,nNodes):
            if nodes[node][2] != nodes[second_node][2]: # Nao queremos arestas entre os nós de um mesmo grupo
                x1, x2 = nodes[node][0], nodes[second_node][0]
                y1, y2 = nodes[node][1], nodes[second_node][1]
                distance2 = (x2 - x1)**2 + (y2 - y1)**2
                distances[node][second_node] = math.sqrt(distance2)
                distances[second_node][node] = math.sqrt(distance2)
                
    #print('Ending processing file :', instance)
    return instance, nNodes, nVehicles, nGroups, capacity, nodes, groups, globalDemands, distances

def printgraph():
    '''Prints graph for visualization purposes'''
    G = nx.DiGraph()
    node = 1
    nodecolors = []
    count_nodes = 0
    for node in nodes:
        pos = [node[0],node[1]]
        G.add_node(count_nodes, pos = pos)
        G.node[count_nodes]['group'] = node[2]
        count_nodes+=1
        
    colors = ['cadetblue','red','blue','green','yellow', 'darkgreen', 'mediumpurple', 'purple', 'grey', 'greenyellow', 'pink', 'lightblue']
    pos=nx.get_node_attributes(G,'pos')

    labels=dict((n,d['group']) for n,d in G.nodes(data=True))
    nx.draw(G, pos, labels = labels)
    for i in range(0,len(colors)):
        nodelist = []
        for node in G.nodes():
            if G.node[node]['group'] == i:
                nodelist.append(node)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, node_size=400, node_color=colors[i])
        i+=1

# Função usada para o set up da seed a cada função, pois se não não funcionava
def set_seed(seed_value):
    '''Setting seed'''
    # Set a seed value: 
    seed_value = seed_value  
    # 1. Set PYTHONHASHSEED environment variable at a fixed value: 
    os.environ['PYTHONHASHSEED']=str(seed_value) 
    # 2. Set python built-in pseudo-random generator at a fixed value:
    random.seed(seed_value) 
    # 3. Set numpy pseudo-random generator at a fixed value:
    np.random.seed(seed_value)    
        
def calculateAngle(x,y):
    '''Calculates angle for element of coordinates (x,y)'''
    angle = 0
    pi = math.pi
    arctanXY = math.atan(y/x)
    if x == 0:
        if y > 0:
            angle = pi/2
        else :
            angle = 3*pi/2
    elif x > 0:
        if y >= 0:
            angle = arctanXY
        if y < 0:
            angle = arctanXY + 2*pi
    else:
        angle = arctanXY + pi
    return angle      
        

def createInitialSolutionBySweep(nVehicles, nGroups, capacity, demands, nodes):
    '''Creates initial solution with sweep method'''
    angles = {}
    # Create dictionary with "nodeNumber : (angle, group)"
    for i in range(1,len(nodes)): # Not inserting the depot (0)
        node = nodes[i]
        angle = calculateAngle(node[0], node[1])
        angles.update({i : (angle, node[2])}) 
    sortedAngles = sorted(angles.items(), key=lambda x:x[1])
    # Keeping track of groups already assigned
    groupsAssigned = {}
    for group in range(1,nGroups + 1): # the depot is group 0
        groupsAssigned.update({group : False})
    # Inserting routes
    routes = []
    demandDistribution = []
    demandDistributionOfGroups = []
    nextRoute = []
    demandOfRoute = []
    totalDemandOfRoute = 0
    for nodeData in sortedAngles: # gets (node,(angle,group))
        nodeGroup = nodeData[1][1]
        if groupsAssigned[nodeGroup] == False: # not yet assigned
            groupDemand = demands[nodeGroup - 1] 
            if groupDemand + totalDemandOfRoute < capacity:
                nextRoute.append(nodeGroup)
                demandOfRoute.append(groupDemand)
                totalDemandOfRoute += groupDemand
            else:
                routes.append(nextRoute)
                demandDistributionOfGroups.append(demandOfRoute)
                demandDistribution.append(totalDemandOfRoute)
                nextRoute = [nodeGroup]
                demandOfRoute = [groupDemand]
                totalDemandOfRoute = groupDemand
            groupsAssigned[nodeGroup] = True
    routes.append(nextRoute)
    demandDistributionOfGroups.append(demandOfRoute)
    demandDistribution.append(totalDemandOfRoute)

    if len(routes) > nVehicles: 
        '''When we have too many routes assigned'''
        capacityMissing = demandDistribution[-1] - (capacity - demandDistribution[-2])
        route1 = routes[-2]
        demandProblem = demandDistribution[-1]
        for i in range(len(route1)):
            group1 = route1[i]
            demand1 = demandDistributionOfGroups[-2][i]
            for j in range(len(routes[:-2])): # Percorrendo todas as outras rotas para ver se podemos inserir o grupo la
                route2 = routes[j]
                for k in range(len(route2)):
                    group2 = route2[k]
                    demand2 = demandDistributionOfGroups[j][k]
                    diff = demand1 - demand2
                    newDemandGroup2 = demandDistribution[j] - demand2 + demand1
                    newDemandGroup1 = demandDistribution[-2] + demand2 - demand1 + demandProblem
                    if abs(diff) >= capacityMissing and newDemandGroup2 <= capacity and newDemandGroup1 <= capacity:
                        route1[i] = group2
                        route2[k] = group1
                        demandDistributionOfGroups[j][k] = demand1
                        demandDistributionOfGroups[-2][i] = demand2
                        demandDistribution[-2] = newDemandGroup1
                        demandDistribution[j] = newDemandGroup2
                        
                        for group in routes[-1]:
                            routes[-2].append(group)
                        routes = routes[:-1]
                        demandDistribution = demandDistribution[:-1]
       
    if len(routes) > nVehicles: 
        '''If it is still not possible, create another solution by brute force'''
        routes, demandDistribution = createInitialSolutionByBruteForce(nVehicles, nGroups, capacity, demands)
        
    return routes, demandDistribution

def createInitialSolutionByBruteForce(nVehicles, nGroups, capacity, demands):
    '''Creates initial solution by brute force -> looks at every group and puts it in a group that has space'''
    routes = []
    capacities = []
    for vehicle in range(nVehicles): # Preparing routes = [[],[],....,[]]
        routes.append([]) 
    vehicleToFill = 0
    capacityFilled = 0
    for group in range(1,nGroups + 1): # Depot é 0
        oldCapacityFilled = capacityFilled
        groupDemand = demands[group - 1]
        space = capacity - capacityFilled
        if groupDemand < space:
            routes[vehicleToFill].append(group)
            capacityFilled += groupDemand
        else:
            vehicleToFill += 1
            if vehicleToFill == nVehicles :
                conseguiu = False
                for i in range(nVehicles - 1):
                    if capacities[i] + groupDemand < capacity:
                        capacities.append(capacityFilled)
                        routes[i].append(group)
                        capacities[i] += groupDemand 
                        vehicleToFill -= 1
                        capacityFilled = oldCapacityFilled
                        conseguiu = True
                        break
                if conseguiu == False:
                    vehicleToFill -=1
                    routes[0].append(group)
                    capacities[0] += groupDemand
                    #print("couldnt put it in any group")
            else:
                capacities.append(capacityFilled)
                capacityFilled = groupDemand
                routes[vehicleToFill].append(group)
    if capacityFilled != 0 and capacityFilled != oldCapacityFilled:
        capacities.append(capacityFilled)
    return routes, capacities


def localSearch(routes,distancesPerVehicle, timePassed, timeMax):
    '''Local search algorithm.'''
    routes_ls = routes
    min_distances = distancesPerVehicle
    end = False
    iterations = 0
    
    timeIni = timePassed
    time = timePassed
    inicio = timeit.default_timer()
    fim = timeit.default_timer()
    diferenca = fim - inicio
    newTime = time + diferenca
    if newTime >= timeMax or newTime + 2*diferenca > timeMax:
        return routes_ls, min_distances, newTime, newTime - timeIni
    
    # Intra-route neighborhood
    routes_ls, min_distances, changedTwoOpt = twoOptLocalSearch(routes_ls,min_distances) # Two Opt
    
    # Inter-route neighborhood
    fim = timeit.default_timer()
    diferenca = fim - inicio
    newTime = time + diferenca
    if newTime >= timeMax or newTime + 2*diferenca > timeMax:
        return routes_ls, min_distances, newTime, newTime - timeIni
    routes_ls, min_distances, changedRelocate = relocateLocalSearch(routes_ls, min_distances) # Relocate
    fim = timeit.default_timer()
    diferenca = fim - inicio
    newTime = time + diferenca
    if newTime >= timeMax or newTime + diferenca > timeMax:
        return routes_ls, min_distances, newTime, newTime - timeIni
    if changedRelocate == False:
        routes_ls, min_distances, changedSwap11 = swap11LocalSearch(routes_ls, min_distances) # Swap 11
        fim = timeit.default_timer()
        diferenca = fim - inicio
        newTime = time + diferenca
        if newTime >= timeMax or newTime + diferenca > timeMax:
            return routes_ls, min_distances, newTime, newTime - timeIni
        if changedSwap11 == False:
            routes_ls, min_distances, changedTwoOptStar = twoOptStarLocalSearch(routes_ls, min_distances) # 2 opt star
            fim = timeit.default_timer()
            diferenca = fim - inicio
            newTime = time + diferenca
            if newTime >= timeMax or newTime + diferenca > timeMax:
                return routes_ls, min_distances, newTime, newTime - timeIni
            if changedTwoOptStar == False:
                end = True
    iterations +=1
    fim = timeit.default_timer()
    diferenca = fim - inicio
    time += diferenca
    timeOfLocalSearch = time-timeIni
    return routes_ls, min_distances, time, timeOfLocalSearch

def twoOptLocalSearch(routes_final,min_distances):
    '''Tries twoOpt for every two nodes of a group'''
    changed = False
    for num_route in range(len(routes_final)):
        route_iteration = routes_final[num_route]
        min_distance = min_distances[num_route]
        group_i, group_j = -1, -1
        for group1 in route_iteration:
            group_i +=1
            for group2 in route_iteration: # tentando twoOpt com todos os outros da rota
                group_j +=1
                if group_i != group_j :
                    new_route = twoOptIntra(route_iteration, group_i, group_j)
                    new_distance = calculateDistanceAfter2Opt(new_route, group_i, group_j)
                    if new_distance < min_distance:
                        route_iteration = new_route
                        min_distance = new_distance
                        routes_final[num_route] = route_iteration
                        min_distances[num_route] = min_distance
                        changed = True
            group_j = -1
    return routes_final, min_distances, changed

def relocateLocalSearch(routes_relocate, min_distances_relocate):
    '''Tries to relocate every element of a route in every space of another route'''
    routes_final = routes_relocate
    min_distances = min_distances_relocate
    changed = False
    route_i,route_j = -1,-1  # servem para contar onde estamos ao percorrer as rotas
    group_i, group_j = -1,-1 # servem para contar onde estamos ao percorrer os grupos de cada rota
    for route1 in routes_final:
        route_i += 1
        for route2 in routes_final:
            route_j += 1
            if route_i != route_j:
                for group1 in route1:
                    group_i += 1
                    for group2 in route2:
                        group_j += 1
                        if route_i != route_j:
                            new_route1, new_route2 = relocate(route1,route2,group_i,group_j)
                            # We need to check if the new_routes are feasible
                            if isFeasible(new_route1) and isFeasible(new_route2):
                                new_distance1, new_distance2 = calculateDistanceAfterRelocate(new_route1, new_route2)
                                if new_distance1 + new_distance2 < min_distances[route_i] + min_distances[route_j]:
                                    routes_final,min_distances,changed = changeRoutes(routes_final,min_distances,new_route1,new_route2,route_i,route_j,group_i,group_j)
                                    route1 = new_route1
                                    route2 = new_route2
                                    return routes_final,min_distances,changed
                    group_j = -1
                group_i = -1
        group_i, group_j = -1,-1
        route_j = -1
    return routes_final,min_distances,changed

def swap11LocalSearch(routes_final, min_distances):
    '''Tries swap11 with every element of a route and every element of another route'''
    changed = False
    route_i,route_j = -1,-1  # servem para contar onde estamos ao percorrer as rotas
    group_i, group_j = -1,-1 # servem para contar onde estamos ao percorrer os grupos de cada rota
    for route1 in routes_final:
        route_i += 1
        endRoute1 = route1[-1]
        for route2 in routes_final:
            route_j += 1
            if route1 != route2:
                for group1 in route1:
                    if group1 != endRoute1: # swap até o antepenultimo
                        group_i += 1
                        for group2 in route2:
                            group_j += 1
                            new_route1, new_route2 = swap11(route1,route2,group_i,group_j)
                            # We need to check if the new_routes are feasible
                            if isFeasible(new_route1) and isFeasible(new_route2):
                                new_distance1, new_distance2 = calculateDistanceAfterSwap11(new_route1, new_route2)
                                if new_distance1 + new_distance2 < min_distances[route_i] + min_distances[route_j]:
                                    routes_final,min_distances,changed = changeRoutes(routes_final,min_distances,new_route1,new_route2,route_i,route_j,group_i,group_j)
                                    route1 = new_route1
                                    route2 = new_route2
                                    return routes_final,min_distances,changed
                        group_j = -1
                group_i = -1
        group_i, group_j = -1,-1
        route_j = -1
    return routes_final,min_distances,changed


def twoOptStarLocalSearch(routes_final, min_distances):
    '''Tries twoOpt* with every element of a route and every element of another route'''
    changed = False
    route_i,route_j = -1,-1  # servem para contar onde estamos ao percorrer as rotas
    group_i, group_j = -1,-1 # servem para contar onde estamos ao percorrer os grupos de cada rota
    for route1 in routes_final:
        route_i += 1
        for route2 in routes_final:
            route_j += 1
            if route1 != route2:
                for group1 in route1:
                    group_i += 1
                    for group2 in route2:
                        group_j += 1
                        new_route1, new_route2 = twoOptStar(route1,route2,group_j)
                        # We need to check if the new_routes are feasible
                        if isFeasible(new_route1) and isFeasible(new_route2):
                            new_distance1, new_distance2 = calculateDistanceAfterTwoOptStar(new_route1, new_route2, group_i, group_j)
                            if new_distance1 + new_distance2 < min_distances[route_i] + min_distances[route_j]:
                                routes_final,min_distances,changed = changeRoutes(routes_final,min_distances,new_route1,new_route2,route_i,route_j,group_i,group_j)
                                route1 = new_route1
                                route2 = new_route2
                                return routes_final,min_distances,changed
                    group_j = -1
                group_i = -1
        group_i, group_j = -1,-1
        route_j = -1
    return routes_final,min_distances,changed

def calculateDistancesForRoute(route, distances):
    fullRoute = [0] + route + [0]
    p = FloydWarshall(distances.copy(), fullRoute.copy())
    totalDistanceOfVehicle = p[0][0]
    return totalDistanceOfVehicle

def calculateDistancesForRoutes(routes, distances):
    distancesPerVehicle = np.zeros(len(routes))
    for vehicle in range(len(routes)):
        fullRoute = [0] + routes[vehicle] + [0]
        p = FloydWarshall(distances.copy(), fullRoute.copy())
        totalDistanceOfVehicle = p[0][0]
        distancesPerVehicle[vehicle] = totalDistanceOfVehicle
    return distancesPerVehicle

def calculateDistanceAfter2Opt(new_route, group1, group2):
    new_distance = calculateDistancesForRoute(new_route, distances)
    return new_distance
  
def calculateDistanceAfterTwoOptStar(new_route1, new_route2, group_i, group_j):
    new_distance1, new_distance2 = calculateDistancesForRoute(new_route1, distances), calculateDistancesForRoute(new_route2, distances)
    return new_distance1, new_distance2

def calculateDistanceAfterCross(new_route1, new_route2, group1, group2):
    new_distance1, new_distance2 = calculateDistancesForRoute(new_route1, distances), calculateDistancesForRoute(new_route2, distances)
    return new_distance1, new_distance2

def calculateDistanceAfterExchange(new_route1, new_route2):
    new_distance1, new_distance2 = calculateDistancesForRoute(new_route1, distances), calculateDistancesForRoute(new_route2, distances)
    return new_distance1, new_distance2

def calculateDistanceAfterRelocate(new_route1, new_route2):
    new_distance1, new_distance2 = calculateDistancesForRoute(new_route1, distances), calculateDistancesForRoute(new_route2, distances)
    return new_distance1, new_distance2

def calculateDistanceAfterSwap11(new_route1, new_route2):
    new_distance1, new_distance2 = calculateDistancesForRoute(new_route1, distances), calculateDistancesForRoute(new_route2, distances)
    return new_distance1, new_distance2

### Intra-route neighborhood functions
def twoOptIntra(route, group1, group2):
    '''Will remove and replace 2 edges in the route
    example: with route = [1,2,3,4,5,6,7,8], twoOptIntra(route, 2, 5) returns [1 2 3 6 5 4 7 8]'''
    if (group2 < group1):
        temp = group1
        group1 = group2
        group2 = temp
    temporary = route[group1 + 1 : group2]
    if temporary != []:
        new_route = route[:group1 + 1] + [route[group2]] + temporary[::-1] + route[group2 + 1 : ]
    else:
        return route
    return new_route

### Inter-route neighborhood functions
def twoOptStar(route1, route2, group):
    '''Will remove and replace 2 consecutive edges in two different routes
    example : with route1 = [1,2,3,4,5,6,7,8] and route2 = [9,10,11,12]
    twoOptInter(route1, route2, 1)
    will give (array([ 1,  2, 11, 12]), array([ 9, 10,  3,  4,  5,  6,  7,  8]))'''
    new_route1 = route1[:group + 1] + route2[group + 1 :]
    new_route2 = route2[:group + 1] + route1[group + 1 :]
    return new_route1, new_route2

def twoOptStarForCross(route1, route2, group):
    '''Will remove and replace 2 consecutive edges in two different routes
    example : with route1 = [1,2,3,4,5,6,7,8] and route2 = [9,10,11,12]
    twoOptInter(route1, route2, 1)
    will give (array([ 1,  2, 11, 12]), array([ 9, 10,  3,  4,  5,  6,  7,  8]))'''
    group +=1
    new_route1 = route1[:group + 1] + route2[group + 1 :]
    new_route2 = route2[:group + 1] + route1[group + 1 :]
    return new_route1, new_route2

def cross(route1, route2, group1, group2):
    '''Will exchange 2 sequences of visits
    example : with route1 = [1,2,3,4,5,6,7,8] and route2 = [9,10,11,12]
    cross(route1, route2, 1, 3)
    will return ([1, 2, 11, 12, 5, 6, 7, 8], [9, 10, 3, 4])'''
    if (group2 < group1):
        temp = group1
        group1 = group2
        group2 = temp
    route1_temporary, route2_temporary = twoOptStar(route1, route2, group1)
    route1_final, route2_final = twoOptStar(route1_temporary,route2_temporary, group2)
    return route1_final, route2_final

def exchange(route1, route2, group1, group2):
    '''Will exchange 2 groups of 2 routes
    example : with route1 = [1,2,3,4,5,6,7,8] and route2 = [9,10,11,12]
    exchange(route1, route2, 2, 3)
    will return ([1,2,12,4,5,6,7,8], [9,10,11,3])'''  
    tempGroupRoute1 = route1[group1]
    tempGroupRoute2 = route2[group2]
    route1[group1] = tempGroupRoute2
    route2[group2] = tempGroupRoute1
    return route1, route2

def relocate(route1, route2, i, j):
    '''Moves group from route1 (position i) to route2 (position j)'''
    route1Esq, group, route1Dir = route1[:i], route1[i], route1[i+1:]
    route2Esq, route2Dir = route2[:j], route2[j:]
    route2 = route2Esq + [group] + route2Dir
    route1 = route1Esq + route1Dir
    return route1, route2

def swap11(route1,route2, i,j):
    '''Exchanges element i from route1 with element j from route2'''
    route1Temp, route2Temp = relocate(route1,route2,i,j)
    route2, route1 = relocate(route2Temp,route1Temp,j+1,i)
    return route1,route2
    
def swap21(route1,route2,i, j):
    '''Exchanges elements i and i + 1 from route1 with element j from route2'''
    route1Temp, route2Temp = relocate(route1,route2,i,j)
    route1Temp, route2Temp = relocate(route1Temp,route2Temp,i,j+1)
    route2, route1 = relocate(route2Temp,route1Temp,j+2,i)
    return route1,route2
    
def insert(route1, group_i, j):
    '''Inserts group i in route1 position j'''
    route1Esq, route1Dir = route1[:j], route1[j:]
    route1 = route1Esq + group_i + route1Dir
    return route1

def changeRoutes(routes_final,min_distances,new_route1,new_route2,route_i,route_j,group_i,group_j):
    '''Effectively changes the routes'''
    new_distance1, new_distance2 = calculateDistanceAfterCross(new_route1, new_route2, group_i, group_j)
    routes_final[route_i] = new_route1
    routes_final[route_j] = new_route2
    min_distances[route_i] = new_distance1
    min_distances[route_j] = new_distance2
    changed = True
    return routes_final,min_distances,changed
    
def crossPerturbate(routes_final, min_distances, demandsRoutes):
    changed = False
    route_i,route_j = -1,-1  
    group_i, group_j = -1,1 
    for route1 in routes_final:
        route_i += 1
        for route2 in routes_final:
            route_j += 1
            if route1 != route2:
                for group1 in route1:
                    group_i += 1
                    for group2 in route2:
                        group_j += 1
                        if group_i != group_j and group_j < 3: # no more than 3 groups in the cross
                            new_route1, new_route2 = cross(route1,route2,group_i,group_j)
                            # We need to check if the new_routes are feasible
                            if isFeasible(new_route1) and isFeasible(new_route2):
                                routes_final,min_distances,changed = changeRoutes(routes_final,min_distances,new_route1,new_route2,route_i,route_j,group_i,group_j)
                                return routes_final,min_distances,changed
                    group_j = 1
                group_i = -1
        group_i, group_j = -1,1
        route_j = -1
    return routes_final,min_distances,changed

def perturbate(routes_final,distancesPerVehicle, demandsPerVehicle, timePassed, timeMax):
    '''Tries perturbating the solution until finding a feasible one'''
    feasibleSolution = False
    timeInicial = timePassed
    time = timePassed
    diferenca = 0
    while feasibleSolution == False and time < timeMax - diferenca:
        inicio = timeit.default_timer()
        routes_final,distancesPerVehicle,feasibleSolution = crossPerturbate(routes_final,distancesPerVehicle, demandsPerVehicle)
        fim = timeit.default_timer()
        diferenca = fim - inicio
        time += diferenca
    return routes_final,distancesPerVehicle, time

def isFeasible(route):
    '''Checks if the route is feasible'''
    demandOfRoute = 0
    for group in route:
        demandOfRoute += globalDemands[group - 1] # globalDemands tem indexes de 0 até nGroups - 1 
        if demandOfRoute > capacity :
            return False
    return True

def calculateDemands(routes):
    '''Calculates the demands of the routes, returns a vector'''
    demandsRoutes = []
    routes = routes
    for route in routes:
        demandRoute = 0
        for group in route:
            demandRoute += globalDemands[group - 1] # globalDemands tem indexes de 0 até nGroups - 1 
        demandsRoutes.append(demandRoute)
    return demandsRoutes

########### Floyd-Warshall for shortest path
def novasDistancias(distances, route, nNodes, groups):
    '''Algorithm used in FloydWarshall algo. used above'''
    nos = [0]
    novasDistancias = np.full((nNodes,nNodes), 100000000000) # copiamos matriz de distancias
    for i in range(len(route) - 1): # e mudamos valores para infinito quando nao sao do grupo
        group = route[i]
        next_group = route[i+1]
        nos_grupo = groups[next_group]
        nos += nos_grupo
        grupo1 = groups[group]
        grupo2 = groups[next_group]
        for node in grupo1:
            for i in range(nNodes):
                if i in grupo2:
                    novasDistancias[node][i] = distances[node][i]
    return novasDistancias
        
def FloydWarshall(distances, route):
    '''Implements FW method to calculate the shortest path considering the routes and distances as parameters'''
    newDistances = novasDistancias(distances, route, nNodes, groups)
    parent = []
    v = len(newDistances)
    # path reconstruction matrix
    p = np.zeros(newDistances.shape)
    for i in range(0,v):
        for j in range(0,v):
            p[i,j] = newDistances[i,j]

    # initialize to infinity
    for i in range (0, v):
        parent.append([])
        for j in range (0, v):
            parent[i].append(0)

    # initialize the path matrix
    for i in range (0,v):
        for j in range (0,v):
            if newDistances[i][j] == float("inf"):
                parent[i][j] = 0
            else:
                parent[i][j] = i

    changed = 0
    for k in range(0,v):
        for i in range(0,v):
            for j in range(0,v):
                if p[i,j] > p[i,k] + p[k,j]:
                    p[i,j] = p[i,k] + p[k,j]
                    parent[i][j] = parent[k][j]
                    changed +=1
                
    return p
                
# Recursive function to obtain the path as a string
def obtainPath(i, j, parent):
    '''Returns a path between i and j'''
    if newDistances[i][j] == float("inf"):
        return " no path to "
    if parent[i][j] == i:
        return " "
    else :
        return obtainPath(i, parent[i][j]) + str(parent[i][j]) + obtainPath(parent[i][j], j)

        
####### GVRP
def gvrp(filename):
    '''Implements the gvrp problem and tries to solve it'''
    global instance, nNodes, nVehicles, nGroups, capacity, nodes, groups, globalDemands, distances
    instance, nNodes, nVehicles, nGroups, capacity, nodes, groups, globalDemands, distances = processFile(filename)
    
    # Time setting
    inicio = timeit.default_timer()
    fim = timeit.default_timer() - 15
    time = fim - inicio
    timeMax = 120

    ## Data structures
    set_seed(12345678)
    iterations = 0
    min_total_distance = 1000000
    routes = []
    demandsRoutes =  []
    
    ## Create initial solution
    routes_initial, demand_distribution = createInitialSolutionBySweep(nVehicles, nGroups, capacity, globalDemands,nodes)
    distancesPerVehicle = calculateDistancesForRoutes(routes_initial, distances)
    total_distance = sum(distancesPerVehicle)
    initialDistance = total_distance
    
    # Do Local Search on the initial solution
    routes, distancesPerRoute, time, durationOfLastLocalSearch = localSearch(routes_initial, distancesPerVehicle, time,timeMax)
    min_total_distance = sum(distancesPerRoute)
    demandsRoutes = calculateDemands(routes)
    distancesPerVehicle = calculateDistancesForRoutes(routes, distances)
    
    ###### ILS
    
    while (time < timeMax - durationOfLastLocalSearch):
        iterations += 1
        fim = timeit.default_timer()
        time = fim - inicio
        perturbated_routes, distancesPerVehicleIteration, time = perturbate(routes, distancesPerVehicle, demandsRoutes,time,timeMax)
        if time > timeMax:
            break
        new_routes, new_distances, time, durationOfLastLocalSearch = localSearch(perturbated_routes, distancesPerVehicleIteration, time,timeMax)
        if time > timeMax:
            break
            
        new_distance = sum(new_distances)
        
        # Checking if the solution of this iteration is better than the global solution
        if new_distance < min_total_distance:
            min_total_distance = new_distance
            routes = new_routes
            distancesPerVehicle = new_distances
            demandsRoutes = calculateDemands(routes)

        fim = timeit.default_timer()
        time = fim - inicio

    improvement = min_total_distance - initialDistance
    exceedCapacities = False
    for demand in demandsRoutes:
        if demand > capacity:
            exceedCapacities = True
    
    # Keeping results in Excel file
    #writeToExcel(instance, nNodes, nVehicles, nGroups, capacity, min_total_distance, time, exceedCapacities, iterations, improvement)

    # Print graph
    #printgraph()

#### WRITE TO EXCEL (results)
def writeToExcel(instance, nNodes, nVehicles, nGroups, capacity, min_total_distance, time, exceedCapacities, iterations, improvement):
    '''writes results to an Excel file called GVRP.csv'''
    #rowDataCSV = "Instance|Nodes|Groups|Capacity|Vehicles|Solution|Time|ExceededCapacities|Iterations|Improvement"
    rowDataCSV = instance + "|" + str(nNodes) + "|" + str(nGroups) + "|" + str(nVehicles) + "|" + str(capacity) + "|" + str(min_total_distance) + "|" + str(time) + "|" + str(exceedCapacities) + "|" + str(iterations) + "|" + str(improvement)
    with open('GVRP.csv','a') as fd:
        fd.write(rowDataCSV)
        fd.write("\n")

if __name__ == "__main__":
    gvrp('A-n38-k5-C13-V2.gvrp')
