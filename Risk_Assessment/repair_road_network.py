import os
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, mapping
import time

########################## Definitions ########################
#set wd
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Measles DRC/Data')

# Load road data
roads = gpd.read_file('Datasets_Directory/roads.shp')
roads = roads.to_crs(epsg=32734)

# Create network graph from road network
def simplify_geometry(geom, tolerance):
    if geom.type == 'LineString':
        simplified_geom = geom.simplify(tolerance)
        return simplified_geom
    else:
        return geom

def connect_unconnected_to_main(graph, main_component, unconnected_components):
    count = 0
    for unconnected_component in unconnected_components:
        count = count + 1
        print(f"Connecting unconnected island {count}")
        start_time = time.time()
        max_distance = 1000000
        random_node_unconnected = np.asarray(list(unconnected_component.nodes)[0])
        all_connected_nodes = np.asarray(list(main_component.nodes))
        R = 1000000000
        closest_main_nodes = all_connected_nodes[(cdist(all_connected_nodes[:, :2], random_node_unconnected[None]) < R).ravel()]
        print(len(closest_main_nodes))
        while len(closest_main_nodes) == 0:
            R+=1000000000
            closest_main_nodes = all_connected_nodes[(cdist(all_connected_nodes[:, :2], random_node_unconnected[None]) < R).ravel()]
        print(len(closest_main_nodes))

        tuples = list(map(tuple, closest_main_nodes))
        for node_unconnected in unconnected_component.nodes:
            for node_main in tuples:
                coords_node_main = graph.nodes[node_main].get('pos', None)
                coords_node_unconnected = graph.nodes[node_unconnected].get('pos', None)
                distance = euclidean(coords_node_unconnected, coords_node_main)
                if distance <= max_distance:
                    max_distance = distance
                    best_node_unconnected = node_unconnected
                    best_node_main = node_main

        graph.add_edge(best_node_unconnected, best_node_main)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {count} took {elapsed_time} seconds")

def create_road_network(roads_gdf, tolerance):
    G = nx.Graph()
    for idx, row in roads_gdf.iterrows():
        geom = row['geometry']
        line = geom.coords if geom.type == 'LineString' else geom.geoms[0].coords
        for i in range(len(line) - 1):
            node1 = (int(line[i][0] / tolerance), int(line[i][1] / tolerance))
            node2 = (int(line[i + 1][0] / tolerance), int(line[i + 1][1] / tolerance))

            # Add nodes with 'pos' attribute
            G.add_node(node1, pos=line[i])
            G.add_node(node2, pos=line[i + 1])

            # Add edge
            G.add_edge(
                node1,
                node2,
                fid=row['osm_id']  # Replace 'fid' with the actual identifier column name
            )
    return G

def identify_unconnected_components(graph):
    connected_components = list(nx.connected_components(graph))
    main_network = max(connected_components, key=len)
    unconnected_components = [c for c in connected_components if c != main_network]
    return graph.subgraph(main_network), [graph.subgraph(c) for c in unconnected_components]

# Set the tolerance value
tolerance_value = 0.000001
roads['geometry'] = roads['geometry'].apply(lambda x: simplify_geometry(x, tolerance_value))

# Create the road network graph
road_network = create_road_network(roads, tolerance_value)

# Identify connected and unconnected components
main_network, unconnected_components = identify_unconnected_components(road_network)

# Print information about the network components
print(f"Number of nodes in the main network: {len(main_network.nodes)}")
print(f"Number of edges in the main network: {len(main_network.edges)}")

print(f"Number of unconnected islands: {len(unconnected_components)}")
for i, island in enumerate(unconnected_components):
    print(f"Island {i + 1}: Number of nodes - {len(island.nodes)}, Number of edges - {len(island.edges)}")

# Connect unconnected islands to the main network within 10 km
connect_unconnected_to_main(road_network, main_network, unconnected_components)

# Print information about the updated network
print(f"Updated number of nodes in the main network: {len(road_network.nodes())}")
print(f"Updated number of edges in the main network: {len(road_network.edges())}")

# Identify connected and unconnected components in the updated network
main_network_updated, unconnected_components_updated = identify_unconnected_components(road_network)

# Print the number of unconnected components in the updated network
print(f"Number of unconnected components in the updated network: {len(unconnected_components_updated)}")

# Extract node positions and edges from the network graph
node_positions = {node: data['pos'] for node, data in road_network.nodes(data=True)}
edges = road_network.edges()

# Create GeoDataFrames for nodes and edges
nodes_gdf = gpd.GeoDataFrame(geometry=[Point(pos) for pos in node_positions.values()])
edges_gdf = gpd.GeoDataFrame(geometry=[LineString([node_positions[edge[0]], node_positions[edge[1]]]) for edge in edges])

# Save nodes and edges as shapefiles
nodes_gdf.to_file("nodes.shp")
edges_gdf.to_file("edges.shp")