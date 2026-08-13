# Project Summary
This project is a tool for Sythetic Knowledge Graph generation. The code for the tool is in the 'src' folder.

This tool creates a Knowledge Graph (KG) from topological metrics (number of nodes, relations, frequencies of the elements in the domain and range of each relation, etc.) and a set of rules represented in the form of Horn Rules.

The graphs are always stored in a graphical database, avoiding the usage of memmory intense libraries like RDFlib. We work with RDF format graphs and communicate with the graphical database with SPARQL queries through the SPARQLWrapper library.

# Building blocks
 - **config.py**: The script 'config.py' implements a general configuration dataclass that stores all relevant parameters for the execution of an experiment. The parameters are read from '.json' files from the folder "configurations".
 - **rules.py**: The script 'rules.py' implements the dataclasses that represent a rule and the functionalities to parse a set of rules from a .csv file.
 - **utils.py**: The script 'utils.py' implements some common functionalities.
 - **queries.py**: The script 'queries.py' implements the comunication functions with the graphical database and the construction of SPARQL queries.
 - **graph_metrics.py**: The script 'graph_metrics.py' implements the 'GraphMetrics' dataclass. This class stores the topological descriptors of a graph. It counts the frequency of each relation (often refered as 'predicate') and the frequency of the elements in its domain and range.

# Main functionality
The main functionality of this tool is to create a synthetic copy of a set of rules and topological descriptors. 

The current implementation of the tool obtains the topological descriptors from the graph itself through the 'graph_metrics.py' script. Ideally, the synthetic generator does not need to access the original graph.