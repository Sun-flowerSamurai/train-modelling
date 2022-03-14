from typing import Tuple, List
from matplotlib import style
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import math
import operator
from collections import defaultdict
from trainconstants import *

from cycler import cycler
colors = ['#2E75B6', '#763870', '#C8191A', '#101073', '#6A7A94', '#66968C', '#385765']
line_cycler   = (cycler(color=colors) +
                 cycler(linestyle=['-', '--', '-.', ':', '-', '--', '-.']))

matplotlib.rcParams.update({'font.size': 20,
            'legend.fontsize': 20,
            'axes.titlesize': 25,
            'axes.labelsize': 25,
            'axes.titleweight': 'bold',
            'lines.linewidth': 3,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'axes.prop_cycle': line_cycler})


TrainStop = Tuple[int, int]
TrainType = Tuple[int, int]


def read_schedule(filename: str) -> pd.DataFrame:
    """Read a schedule from a file.
    """

    df = pd.read_csv(filename,  delimiter=r"\s+")
    df['departure_time'] = df['departure_time'].map(to_minutes_past_midnight)
    df['arrival_time'] = df['arrival_time'].map(to_minutes_past_midnight)

    return df


def to_minutes_past_midnight(time: int) -> int:
    """Converts the time as used in the data-files to minutes past midnight.
    """

    assert type(time) == int, "The time must be an integer!"

    hours = time // 100
    minutes = time % 100

    return 60 * hours + minutes


def to_regular_time(time: int) -> int:
    """Converts the time in minutes past midnight back to the regular time format.
    """

    assert type(time) == int, "The time must be an integer!"

    hours = time // 60
    minutes = time % 60

    return hours * 100 + minutes


def minimum_number_of_trains(row:pd.Series, train_type: Tuple[int, int])->  int:
    """Finds the minimum number of trains needed in row.
    """
    
    min_first_class = row['first_class']/train_type[0]
    min_second_class = row['second_class']/train_type[1]

    min_trains = math.ceil(max(min_first_class, min_second_class))

    return min_trains


def minimum_number_of_trains_at_time_t(schedule: pd.DataFrame, train_type: Tuple[int, int]) -> pd.DataFrame:
    """Creates a new DataFrame whoes index is number of minutes past midnight and 
    for each minute contains the minimum number of trains needed to serve the schedule.
    """
    
    df = pd.DataFrame(np.zeros((60*24+1, 1)))
    df.columns = ['number of trains']

    for index, row in schedule.iterrows():
        start = row['departure_time']
        stop = row['arrival_time']
        
        trains = minimum_number_of_trains(row, train_type=train_type)
        df.loc[start:stop, 'number of trains'] += trains

    return df


def create_trains_dict(schedulce: pd.DataFrame) -> dict:
    """Creates a dict with an entry for every train_number in schedule
    with the train_number as key and that trains schedule as value.
    """

    trains = defaultdict(pd.DataFrame)

    for index, row in schedulce.iterrows():
        trains[row[0]] = pd.concat([trains[row[0]], pd.DataFrame(row).T])
    
    return trains


class Train:
    def __init__(self, train_number: int, schedule: pd.DataFrame, ax:plt.Axes) -> None:
        """Create train object based on the train number and its schedule.
        """
        
        self.ax = ax
        self.train_number = train_number
        self._schedule = schedule
        df = pd.DataFrame()
        df['time'] = pd.concat([schedule['departure_time'], schedule['arrival_time']])
        df['place'] = pd.concat([schedule['start'], schedule['end']])
        self.schedule = df.sort_values(by='time').reset_index()


    def plot(self) -> None:
        """Plot the space time graph of the train line.
        """
        
        ax=self.ax
            
        self.schedule.plot(kind='line', x='time', y='place', ax=ax) 
         
        for index, row in self._schedule.iterrows():
            offset_t = 10
            offset_p = 0
            place = row['start'] - (row['start'] - row['end'])/4 + offset_p
            time = row['departure_time'] - (row['departure_time'] - row['arrival_time'])/4 + offset_t

            ax.text(time, place, f'{row[5]}\n{row[6]}')
            ax.text(
                time, 
                place - 0.1, 
                str((
                    minimum_number_of_trains(row,  train_type=TYPE_3_TRAIN), 
                    minimum_number_of_trains(row,  train_type=COMPARTMENT_BASED_ON_3)
                )), 
                color='r'
                )
            ax.text(
                time, 
                place - 0.2, 
                str((
                    minimum_number_of_trains(row, train_type=TYPE_4_TRAIN), 
                    minimum_number_of_trains(row,  train_type=COMPARTMENT_BASED_ON_4)
                )), 
                color='g'
                )


class VisualizeSchedule:
    def __init__(self, trains_to_visualize: dict) -> None:
        """Setup the schedule that needs to be visualized.
        """

        self.trains = trains_to_visualize

    def visualize(self) -> None:
        """Show the space-time plot of the train schedule.
        """

        fig, ax = plt.subplots(figsize=(30, 10))

        for station in range(1, 5):
            plt.plot([300, 1440], [station, station], color='k', linestyle="-")

        for key, value in self.trains.items():
            train = Train(key, value, ax)
            train.plot()
            ax.text(300, AMSTERDAM + 0.1, 'Amsterdam', fontsize=15)
            ax.text(300, ROTTERDAM + 0.1, 'Rotterdam', fontsize=15)
            ax.text(300, ROOSENDAAL + 0.1, 'Roosendaal', fontsize=15)
            ax.text(300, VLISSINGEN + 0.1, 'Vlissingen', fontsize=15)

        ax.set_xlim(250, 1450)
        ax.set_ylim(0.5, 4.5)
        ax.set_xlabel('Minutes past midnight')
        ax.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.legend().remove()


def find_starting_trainstops(G: nx.DiGraph) -> Tuple[TrainStop]:
    """Creates a tuple of the first trainstop per station.
    """

    return (stops[0] for stops in find_all_stops_per_station(G).values())


def find_ending_trainstops(G: nx.DiGraph) -> Tuple[TrainStop]:
    """Creates a tuple of the last trainstop per station.
    """

    return (stops[-1] for stops in find_all_stops_per_station(G).values())


def create_network_schedule(df: pd.DataFrame, train_type: Tuple[int, int]) -> pd.DataFrame:
    """Creates a DataFrame which can be turned into a networkx graph.

    The DataFrame consists of 4 columns, with for each train ride a separate row.
    The columns are
    first_class: minimum number of seats for first class passengers 
    second_class: minimum number of seats for second class passengers
    start: Tuple[station, departure time]
    end: Tuple[station, arrival time]
    """
    def find_bottleneck(row: pd.Series):
        """Find the bottleneck which will decide what the minimum number of trains are.
        """
        return math.ceil(max(row[0], row[1]))
    
    network_schedule = df[['first_class', 'second_class']].copy()
    network_schedule['start'] = list(zip(df.start, df.departure_time))
    network_schedule['end'] = list(zip(df.end, df.arrival_time))
    temp = network_schedule[['first_class', 'second_class']].copy()
    temp['trains_first'] = temp[['first_class']].apply(operator.truediv, args=(train_type[0],))
    temp['trains_second'] = temp[['second_class']].apply(operator.truediv, args=(train_type[1],))
    network_schedule['min_trains'] = temp[['trains_first', 'trains_second']].apply(find_bottleneck, axis=1)

    return network_schedule


def find_all_stops_per_station(G: nx.DiGraph) -> List[TrainStop]:
    """Creats dict with each station as a key and all stops to / from that station in a list as the value.
    The stops appear in chronological order in the list.
    """

    nodes_per_station = defaultdict(list)

    # create dict with each station as a key and all stops to / from that station in a list as the value 
    for node in G.nodes:
        nodes_per_station[node[0]].append(node)

    for station in nodes_per_station.keys():
        # sort the stations so they appear in chronological order in the list
        nodes_per_station[station].sort()

    return nodes_per_station


def connect_stationary_nodes(G: nx.DiGraph) -> None:
    """Connects all nodes to the next node in time at the same station.
    Corresponds to the train staying at the same station, i.e. not being used.
    """

    nodes_per_station = find_all_stops_per_station(G)
    
    for station in nodes_per_station.keys():
        # connect each stop to the next stop in time at the same station
        for index, stop in enumerate(nodes_per_station[station][:-1]):
            G.add_edge(stop, nodes_per_station[station][index + 1], min_trains=0)


def graph_from_schedule(df: pd.DataFrame, train_type: TrainType) -> nx.DiGraph:
    """Creates a digraph from the schedule.
    """
    
    # create the dataframe in the needed format for the creation of a graph
    schedule_network = create_network_schedule(df, train_type)

    # create the graph
    G = nx.from_pandas_edgelist(
        df=schedule_network, 
        source='start', 
        target='end', 
        edge_attr=['min_trains'], 
        create_using=nx.DiGraph
        )
    
    # add the stationary connections that correspond to staying at any given station
    connect_stationary_nodes(G)

    return G