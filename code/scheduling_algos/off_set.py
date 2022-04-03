import networkx as nx
from collections import defaultdict, Counter
from typing import List, Tuple
import traintools

TrainStop = Tuple[int, int]
TrainType = Tuple[int, int]


def solve(filename: str, train_type: TrainType) -> None:
    """
    """

    df = traintools.read_schedule(filename)

    lowerbound = traintools.minimum_number_of_trains_at_time_t(df, train_type).max()[0]

    G = traintools.graph_from_schedule(df, train_type, off_set=0.01)
    all_trains, number_of_trains, starting_trains, ending_trains, G = schedule_with_offset(G.copy())

    print({station: len(trains) for station, trains in starting_trains.items()}, {station: len(trains) for station, trains in ending_trains.items()})
    print('Endpoints match!'*({station: len(trains) for station, trains in starting_trains.items()} == {station: len(trains) for station, trains in ending_trains.items()}))
    print('using:', number_of_trains)

    df_ = nx.to_pandas_edgelist(G.copy())
    df_['over_scheduled'] = df_['trains_scheduled'] - df_['trains_needed']
    G_ = nx.from_pandas_edgelist(
            df=df_, 
            source='source', 
            target='target', 
            edge_attr=['over_scheduled'], 
            create_using=nx.DiGraph
            )
    all_trains2, number_of_trains2, starting_trains2, ending_trains2, G2 = traintools.find_over_scheduled(G_)

    print(number_of_trains2, 'train(s) can be subtracted')
    print(f'Total number of trains scheduled: {number_of_trains - number_of_trains2}')
    print(f'Lowerbound: {lowerbound}')
    if lowerbound == number_of_trains - number_of_trains2:
        print('Optimal solution found!')



def schedule_with_offset(G: nx.DiGraph) -> Tuple[int, defaultdict, defaultdict]:
    """Attempt at a solution where the end points don't need to be fixed.
    """

    def do_bookkeeping(path: List[TrainStop]) -> None:
        """Update the variables that keep track of the total number of trains 
        and where they start and end.
        """
        nonlocal all_trains, starting_trains, ending_trains, number_of_trains

        all_trains.append(path)
        starting_trains[path[0][0]].append(path)
        ending_trains[path[-1][0]].append(path)
        number_of_trains += 1

    def fix_start(node: TrainStop) -> None:
        """Add temp node to fix the starting point of the longest path.
        """
        G.add_edge('temp_start', node, min_trains=100)

    def fix_end(node: TrainStop) -> None:
        """Add temp node to fix ending point of the longest path.
        """
        G.add_edge(node, 'temp_end', min_trains=100)

    def remove_temp_start() -> None:
        """Remove temp node to get the 'true' graph back.
        """
        G.remove_node('temp_start')
        
    def remove_temp_end() -> None:
        """Remove temp node to get the 'true' graph back.
        """
        G.remove_node('temp_end')  


    # find the end points of the graph
    ends, starts = traintools.find_ending_trainstops(G), traintools.find_starting_trainstops(G)

    # set up variables to keep track of the number of trains and at which stations the trains start and end
    starting_trains, ending_trains, number_of_trains = defaultdict(list), defaultdict(list), 0
    all_trains = []

    starting_stops = {stop[0]: stop for stop in traintools.find_starting_trainstops(G)}
    ending_stops = {stop[0]: stop for stop in traintools.find_ending_trainstops(G)}


   # find the longest path in the graph
    path = nx.dag_longest_path(G, weight='min_trains')    

    # repeat until the longest path consists of only one node
    next_random = False
    while G.size(weight='min_trains') >= 1:

        
        do_bookkeeping(path)
        
        # go through the longest path and pick up the maximal number of passengers on the way
        for index, current_stop in enumerate(path[:-1]):
            next_stop = path[index + 1]
            G[current_stop][next_stop]['trains_scheduled'] += 1
            if G[current_stop][next_stop]['min_trains'] > 0:
                G[current_stop][next_stop]['min_trains'] -= 1
    

        # recompute the longest path 
        if next_random:
            path = nx.dag_longest_path(G, weight='min_trains')
            next_random = False
        else:
            prev_end = starting_stops[path[-1][0]]
            fix_start(prev_end)
            path = nx.dag_longest_path(G, weight='min_trains')[1:]
            fix_end(ending_stops[path[-1][0]])
            path = nx.dag_longest_path(G, weight='min_trains')[1:-1]
            remove_temp_end()
            remove_temp_start()  

        if nx.path_weight(G, path, weight='min_trains') == 0:
            next_random = True
        
    while stations_diff := Counter({station: len(trains) for station, trains in ending_trains.items()}) - Counter({station: len(trains) for station, trains in starting_trains.items()}):
        stations_diff2 = Counter({station: len(trains) for station, trains in starting_trains.items()}) - Counter({station: len(trains) for station, trains in ending_trains.items()})
        fix_start(starting_stops[stations_diff.most_common(1)[0][0]])
        fix_end(ending_stops[stations_diff2.most_common(1)[0][0]])
        path = nx.dag_longest_path(G, weight='min_trains')[1:-1]
        remove_temp_start()  
        remove_temp_end()

        do_bookkeeping(path)
        
    return(
        all_trains, 
        number_of_trains, 
        starting_trains, 
        ending_trains,
        G
        )

