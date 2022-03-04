from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from collections import defaultdict
from trainconstants import *


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


colors = ['b', 'r', 'k', 'g', 'm', 'c']
color_cycle = itertools.cycle(colors)


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
        """Plot the space time graph of the train line
        """
        col = next(color_cycle)

        ax=self.ax
            
        self.schedule.plot(kind='line', x='time', y='place', ax=ax, color=col) 
         
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
            plt.plot([300, 1450], [station, station], color='k')

        for key, value in self.trains.items():
            train = Train(key, value, ax)
            train.plot()
            ax.text(300, AMSTERDAM + 0.1, 'Amsterdam')
            ax.text(300, ROTTERDAM + 0.1, 'Rotterdam')
            ax.text(300, ROOSENDAAL + 0.1, 'Roosendaal')
            ax.text(300, VLISSINGEN + 0.1, 'Vlissingen')

        ax.set_xlim(300, 1450)
        ax.set_xlabel('Minutes past midnight')
        ax.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.legend().remove()