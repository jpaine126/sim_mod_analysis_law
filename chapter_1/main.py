"""Basic single server queue simulation"""

import bisect
import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Server:
    status: Literal["idle", "busy"] = "idle"

    @property
    def is_busy(self):
        return self.status == "busy"


@dataclass
class Event:
    event: str
    time: float


_event_register = dict()


class Sim:

    def __init__(self):
        # params
        self.simtime = 0.0
        self.event_queue = deque()
        self.max_service_queue_size = 100
        self.mean_arrival_time = 1.0
        self.mean_service_time = 0.5

        # sim objects
        self.service_queue = deque()
        self.server = Server()
        self.rng_stream = np.random.default_rng(10)

        # stats
        self.arrival_log = []
        self.departure_log = []

    def event_dispatcher(func):
        """Register sim event callback"""
        _event_register[func.__name__] = func
        return func

    def insert_event(self, event: Event):
        event.time = event.time + self.simtime
        bisect.insort(self.event_queue, event, key=lambda e: e.time)

    @event_dispatcher
    def arrival(self):
        logger.info("arrival")

        log = {"time": self.simtime, "delayed": False, "queue_size": self.service_queue}

        # schedule next arrival
        self.insert_event(
            Event("arrival", self.rng_stream.exponential(self.mean_arrival_time))
        )

        if self.server.is_busy:
            self.service_queue.append(self.simtime)

            if len(self.service_queue) > self.max_service_queue_size:
                raise RuntimeError("Service queue size exceeded maximum")

            log["delayed"] = True
        else:
            self.server.status = "busy"

            # schedule next departure
            self.insert_event(
                Event("departure", self.rng_stream.exponential(self.mean_service_time))
            )

        self.arrival_log.append(log)

    @event_dispatcher
    def departure(self):
        logger.info("departure")

        log = {"time": self.simtime, "wait_time": np.nan}

        if len(self.service_queue) == 0:
            self.server.status = "idle"
        else:
            this_customer_arrival_time = self.service_queue.popleft()

            log["wait_time"] = self.simtime - this_customer_arrival_time

            # schedule next departure
            self.insert_event(
                Event("departure", self.rng_stream.exponential(self.mean_service_time))
            )

        self.departure_log.append(log)

    def run(self):
        # first arrival
        self.insert_event(
            Event("arrival", self.rng_stream.exponential(self.mean_arrival_time))
        )

        while len(self.arrival_log) < 100:
            logger.info(f"STEP simtime={self.simtime}")
            next_event = self.event_queue.popleft()

            self.simtime = next_event.time

            result = _event_register[next_event.event](self)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = Sim()
    sim.run()

    import pandas as pd

    arrivals = pd.DataFrame.from_records(sim.arrival_log)
    departures = pd.DataFrame.from_records(sim.departure_log)

    total_arrivals = len(arrivals)
    total_departures = len(departures)

    num_arrivals_delayed = arrivals["delayed"].sum()

    print(total_arrivals)
    print(total_departures)
    print(num_arrivals_delayed)
    print(departures["wait_time"].mean())
