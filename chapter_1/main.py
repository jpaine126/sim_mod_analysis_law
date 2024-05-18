"""Basic single server queue simulation"""

import bisect
import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class SimEnd(RuntimeError):
    """Simulation stop signal"""


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

        self.num_delays = 0
        self.num_arrivals = 0
        self.cumulative_delay = 0

        # end conditions
        self.stop_rule = "end_time"  # num_arrivals, num_delays, cum_delay, end_time
        self.max_num_arrivals = 1027
        self.max_num_delays = 1000
        self.max_cum_delay = 1000
        self.stop_time = 480

        # stop time is deterministic, so handle now
        if self.stop_rule == "end_time":
            self.insert_event(Event("stop", self.stop_time))

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
        self.num_arrivals += 1

        log = {"time": self.simtime, "delayed": False}

        # schedule next arrival
        self.insert_event(
            Event("arrival", self.rng_stream.exponential(self.mean_arrival_time))
        )

        if self.server.is_busy:
            self.service_queue.append(self.simtime)

            if len(self.service_queue) > self.max_service_queue_size:
                raise RuntimeError("Service queue size exceeded maximum")

            self.num_delays += 1
            log["delayed"] = True
        else:
            self.server.status = "busy"

            # schedule next departure
            self.insert_event(
                Event("departure", self.rng_stream.exponential(self.mean_service_time))
            )

        log["server_status"] = self.server.status
        self.arrival_log.append(log)

    @event_dispatcher
    def departure(self):
        logger.info("departure")

        log = {"time": self.simtime, "wait_time": np.nan}

        if len(self.service_queue) == 0:
            self.server.status = "idle"
        else:
            this_customer_arrival_time = self.service_queue.popleft()

            delay = self.simtime - this_customer_arrival_time
            self.cumulative_delay += delay
            log["wait_time"] = delay

            # schedule next departure
            self.insert_event(
                Event("departure", self.rng_stream.exponential(self.mean_service_time))
            )

        log["server_status"] = self.server.status

        self.departure_log.append(log)

    @event_dispatcher
    def stop(self):
        raise SimEnd

    def num_arrivals_stop_rule(self):
        """Stop when arrivals exceeds a given amount."""
        return self.num_arrivals >= self.max_num_arrivals

    def num_delays_stop_rule(self):
        """Stop when number of delayed customers exceeds a given amount."""
        return self.num_delays >= self.max_num_delays

    def cum_delay_stop_rule(self):
        """Stop when cumulative delay time exceeds a given value."""
        return self.cumulative_delay >= self.max_cum_delay

    def check_stop_conditions(self):
        """Check stop conditions and register stop event."""
        if (
            (self.stop_rule == "num_arrivals" and self.num_arrivals_stop_rule())
            or (self.stop_rule == "num_delays" and self.num_delays_stop_rule())
            or (self.stop_rule == "cum_delay" and self.cum_delay_stop_rule())
        ):
            self.insert_event(Event("stop", 0))

    def run(self):
        n = 0
        # first arrival
        self.insert_event(
            Event("arrival", self.rng_stream.exponential(self.mean_arrival_time))
        )

        while True:
            self.check_stop_conditions()

            next_event = self.event_queue.popleft()

            # update simtime for the coming time step
            self.simtime = next_event.time

            try:
                # run event callback function
                _ = _event_register[next_event.event](self)

                n += 1
                logger.info(f"STEP {n} simtime={self.simtime}")
            except SimEnd:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sim = Sim()
    sim.run()

    # analysis
    import pandas as pd

    arrivals = pd.DataFrame.from_records(sim.arrival_log)
    arrivals["event_type"] = "arrival"
    departures = pd.DataFrame.from_records(sim.departure_log)
    departures["event_type"] = "departure"

    total_arrivals = len(arrivals)
    total_departures = len(departures)
    total_delays = arrivals["delayed"].sum()
    cumulative_delay = departures["wait_time"].sum()

    num_arrivals_delayed = arrivals["delayed"].sum()

    average_wait_time = departures["wait_time"].mean()

    queue_size_integrator = (
        pd.concat(
            (
                arrivals[["time", "event_type", "server_status"]],
                departures[["time", "event_type", "server_status"]],
            ),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    # time average queue size
    queue_size_integrator["queue_size"] = queue_size_integrator["event_type"].apply(
        lambda x: 1 if x == "arrival" else -1,
    )
    queue_size_integrator["queue_size"] = (
        queue_size_integrator["queue_size"].cumsum() - 1
    )
    queue_size_integrator.loc[queue_size_integrator["queue_size"] < 0, "queue_size"] = 0

    queue_size_integrator["time_diff"] = (
        queue_size_integrator["time"]
        .diff()
        .fillna(queue_size_integrator["time"].iloc[0])
    )
    time_average_queue_length = (
        queue_size_integrator["time_diff"] * queue_size_integrator["queue_size"]
    ).sum() / queue_size_integrator["time"].max()

    # server time spent busy
    queue_size_integrator["server_status"] = queue_size_integrator[
        "server_status"
    ].apply(lambda x: 1 if x == "busy" else 0)
    time_average_server_busy = (
        queue_size_integrator["time_diff"]
        * queue_size_integrator["server_status"].shift(1, fill_value=0)
    ).sum() / queue_size_integrator["time"].max()

    # report
    print(f"{total_arrivals=}")
    print(f"{total_departures=}")
    print(f"{num_arrivals_delayed=}")
    print(f"{total_delays=}")
    print(f"{cumulative_delay=}")
    print(f"{time_average_queue_length=:.2f}")
    print(f"{time_average_server_busy=:.2f}")
    print(f"sim_end_time={queue_size_integrator['time'].max():.2f}")
