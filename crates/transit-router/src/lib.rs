//! A timetable-aware, round-based one-to-all router.
//!
//! The implementation keeps the route scan simple and auditable: each round
//! adds at most one vehicle boarding, while static transfer edges are relaxed
//! with a small Dijkstra closure. Independent origin/departure/intervention
//! queries can therefore be parallelized by the labels crate.

use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;
use transit_domain::{
    CanonicalPattern, CompiledNetwork, Intervention, LineIndex, LineMask, StationIndex, StopTime,
    INF_RIDES, INF_TIME,
};

#[derive(Clone, Debug)]
pub struct RoutingTrip {
    pub stop_times: Vec<StopTime>,
}

#[derive(Clone, Debug)]
pub struct RoutingPattern {
    pub line: LineIndex,
    pub stops: Vec<StationIndex>,
    pub trips: Vec<RoutingTrip>,
}

#[derive(Clone, Debug)]
pub struct RoutingTransfer {
    pub from: StationIndex,
    pub to: StationIndex,
    pub seconds: u32,
}

#[derive(Clone, Debug)]
pub struct RoutingData {
    pub station_count: usize,
    pub line_count: usize,
    pub patterns: Vec<RoutingPattern>,
    pub transfers: Vec<RoutingTransfer>,
}

impl RoutingData {
    pub fn from_network(network: &CompiledNetwork) -> Result<Self> {
        network.validate_indices()?;
        let mut patterns = Vec::with_capacity(network.patterns.len());
        for pattern in &network.patterns {
            validate_pattern(pattern, network.station_count(), network.line_count())?;
            patterns.push(RoutingPattern {
                line: pattern.signature.line,
                stops: pattern.signature.stops.clone(),
                trips: pattern
                    .trips
                    .iter()
                    .map(|trip| RoutingTrip {
                        stop_times: trip.stop_times.clone(),
                    })
                    .collect(),
            });
        }
        let transfers = network
            .transfers
            .iter()
            .map(|transfer| RoutingTransfer {
                from: transfer.from,
                to: transfer.to,
                seconds: transfer.minimum_transfer_seconds,
            })
            .collect();
        Ok(Self {
            station_count: network.stations.len(),
            line_count: network.lines.len(),
            patterns,
            transfers,
        })
    }
}

fn validate_pattern(
    pattern: &CanonicalPattern,
    station_count: usize,
    line_count: usize,
) -> Result<()> {
    if pattern.signature.stops.len() < 2 {
        bail!("pattern {} has fewer than two stations", pattern.index);
    }
    if pattern.signature.line.0 as usize >= line_count {
        bail!("pattern {} references an invalid line", pattern.index);
    }
    if pattern
        .signature
        .stops
        .iter()
        .any(|station| station.0 as usize >= station_count)
    {
        bail!("pattern {} references an invalid station", pattern.index);
    }
    if pattern
        .trips
        .iter()
        .any(|trip| trip.stop_times.len() != pattern.signature.stops.len())
    {
        bail!(
            "pattern {} has a trip with the wrong stop-time width",
            pattern.index
        );
    }
    for trip in &pattern.trips {
        for (position, stop_time) in trip.stop_times.iter().enumerate() {
            if stop_time.arrival > stop_time.departure {
                bail!(
                    "pattern {} trip {} has arrival after departure at stop {}",
                    pattern.index,
                    trip.trip_id,
                    position
                );
            }
            if let Some(previous) = position
                .checked_sub(1)
                .and_then(|index| trip.stop_times.get(index))
            {
                if stop_time.arrival < previous.departure {
                    bail!(
                        "pattern {} trip {} has non-monotonic stop times at stop {}",
                        pattern.index,
                        trip.trip_id,
                        position
                    );
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterConfig {
    pub maximum_transfers: u8,
    pub maximum_journey_seconds: u32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            maximum_transfers: 4,
            maximum_journey_seconds: 120 * 60,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OneToAllResult {
    pub arrival_seconds: Vec<u32>,
    pub transfers: Vec<u8>,
}

impl OneToAllResult {
    pub fn reachable(&self, station: StationIndex) -> bool {
        self.arrival_seconds
            .get(station.0 as usize)
            .copied()
            .unwrap_or(INF_TIME)
            != INF_TIME
    }
}

/// Per-thread scratch memory for repeated route queries. A workspace is not
/// shared between threads; the immutable `Router` remains safe to share.
#[derive(Clone, Debug)]
pub struct RoutingWorkspace {
    arrivals_by_rides: Vec<Vec<u32>>,
    station_count: usize,
}

impl RoutingWorkspace {
    pub fn new(station_count: usize, maximum_transfers: u8) -> Self {
        let rounds = maximum_transfers as usize + 2;
        Self {
            arrivals_by_rides: (0..rounds).map(|_| vec![INF_TIME; station_count]).collect(),
            station_count,
        }
    }

    fn prepare(&mut self, station_count: usize, rounds: usize) {
        self.station_count = station_count;
        if self.arrivals_by_rides.len() < rounds {
            self.arrivals_by_rides.extend(
                (self.arrivals_by_rides.len()..rounds).map(|_| vec![INF_TIME; station_count]),
            );
        }
        self.arrivals_by_rides.truncate(rounds);
        for values in &mut self.arrivals_by_rides {
            if values.len() != station_count {
                values.resize(station_count, INF_TIME);
            }
            values.fill(INF_TIME);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingBenchmarkReport {
    pub warmup_queries: usize,
    pub measured_queries: usize,
    pub median_milliseconds: f64,
    pub p95_milliseconds: f64,
    pub queries_per_second: f64,
    pub station_count: usize,
    pub line_count: usize,
    pub pattern_count: usize,
    pub maximum_transfers: u8,
}

#[derive(Clone, Debug)]
pub struct Router {
    pub data: RoutingData,
    pub config: RouterConfig,
    transfer_offsets: Vec<usize>,
    transfer_indices: Vec<usize>,
    patterns_by_station: Vec<Vec<usize>>,
}

impl Router {
    pub fn new(data: RoutingData, config: RouterConfig) -> Self {
        let mut transfer_counts = vec![0_usize; data.station_count];
        for transfer in &data.transfers {
            if (transfer.from.0 as usize) < data.station_count {
                transfer_counts[transfer.from.0 as usize] += 1;
            }
        }
        let mut transfer_offsets = vec![0_usize; data.station_count + 1];
        for station in 0..data.station_count {
            transfer_offsets[station + 1] = transfer_offsets[station] + transfer_counts[station];
        }
        let mut transfer_indices = vec![0_usize; transfer_offsets[data.station_count]];
        let mut cursors = transfer_offsets[..data.station_count].to_vec();
        for (index, transfer) in data.transfers.iter().enumerate() {
            let station = transfer.from.0 as usize;
            if station >= data.station_count {
                continue;
            }
            transfer_indices[cursors[station]] = index;
            cursors[station] += 1;
        }
        let mut patterns_by_station = vec![Vec::new(); data.station_count];
        for (pattern_index, pattern) in data.patterns.iter().enumerate() {
            for station in &pattern.stops {
                if let Some(patterns) = patterns_by_station.get_mut(station.0 as usize) {
                    patterns.push(pattern_index);
                }
            }
        }
        for patterns in &mut patterns_by_station {
            patterns.sort_unstable();
            patterns.dedup();
        }
        Self {
            data,
            config,
            transfer_offsets,
            transfer_indices,
            patterns_by_station,
        }
    }

    pub fn from_network(network: &CompiledNetwork, config: RouterConfig) -> Result<Self> {
        Ok(Self::new(RoutingData::from_network(network)?, config))
    }

    pub fn one_to_all(
        &self,
        origin: StationIndex,
        departure: u32,
        disabled_lines: &LineMask,
    ) -> OneToAllResult {
        self.one_to_all_internal(origin, departure, disabled_lines, None, None, None)
    }

    pub fn workspace(&self) -> RoutingWorkspace {
        RoutingWorkspace::new(self.data.station_count, self.config.maximum_transfers)
    }

    pub fn one_to_all_with_workspace(
        &self,
        origin: StationIndex,
        departure: u32,
        disabled_lines: &LineMask,
        workspace: &mut RoutingWorkspace,
    ) -> OneToAllResult {
        self.one_to_all_internal(
            origin,
            departure,
            disabled_lines,
            None,
            None,
            Some(workspace),
        )
    }

    fn one_to_all_internal(
        &self,
        origin: StationIndex,
        departure: u32,
        disabled_lines: &LineMask,
        closed_station: Option<StationIndex>,
        frequency_multipliers: Option<&[f32]>,
        mut workspace: Option<&mut RoutingWorkspace>,
    ) -> OneToAllResult {
        let station_count = self.data.station_count;
        if departure == INF_TIME {
            return OneToAllResult {
                arrival_seconds: vec![INF_TIME; station_count],
                transfers: vec![INF_RIDES; station_count],
            };
        }
        if origin.0 as usize >= station_count {
            return OneToAllResult {
                arrival_seconds: vec![INF_TIME; station_count],
                transfers: vec![INF_RIDES; station_count],
            };
        }
        if closed_station == Some(origin) {
            return OneToAllResult {
                arrival_seconds: vec![INF_TIME; station_count],
                transfers: vec![INF_RIDES; station_count],
            };
        }

        let maximum_rides = self.config.maximum_transfers as usize + 1;
        let rounds = maximum_rides.saturating_add(1);
        let mut owned_arrivals;
        let arrivals_by_rides = if let Some(workspace) = workspace.as_deref_mut() {
            workspace.prepare(station_count, rounds);
            &mut workspace.arrivals_by_rides
        } else {
            owned_arrivals = Some(vec![vec![INF_TIME; station_count]; rounds]);
            owned_arrivals
                .as_mut()
                .expect("owned routing workspace exists")
        };
        arrivals_by_rides[0][origin.0 as usize] = departure;
        let mut marked_stations = vec![origin.0 as usize];
        let mut initial_seeds = vec![origin.0 as usize];
        self.relax_transfers(
            departure,
            &mut arrivals_by_rides[0],
            &mut initial_seeds,
            closed_station,
            &mut marked_stations,
        );

        for ride_count in 1..=maximum_rides {
            if marked_stations.is_empty() {
                break;
            }
            let (previous_rounds, current_rounds) = arrivals_by_rides.split_at_mut(ride_count);
            let previous_arrivals = &previous_rounds[ride_count - 1];
            let current_arrivals = &mut current_rounds[0];
            let mut route_seeds = Vec::new();
            let mut marked_patterns = vec![false; self.data.patterns.len()];
            for station in &marked_stations {
                for pattern_index in self.patterns_by_station.get(*station).into_iter().flatten() {
                    marked_patterns[*pattern_index] = true;
                }
            }

            for (pattern_index, pattern) in self.data.patterns.iter().enumerate() {
                if !marked_patterns[pattern_index] {
                    continue;
                }
                if disabled_lines.contains(pattern.line) {
                    continue;
                }
                let multiplier = frequency_multipliers
                    .and_then(|values| values.get(pattern.line.0 as usize))
                    .copied()
                    .unwrap_or(1.0);
                // Scan each marked pattern once. This is the marked-route
                // RAPTOR step: a route is considered only when it serves a
                // station improved in the previous round.
                let mut boarded_trip: Option<(usize, u32)> = None;
                for position in 0..pattern.stops.len() {
                    let station = pattern.stops[position];
                    let station_slot = station.0 as usize;
                    if station_slot >= station_count {
                        continue;
                    }
                    if closed_station != Some(station)
                        && previous_arrivals[station_slot] != INF_TIME
                    {
                        if let Some((trip_index, trip)) = earliest_boardable_trip_with_multiplier(
                            pattern,
                            position,
                            previous_arrivals[station_slot],
                            multiplier,
                        ) {
                            let departure_time = trip.stop_times[position].departure;
                            if boarded_trip
                                .as_ref()
                                .map(|(_, current_departure)| departure_time < *current_departure)
                                .unwrap_or(true)
                            {
                                boarded_trip = Some((trip_index, departure_time));
                            }
                        }
                    }
                    let Some((trip_index, _)) = boarded_trip else {
                        continue;
                    };
                    let Some(stop_time) = pattern
                        .trips
                        .get(trip_index)
                        .and_then(|trip| trip.stop_times.get(position))
                    else {
                        continue;
                    };
                    if stop_time.dropoff_type == 1 || closed_station == Some(station) {
                        continue;
                    }
                    let arrival = stop_time.arrival;
                    if arrival == INF_TIME
                        || arrival < departure
                        || arrival.saturating_sub(departure) > self.config.maximum_journey_seconds
                    {
                        continue;
                    }
                    if better(arrival, 0, current_arrivals[station_slot], 0) {
                        current_arrivals[station_slot] = arrival;
                        route_seeds.push(station_slot);
                    }
                }
            }

            if route_seeds.is_empty() {
                marked_stations.clear();
                continue;
            }
            route_seeds.sort_unstable();
            route_seeds.dedup();
            marked_stations.clear();
            marked_stations.extend(route_seeds.iter().copied());
            self.relax_transfers(
                departure,
                current_arrivals,
                &mut route_seeds,
                closed_station,
                &mut marked_stations,
            );
        }

        let mut arrivals = vec![INF_TIME; station_count];
        let mut transfers = vec![INF_RIDES; station_count];
        let mut best_rides = vec![usize::MAX; station_count];
        for station in 0..station_count {
            for (ride_count, round) in arrivals_by_rides.iter().enumerate() {
                let arrival = round[station];
                if arrival == INF_TIME
                    || !better(arrival, ride_count, arrivals[station], best_rides[station])
                {
                    continue;
                }
                arrivals[station] = arrival;
                best_rides[station] = ride_count;
                transfers[station] = ride_count.saturating_sub(1).min(u8::MAX as usize) as u8;
            }
        }
        OneToAllResult {
            arrival_seconds: arrivals,
            transfers,
        }
    }

    pub fn one_to_all_intervention(
        &self,
        origin: StationIndex,
        departure: u32,
        intervention: &Intervention,
    ) -> OneToAllResult {
        match intervention {
            Intervention::DisableLine(line) => self.one_to_all(
                origin,
                departure,
                &LineMask::single(self.data.line_count, *line),
            ),
            Intervention::DisableLines(lines) => self.one_to_all(
                origin,
                departure,
                &LineMask::from_lines(self.data.line_count, lines.iter().copied()),
            ),
            Intervention::CloseStation(station) => {
                self.one_to_all_with_closed_station(origin, departure, *station)
            }
            Intervention::ScaleLineFrequency { line, multiplier } => {
                if *multiplier <= 0.0 {
                    self.one_to_all(
                        origin,
                        departure,
                        &LineMask::single(self.data.line_count, *line),
                    )
                } else {
                    let mut multipliers = vec![1.0_f32; self.data.line_count];
                    if let Some(value) = multipliers.get_mut(line.0 as usize) {
                        *value = *multiplier;
                    }
                    self.one_to_all_internal(
                        origin,
                        departure,
                        &LineMask::empty(self.data.line_count),
                        None,
                        Some(&multipliers),
                        None,
                    )
                }
            }
        }
    }

    fn one_to_all_with_closed_station(
        &self,
        origin: StationIndex,
        departure: u32,
        closed: StationIndex,
    ) -> OneToAllResult {
        self.one_to_all_internal(
            origin,
            departure,
            &LineMask::empty(self.data.line_count),
            Some(closed),
            None,
            None,
        )
    }

    /// Measure complete route queries, including timetable scans and transfer
    /// closure, after a warm-up period. The result is used by the local ETA
    /// estimator and intentionally reports a distribution rather than one
    /// optimistic sample.
    pub fn benchmark(
        &self,
        origins: &[StationIndex],
        departures: &[u32],
        disabled_line: Option<LineIndex>,
        warmup_queries: usize,
        measured_queries: usize,
    ) -> Result<RoutingBenchmarkReport> {
        self.benchmark_with_threads(
            origins,
            departures,
            disabled_line,
            warmup_queries,
            measured_queries,
            1,
        )
    }

    /// Benchmark route queries in a dedicated pool. This keeps the measured
    /// thread configuration explicit and prevents callers from accidentally
    /// sharing a global Rayon pool with unrelated work.
    pub fn benchmark_with_threads(
        &self,
        origins: &[StationIndex],
        departures: &[u32],
        disabled_line: Option<LineIndex>,
        warmup_queries: usize,
        measured_queries: usize,
        thread_count: usize,
    ) -> Result<RoutingBenchmarkReport> {
        if origins.is_empty() || departures.is_empty() {
            bail!("routing benchmark needs at least one origin and departure");
        }
        if measured_queries == 0 {
            bail!("routing benchmark needs measured queries");
        }
        if thread_count == 0 {
            bail!("routing benchmark thread count must be positive");
        }
        let mask = disabled_line
            .map(|line| LineMask::single(self.data.line_count, line))
            .unwrap_or_else(|| LineMask::empty(self.data.line_count));
        let query = |index: usize, workspace: &mut RoutingWorkspace| {
            let origin = origins[index % origins.len()];
            let departure = departures[(index / origins.len()) % departures.len()];
            self.one_to_all_with_workspace(origin, departure, &mask, workspace)
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|error| anyhow::anyhow!("building routing benchmark pool: {error}"))?;
        pool.install(|| {
            (0..warmup_queries)
                .into_par_iter()
                .map_init(
                    || self.workspace(),
                    |workspace, index| {
                        let _ = query(index, workspace);
                    },
                )
                .count();
        });
        let durations = pool.install(|| {
            (0..measured_queries)
                .into_par_iter()
                .map_init(
                    || (self.workspace(), Instant::now()),
                    |(workspace, started), index| {
                        *started = Instant::now();
                        let _ = query(warmup_queries + index, workspace);
                        started.elapsed().as_secs_f64() * 1_000.0
                    },
                )
                .collect::<Vec<_>>()
        });
        let mut durations = durations;
        durations.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            let index = ((durations.len() as f64 * fraction).ceil() as usize)
                .saturating_sub(1)
                .min(durations.len() - 1);
            durations[index]
        };
        let seconds = durations.iter().sum::<f64>() / 1_000.0;
        Ok(RoutingBenchmarkReport {
            warmup_queries,
            measured_queries,
            median_milliseconds: percentile(0.50),
            p95_milliseconds: percentile(0.95),
            queries_per_second: if seconds > 0.0 {
                measured_queries as f64 / seconds
            } else {
                0.0
            },
            station_count: self.data.station_count,
            line_count: self.data.line_count,
            pattern_count: self.data.patterns.len(),
            maximum_transfers: self.config.maximum_transfers,
        })
    }

    fn relax_transfers(
        &self,
        departure: u32,
        arrivals: &mut [u32],
        seeds: &mut Vec<usize>,
        closed_station: Option<StationIndex>,
        changed_stations: &mut Vec<usize>,
    ) {
        let mut queue = BinaryHeap::<Reverse<(u32, usize)>>::new();
        for station in seeds.drain(..) {
            if station < arrivals.len() && arrivals[station] != INF_TIME {
                queue.push(Reverse((arrivals[station], station)));
            }
        }
        while let Some(Reverse((time, station))) = queue.pop() {
            if arrivals[station] != time {
                continue;
            }
            let start = self.transfer_offsets.get(station).copied().unwrap_or(0);
            let end = self
                .transfer_offsets
                .get(station + 1)
                .copied()
                .unwrap_or(start);
            for transfer_index in self.transfer_indices.get(start..end).unwrap_or(&[]) {
                let transfer = &self.data.transfers[*transfer_index];
                if closed_station == Some(transfer.from) || closed_station == Some(transfer.to) {
                    continue;
                }
                let target = transfer.to.0 as usize;
                if target >= arrivals.len() {
                    continue;
                }
                let arrival = time.saturating_add(transfer.seconds);
                if arrival < departure
                    || arrival.saturating_sub(departure) > self.config.maximum_journey_seconds
                {
                    continue;
                }
                if arrival < arrivals[target] {
                    arrivals[target] = arrival;
                    queue.push(Reverse((arrival, target)));
                    changed_stations.push(target);
                }
            }
        }
    }
}

fn earliest_boardable_trip_with_multiplier(
    pattern: &RoutingPattern,
    position: usize,
    ready: u32,
    multiplier: f32,
) -> Option<(usize, &RoutingTrip)> {
    if pattern.trips.is_empty() || !multiplier.is_finite() || multiplier <= 0.0 {
        return None;
    }
    let multiplier = multiplier.min(1.0);
    let keep_count = if multiplier >= 1.0 {
        pattern.trips.len()
    } else {
        ((pattern.trips.len() as f32 * multiplier).round() as usize)
            .max(1)
            .min(pattern.trips.len())
    };
    pattern
        .trips
        .iter()
        .enumerate()
        .filter(|(trip_index, trip)| {
            trip_is_retained(*trip_index, pattern.trips.len(), keep_count)
                && position < trip.stop_times.len()
                && {
                    let time = &trip.stop_times[position];
                    time.pickup_type != 1 && time.departure >= ready
                }
        })
        .min_by_key(|(_, trip)| {
            let time = &trip.stop_times[position];
            (time.departure, time.arrival)
        })
}

fn trip_is_retained(index: usize, trip_count: usize, keep_count: usize) -> bool {
    if keep_count >= trip_count {
        return true;
    }
    ((index * keep_count) % trip_count) < keep_count
}

fn better(new_arrival: u32, new_rides: usize, old_arrival: u32, old_rides: usize) -> bool {
    new_arrival < old_arrival || (new_arrival == old_arrival && new_rides < old_rides)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(arrival: u32, departure: u32) -> StopTime {
        StopTime {
            arrival,
            departure,
            pickup_type: 0,
            dropoff_type: 0,
        }
    }

    fn trip(times: &[(u32, u32)]) -> RoutingTrip {
        RoutingTrip {
            stop_times: times
                .iter()
                .map(|(arrival, departure)| time(*arrival, *departure))
                .collect(),
        }
    }

    #[test]
    fn disabled_line_cannot_be_used_and_transfer_rounds_work() {
        let data = RoutingData {
            station_count: 4,
            line_count: 2,
            patterns: vec![
                RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1), StationIndex(2)],
                    trips: vec![trip(&[(100, 100), (200, 210), (300, 300)])],
                },
                RoutingPattern {
                    line: LineIndex(1),
                    stops: vec![StationIndex(2), StationIndex(3)],
                    trips: vec![trip(&[(360, 360), (500, 500)])],
                },
            ],
            transfers: Vec::new(),
        };
        let router = Router::new(
            data,
            RouterConfig {
                maximum_transfers: 2,
                ..RouterConfig::default()
            },
        );
        let intact = router.one_to_all(StationIndex(0), 90, &LineMask::empty(2));
        assert_eq!(intact.arrival_seconds[2], 300);
        assert_eq!(intact.arrival_seconds[3], 500);
        assert_eq!(intact.transfers[3], 1);

        let disabled = router.one_to_all(StationIndex(0), 90, &LineMask::single(2, LineIndex(0)));
        assert_eq!(disabled.arrival_seconds[1], INF_TIME);
        assert_eq!(disabled.arrival_seconds[2], INF_TIME);
        assert_eq!(disabled.arrival_seconds[3], INF_TIME);
    }

    #[test]
    fn parallel_service_provides_a_slower_fallback() {
        let data = RoutingData {
            station_count: 2,
            line_count: 2,
            patterns: vec![
                RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1)],
                    trips: vec![trip(&[(100, 100), (200, 200)])],
                },
                RoutingPattern {
                    line: LineIndex(1),
                    stops: vec![StationIndex(0), StationIndex(1)],
                    trips: vec![trip(&[(150, 150), (250, 250)])],
                },
            ],
            transfers: Vec::new(),
        };
        let router = Router::new(data, RouterConfig::default());
        let intact = router.one_to_all(StationIndex(0), 90, &LineMask::empty(2));
        let damaged = router.one_to_all(StationIndex(0), 90, &LineMask::single(2, LineIndex(0)));
        assert_eq!(intact.arrival_seconds[1], 200);
        assert_eq!(damaged.arrival_seconds[1], 250);
        assert!(damaged.arrival_seconds[1] >= intact.arrival_seconds[1]);
    }

    #[test]
    fn station_closure_blocks_boarding_and_alighting() {
        let data = RoutingData {
            station_count: 3,
            line_count: 1,
            patterns: vec![RoutingPattern {
                line: LineIndex(0),
                stops: vec![StationIndex(0), StationIndex(1), StationIndex(2)],
                trips: vec![trip(&[(100, 100), (200, 200), (300, 300)])],
            }],
            transfers: Vec::new(),
        };
        let router = Router::new(data, RouterConfig::default());
        let result = router.one_to_all_intervention(
            StationIndex(0),
            90,
            &Intervention::CloseStation(StationIndex(1)),
        );
        assert_eq!(result.arrival_seconds[1], INF_TIME);
        assert_eq!(result.arrival_seconds[2], 300);
    }

    #[test]
    fn frequency_scaling_thins_trips_without_rebuilding_patterns() {
        let data = RoutingData {
            station_count: 2,
            line_count: 1,
            patterns: vec![RoutingPattern {
                line: LineIndex(0),
                stops: vec![StationIndex(0), StationIndex(1)],
                trips: vec![
                    trip(&[(100, 100), (150, 150)]),
                    trip(&[(200, 200), (250, 250)]),
                ],
            }],
            transfers: Vec::new(),
        };
        let router = Router::new(data, RouterConfig::default());
        let scaled = router.one_to_all_intervention(
            StationIndex(0),
            90,
            &Intervention::ScaleLineFrequency {
                line: LineIndex(0),
                multiplier: 0.5,
            },
        );
        assert_eq!(scaled.arrival_seconds[1], 150);
        let disabled = router.one_to_all_intervention(
            StationIndex(0),
            90,
            &Intervention::ScaleLineFrequency {
                line: LineIndex(0),
                multiplier: 0.0,
            },
        );
        assert_eq!(disabled.arrival_seconds[1], INF_TIME);
    }

    #[test]
    fn maximum_transfer_limit_is_enforced() {
        let data = RoutingData {
            station_count: 3,
            line_count: 2,
            patterns: vec![
                RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1)],
                    trips: vec![trip(&[(100, 100), (150, 150)])],
                },
                RoutingPattern {
                    line: LineIndex(1),
                    stops: vec![StationIndex(1), StationIndex(2)],
                    trips: vec![trip(&[(160, 160), (210, 210)])],
                },
            ],
            transfers: Vec::new(),
        };
        let router = Router::new(
            data,
            RouterConfig {
                maximum_transfers: 0,
                ..RouterConfig::default()
            },
        );
        let result = router.one_to_all(StationIndex(0), 90, &LineMask::empty(2));
        assert_eq!(result.arrival_seconds[1], 150);
        assert_eq!(result.arrival_seconds[2], INF_TIME);
    }
}
