//! A timetable-aware, round-based one-to-all router.
//!
//! The implementation keeps the route scan simple and auditable: each round
//! adds at most one vehicle boarding, while static transfer edges are relaxed
//! with a small Dijkstra closure. Independent origin/departure/intervention
//! queries can therefore be parallelized by the labels crate.

/// Semantic version for routing results consumed by downstream artifacts.
///
/// This must change whenever routing semantics change in a way that can alter
/// arrivals, transfer counts, or intervention outcomes. Consumers include the
/// version in their fingerprints so labels from an older algorithm cannot be
/// reused silently.
pub const ROUTER_ALGORITHM_VERSION: &str = "transit-router-v2";

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
        workspace: Option<&mut RoutingWorkspace>,
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
        let arrivals_by_rides = if let Some(workspace) = workspace {
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
                // station improved in the previous round. Keep all boardable
                // trips rather than assuming the timetable is non-overtaking:
                // a later departure can arrive earlier downstream.
                let retained_trip_indices = retained_trip_indices(pattern, multiplier);
                let mut boarded_trips: Vec<usize> = Vec::new();
                let mut trip_is_boarded = vec![false; pattern.trips.len()];
                for position in 0..pattern.stops.len() {
                    let station = pattern.stops[position];
                    let station_slot = station.0 as usize;
                    if station_slot >= station_count {
                        continue;
                    }

                    // A trip selected at an earlier position can be used to
                    // alight here. A trip selected at this position must not
                    // make its own boarding stop reachable at its scheduled
                    // arrival time.
                    for &trip_index in &boarded_trips {
                        if let Some(stop_time) = pattern
                            .trips
                            .get(trip_index)
                            .and_then(|trip| trip.stop_times.get(position))
                        {
                            let arrival = stop_time.arrival;
                            if stop_time.dropoff_type != 1
                                && closed_station != Some(station)
                                && arrival != INF_TIME
                                && arrival >= departure
                                && arrival.saturating_sub(departure)
                                    <= self.config.maximum_journey_seconds
                                && better(arrival, 0, current_arrivals[station_slot], 0)
                            {
                                current_arrivals[station_slot] = arrival;
                                route_seeds.push(station_slot);
                            }
                        }
                    }

                    // Boarding happens after alighting at the current stop,
                    // and affects only positions after this one. Compare
                    // departures at this same position; a departure at an
                    // earlier boarding stop is not comparable.
                    if closed_station != Some(station)
                        && previous_arrivals[station_slot] != INF_TIME
                    {
                        for &trip_index in &retained_trip_indices {
                            let trip = &pattern.trips[trip_index];
                            let stop_time = &trip.stop_times[position];
                            if !trip_is_boarded[trip_index]
                                && stop_time.pickup_type != 1
                                && stop_time.departure >= previous_arrivals[station_slot]
                            {
                                trip_is_boarded[trip_index] = true;
                                boarded_trips.push(trip_index);
                            }
                        }
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
        let batch_started = Instant::now();
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
        let batch_seconds = batch_started.elapsed().as_secs_f64();
        let mut durations = durations;
        durations.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            let index = ((durations.len() as f64 * fraction).ceil() as usize)
                .saturating_sub(1)
                .min(durations.len() - 1);
            durations[index]
        };
        Ok(RoutingBenchmarkReport {
            warmup_queries,
            measured_queries,
            median_milliseconds: percentile(0.50),
            p95_milliseconds: percentile(0.95),
            queries_per_second: if batch_seconds > 0.0 {
                measured_queries as f64 / batch_seconds
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

fn retained_trip_indices(pattern: &RoutingPattern, multiplier: f32) -> Vec<usize> {
    if pattern.trips.is_empty() || !multiplier.is_finite() || multiplier <= 0.0 {
        return Vec::new();
    }
    let multiplier = multiplier.min(1.0);
    let keep_count = if multiplier >= 1.0 {
        pattern.trips.len()
    } else {
        ((pattern.trips.len() as f32 * multiplier).round() as usize)
            .max(1)
            .min(pattern.trips.len())
    };
    (0..pattern.trips.len())
        .filter(|trip_index| trip_is_retained(*trip_index, pattern.trips.len(), keep_count))
        .collect()
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
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

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

    fn restricted_trip(times: &[(u32, u32, u8, u8)]) -> RoutingTrip {
        RoutingTrip {
            stop_times: times
                .iter()
                .map(|(arrival, departure, pickup_type, dropoff_type)| StopTime {
                    arrival: *arrival,
                    departure: *departure,
                    pickup_type: *pickup_type,
                    dropoff_type: *dropoff_type,
                })
                .collect(),
        }
    }

    fn exhaustive_transfer_closure(
        data: &RoutingData,
        config: &RouterConfig,
        arrivals: &mut [u32],
        seeds: &[usize],
        departure: u32,
        closed_station: Option<StationIndex>,
    ) {
        let mut queue = BinaryHeap::<Reverse<(u32, usize)>>::new();
        for &station in seeds {
            if station < arrivals.len() && arrivals[station] != INF_TIME {
                queue.push(Reverse((arrivals[station], station)));
            }
        }
        while let Some(Reverse((time, station))) = queue.pop() {
            if arrivals[station] != time {
                continue;
            }
            for transfer in &data.transfers {
                if transfer.from.0 as usize != station
                    || closed_station == Some(transfer.from)
                    || closed_station == Some(transfer.to)
                {
                    continue;
                }
                let target = transfer.to.0 as usize;
                if target >= arrivals.len() {
                    continue;
                }
                let arrival = time.saturating_add(transfer.seconds);
                if arrival < departure
                    || arrival.saturating_sub(departure) > config.maximum_journey_seconds
                {
                    continue;
                }
                if arrival < arrivals[target] {
                    arrivals[target] = arrival;
                    queue.push(Reverse((arrival, target)));
                }
            }
        }
    }

    fn exhaustive_trip_is_retained(index: usize, trip_count: usize, keep_count: usize) -> bool {
        keep_count >= trip_count || ((index * keep_count) % trip_count) < keep_count
    }

    fn exhaustive_keep_count(trip_count: usize, multiplier: f32) -> Option<usize> {
        if trip_count == 0 || !multiplier.is_finite() || multiplier <= 0.0 {
            return None;
        }
        let multiplier = multiplier.min(1.0);
        Some(if multiplier >= 1.0 {
            trip_count
        } else {
            ((trip_count as f32 * multiplier).round() as usize)
                .max(1)
                .min(trip_count)
        })
    }

    /// Deliberately slow reference implementation. It considers every valid
    /// boarding/alighting pair instead of sharing the production marked-route
    /// scan, so differential failures identify route-scan errors rather than
    /// duplicated implementation mistakes.
    fn exhaustive_one_to_all(
        data: &RoutingData,
        config: &RouterConfig,
        origin: StationIndex,
        departure: u32,
        disabled_lines: &LineMask,
        closed_station: Option<StationIndex>,
        frequency_multipliers: Option<&[f32]>,
    ) -> OneToAllResult {
        if departure == INF_TIME
            || origin.0 as usize >= data.station_count
            || closed_station == Some(origin)
        {
            return OneToAllResult {
                arrival_seconds: vec![INF_TIME; data.station_count],
                transfers: vec![INF_RIDES; data.station_count],
            };
        }

        let maximum_rides = config.maximum_transfers as usize + 1;
        let mut arrivals_by_rides =
            vec![vec![INF_TIME; data.station_count]; maximum_rides.saturating_add(1)];
        arrivals_by_rides[0][origin.0 as usize] = departure;
        exhaustive_transfer_closure(
            data,
            config,
            &mut arrivals_by_rides[0],
            &[origin.0 as usize],
            departure,
            closed_station,
        );

        for ride_count in 1..=maximum_rides {
            let previous = arrivals_by_rides[ride_count - 1].clone();
            let current = &mut arrivals_by_rides[ride_count];
            for pattern in &data.patterns {
                if disabled_lines.contains(pattern.line) {
                    continue;
                }
                let multiplier = frequency_multipliers
                    .and_then(|values| values.get(pattern.line.0 as usize))
                    .copied()
                    .unwrap_or(1.0);
                let Some(keep_count) = exhaustive_keep_count(pattern.trips.len(), multiplier)
                else {
                    continue;
                };
                for (trip_index, trip) in pattern.trips.iter().enumerate() {
                    if !exhaustive_trip_is_retained(trip_index, pattern.trips.len(), keep_count) {
                        continue;
                    }
                    for board in 0..pattern.stops.len() {
                        let board_station = pattern.stops[board];
                        let board_slot = board_station.0 as usize;
                        if board_slot >= data.station_count
                            || closed_station == Some(board_station)
                            || previous[board_slot] == INF_TIME
                        {
                            continue;
                        }
                        let board_time = &trip.stop_times[board];
                        if board_time.pickup_type == 1
                            || board_time.departure < previous[board_slot]
                        {
                            continue;
                        }
                        for alight in (board + 1)..pattern.stops.len() {
                            let station = pattern.stops[alight];
                            let station_slot = station.0 as usize;
                            let stop_time = &trip.stop_times[alight];
                            if station_slot >= data.station_count
                                || closed_station == Some(station)
                                || stop_time.dropoff_type == 1
                            {
                                continue;
                            }
                            let arrival = stop_time.arrival;
                            if arrival == INF_TIME
                                || arrival < departure
                                || arrival.saturating_sub(departure)
                                    > config.maximum_journey_seconds
                            {
                                continue;
                            }
                            current[station_slot] = current[station_slot].min(arrival);
                        }
                    }
                }
            }
            let seeds = current
                .iter()
                .enumerate()
                .filter_map(|(station, arrival)| (*arrival != INF_TIME).then_some(station))
                .collect::<Vec<_>>();
            exhaustive_transfer_closure(data, config, current, &seeds, departure, closed_station);
        }

        let mut result = OneToAllResult {
            arrival_seconds: vec![INF_TIME; data.station_count],
            transfers: vec![INF_RIDES; data.station_count],
        };
        for station in 0..data.station_count {
            let mut best_rides = usize::MAX;
            for (ride_count, arrivals) in arrivals_by_rides.iter().enumerate() {
                let arrival = arrivals[station];
                if arrival < result.arrival_seconds[station]
                    || (arrival == result.arrival_seconds[station] && ride_count < best_rides)
                {
                    result.arrival_seconds[station] = arrival;
                    best_rides = ride_count;
                }
            }
            if result.arrival_seconds[station] != INF_TIME {
                result.transfers[station] =
                    best_rides.saturating_sub(1).min(u8::MAX as usize) as u8;
            }
        }
        result
    }

    #[derive(Clone, Copy)]
    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
            }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state ^ (self.state >> 29)
        }

        fn range(&mut self, upper: usize) -> usize {
            debug_assert!(upper > 0);
            (self.next() as usize) % upper
        }
    }

    fn generated_routing_data(seed: u64) -> RoutingData {
        let mut rng = TestRng::new(seed);
        let station_count = 2 + rng.range(7);
        let line_count = 1 + rng.range(5);
        let mut patterns = Vec::new();
        for line in 0..line_count {
            for _ in 0..(1 + rng.range(4)) {
                let stop_count = 2 + rng.range(station_count - 1);
                let mut stops = Vec::with_capacity(stop_count);
                while stops.len() < stop_count {
                    let station = StationIndex(rng.range(station_count) as u32);
                    if !stops.contains(&station) {
                        stops.push(station);
                    }
                }
                let trip_count = 1 + rng.range(6);
                let mut trips = Vec::with_capacity(trip_count);
                for trip_index in 0..trip_count {
                    let mut stop_times = Vec::with_capacity(stop_count);
                    let mut departure = 86_400 + rng.range(12_000) as u32;
                    if trip_index == 1 && trip_count > 1 {
                        // Force some generated patterns to contain a later
                        // departure that overtakes an earlier trip downstream.
                        departure = departure.saturating_add(600);
                    }
                    for position in 0..stop_count {
                        let arrival = departure;
                        let dwell = rng.range(90) as u32;
                        let pickup_type = (rng.range(25) == 0) as u8;
                        let dropoff_type = (rng.range(25) == 0) as u8;
                        stop_times.push(StopTime {
                            arrival,
                            departure: arrival + dwell,
                            pickup_type,
                            dropoff_type,
                        });
                        if position + 1 < stop_count {
                            departure = arrival
                                .saturating_add(dwell)
                                .saturating_add(30 + rng.range(700) as u32);
                        }
                    }
                    trips.push(RoutingTrip { stop_times });
                }
                patterns.push(RoutingPattern {
                    line: LineIndex(line as u32),
                    stops,
                    trips,
                });
            }
        }
        let transfer_count = rng.range(station_count * 3 + 1);
        let mut transfers = Vec::with_capacity(transfer_count);
        for _ in 0..transfer_count {
            let from = rng.range(station_count);
            let mut to = rng.range(station_count);
            if to == from {
                to = (to + 1) % station_count;
            }
            transfers.push(RoutingTransfer {
                from: StationIndex(from as u32),
                to: StationIndex(to as u32),
                seconds: rng.range(360) as u32,
            });
        }
        RoutingData {
            station_count,
            line_count,
            patterns,
            transfers,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_matches_oracle(
        router: &Router,
        origin: StationIndex,
        departure: u32,
        disabled_lines: &LineMask,
        closed_station: Option<StationIndex>,
        frequency_multipliers: Option<&[f32]>,
        actual: OneToAllResult,
        context: &str,
    ) {
        let expected = exhaustive_one_to_all(
            &router.data,
            &router.config,
            origin,
            departure,
            disabled_lines,
            closed_station,
            frequency_multipliers,
        );
        assert_eq!(
            actual.arrival_seconds, expected.arrival_seconds,
            "{context}: arrival times differ"
        );
        assert_eq!(
            actual.transfers, expected.transfers,
            "{context}: transfer counts differ"
        );
    }

    #[test]
    fn marked_route_scan_matches_exhaustive_oracle_on_generated_networks() {
        for seed in 0..100_u64 {
            let data = generated_routing_data(seed);
            let router = Router::new(
                data,
                RouterConfig {
                    maximum_transfers: (seed % 4) as u8,
                    maximum_journey_seconds: 45 * 60 + (seed as u32 % 4) * 15 * 60,
                },
            );
            let origins = (0..router.data.station_count.min(5))
                .map(|station| StationIndex(station as u32))
                .collect::<Vec<_>>();
            let departures = [86_400, 90_000, 96_000];
            for origin in origins {
                for departure in departures {
                    let empty = LineMask::empty(router.data.line_count);
                    assert_matches_oracle(
                        &router,
                        origin,
                        departure,
                        &empty,
                        None,
                        None,
                        router.one_to_all(origin, departure, &empty),
                        &format!(
                            "seed {seed}, origin {}, departure {departure}, intact",
                            origin.0
                        ),
                    );

                    let disabled_line =
                        LineIndex(((seed as usize) % router.data.line_count) as u32);
                    let disabled = LineMask::single(router.data.line_count, disabled_line);
                    assert_matches_oracle(
                        &router,
                        origin,
                        departure,
                        &disabled,
                        None,
                        None,
                        router.one_to_all(origin, departure, &disabled),
                        &format!(
                            "seed {seed}, origin {}, departure {departure}, disabled",
                            origin.0
                        ),
                    );

                    let closed_station =
                        StationIndex(((seed as usize + 1) % router.data.station_count) as u32);
                    assert_matches_oracle(
                        &router,
                        origin,
                        departure,
                        &empty,
                        Some(closed_station),
                        None,
                        router.one_to_all_intervention(
                            origin,
                            departure,
                            &Intervention::CloseStation(closed_station),
                        ),
                        &format!(
                            "seed {seed}, origin {}, departure {departure}, closed",
                            origin.0
                        ),
                    );

                    let frequency_line =
                        LineIndex(((seed as usize + 1) % router.data.line_count) as u32);
                    let mut multipliers = vec![1.0_f32; router.data.line_count];
                    multipliers[frequency_line.0 as usize] = 0.5;
                    assert_matches_oracle(
                        &router,
                        origin,
                        departure,
                        &empty,
                        None,
                        Some(&multipliers),
                        router.one_to_all_intervention(
                            origin,
                            departure,
                            &Intervention::ScaleLineFrequency {
                                line: frequency_line,
                                multiplier: 0.5,
                            },
                        ),
                        &format!(
                            "seed {seed}, origin {}, departure {departure}, frequency",
                            origin.0
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn boarding_does_not_move_arrival_time_backward() {
        let data = RoutingData {
            station_count: 4,
            line_count: 1,
            patterns: vec![RoutingPattern {
                line: LineIndex(0),
                stops: vec![StationIndex(1), StationIndex(2)],
                trips: vec![trip(&[(90, 110), (200, 200)])],
            }],
            transfers: vec![
                RoutingTransfer {
                    from: StationIndex(0),
                    to: StationIndex(1),
                    seconds: 100,
                },
                RoutingTransfer {
                    from: StationIndex(1),
                    to: StationIndex(3),
                    seconds: 20,
                },
            ],
        };
        let router = Router::new(
            data,
            RouterConfig {
                maximum_transfers: 0,
                ..RouterConfig::default()
            },
        );

        let result = router.one_to_all(StationIndex(0), 0, &LineMask::empty(1));

        assert_eq!(result.arrival_seconds[1], 100);
        assert_eq!(result.arrival_seconds[2], 200);
        assert_eq!(result.arrival_seconds[3], 120);
    }

    #[test]
    fn downstream_boarding_can_replace_an_earlier_boarded_trip() {
        let data = RoutingData {
            station_count: 4,
            line_count: 1,
            patterns: vec![RoutingPattern {
                line: LineIndex(0),
                stops: vec![StationIndex(1), StationIndex(2), StationIndex(3)],
                trips: vec![
                    trip(&[(100, 100), (210, 220), (320, 320)]),
                    trip(&[(160, 160), (170, 180), (230, 230)]),
                ],
            }],
            transfers: vec![
                RoutingTransfer {
                    from: StationIndex(0),
                    to: StationIndex(1),
                    seconds: 0,
                },
                RoutingTransfer {
                    from: StationIndex(0),
                    to: StationIndex(2),
                    seconds: 150,
                },
            ],
        };
        let router = Router::new(
            data,
            RouterConfig {
                maximum_transfers: 0,
                ..RouterConfig::default()
            },
        );

        let result = router.one_to_all(StationIndex(0), 0, &LineMask::empty(1));

        assert_eq!(result.arrival_seconds[2], 150);
        assert_eq!(result.arrival_seconds[3], 230);
    }

    #[test]
    fn boarding_and_alighting_restrictions_apply_at_the_current_stop() {
        let pickup_blocked = Router::new(
            RoutingData {
                station_count: 3,
                line_count: 1,
                patterns: vec![RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1), StationIndex(2)],
                    trips: vec![restricted_trip(&[
                        (100, 100, 1, 0),
                        (200, 200, 0, 0),
                        (300, 300, 0, 0),
                    ])],
                }],
                transfers: Vec::new(),
            },
            RouterConfig::default(),
        );
        let pickup_result = pickup_blocked.one_to_all(StationIndex(0), 90, &LineMask::empty(1));
        assert_eq!(pickup_result.arrival_seconds[1], INF_TIME);
        assert_eq!(pickup_result.arrival_seconds[2], INF_TIME);

        let dropoff_blocked = Router::new(
            RoutingData {
                station_count: 3,
                line_count: 1,
                patterns: vec![RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1), StationIndex(2)],
                    trips: vec![restricted_trip(&[
                        (100, 100, 0, 0),
                        (200, 200, 0, 1),
                        (300, 300, 0, 0),
                    ])],
                }],
                transfers: Vec::new(),
            },
            RouterConfig::default(),
        );
        let dropoff_result = dropoff_blocked.one_to_all(StationIndex(0), 90, &LineMask::empty(1));
        assert_eq!(dropoff_result.arrival_seconds[1], INF_TIME);
        assert_eq!(dropoff_result.arrival_seconds[2], 300);
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
