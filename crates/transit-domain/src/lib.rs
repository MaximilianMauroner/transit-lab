//! Stable, feed-independent contracts shared by the compiler, router, and
//! learning pipeline.

use anyhow::{bail, Context, Result};
use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const SERVICE_DAY_BINS: usize = 128;
pub const BIN_SECONDS: u32 = 15 * 60;
pub const INF_TIME: u32 = u32::MAX;
pub const INF_RIDES: u8 = u8::MAX;

macro_rules! index_type {
    ($name:ident) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            Serialize,
            Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(pub u32);

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                Self(value)
            }
        }

        impl From<$name> for usize {
            fn from(value: $name) -> Self {
                value.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

index_type!(StationIndex);
index_type!(LineIndex);
index_type!(PatternIndex);
index_type!(TripIndex);

/// Parse a GTFS clock value without wrapping hours at 24.
pub fn parse_gtfs_time(value: &str) -> Result<u32> {
    let mut parts = value.trim().split(':');
    let hours: u32 = parts.next().context("missing GTFS time hours")?.parse()?;
    let minutes: u32 = parts.next().context("missing GTFS time minutes")?.parse()?;
    let seconds: u32 = parts.next().context("missing GTFS time seconds")?.parse()?;

    if parts.next().is_some() {
        bail!("GTFS time has too many components: {value}");
    }
    if minutes >= 60 || seconds >= 60 {
        bail!("invalid GTFS time: {value}");
    }
    Ok(hours * 3600 + minutes * 60 + seconds)
}

/// Parse a CLI/configuration clock. Human-facing experiment configs commonly
/// use `HH:MM`, while GTFS table cells use `HH:MM:SS`.
pub fn parse_departure_time(value: &str) -> Result<u32> {
    let trimmed = value.trim();
    if trimmed.split(':').count() == 2 {
        parse_gtfs_time(&format!("{trimmed}:00"))
    } else {
        parse_gtfs_time(trimmed)
    }
}

pub fn parse_gtfs_date(value: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(value.trim(), "%Y%m%d")
        .with_context(|| format!("invalid GTFS date: {value}"))
}

pub fn service_day_bin(seconds: u32) -> usize {
    ((seconds / BIN_SECONDS) as usize).min(SERVICE_DAY_BINS - 1)
}

pub fn mode_bucket(route_type: u16) -> usize {
    match route_type {
        0 | 5 => 0,          // tram / cable tram
        1 | 6 | 7 | 12 => 1, // subway / aerial / funicular / monorail
        2 | 100..=199 => 2,  // rail
        3 | 11 => 3,         // bus / trolleybus
        4 | 8..=10 => 4,     // ferry / cable / gondola
        _ => 4,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NetworkSnapshotDescriptor {
    pub feed_hashes: Vec<[u8; 32]>,
    pub service_date: NaiveDate,
    pub scope_hash: [u8; 32],
    pub compiler_version: String,
    pub transfer_policy_version: String,
    pub line_grouping_version: String,
}

impl NetworkSnapshotDescriptor {
    pub fn snapshot_id(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("snapshot descriptor is serializable");
        let digest = Sha256::digest(encoded);
        hex_digest(&digest)
    }
}

pub fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}

pub fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub checked_files: Vec<String>,
    pub row_counts: BTreeMap<String, usize>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub snapshot_id: String,
    pub descriptor: NetworkSnapshotDescriptor,
    pub source_name: String,
    pub source_path: String,
    pub downloaded_at: Option<String>,
    pub licence: Option<String>,
    pub geographical_scope: String,
    pub transfer_policy: String,
    pub line_grouping_policy: String,
    pub validation: ValidationReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StationMergeMethod {
    ParentStation,
    Pathway,
    ExplicitTransfer,
    ExactNameRadius,
    FuzzyNameRadius,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StationMergeEvidence {
    pub method: StationMergeMethod,
    pub confidence: f32,
    pub distance_metres: Option<f32>,
    pub source_stop_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalStation {
    pub index: StationIndex,
    pub canonical_id: String,
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub raw_stop_ids: Vec<String>,
    pub merge_confidence: f32,
    pub platform_count: u32,
    pub line_count: u32,
    pub pattern_count: u32,
    pub first_departure: u32,
    pub last_departure: u32,
    pub daily_departures: u32,
    pub daily_arrivals: u32,
    pub transfer_degree: u32,
    pub terminal: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalLine {
    pub index: LineIndex,
    pub canonical_id: String,
    pub display_name: String,
    pub agency_key: String,
    pub mode: u16,
    pub raw_route_ids: Vec<String>,
    pub station_count: u32,
    pub pattern_count: u32,
    pub route_length_metres: f32,
    pub end_to_end_distance_metres: f32,
    pub branching_factor: f32,
    pub service_span_seconds: u32,
    pub daily_trip_count: u32,
    pub median_headway_seconds: f32,
    pub peak_headway_seconds: f32,
    pub off_peak_headway_seconds: f32,
    pub transfer_station_count: u32,
    pub unique_station_fraction: f32,
    pub shared_segment_fraction: f32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PatternSignature {
    pub line: LineIndex,
    pub direction_id: Option<u8>,
    pub stops: Vec<StationIndex>,
    pub pickup_types: Vec<u8>,
    pub dropoff_types: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StopTime {
    pub arrival: u32,
    pub departure: u32,
    pub pickup_type: u8,
    pub dropoff_type: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalTrip {
    pub trip_id: String,
    pub service_id: String,
    pub stop_times: Vec<StopTime>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalPattern {
    pub index: PatternIndex,
    pub signature: PatternSignature,
    pub trips: Vec<CanonicalTrip>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalTransitEdge {
    pub from: StationIndex,
    pub to: StationIndex,
    pub line: LineIndex,
    pub distance_metres: f32,
    pub median_travel_seconds: u32,
    pub minimum_travel_seconds: u32,
    pub active_trip_count: u32,
    pub relative_position: f32,
    pub bearing_sin: f32,
    pub bearing_cos: f32,
    pub departures_by_bin: Vec<f32>,
    pub median_runtime_by_bin: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalTransfer {
    pub from: StationIndex,
    pub to: StationIndex,
    pub minimum_transfer_seconds: u32,
    pub walking_distance_metres: Option<f32>,
    pub explicit: bool,
    pub confidence: f32,
    pub same_physical_station: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineInterchange {
    pub from: LineIndex,
    pub to: LineIndex,
    pub shared_station_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompiledNetwork {
    pub snapshot_id: String,
    pub manifest: SnapshotManifest,
    pub stations: Vec<CanonicalStation>,
    pub lines: Vec<CanonicalLine>,
    pub patterns: Vec<CanonicalPattern>,
    pub transit_edges: Vec<CanonicalTransitEdge>,
    pub transfers: Vec<CanonicalTransfer>,
    pub interchanges: Vec<LineInterchange>,
    pub station_merge_evidence: Vec<StationMergeEvidence>,
    pub stop_to_station: BTreeMap<String, StationIndex>,
}

impl CompiledNetwork {
    pub fn station_count(&self) -> usize {
        self.stations.len()
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub fn validate_indices(&self) -> Result<()> {
        for (expected, station) in self.stations.iter().enumerate() {
            if station.index.0 as usize != expected {
                bail!(
                    "station index {} is not contiguous at {expected}",
                    station.index.0
                );
            }
        }
        for (expected, line) in self.lines.iter().enumerate() {
            if line.index.0 as usize != expected {
                bail!(
                    "line index {} is not contiguous at {expected}",
                    line.index.0
                );
            }
        }
        for (expected, pattern) in self.patterns.iter().enumerate() {
            if pattern.index.0 as usize != expected {
                bail!("pattern index out of bounds: {}", pattern.index.0);
            }
            if pattern.signature.line.0 as usize >= self.lines.len() {
                bail!(
                    "pattern line index out of bounds: {}",
                    pattern.signature.line
                );
            }
            for station in &pattern.signature.stops {
                if station.0 as usize >= self.stations.len() {
                    bail!("pattern station index out of bounds: {station}");
                }
            }
            if pattern
                .trips
                .iter()
                .any(|trip| trip.stop_times.len() != pattern.signature.stops.len())
            {
                bail!(
                    "pattern {} has a trip with a mismatched stop-time width",
                    pattern.index
                );
            }
        }
        for edge in &self.transit_edges {
            if edge.from.0 as usize >= self.stations.len()
                || edge.to.0 as usize >= self.stations.len()
                || edge.line.0 as usize >= self.lines.len()
            {
                bail!("transit edge contains a dangling index");
            }
            if edge.departures_by_bin.len() != SERVICE_DAY_BINS
                || edge.median_runtime_by_bin.len() != SERVICE_DAY_BINS
            {
                bail!("transit edge has an invalid temporal feature width");
            }
        }
        for transfer in &self.transfers {
            if transfer.from.0 as usize >= self.stations.len()
                || transfer.to.0 as usize >= self.stations.len()
            {
                bail!("transfer edge contains a dangling index");
            }
        }
        for interchange in &self.interchanges {
            if interchange.from.0 as usize >= self.lines.len()
                || interchange.to.0 as usize >= self.lines.len()
            {
                bail!("interchange edge contains a dangling index");
            }
        }
        for (stop_id, station) in &self.stop_to_station {
            if station.0 as usize >= self.stations.len() {
                bail!("stop {stop_id} maps to a dangling station index");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LineMask {
    line_count: usize,
    words: Vec<u64>,
}

impl LineMask {
    pub fn empty(line_count: usize) -> Self {
        Self {
            line_count,
            words: vec![0; line_count.div_ceil(64)],
        }
    }

    pub fn single(line_count: usize, line: LineIndex) -> Self {
        let mut mask = Self::empty(line_count);
        mask.disable(line);
        mask
    }

    pub fn from_lines(line_count: usize, lines: impl IntoIterator<Item = LineIndex>) -> Self {
        let mut mask = Self::empty(line_count);
        for line in lines {
            mask.disable(line);
        }
        mask
    }

    pub fn disable(&mut self, line: LineIndex) {
        let index = line.0 as usize;
        if index >= self.line_count {
            return;
        }
        self.words[index / 64] |= 1_u64 << (index % 64);
    }

    pub fn contains(&self, line: LineIndex) -> bool {
        let index = line.0 as usize;
        index < self.line_count && (self.words[index / 64] & (1_u64 << (index % 64))) != 0
    }

    pub fn line_count(&self) -> usize {
        self.line_count
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Intervention {
    DisableLine(LineIndex),
    DisableLines(Vec<LineIndex>),
    CloseStation(StationIndex),
    ScaleLineFrequency { line: LineIndex, multiplier: f32 },
}

impl NetworkSnapshotDescriptor {
    pub fn weekday_matches(&self, date: NaiveDate) -> bool {
        self.service_date.weekday() == date.weekday()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_gtfs_times_after_midnight() {
        assert_eq!(parse_gtfs_time("25:15:00").unwrap(), 90_900);
        assert_eq!(parse_departure_time("07:30").unwrap(), 27_000);
    }

    #[test]
    fn rejects_invalid_clock_components() {
        assert!(parse_gtfs_time("07:60:00").is_err());
        assert!(parse_gtfs_time("07:30:60").is_err());
        assert!(parse_gtfs_time("07:30:00:00").is_err());
    }

    #[test]
    fn line_mask_handles_word_boundaries() {
        let mut mask = LineMask::single(65, LineIndex(64));
        mask.disable(LineIndex(0));
        assert!(mask.contains(LineIndex(64)));
        assert!(mask.contains(LineIndex(0)));
        assert!(!mask.contains(LineIndex(1)));
        assert_eq!(mask.line_count(), 65);
    }
}
