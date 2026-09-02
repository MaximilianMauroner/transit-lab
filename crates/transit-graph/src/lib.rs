//! Numeric graph tensors and compact on-disk serialization.

use anyhow::{bail, Context, Result};
use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use transit_domain::{
    mode_bucket, CanonicalLine, CanonicalStation, CompiledNetwork, SERVICE_DAY_BINS,
};

pub const GRAPH_SCHEMA_VERSION: &str = "station-line-relational-v2";
pub const TEMPORAL_CHANNELS: usize = 4;
pub const EDGE_FEATURES: usize = 7;
pub const TRANSFER_FEATURES: usize = 5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl FeatureMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }

    pub fn from_rows(rows: Vec<Vec<f32>>) -> Result<Self> {
        let cols = rows.first().map(Vec::len).unwrap_or(0);
        if rows.iter().any(|row| row.len() != cols) {
            bail!("feature rows have inconsistent widths");
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            values: rows.into_iter().flatten().collect(),
        })
    }

    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.values[start..start + self.cols]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.values[start..start + self.cols]
    }

    pub fn validate(&self) -> Result<()> {
        if self.values.len() != self.rows * self.cols {
            bail!("matrix values do not match declared shape");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphManifest {
    pub schema_version: String,
    pub snapshot_id: String,
    pub station_count: usize,
    pub line_count: usize,
    pub transit_edge_count: usize,
    pub transfer_edge_count: usize,
    pub interchange_edge_count: usize,
    pub pattern_count: usize,
    pub pattern_stop_count: usize,
    pub pattern_segment_count: usize,
    pub temporal_bins: usize,
    pub temporal_bin_seconds: u32,
    pub station_feature_names: Vec<String>,
    pub line_feature_names: Vec<String>,
    pub temporal_channel_names: Vec<String>,
    pub transit_edge_feature_names: Vec<String>,
    pub transfer_feature_names: Vec<String>,
    pub pattern_stop_feature_names: Vec<String>,
    pub pattern_segment_feature_names: Vec<String>,
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct GraphTensor {
    pub manifest: GraphManifest,
    /// Passenger-facing labels for search and explanations. These strings
    /// are lookup metadata and are never included in model features.
    pub line_names: Vec<String>,
    pub station_features: FeatureMatrix,
    pub station_temporal: FeatureMatrix,
    pub line_features: FeatureMatrix,
    pub line_temporal: FeatureMatrix,
    pub serves_src: Vec<u32>,
    pub serves_dst: Vec<u32>,
    pub transit_src: Vec<u32>,
    pub transit_dst: Vec<u32>,
    pub transit_line: Vec<u32>,
    pub transit_features: FeatureMatrix,
    pub transit_temporal: FeatureMatrix,
    pub transfer_src: Vec<u32>,
    pub transfer_dst: Vec<u32>,
    pub transfer_features: FeatureMatrix,
    pub interchange_src: Vec<u32>,
    pub interchange_dst: Vec<u32>,
    /// CSR-like ordered pattern sequences. `pattern_offsets[p]..pattern_offsets[p + 1]`
    /// indexes the stop and stop-feature arrays for pattern `p`.
    pub pattern_offsets: Vec<u32>,
    pub pattern_stops: Vec<u32>,
    pub pattern_lines: Vec<u32>,
    /// GTFS direction IDs, with `u32::MAX` representing an absent direction.
    pub pattern_directions: Vec<u32>,
    pub pattern_trip_counts: Vec<u32>,
    pub pattern_stop_features: FeatureMatrix,
    /// One row per ordered pattern segment, aligned with the segment order
    /// implied by the pattern offsets.
    pub pattern_segment_features: FeatureMatrix,
}

impl GraphTensor {
    pub fn from_network(network: &CompiledNetwork) -> Result<Self> {
        network.validate_indices()?;
        let station_feature_names = station_feature_names();
        let line_feature_names = line_feature_names();
        let temporal_channel_names = vec![
            "departures".into(),
            "arrivals".into(),
            "active_lines_or_trips".into(),
            "median_wait_or_headway".into(),
        ];
        let transit_edge_feature_names = vec![
            "distance_metres".into(),
            "median_travel_seconds".into(),
            "minimum_travel_seconds".into(),
            "active_trip_count".into(),
            "relative_route_position".into(),
            "bearing_sin".into(),
            "bearing_cos".into(),
        ];
        let transfer_feature_names = vec![
            "minimum_transfer_seconds".into(),
            "walking_distance_metres".into(),
            "explicit".into(),
            "confidence".into(),
            "same_physical_station".into(),
        ];
        let pattern_stop_feature_names = vec![
            "pickup_type".into(),
            "drop_off_type".into(),
            "relative_pattern_position".into(),
        ];
        let pattern_segment_feature_names = transit_edge_feature_names.clone();
        let station_feature_count = station_feature_names.len();
        let line_feature_count = line_feature_names.len();

        let station_lines = station_line_sets(network);
        let (min_lat, max_lat, min_lon, max_lon) = coordinate_bounds(network);
        let mut station_rows = Vec::with_capacity(network.stations.len());
        for (station_index, station) in network.stations.iter().enumerate() {
            let (x, y) =
                normalized_station_coordinate(station, (min_lat, max_lat, min_lon, max_lon));
            let mut row = vec![
                x,
                y,
                station.platform_count as f32,
                station.line_count as f32,
                station.pattern_count as f32,
            ];
            let mut mode_counts = [0_u32; 5];
            for line_index in &station_lines[station_index] {
                mode_counts[mode_bucket(network.lines[*line_index as usize].mode)] += 1;
            }
            row.extend(mode_counts.into_iter().map(|value| value as f32));
            row.extend([
                normalized_time(station.first_departure),
                normalized_time(station.last_departure),
                (station.daily_departures as f32 + 1.0).ln(),
                (station.daily_arrivals as f32 + 1.0).ln(),
                station.transfer_degree as f32,
                f32::from(station.terminal),
            ]);
            debug_assert_eq!(row.len(), station_feature_names.len());
            station_rows.push(row);
        }

        let mut line_rows = Vec::with_capacity(network.lines.len());
        for line in &network.lines {
            let mut row = vec![0.0; line_feature_names.len()];
            row[mode_bucket(line.mode)] = 1.0;
            row[5] = (line.station_count as f32 + 1.0).ln();
            row[6] = (line.pattern_count as f32 + 1.0).ln();
            row[7] = line.route_length_metres / 10_000.0;
            row[8] = line.end_to_end_distance_metres / 10_000.0;
            row[9] = line.branching_factor;
            row[10] = line.service_span_seconds as f32 / (32.0 * 3600.0);
            row[11] = (line.daily_trip_count as f32 + 1.0).ln();
            row[12] = line.median_headway_seconds / 3600.0;
            row[13] = line.peak_headway_seconds / 3600.0;
            row[14] = line.off_peak_headway_seconds / 3600.0;
            row[15] = (line.transfer_station_count as f32 + 1.0).ln();
            row[16] = line.unique_station_fraction;
            row[17] = line.shared_segment_fraction;
            line_rows.push(row);
        }

        let station_temporal = station_temporal(network);
        let line_temporal = line_temporal(network);
        let mut serves = BTreeSet::new();
        for pattern in &network.patterns {
            for station in &pattern.signature.stops {
                serves.insert((station.0, pattern.signature.line.0));
            }
        }
        let (serves_src, serves_dst): (Vec<_>, Vec<_>) = serves.into_iter().unzip();

        let transit_src: Vec<u32> = network
            .transit_edges
            .iter()
            .map(|edge| edge.from.0)
            .collect();
        let transit_dst: Vec<u32> = network.transit_edges.iter().map(|edge| edge.to.0).collect();
        let transit_line: Vec<u32> = network
            .transit_edges
            .iter()
            .map(|edge| edge.line.0)
            .collect();
        let transit_features = feature_matrix_or_zeros(
            network
                .transit_edges
                .iter()
                .map(|edge| {
                    vec![
                        edge.distance_metres / 10_000.0,
                        edge.median_travel_seconds as f32 / 3600.0,
                        edge.minimum_travel_seconds as f32 / 3600.0,
                        (edge.active_trip_count as f32 + 1.0).ln(),
                        edge.relative_position,
                        edge.bearing_sin,
                        edge.bearing_cos,
                    ]
                })
                .collect(),
            EDGE_FEATURES,
        )?;
        let transit_temporal = feature_matrix_or_zeros(
            network
                .transit_edges
                .iter()
                .map(|edge| {
                    let mut row = Vec::with_capacity(SERVICE_DAY_BINS * 2);
                    row.extend(&edge.departures_by_bin);
                    row.extend(&edge.median_runtime_by_bin);
                    row
                })
                .collect(),
            SERVICE_DAY_BINS * 2,
        )?;
        let transfer_src: Vec<u32> = network.transfers.iter().map(|edge| edge.from.0).collect();
        let transfer_dst: Vec<u32> = network.transfers.iter().map(|edge| edge.to.0).collect();
        let transfer_features = feature_matrix_or_zeros(
            network
                .transfers
                .iter()
                .map(|edge| {
                    vec![
                        edge.minimum_transfer_seconds as f32 / 3600.0,
                        edge.walking_distance_metres.unwrap_or(0.0) / 1_000.0,
                        f32::from(edge.explicit),
                        edge.confidence,
                        f32::from(edge.same_physical_station),
                    ]
                })
                .collect(),
            TRANSFER_FEATURES,
        )?;
        let interchange_src: Vec<u32> = network
            .interchanges
            .iter()
            .map(|edge| edge.from.0)
            .collect();
        let interchange_dst: Vec<u32> = network.interchanges.iter().map(|edge| edge.to.0).collect();
        let pattern_data = pattern_arrays(network, &transit_features);

        let mut files = BTreeMap::new();
        for (key, file) in [
            ("station_features", "station_features.f32"),
            ("station_temporal", "station_temporal.f16"),
            ("line_features", "line_features.f32"),
            ("line_temporal", "line_temporal.f16"),
            ("serves_src", "serves_src.u32"),
            ("serves_dst", "serves_dst.u32"),
            ("transit_src", "transit_src.u32"),
            ("transit_dst", "transit_dst.u32"),
            ("transit_line", "transit_line.u32"),
            ("transit_features", "transit_features.f32"),
            ("transit_temporal", "transit_temporal.f16"),
            ("transfer_src", "transfer_src.u32"),
            ("transfer_dst", "transfer_dst.u32"),
            ("transfer_features", "transfer_features.f32"),
            ("interchange_src", "interchange_src.u32"),
            ("interchange_dst", "interchange_dst.u32"),
            ("pattern_offsets", "pattern_offsets.u32"),
            ("pattern_stops", "pattern_stops.u32"),
            ("pattern_lines", "pattern_lines.u32"),
            ("pattern_directions", "pattern_directions.u32"),
            ("pattern_trip_counts", "pattern_trip_counts.u32"),
            ("pattern_stop_features", "pattern_stop_features.f32"),
            ("pattern_segment_features", "pattern_segment_features.f32"),
            ("lookup_stations", "lookup_stations.json"),
            ("lookup_lines", "lookup_lines.json"),
        ] {
            files.insert(key.into(), file.into());
        }
        let manifest = GraphManifest {
            schema_version: GRAPH_SCHEMA_VERSION.into(),
            snapshot_id: network.snapshot_id.clone(),
            station_count: network.stations.len(),
            line_count: network.lines.len(),
            transit_edge_count: transit_src.len(),
            transfer_edge_count: transfer_src.len(),
            interchange_edge_count: interchange_src.len(),
            pattern_count: network.patterns.len(),
            pattern_stop_count: pattern_data.stops.len(),
            pattern_segment_count: pattern_data.segment_features.rows,
            temporal_bins: SERVICE_DAY_BINS,
            temporal_bin_seconds: 15 * 60,
            station_feature_names,
            line_feature_names,
            temporal_channel_names,
            transit_edge_feature_names,
            transfer_feature_names,
            pattern_stop_feature_names,
            pattern_segment_feature_names,
            files,
        };
        let graph = Self {
            manifest,
            line_names: network
                .lines
                .iter()
                .map(|line| line.display_name.clone())
                .collect(),
            station_features: feature_matrix_or_zeros(station_rows, station_feature_count)?,
            station_temporal,
            line_features: feature_matrix_or_zeros(line_rows, line_feature_count)?,
            line_temporal,
            serves_src,
            serves_dst,
            transit_src,
            transit_dst,
            transit_line,
            transit_features,
            transit_temporal,
            transfer_src,
            transfer_dst,
            transfer_features,
            interchange_src,
            interchange_dst,
            pattern_offsets: pattern_data.offsets,
            pattern_stops: pattern_data.stops,
            pattern_lines: pattern_data.lines,
            pattern_directions: pattern_data.directions,
            pattern_trip_counts: pattern_data.trip_counts,
            pattern_stop_features: pattern_data.stop_features,
            pattern_segment_features: pattern_data.segment_features,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<()> {
        if self.manifest.temporal_bins == 0 {
            bail!("graph manifest has no temporal bins");
        }
        if self.manifest.temporal_bins != SERVICE_DAY_BINS {
            bail!(
                "graph uses {} temporal bins, expected {}",
                self.manifest.temporal_bins,
                SERVICE_DAY_BINS
            );
        }
        if self.manifest.temporal_bin_seconds != 15 * 60 {
            bail!("graph temporal bin width is not 15 minutes");
        }
        if self.station_features.rows != self.manifest.station_count
            || self.line_features.rows != self.manifest.line_count
            || self.station_temporal.rows != self.manifest.station_count
            || self.line_temporal.rows != self.manifest.line_count
        {
            bail!("graph feature matrix row counts do not match the manifest");
        }
        if self.line_names.len() != self.manifest.line_count {
            bail!("graph line names do not match the manifest count");
        }
        if self.station_features.cols != self.manifest.station_feature_names.len()
            || self.line_features.cols != self.manifest.line_feature_names.len()
            || self.station_temporal.cols != self.manifest.temporal_bins * TEMPORAL_CHANNELS
            || self.line_temporal.cols != self.manifest.temporal_bins * TEMPORAL_CHANNELS
            || self.transit_features.cols != self.manifest.transit_edge_feature_names.len()
            || self.transit_temporal.cols != self.manifest.temporal_bins * 2
            || self.transfer_features.cols != self.manifest.transfer_feature_names.len()
            || self.pattern_stop_features.cols != self.manifest.pattern_stop_feature_names.len()
            || self.pattern_segment_features.cols
                != self.manifest.pattern_segment_feature_names.len()
        {
            bail!("graph feature matrix columns do not match the manifest");
        }
        if self.transit_src.len() != self.manifest.transit_edge_count
            || self.transfer_src.len() != self.manifest.transfer_edge_count
            || self.interchange_src.len() != self.manifest.interchange_edge_count
            || self.pattern_offsets.len() != self.manifest.pattern_count + 1
            || self.pattern_lines.len() != self.manifest.pattern_count
            || self.pattern_directions.len() != self.manifest.pattern_count
            || self.pattern_trip_counts.len() != self.manifest.pattern_count
            || self.pattern_stops.len() != self.manifest.pattern_stop_count
            || self.pattern_stop_features.rows != self.manifest.pattern_stop_count
            || self.pattern_segment_features.rows != self.manifest.pattern_segment_count
        {
            bail!("graph relation counts do not match the manifest");
        }
        if self.pattern_offsets.first().copied().unwrap_or(1) != 0
            || self
                .pattern_offsets
                .windows(2)
                .any(|window| window[0] > window[1])
            || self.pattern_offsets.last().copied().unwrap_or(0)
                != self.manifest.pattern_stop_count as u32
        {
            bail!("pattern offsets do not describe the pattern stop array");
        }
        let expected_segments = self
            .pattern_offsets
            .windows(2)
            .map(|window| (window[1] - window[0]).saturating_sub(1) as usize)
            .sum::<usize>();
        if expected_segments != self.manifest.pattern_segment_count {
            bail!("pattern segment count does not match pattern offsets");
        }
        self.station_features.validate()?;
        self.station_temporal.validate()?;
        self.line_features.validate()?;
        self.line_temporal.validate()?;
        self.transit_features.validate()?;
        self.transit_temporal.validate()?;
        self.transfer_features.validate()?;
        self.pattern_stop_features.validate()?;
        self.pattern_segment_features.validate()?;
        if self.serves_src.len() != self.serves_dst.len()
            || self.transit_src.len() != self.transit_dst.len()
            || self.transit_src.len() != self.transit_line.len()
            || self.transit_src.len() != self.transit_features.rows
            || self.transit_src.len() != self.transit_temporal.rows
            || self.transfer_src.len() != self.transfer_dst.len()
            || self.transfer_src.len() != self.transfer_features.rows
            || self.interchange_src.len() != self.interchange_dst.len()
        {
            bail!("graph relation arrays have inconsistent lengths");
        }
        for index in self
            .serves_src
            .iter()
            .chain(self.transit_src.iter())
            .chain(self.transit_dst.iter())
            .chain(self.transfer_src.iter())
            .chain(self.transfer_dst.iter())
            .chain(self.pattern_stops.iter())
        {
            if *index as usize >= self.manifest.station_count {
                bail!("station relation index out of bounds: {index}");
            }
        }
        for index in self
            .serves_dst
            .iter()
            .chain(self.transit_line.iter())
            .chain(self.interchange_src.iter())
            .chain(self.interchange_dst.iter())
            .chain(self.pattern_lines.iter())
        {
            if *index as usize >= self.manifest.line_count {
                bail!("line relation index out of bounds: {index}");
            }
        }
        Ok(())
    }

    pub fn save(&self, directory: &Path, network: &CompiledNetwork) -> Result<()> {
        network.validate_indices()?;
        if network.snapshot_id != self.manifest.snapshot_id
            || network.stations.len() != self.manifest.station_count
            || network.lines.len() != self.manifest.line_count
        {
            bail!("graph and compiled network refer to different snapshot shapes");
        }
        self.validate()?;
        fs::create_dir_all(directory)
            .with_context(|| format!("creating {}", directory.display()))?;
        write_json(directory.join("manifest.json"), &self.manifest)?;
        write_f32(
            directory.join("station_features.f32"),
            &self.station_features.values,
        )?;
        write_f16(
            directory.join("station_temporal.f16"),
            &self.station_temporal.values,
        )?;
        write_f32(
            directory.join("line_features.f32"),
            &self.line_features.values,
        )?;
        write_f16(
            directory.join("line_temporal.f16"),
            &self.line_temporal.values,
        )?;
        write_u32(directory.join("serves_src.u32"), &self.serves_src)?;
        write_u32(directory.join("serves_dst.u32"), &self.serves_dst)?;
        write_u32(directory.join("transit_src.u32"), &self.transit_src)?;
        write_u32(directory.join("transit_dst.u32"), &self.transit_dst)?;
        write_u32(directory.join("transit_line.u32"), &self.transit_line)?;
        write_f32(
            directory.join("transit_features.f32"),
            &self.transit_features.values,
        )?;
        write_f16(
            directory.join("transit_temporal.f16"),
            &self.transit_temporal.values,
        )?;
        write_u32(directory.join("transfer_src.u32"), &self.transfer_src)?;
        write_u32(directory.join("transfer_dst.u32"), &self.transfer_dst)?;
        write_f32(
            directory.join("transfer_features.f32"),
            &self.transfer_features.values,
        )?;
        write_u32(directory.join("interchange_src.u32"), &self.interchange_src)?;
        write_u32(directory.join("interchange_dst.u32"), &self.interchange_dst)?;
        write_u32(directory.join("pattern_offsets.u32"), &self.pattern_offsets)?;
        write_u32(directory.join("pattern_stops.u32"), &self.pattern_stops)?;
        write_u32(directory.join("pattern_lines.u32"), &self.pattern_lines)?;
        write_u32(
            directory.join("pattern_directions.u32"),
            &self.pattern_directions,
        )?;
        write_u32(
            directory.join("pattern_trip_counts.u32"),
            &self.pattern_trip_counts,
        )?;
        write_f32(
            directory.join("pattern_stop_features.f32"),
            &self.pattern_stop_features.values,
        )?;
        write_f32(
            directory.join("pattern_segment_features.f32"),
            &self.pattern_segment_features.values,
        )?;
        write_json(directory.join("lookup_stations.json"), &network.stations)?;
        write_json(directory.join("lookup_lines.json"), &network.lines)?;
        Ok(())
    }

    pub fn load(directory: &Path) -> Result<Self> {
        let manifest: GraphManifest = read_json(directory.join("manifest.json"))?;
        if manifest.schema_version != GRAPH_SCHEMA_VERSION {
            bail!(
                "unsupported graph schema {}; expected {}",
                manifest.schema_version,
                GRAPH_SCHEMA_VERSION
            );
        }
        let lookup_stations: Vec<CanonicalStation> =
            read_json(directory.join("lookup_stations.json"))?;
        let lookup_lines: Vec<CanonicalLine> = read_json(directory.join("lookup_lines.json"))?;
        if lookup_stations.len() != manifest.station_count
            || lookup_lines.len() != manifest.line_count
        {
            bail!("graph lookup tables do not match manifest counts");
        }
        let graph = Self {
            station_features: read_matrix_f32(
                directory.join("station_features.f32"),
                manifest.station_count,
                manifest.station_feature_names.len(),
            )?,
            station_temporal: read_matrix_f16(
                directory.join("station_temporal.f16"),
                manifest.station_count,
                manifest.temporal_bins * TEMPORAL_CHANNELS,
            )?,
            line_features: read_matrix_f32(
                directory.join("line_features.f32"),
                manifest.line_count,
                manifest.line_feature_names.len(),
            )?,
            line_temporal: read_matrix_f16(
                directory.join("line_temporal.f16"),
                manifest.line_count,
                manifest.temporal_bins * TEMPORAL_CHANNELS,
            )?,
            serves_src: read_u32(directory.join("serves_src.u32"))?,
            serves_dst: read_u32(directory.join("serves_dst.u32"))?,
            transit_src: read_u32(directory.join("transit_src.u32"))?,
            transit_dst: read_u32(directory.join("transit_dst.u32"))?,
            transit_line: read_u32(directory.join("transit_line.u32"))?,
            transit_features: read_matrix_f32(
                directory.join("transit_features.f32"),
                manifest.transit_edge_count,
                manifest.transit_edge_feature_names.len(),
            )?,
            transit_temporal: read_matrix_f16(
                directory.join("transit_temporal.f16"),
                manifest.transit_edge_count,
                manifest.temporal_bins * 2,
            )?,
            transfer_src: read_u32(directory.join("transfer_src.u32"))?,
            transfer_dst: read_u32(directory.join("transfer_dst.u32"))?,
            transfer_features: read_matrix_f32(
                directory.join("transfer_features.f32"),
                manifest.transfer_edge_count,
                manifest.transfer_feature_names.len(),
            )?,
            interchange_src: read_u32(directory.join("interchange_src.u32"))?,
            interchange_dst: read_u32(directory.join("interchange_dst.u32"))?,
            pattern_offsets: read_u32(directory.join("pattern_offsets.u32"))?,
            pattern_stops: read_u32(directory.join("pattern_stops.u32"))?,
            pattern_lines: read_u32(directory.join("pattern_lines.u32"))?,
            pattern_directions: read_u32(directory.join("pattern_directions.u32"))?,
            pattern_trip_counts: read_u32(directory.join("pattern_trip_counts.u32"))?,
            pattern_stop_features: read_matrix_f32(
                directory.join("pattern_stop_features.f32"),
                manifest.pattern_stop_count,
                manifest.pattern_stop_feature_names.len(),
            )?,
            pattern_segment_features: read_matrix_f32(
                directory.join("pattern_segment_features.f32"),
                manifest.pattern_segment_count,
                manifest.pattern_segment_feature_names.len(),
            )?,
            line_names: lookup_lines
                .iter()
                .map(|line| line.display_name.clone())
                .collect(),
            manifest,
        };
        graph.validate()?;
        Ok(graph)
    }
}

struct PatternArrays {
    offsets: Vec<u32>,
    stops: Vec<u32>,
    lines: Vec<u32>,
    directions: Vec<u32>,
    trip_counts: Vec<u32>,
    stop_features: FeatureMatrix,
    segment_features: FeatureMatrix,
}

fn pattern_arrays(network: &CompiledNetwork, transit_features: &FeatureMatrix) -> PatternArrays {
    let edge_indices: BTreeMap<(u32, u32, u32), usize> = network
        .transit_edges
        .iter()
        .enumerate()
        .map(|(index, edge)| ((edge.from.0, edge.to.0, edge.line.0), index))
        .collect();
    let mut offsets = vec![0_u32];
    let mut stops = Vec::new();
    let mut lines = Vec::with_capacity(network.patterns.len());
    let mut directions = Vec::with_capacity(network.patterns.len());
    let mut trip_counts = Vec::with_capacity(network.patterns.len());
    let mut stop_rows = Vec::new();
    let mut segment_rows = Vec::new();

    for pattern in &network.patterns {
        let pattern_stops = &pattern.signature.stops;
        let stop_count = pattern_stops.len().max(1);
        stops.extend(pattern_stops.iter().map(|station| station.0));
        lines.push(pattern.signature.line.0);
        directions.push(
            pattern
                .signature
                .direction_id
                .map(u32::from)
                .unwrap_or(u32::MAX),
        );
        trip_counts.push(pattern.trips.len() as u32);
        for position in 0..pattern_stops.len() {
            stop_rows.push(vec![
                pattern
                    .signature
                    .pickup_types
                    .get(position)
                    .copied()
                    .unwrap_or(0) as f32
                    / 3.0,
                pattern
                    .signature
                    .dropoff_types
                    .get(position)
                    .copied()
                    .unwrap_or(0) as f32
                    / 3.0,
                position as f32 / stop_count.saturating_sub(1).max(1) as f32,
            ]);
        }
        for pair in pattern_stops.windows(2) {
            let row = edge_indices
                .get(&(pair[0].0, pair[1].0, pattern.signature.line.0))
                .map(|index| transit_features.row(*index).to_vec())
                .unwrap_or_else(|| vec![0.0; EDGE_FEATURES]);
            segment_rows.push(row);
        }
        offsets.push(stops.len() as u32);
    }

    PatternArrays {
        offsets,
        stops,
        lines,
        directions,
        trip_counts,
        stop_features: feature_matrix_or_zeros(stop_rows, 3)
            .expect("pattern stop feature rows have a stable width"),
        segment_features: feature_matrix_or_zeros(segment_rows, EDGE_FEATURES)
            .expect("pattern segment feature rows have a stable width"),
    }
}

pub fn station_feature_names() -> Vec<String> {
    [
        "x",
        "y",
        "platform_count",
        "line_count",
        "pattern_count",
        "mode_0",
        "mode_1",
        "mode_2",
        "mode_3",
        "mode_4",
        "first_departure",
        "last_departure",
        "daily_departures",
        "daily_arrivals",
        "transfer_degree",
        "terminal",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

pub fn line_feature_names() -> Vec<String> {
    [
        "mode_0",
        "mode_1",
        "mode_2",
        "mode_3",
        "mode_4",
        "station_count",
        "pattern_count",
        "route_length_metres",
        "end_to_end_distance_metres",
        "branching_factor",
        "service_span_seconds",
        "daily_trip_count_log1p",
        "median_headway_seconds",
        "peak_headway_seconds",
        "off_peak_headway_seconds",
        "transfer_station_count",
        "unique_station_fraction",
        "shared_segment_fraction",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn station_line_sets(network: &CompiledNetwork) -> Vec<BTreeSet<u32>> {
    let mut output = vec![BTreeSet::new(); network.stations.len()];
    for pattern in &network.patterns {
        for station in &pattern.signature.stops {
            output[station.0 as usize].insert(pattern.signature.line.0);
        }
    }
    output
}

fn coordinate_bounds(network: &CompiledNetwork) -> (f64, f64, f64, f64) {
    let coordinates: Vec<(f64, f64)> = network
        .stations
        .iter()
        .filter_map(|station| {
            (station.latitude.is_finite()
                && station.longitude.is_finite()
                && (station.latitude != 0.0 || station.longitude != 0.0))
                .then_some((station.latitude, station.longitude))
        })
        .collect();
    if coordinates.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }
    let min_lat = coordinates
        .iter()
        .map(|(lat, _)| *lat)
        .fold(f64::INFINITY, f64::min);
    let max_lat = coordinates
        .iter()
        .map(|(lat, _)| *lat)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lon = coordinates
        .iter()
        .map(|(_, lon)| *lon)
        .fold(f64::INFINITY, f64::min);
    let max_lon = coordinates
        .iter()
        .map(|(_, lon)| *lon)
        .fold(f64::NEG_INFINITY, f64::max);
    (min_lat, max_lat, min_lon, max_lon)
}

fn normalized_station_coordinate(
    station: &transit_domain::CanonicalStation,
    (min_lat, max_lat, min_lon, max_lon): (f64, f64, f64, f64),
) -> (f32, f32) {
    if !station.latitude.is_finite()
        || !station.longitude.is_finite()
        || (station.latitude == 0.0 && station.longitude == 0.0)
    {
        return (0.0, 0.0);
    }
    let lat_span = (max_lat - min_lat).max(f64::EPSILON);
    let lon_span = (max_lon - min_lon).max(f64::EPSILON);
    (
        ((station.longitude - min_lon) / lon_span).clamp(0.0, 1.0) as f32,
        ((station.latitude - min_lat) / lat_span).clamp(0.0, 1.0) as f32,
    )
}

fn normalized_time(value: u32) -> f32 {
    if value == u32::MAX {
        0.0
    } else {
        value as f32 / (32.0 * 3600.0)
    }
}

fn station_temporal(network: &CompiledNetwork) -> FeatureMatrix {
    let mut values =
        vec![vec![0.0_f32; SERVICE_DAY_BINS * TEMPORAL_CHANNELS]; network.stations.len()];
    let mut active_lines =
        vec![vec![BTreeSet::<u32>::new(); SERVICE_DAY_BINS]; network.stations.len()];
    let mut departures = vec![vec![Vec::<u32>::new(); SERVICE_DAY_BINS]; network.stations.len()];
    for pattern in &network.patterns {
        for trip in &pattern.trips {
            for (position, time) in trip.stop_times.iter().enumerate() {
                let station = pattern.signature.stops[position].0 as usize;
                let departure_bin =
                    ((time.departure / (15 * 60)) as usize).min(SERVICE_DAY_BINS - 1);
                let arrival_bin = ((time.arrival / (15 * 60)) as usize).min(SERVICE_DAY_BINS - 1);
                values[station][departure_bin] += 1.0;
                values[station][SERVICE_DAY_BINS + arrival_bin] += 1.0;
                active_lines[station][departure_bin].insert(pattern.signature.line.0);
                departures[station][departure_bin].push(time.departure);
            }
        }
    }
    for station in 0..network.stations.len() {
        for bin in 0..SERVICE_DAY_BINS {
            values[station][2 * SERVICE_DAY_BINS + bin] = active_lines[station][bin].len() as f32;
            values[station][3 * SERVICE_DAY_BINS + bin] =
                median_gap(&mut departures[station][bin]) as f32;
        }
    }
    FeatureMatrix {
        rows: network.stations.len(),
        cols: SERVICE_DAY_BINS * TEMPORAL_CHANNELS,
        values: values.into_iter().flatten().collect(),
    }
}

fn line_temporal(network: &CompiledNetwork) -> FeatureMatrix {
    let mut values = vec![vec![0.0_f32; SERVICE_DAY_BINS * TEMPORAL_CHANNELS]; network.lines.len()];
    let mut departures = vec![vec![Vec::<u32>::new(); SERVICE_DAY_BINS]; network.lines.len()];
    for pattern in &network.patterns {
        let line = pattern.signature.line.0 as usize;
        for trip in &pattern.trips {
            if let Some(first) = trip.stop_times.first() {
                let bin = ((first.departure / (15 * 60)) as usize).min(SERVICE_DAY_BINS - 1);
                values[line][bin] += 1.0;
                values[line][SERVICE_DAY_BINS + bin] += 1.0;
                departures[line][bin].push(first.departure);
            }
        }
    }
    for line in 0..network.lines.len() {
        for bin in 0..SERVICE_DAY_BINS {
            values[line][2 * SERVICE_DAY_BINS + bin] = departures[line][bin].len() as f32;
            values[line][3 * SERVICE_DAY_BINS + bin] =
                median_gap(&mut departures[line][bin]) as f32;
        }
    }
    FeatureMatrix {
        rows: network.lines.len(),
        cols: SERVICE_DAY_BINS * TEMPORAL_CHANNELS,
        values: values.into_iter().flatten().collect(),
    }
}

fn median_gap(values: &mut [u32]) -> u32 {
    if values.len() < 2 {
        return 0;
    }
    values.sort_unstable();
    let mut gaps: Vec<u32> = values
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    gaps.sort_unstable();
    gaps[gaps.len() / 2]
}

fn feature_matrix_or_zeros(rows: Vec<Vec<f32>>, columns: usize) -> Result<FeatureMatrix> {
    if rows.is_empty() {
        Ok(FeatureMatrix::zeros(0, columns))
    } else {
        FeatureMatrix::from_rows(rows)
    }
}

fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("encoding JSON")?;
    fs::write(path.as_ref(), bytes)
        .with_context(|| format!("writing {}", path.as_ref().display()))?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let bytes =
        fs::read(path.as_ref()).with_context(|| format!("reading {}", path.as_ref().display()))?;
    serde_json::from_slice(&bytes).context("decoding JSON")
}

fn write_u32(path: impl AsRef<Path>, values: &[u32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    fs::write(path.as_ref(), bytes)
        .with_context(|| format!("writing {}", path.as_ref().display()))?;
    Ok(())
}

fn read_u32(path: impl AsRef<Path>) -> Result<Vec<u32>> {
    let bytes =
        fs::read(path.as_ref()).with_context(|| format!("reading {}", path.as_ref().display()))?;
    if bytes.len() % 4 != 0 {
        bail!("u32 array has a partial value: {}", path.as_ref().display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn write_f32(path: impl AsRef<Path>, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    fs::write(path.as_ref(), bytes)
        .with_context(|| format!("writing {}", path.as_ref().display()))?;
    Ok(())
}

fn write_f16(path: impl AsRef<Path>, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend(f16::from_f32(*value).to_le_bytes());
    }
    fs::write(path.as_ref(), bytes)
        .with_context(|| format!("writing {}", path.as_ref().display()))?;
    Ok(())
}

fn read_matrix_f32(path: impl AsRef<Path>, rows: usize, cols: usize) -> Result<FeatureMatrix> {
    let bytes =
        fs::read(path.as_ref()).with_context(|| format!("reading {}", path.as_ref().display()))?;
    if bytes.len() % 4 != 0 {
        bail!(
            "f32 matrix has a partial value: {}",
            path.as_ref().display()
        );
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let matrix = FeatureMatrix { rows, cols, values };
    matrix.validate()?;
    Ok(matrix)
}

fn read_matrix_f16(path: impl AsRef<Path>, rows: usize, cols: usize) -> Result<FeatureMatrix> {
    let bytes =
        fs::read(path.as_ref()).with_context(|| format!("reading {}", path.as_ref().display()))?;
    if bytes.len() % 2 != 0 {
        bail!(
            "f16 matrix has a partial value: {}",
            path.as_ref().display()
        );
    }
    let values: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes(chunk.try_into().expect("two-byte chunk")).to_f32())
        .collect();
    let matrix = FeatureMatrix { rows, cols, values };
    matrix.validate()?;
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use gtfs_compile::{compile, CompileOptions};
    use gtfs_ingest::GtfsFeed;

    #[test]
    fn materializes_and_round_trips_graph_arrays() {
        let feed = GtfsFeed::from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/synthetic-feeds/basic"
        ))
        .unwrap();
        let network = compile(
            &feed,
            &CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap()),
        )
        .unwrap();
        let graph = GraphTensor::from_network(&network).unwrap();
        assert_eq!(graph.manifest.station_count, 5);
        assert_eq!(
            graph.station_temporal.cols,
            SERVICE_DAY_BINS * TEMPORAL_CHANNELS
        );
        assert_eq!(graph.line_features.cols, line_feature_names().len());
        assert!(!graph.serves_src.is_empty());
        assert_eq!(
            graph.pattern_offsets.len(),
            graph.manifest.pattern_count + 1
        );
        assert_eq!(
            graph.pattern_stop_features.rows,
            graph.manifest.pattern_stop_count
        );
        assert!(graph.pattern_segment_features.rows > 0);
        graph.validate().unwrap();

        for (station_index, station) in network.stations.iter().enumerate() {
            let temporal = graph.station_temporal.row(station_index);
            let departure_count: f32 = temporal[..SERVICE_DAY_BINS].iter().sum();
            let arrival_count: f32 = temporal[SERVICE_DAY_BINS..2 * SERVICE_DAY_BINS]
                .iter()
                .sum();
            assert_eq!(departure_count, station.daily_departures as f32);
            assert_eq!(arrival_count, station.daily_arrivals as f32);
        }

        let directory = tempfile::tempdir().unwrap();
        graph.save(directory.path(), &network).unwrap();
        let loaded = GraphTensor::load(directory.path()).unwrap();
        assert_eq!(loaded.manifest.snapshot_id, graph.manifest.snapshot_id);
        assert_eq!(loaded.line_names, graph.line_names);
        assert_eq!(loaded.serves_src, graph.serves_src);
        assert_eq!(loaded.pattern_offsets, graph.pattern_offsets);
        assert_eq!(loaded.pattern_stops, graph.pattern_stops);
        assert_eq!(
            loaded.pattern_segment_features.values,
            graph.pattern_segment_features.values
        );
        assert_eq!(
            loaded.station_features.values,
            graph.station_features.values
        );
        assert!(
            (loaded.station_temporal.values[0] - graph.station_temporal.values[0]).abs() < 0.01
        );
    }

    #[test]
    fn preserves_feature_widths_for_empty_relations() {
        let feed = GtfsFeed::from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/synthetic-feeds/basic"
        ))
        .unwrap();
        let mut network = compile(
            &feed,
            &CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap()),
        )
        .unwrap();
        network.transfers.clear();
        let graph = GraphTensor::from_network(&network).unwrap();
        assert_eq!(graph.transfer_features.rows, 0);
        assert_eq!(graph.transfer_features.cols, TRANSFER_FEATURES);
        graph.validate().unwrap();
    }
}
