//! Compile a GTFS feed into a deterministic, integer-indexed transit network.

use anyhow::{bail, Context, Result};
use chrono::NaiveDate;
use gtfs_ingest::{GtfsFeed, PathwayRecord, RouteRecord, StopRecord};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use transit_domain::{
    CanonicalLine, CanonicalPattern, CanonicalStation, CanonicalTransfer, CanonicalTransitEdge,
    CanonicalTrip, CompiledNetwork, LineIndex, LineInterchange, NetworkSnapshotDescriptor,
    PatternSignature, SnapshotManifest, StationIndex, StationMergeEvidence, StationMergeMethod,
    StopTime, SERVICE_DAY_BINS,
};
use transit_spatial::{
    bearing_radians, coordinate_or_zero, distance_metres, name_similarity, normalize_name,
    SpatialPoint,
};

pub const COMPILER_VERSION: &str = "transit-lab-compiler-v1";

/// Reproducible city/network scope applied before compilation. The optional
/// GeoJSON boundary is intentionally parsed here instead of introducing a
/// heavyweight GIS dependency: GTFS stops are points and a deterministic
/// point-in-polygon test is sufficient for the compiler boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScopeDefinition {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub boundary: Option<String>,
    #[serde(default)]
    pub buffer_km: f32,
    #[serde(default)]
    pub modes: Vec<String>,
    #[serde(default = "default_minimum_stops_inside")]
    pub minimum_stops_inside: usize,
    #[serde(skip)]
    polygons: Vec<Vec<Vec<(f64, f64)>>>,
    #[serde(skip)]
    boundary_bytes: Vec<u8>,
}

fn default_minimum_stops_inside() -> usize {
    2
}

impl Default for ScopeDefinition {
    fn default() -> Self {
        Self {
            name: "unscoped".into(),
            description: String::new(),
            boundary: None,
            buffer_km: 0.0,
            modes: Vec::new(),
            minimum_stops_inside: default_minimum_stops_inside(),
            polygons: Vec::new(),
            boundary_bytes: Vec::new(),
        }
    }
}

impl ScopeDefinition {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading scope {}", path.display()))?;
        let mut definition: Self = if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("yaml") | Some("yml")
        ) {
            serde_yaml::from_slice(&bytes).context("decoding scope YAML")?
        } else {
            serde_json::from_slice(&bytes).context("decoding scope JSON")?
        };
        if definition.name.trim().is_empty() {
            definition.name = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("scope")
                .to_owned();
        }
        if definition.buffer_km.is_sign_negative() || !definition.buffer_km.is_finite() {
            bail!("scope buffer_km must be finite and non-negative");
        }
        if definition.minimum_stops_inside == 0 {
            definition.minimum_stops_inside = 1;
        }
        definition.modes = definition
            .modes
            .iter()
            .map(|mode| normalize_scope_mode(mode))
            .filter(|mode| !mode.is_empty())
            .collect();
        if let Some(boundary) = definition.boundary.clone() {
            let boundary_path = if Path::new(&boundary).is_absolute() {
                PathBuf::from(boundary)
            } else {
                path.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(boundary)
            };
            definition.boundary_bytes = fs::read(&boundary_path)
                .with_context(|| format!("reading scope boundary {}", boundary_path.display()))?;
            definition.polygons = parse_geojson_polygons(&definition.boundary_bytes)?;
            if definition.polygons.is_empty() {
                bail!(
                    "scope boundary {} contains no Polygon geometry",
                    boundary_path.display()
                );
            }
        }
        Ok(definition)
    }

    pub fn fingerprint(&self) -> [u8; 32] {
        let config = serde_json::json!({
            "name": self.name,
            "description": self.description,
            "boundary": self.boundary,
            "buffer_km": self.buffer_km,
            "modes": self.modes,
            "minimum_stops_inside": self.minimum_stops_inside,
            "boundary_sha256": Sha256::digest(&self.boundary_bytes).to_vec()
        });
        let digest =
            Sha256::digest(serde_json::to_vec(&config).expect("scope definition is serializable"));
        let mut output = [0_u8; 32];
        output.copy_from_slice(&digest);
        output
    }

    pub fn display_name(&self) -> String {
        if self.buffer_km > 0.0 {
            format!("{} (+{} km buffer)", self.name, self.buffer_km)
        } else {
            self.name.clone()
        }
    }

    /// Filter routes, trips, stop times, and transfer references while keeping
    /// the source hash intact. The scope fingerprint in CompileOptions makes
    /// the resulting snapshot distinct from the unscoped feed.
    pub fn apply(&self, feed: &GtfsFeed) -> Result<GtfsFeed> {
        let route_ids: BTreeSet<String> = feed
            .routes
            .iter()
            .filter(|route| {
                self.modes.is_empty()
                    || self
                        .modes
                        .iter()
                        .any(|mode| mode_matches(route.route_type.as_deref(), mode))
            })
            .map(|route| route.route_id.clone())
            .collect();
        let stops = feed
            .stops
            .iter()
            .map(|stop| (stop.stop_id.as_str(), stop))
            .collect::<HashMap<_, _>>();
        let mut kept_trip_ids = BTreeSet::new();
        let mut kept_stop_ids = BTreeSet::new();
        let mut stop_times = Vec::new();
        let mut rows_by_trip = HashMap::<&str, Vec<&gtfs_ingest::StopTimeRecord>>::new();
        for row in &feed.stop_times {
            rows_by_trip
                .entry(row.trip_id.as_str())
                .or_default()
                .push(row);
        }
        for trip in &feed.trips {
            if !route_ids.contains(&trip.route_id) {
                continue;
            }
            let rows = rows_by_trip
                .get(trip.trip_id.as_str())
                .cloned()
                .unwrap_or_default();
            let inside_count = rows
                .iter()
                .filter(|row| {
                    stops
                        .get(row.stop_id.as_str())
                        .is_some_and(|stop| self.contains_stop(stop))
                })
                .count();
            if inside_count < self.minimum_stops_inside {
                continue;
            }
            kept_trip_ids.insert(trip.trip_id.clone());
            // A regional feed can contain a qualifying route whose trip runs
            // well outside the requested city. Keep only the stop-time rows
            // in the boundary/buffer. Parent stations are added below so
            // canonical station construction still has the hierarchy it
            // needs, without letting out-of-scope platforms leak into the
            // compiled network.
            for row in rows.into_iter().filter(|row| {
                stops
                    .get(row.stop_id.as_str())
                    .is_some_and(|stop| self.contains_stop(stop))
            }) {
                kept_stop_ids.insert(row.stop_id.clone());
                stop_times.push(row.clone());
            }
        }
        // Preserve parent station rows needed by canonical station merging.
        let mut parent = true;
        while parent {
            parent = false;
            for stop_id in kept_stop_ids.clone() {
                if let Some(value) = stops
                    .get(stop_id.as_str())
                    .and_then(|stop| stop.parent_station.as_deref())
                {
                    parent = kept_stop_ids.insert(value.to_owned()) || parent;
                }
            }
        }
        let trips = feed
            .trips
            .iter()
            .filter(|trip| kept_trip_ids.contains(&trip.trip_id))
            .cloned()
            .collect::<Vec<_>>();
        let routes = feed
            .routes
            .iter()
            .filter(|route| {
                route_ids.contains(&route.route_id)
                    && trips.iter().any(|trip| trip.route_id == route.route_id)
            })
            .cloned()
            .collect::<Vec<_>>();
        let stops = feed
            .stops
            .iter()
            .filter(|stop| kept_stop_ids.contains(&stop.stop_id))
            .cloned()
            .collect::<Vec<_>>();
        let service_ids = trips
            .iter()
            .map(|trip| trip.service_id.as_str())
            .collect::<BTreeSet<_>>();
        let calendars = feed
            .calendars
            .iter()
            .filter(|calendar| service_ids.contains(calendar.service_id.as_str()))
            .cloned()
            .collect();
        let calendar_dates = feed
            .calendar_dates
            .iter()
            .filter(|date| service_ids.contains(date.service_id.as_str()))
            .cloned()
            .collect();
        let transfers = feed
            .transfers
            .iter()
            .filter(|edge| {
                kept_stop_ids.contains(&edge.from_stop_id)
                    && kept_stop_ids.contains(&edge.to_stop_id)
            })
            .cloned()
            .collect();
        let pathways = feed
            .pathways
            .iter()
            .filter(|edge| {
                kept_stop_ids.contains(&edge.from_stop_id)
                    && kept_stop_ids.contains(&edge.to_stop_id)
            })
            .cloned()
            .collect();
        if trips.is_empty() {
            bail!("scope {} selected no trips", self.name);
        }
        let mut scoped = feed.clone();
        scoped.routes = routes;
        scoped.trips = trips;
        scoped.stops = stops;
        scoped.stop_times = stop_times;
        scoped.calendars = calendars;
        scoped.calendar_dates = calendar_dates;
        scoped.transfers = transfers;
        scoped.pathways = pathways;
        Ok(scoped)
    }

    fn contains_stop(&self, stop: &gtfs_ingest::StopRecord) -> bool {
        let (Some(latitude), Some(longitude)) = (
            stop.stop_lat
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok()),
            stop.stop_lon
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok()),
        ) else {
            return false;
        };
        if self.polygons.is_empty() {
            return true;
        }
        let buffer_lat = f64::from(self.buffer_km) / 111.0;
        let buffer_lon = buffer_lat / latitude.to_radians().cos().abs().max(0.1);
        self.polygons.iter().any(|polygon| {
            let inside = point_in_ring(
                longitude,
                latitude,
                polygon.first().map(Vec::as_slice).unwrap_or(&[]),
            );
            let inside_hole = polygon
                .iter()
                .skip(1)
                .any(|ring| point_in_ring(longitude, latitude, ring));
            if inside && !inside_hole {
                return true;
            }
            self.buffer_km > 0.0
                && polygon
                    .iter()
                    .any(|ring| point_near_ring(longitude, latitude, ring, buffer_lon, buffer_lat))
        })
    }
}

fn normalize_scope_mode(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn mode_matches(route_type: Option<&str>, wanted: &str) -> bool {
    let mode = match route_type
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(3)
    {
        0 | 5 | 6 | 7 => "tram",
        1 => "subway",
        2 | 12 => "suburban_rail",
        3 => "bus",
        4 => "ferry",
        11 => "trolleybus",
        _ => "other",
    };
    mode == wanted || (wanted == "rail" && mode == "suburban_rail")
}

type Polygon = Vec<Vec<(f64, f64)>>;
type Polygons = Vec<Polygon>;

fn parse_geojson_polygons(bytes: &[u8]) -> Result<Polygons> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("decoding scope GeoJSON")?;
    let mut output = Vec::new();
    collect_geojson_polygons(&value, &mut output)?;
    Ok(output)
}

fn collect_geojson_polygons(value: &serde_json::Value, output: &mut Polygons) -> Result<()> {
    match value.get("type").and_then(serde_json::Value::as_str) {
        Some("FeatureCollection") => {
            for feature in value
                .get("features")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
            {
                collect_geojson_polygons(feature, output)?;
            }
        }
        Some("Feature") => {
            if let Some(geometry) = value.get("geometry") {
                collect_geojson_polygons(geometry, output)?;
            }
        }
        Some("Polygon") => {
            let rings = value
                .get("coordinates")
                .and_then(serde_json::Value::as_array)
                .context("GeoJSON Polygon has no coordinates")?;
            let mut polygon = Vec::new();
            for ring in rings {
                let points = ring
                    .as_array()
                    .context("GeoJSON ring is not an array")?
                    .iter()
                    .map(|point| {
                        let pair = point.as_array().context("GeoJSON point is not an array")?;
                        Ok((
                            pair.first()
                                .and_then(serde_json::Value::as_f64)
                                .context("GeoJSON longitude missing")?,
                            pair.get(1)
                                .and_then(serde_json::Value::as_f64)
                                .context("GeoJSON latitude missing")?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if points.len() >= 3 {
                    polygon.push(points);
                }
            }
            if !polygon.is_empty() {
                output.push(polygon);
            }
        }
        Some("MultiPolygon") => {
            let polygons = value
                .get("coordinates")
                .and_then(serde_json::Value::as_array)
                .context("GeoJSON MultiPolygon has no coordinates")?;
            for coordinates in polygons {
                collect_geojson_polygons(
                    &serde_json::json!({"type": "Polygon", "coordinates": coordinates}),
                    output,
                )?;
            }
        }
        Some(other) => bail!("unsupported GeoJSON geometry type {other}"),
        None => bail!("GeoJSON object has no type"),
    }
    Ok(())
}

fn point_in_ring(x: f64, y: f64, ring: &[(f64, f64)]) -> bool {
    let mut inside = false;
    for index in 0..ring.len() {
        let (x1, y1) = ring[index];
        let (x2, y2) = ring[(index + 1) % ring.len()];
        if ((y1 > y) != (y2 > y)) && x < (x2 - x1) * (y - y1) / (y2 - y1) + x1 {
            inside = !inside;
        }
    }
    inside
}

fn point_near_ring(x: f64, y: f64, ring: &[(f64, f64)], max_x: f64, max_y: f64) -> bool {
    ring.iter()
        .any(|(px, py)| (x - px).abs() <= max_x && (y - py).abs() <= max_y)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineGroupingPolicy {
    #[serde(default)]
    pub route_to_canonical: HashMap<String, String>,
    #[serde(default = "default_line_grouping_version")]
    pub version: String,
}

impl Default for LineGroupingPolicy {
    fn default() -> Self {
        Self {
            route_to_canonical: HashMap::new(),
            version: default_line_grouping_version(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LineMergeRule {
    pub canonical_line: String,
    #[serde(default)]
    pub route_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LineSplitRule {
    pub route_id: String,
    #[serde(default)]
    pub by_pattern: Vec<PatternLineRule>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PatternLineRule {
    pub canonical_line: String,
    #[serde(default)]
    pub patterns: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ExtendedLineGroupingPolicy {
    #[serde(default)]
    route_to_canonical: HashMap<String, String>,
    #[serde(default)]
    merge: Vec<LineMergeRule>,
    #[serde(default)]
    split: Vec<LineSplitRule>,
    #[serde(default = "default_line_grouping_version")]
    version: String,
}

fn default_line_grouping_version() -> String {
    "route-agency-short-name-mode-v1".into()
}

impl LineGroupingPolicy {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let value: ExtendedLineGroupingPolicy = if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("yaml") | Some("yml")
        ) {
            serde_yaml::from_slice(&bytes).context("decoding line grouping YAML")?
        } else {
            serde_json::from_slice(&bytes).context("decoding line grouping JSON")?
        };
        let mut policy = Self {
            route_to_canonical: value.route_to_canonical,
            version: value.version,
        };
        for rule in &value.merge {
            for route_id in &rule.route_ids {
                policy
                    .route_to_canonical
                    .entry(route_id.clone())
                    .or_insert_with(|| rule.canonical_line.clone());
            }
        }
        for rule in &value.split {
            for pattern in &rule.by_pattern {
                for identifier in &pattern.patterns {
                    policy.route_to_canonical.insert(
                        split_override_key(&rule.route_id, identifier),
                        pattern.canonical_line.clone(),
                    );
                }
            }
        }
        Ok(policy)
    }

    fn key_for_route(&self, route: &RouteRecord) -> String {
        if let Some(canonical) = self.canonical_for_route(&route.route_id) {
            return format!("manual:{canonical}");
        }
        let agency = route.agency_id.as_deref().unwrap_or_default().trim();
        let label = route
            .route_short_name
            .as_deref()
            .or(route.route_long_name.as_deref())
            .unwrap_or(route.route_id.as_str())
            .trim();
        let mode = route
            .route_type
            .as_deref()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(3);
        format!("feed:{agency}:{label}:{mode}")
    }

    fn display_for_route(&self, route: &RouteRecord) -> String {
        self.canonical_for_route(&route.route_id)
            .or_else(|| route.route_short_name.clone())
            .or_else(|| route.route_long_name.clone())
            .unwrap_or_else(|| route.route_id.clone())
    }

    fn canonical_for_route(&self, route_id: &str) -> Option<String> {
        self.route_to_canonical.get(route_id).cloned()
    }

    fn key_for_trip(&self, route: &RouteRecord, trip: &gtfs_ingest::TripRecord) -> String {
        for identifier in trip
            .shape_id
            .as_deref()
            .into_iter()
            .chain(std::iter::once(trip.trip_id.as_str()))
        {
            if let Some(canonical) = self
                .route_to_canonical
                .get(&split_override_key(&route.route_id, identifier))
            {
                return format!("manual:{canonical}");
            }
        }
        self.key_for_route(route)
    }

    fn display_for_trip(&self, route: &RouteRecord, trip: &gtfs_ingest::TripRecord) -> String {
        for identifier in trip
            .shape_id
            .as_deref()
            .into_iter()
            .chain(std::iter::once(trip.trip_id.as_str()))
        {
            if let Some(canonical) = self
                .route_to_canonical
                .get(&split_override_key(&route.route_id, identifier))
            {
                return canonical.clone();
            }
        }
        self.display_for_route(route)
    }
}

fn split_override_key(route_id: &str, pattern: &str) -> String {
    format!("\u{0}split:{route_id}:{pattern}")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompileOptions {
    pub service_date: NaiveDate,
    pub geographical_scope: String,
    pub scope_hash: [u8; 32],
    pub transfer_policy_version: String,
    pub line_grouping_policy: LineGroupingPolicy,
    pub compiler_version: String,
    pub station_merge_radius_metres: f32,
    pub exact_name_radius_metres: f32,
    pub fuzzy_name_radius_metres: f32,
    pub fuzzy_similarity_threshold: f32,
    pub source_name: String,
    pub licence: Option<String>,
    pub downloaded_at: Option<String>,
}

impl CompileOptions {
    pub fn for_date(service_date: NaiveDate) -> Self {
        let geographical_scope = "unspecified feed scope".to_owned();
        Self {
            service_date,
            scope_hash: hash_scope(&geographical_scope),
            geographical_scope,
            transfer_policy_version: "explicit-transfers-pathways-conservative-inference-v1".into(),
            line_grouping_policy: LineGroupingPolicy {
                version: default_line_grouping_version(),
                ..LineGroupingPolicy::default()
            },
            compiler_version: COMPILER_VERSION.into(),
            station_merge_radius_metres: 150.0,
            exact_name_radius_metres: 150.0,
            fuzzy_name_radius_metres: 90.0,
            fuzzy_similarity_threshold: 0.92,
            source_name: "unknown GTFS feed".into(),
            licence: None,
            downloaded_at: None,
        }
    }

    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.geographical_scope = scope.into();
        self.scope_hash = hash_scope(&self.geographical_scope);
        self
    }

    pub fn with_scope_definition(mut self, scope: &ScopeDefinition) -> Self {
        self.geographical_scope = scope.display_name();
        self.scope_hash = scope.fingerprint();
        self
    }

    pub fn with_source_name(mut self, source_name: impl Into<String>) -> Self {
        self.source_name = source_name.into();
        self
    }
}

pub fn hash_scope(scope: &str) -> [u8; 32] {
    let digest = Sha256::digest(scope.as_bytes());
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}

#[derive(Clone)]
struct CandidateStop {
    id: String,
    name: String,
    normalized_name: String,
    point: SpatialPoint,
    has_coordinates: bool,
    parent_station: Option<String>,
    location_type: u8,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(length: usize) -> Self {
        Self {
            parent: (0..length).collect(),
            rank: vec![0; length],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) -> usize {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return left_root;
        }
        if self.rank[left_root] < self.rank[right_root]
            || (self.rank[left_root] == self.rank[right_root] && left_root > right_root)
        {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        left_root
    }
}

pub fn compile(feed: &GtfsFeed, options: &CompileOptions) -> Result<CompiledNetwork> {
    if !feed.validation.is_valid() {
        bail!(
            "cannot compile an invalid GTFS feed ({} errors)",
            feed.validation.errors.len()
        );
    }
    let active_trips = feed.active_trips(options.service_date)?;
    if active_trips.is_empty() {
        bail!("no active trips for {}", options.service_date);
    }
    let trip_stop_times = feed.stop_times_for_trip();
    let active_trip_ids: BTreeSet<String> = active_trips
        .iter()
        .map(|trip| trip.trip_id.clone())
        .collect();
    let (mut stations, station_by_stop, station_evidence) =
        build_stations(feed, &trip_stop_times, &active_trip_ids, options)?;

    let route_by_id: HashMap<&str, &RouteRecord> = feed
        .routes
        .iter()
        .map(|route| (route.route_id.as_str(), route))
        .collect();
    let mut trip_line_key = HashMap::<String, String>::new();
    let mut line_builders = BTreeMap::<String, LineBuilder>::new();
    for trip in &active_trips {
        let route = route_by_id
            .get(trip.route_id.as_str())
            .with_context(|| format!("active trip {} has no route", trip.trip_id))?;
        let key = options.line_grouping_policy.key_for_trip(route, trip);
        let mode = route
            .route_type
            .as_deref()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(3);
        let entry = line_builders
            .entry(key.clone())
            .or_insert_with(|| LineBuilder {
                display_name: options.line_grouping_policy.display_for_trip(route, trip),
                agency_key: route.agency_id.clone().unwrap_or_default(),
                mode,
                raw_route_ids: BTreeSet::new(),
            });
        entry.raw_route_ids.insert(route.route_id.clone());
        trip_line_key.insert(trip.trip_id.clone(), key);
    }

    let mut lines = Vec::with_capacity(line_builders.len());
    let mut line_by_key = HashMap::new();
    for (index, (key, builder)) in line_builders.iter().enumerate() {
        let line_index = LineIndex(index as u32);
        line_by_key.insert(key.clone(), line_index);
        lines.push(CanonicalLine {
            index: line_index,
            canonical_id: format!("line:{key}"),
            display_name: builder.display_name.clone(),
            agency_key: builder.agency_key.clone(),
            mode: builder.mode,
            raw_route_ids: builder.raw_route_ids.iter().cloned().collect(),
            station_count: 0,
            pattern_count: 0,
            route_length_metres: 0.0,
            end_to_end_distance_metres: 0.0,
            branching_factor: 0.0,
            service_span_seconds: 0,
            daily_trip_count: 0,
            median_headway_seconds: 0.0,
            peak_headway_seconds: 0.0,
            off_peak_headway_seconds: 0.0,
            transfer_station_count: 0,
            unique_station_fraction: 0.0,
            shared_segment_fraction: 0.0,
        });
    }

    let mut grouped_patterns: HashMap<PatternSignature, Vec<CanonicalTrip>> = HashMap::new();
    for trip in active_trips {
        let line_key = trip_line_key
            .get(&trip.trip_id)
            .with_context(|| format!("line key missing for trip {}", trip.trip_id))?;
        let line = *line_by_key
            .get(line_key)
            .with_context(|| format!("line index missing for {line_key}"))?;
        let rows = trip_stop_times
            .get(trip.trip_id.as_str())
            .with_context(|| format!("active trip {} has no stop_times", trip.trip_id))?;
        let mut station_times = Vec::<(StationIndex, StopTime)>::new();
        let mut previous_sequence = None;
        let mut previous_departure = None;
        for row in rows {
            let (arrival, departure, sequence, pickup_type, dropoff_type) =
                GtfsFeed::parse_stop_time(row)?;
            if previous_sequence.is_some_and(|previous| sequence <= previous) {
                bail!(
                    "trip {} has non-increasing stop_sequence at {}",
                    trip.trip_id,
                    sequence
                );
            }
            if previous_departure.is_some_and(|previous| arrival < previous) {
                bail!(
                    "trip {} has non-monotonic stop times at sequence {}",
                    trip.trip_id,
                    sequence
                );
            }
            previous_sequence = Some(sequence);
            previous_departure = Some(departure);
            let station = station_by_stop
                .get(&row.stop_id)
                .copied()
                .with_context(|| format!("stop {} has no canonical station", row.stop_id))?;
            if let Some((last_station, last_time)) = station_times.last_mut() {
                if *last_station == station {
                    last_time.arrival = last_time.arrival.min(arrival);
                    last_time.departure = last_time.departure.max(departure);
                    last_time.pickup_type = last_time.pickup_type.min(pickup_type);
                    last_time.dropoff_type = last_time.dropoff_type.min(dropoff_type);
                    continue;
                }
            }
            let _ = sequence;
            station_times.push((
                station,
                StopTime {
                    arrival,
                    departure,
                    pickup_type,
                    dropoff_type,
                },
            ));
        }
        if station_times.len() < 2 {
            continue;
        }
        let signature = PatternSignature {
            line,
            direction_id: trip
                .direction_id
                .as_deref()
                .and_then(|value| value.parse().ok()),
            stops: station_times.iter().map(|(station, _)| *station).collect(),
            pickup_types: station_times
                .iter()
                .map(|(_, time)| time.pickup_type)
                .collect(),
            dropoff_types: station_times
                .iter()
                .map(|(_, time)| time.dropoff_type)
                .collect(),
        };
        grouped_patterns
            .entry(signature)
            .or_default()
            .push(CanonicalTrip {
                trip_id: trip.trip_id.clone(),
                service_id: trip.service_id.clone(),
                stop_times: station_times.into_iter().map(|(_, time)| time).collect(),
            });
    }

    let mut pattern_groups: Vec<_> = grouped_patterns.into_iter().collect();
    pattern_groups.sort_by(|left, right| {
        left.0
            .line
            .cmp(&right.0.line)
            .then_with(|| left.0.direction_id.cmp(&right.0.direction_id))
            .then_with(|| left.0.stops.cmp(&right.0.stops))
            .then_with(|| left.0.pickup_types.cmp(&right.0.pickup_types))
    });
    let mut patterns = Vec::with_capacity(pattern_groups.len());
    for (index, (signature, mut trips)) in pattern_groups.into_iter().enumerate() {
        trips.sort_by(|left, right| {
            left.stop_times
                .first()
                .map(|time| time.departure)
                .cmp(&right.stop_times.first().map(|time| time.departure))
                .then_with(|| left.trip_id.cmp(&right.trip_id))
        });
        patterns.push(CanonicalPattern {
            index: (index as u32).into(),
            signature,
            trips,
        });
    }
    if patterns.is_empty() {
        bail!("no usable two-stop patterns were compiled");
    }

    let mut line_stations = vec![BTreeSet::<u32>::new(); lines.len()];
    let mut line_departures = vec![Vec::<u32>::new(); lines.len()];
    let mut line_first_departure = vec![u32::MAX; lines.len()];
    let mut line_last_arrival = vec![0_u32; lines.len()];
    let mut station_lines = vec![BTreeSet::<u32>::new(); stations.len()];
    let mut station_patterns = vec![BTreeSet::<u32>::new(); stations.len()];
    let mut endpoint_stations = vec![false; stations.len()];

    for pattern in &patterns {
        let line = pattern.signature.line;
        let line_slot = line.0 as usize;
        lines[line_slot].pattern_count += 1;
        let stops = &pattern.signature.stops;
        for (position, station) in stops.iter().enumerate() {
            line_stations[line_slot].insert(station.0);
            station_lines[station.0 as usize].insert(line.0);
            station_patterns[station.0 as usize].insert(pattern.index.0);
            if position == 0 || position + 1 == stops.len() {
                endpoint_stations[station.0 as usize] = true;
            }
        }
        for trip in &pattern.trips {
            lines[line_slot].daily_trip_count += 1;
            if let Some(first) = trip.stop_times.first() {
                line_departures[line_slot].push(first.departure);
                line_first_departure[line_slot] =
                    line_first_departure[line_slot].min(first.departure);
            }
            if let Some(last) = trip.stop_times.last() {
                line_last_arrival[line_slot] = line_last_arrival[line_slot].max(last.arrival);
            }
            for (position, time) in trip.stop_times.iter().enumerate() {
                let station = stops[position];
                let target = &mut stations[station.0 as usize];
                target.first_departure = target.first_departure.min(time.departure);
                target.last_departure = target.last_departure.max(time.arrival);
                target.daily_departures = target.daily_departures.saturating_add(1);
                target.daily_arrivals = target.daily_arrivals.saturating_add(1);
            }
        }
    }

    let transit_edges = build_transit_edges(&patterns, &stations);
    let line_segment_sets = line_segments(&patterns, lines.len());
    let segment_lines = segment_line_members(&line_segment_sets);
    for line_slot in 0..lines.len() {
        let station_count = line_stations[line_slot].len();
        lines[line_slot].station_count = station_count as u32;
        lines[line_slot].service_span_seconds = if line_first_departure[line_slot] == u32::MAX {
            0
        } else {
            line_last_arrival[line_slot].saturating_sub(line_first_departure[line_slot])
        };
        lines[line_slot].median_headway_seconds = median_headway(&mut line_departures[line_slot]);
        let peak: Vec<u32> = line_departures[line_slot]
            .iter()
            .copied()
            .filter(|time| in_peak(*time))
            .collect();
        let off_peak: Vec<u32> = line_departures[line_slot]
            .iter()
            .copied()
            .filter(|time| !in_peak(*time))
            .collect();
        lines[line_slot].peak_headway_seconds = headway_from_slice(&peak);
        lines[line_slot].off_peak_headway_seconds = headway_from_slice(&off_peak);
        lines[line_slot].branching_factor = patterns
            .iter()
            .filter(|pattern| pattern.signature.line.0 as usize == line_slot)
            .map(|pattern| {
                pattern
                    .signature
                    .stops
                    .first()
                    .zip(pattern.signature.stops.last())
                    .map(|(start, end)| u32::from(start != end))
                    .unwrap_or(0)
            })
            .sum::<u32>() as f32
            / lines[line_slot].pattern_count.max(1) as f32;
        let unique_stations = line_stations[line_slot]
            .iter()
            .filter(|station| station_lines[**station as usize].len() == 1)
            .count();
        let transfer_stations = line_stations[line_slot]
            .iter()
            .filter(|station| station_lines[**station as usize].len() > 1)
            .count();
        lines[line_slot].unique_station_fraction =
            unique_stations as f32 / station_count.max(1) as f32;
        lines[line_slot].transfer_station_count = transfer_stations as u32;
        let segments = &line_segment_sets[line_slot];
        let shared_segments = segments
            .iter()
            .filter(|segment| {
                segment_lines
                    .get(segment)
                    .map(|members| members.len() > 1)
                    .unwrap_or(false)
            })
            .count();
        lines[line_slot].shared_segment_fraction =
            shared_segments as f32 / segments.len().max(1) as f32;
        lines[line_slot].route_length_metres = segments
            .iter()
            .filter_map(|(from, to)| {
                station_distance(&stations[*from as usize], &stations[*to as usize])
            })
            .sum();
        lines[line_slot].end_to_end_distance_metres = patterns
            .iter()
            .filter(|pattern| pattern.signature.line.0 as usize == line_slot)
            .filter_map(|pattern| {
                pattern
                    .signature
                    .stops
                    .first()
                    .zip(pattern.signature.stops.last())
                    .and_then(|(from, to)| {
                        station_distance(&stations[from.0 as usize], &stations[to.0 as usize])
                    })
            })
            .fold(0.0_f32, f32::max);
    }

    for (station_slot, station) in stations.iter_mut().enumerate() {
        station.line_count = station_lines[station_slot].len() as u32;
        station.pattern_count = station_patterns[station_slot].len() as u32;
        station.terminal = endpoint_stations[station_slot];
    }

    let transfers = build_transfers(feed, &station_by_stop, &stations)?;
    let mut transfer_neighbours = vec![BTreeSet::<u32>::new(); stations.len()];
    for transfer in &transfers {
        transfer_neighbours[transfer.from.0 as usize].insert(transfer.to.0);
        transfer_neighbours[transfer.to.0 as usize].insert(transfer.from.0);
    }
    for (index, station) in stations.iter_mut().enumerate() {
        station.transfer_degree = transfer_neighbours[index].len() as u32;
    }
    let interchanges = build_interchanges(&station_lines);

    let descriptor = NetworkSnapshotDescriptor {
        feed_hashes: vec![feed.source_hash],
        service_date: options.service_date,
        scope_hash: options.scope_hash,
        compiler_version: options.compiler_version.clone(),
        transfer_policy_version: options.transfer_policy_version.clone(),
        line_grouping_version: options.line_grouping_policy.version.clone(),
    };
    let snapshot_id = descriptor.snapshot_id();
    let manifest = SnapshotManifest {
        snapshot_id: snapshot_id.clone(),
        descriptor,
        source_name: options.source_name.clone(),
        source_path: feed.source_path.to_string_lossy().into_owned(),
        downloaded_at: options.downloaded_at.clone(),
        licence: options.licence.clone(),
        geographical_scope: options.geographical_scope.clone(),
        transfer_policy: options.transfer_policy_version.clone(),
        line_grouping_policy: options.line_grouping_policy.version.clone(),
        validation: feed.validation.clone(),
    };
    let network = CompiledNetwork {
        snapshot_id,
        manifest,
        stations,
        lines,
        patterns,
        transit_edges,
        transfers,
        interchanges,
        station_merge_evidence: station_evidence,
        stop_to_station: station_by_stop,
    };
    network.validate_indices()?;
    Ok(network)
}

#[derive(Clone)]
struct LineBuilder {
    display_name: String,
    agency_key: String,
    mode: u16,
    raw_route_ids: BTreeSet<String>,
}

type StationBuild = (
    Vec<CanonicalStation>,
    BTreeMap<String, StationIndex>,
    Vec<StationMergeEvidence>,
);

fn build_stations(
    feed: &GtfsFeed,
    trip_stop_times: &HashMap<&str, Vec<&gtfs_ingest::StopTimeRecord>>,
    active_trip_ids: &BTreeSet<String>,
    options: &CompileOptions,
) -> Result<StationBuild> {
    let stop_by_id: HashMap<&str, &StopRecord> = feed
        .stops
        .iter()
        .map(|stop| (stop.stop_id.as_str(), stop))
        .collect();
    let mut selected = BTreeSet::<String>::new();
    for (trip_id, rows) in trip_stop_times {
        if !active_trip_ids.contains(*trip_id) {
            continue;
        }
        for row in rows {
            selected.insert(row.stop_id.clone());
        }
    }
    for transfer in &feed.transfers {
        selected.insert(transfer.from_stop_id.clone());
        selected.insert(transfer.to_stop_id.clone());
    }
    for pathway in &feed.pathways {
        selected.insert(pathway.from_stop_id.clone());
        selected.insert(pathway.to_stop_id.clone());
    }
    selected.retain(|stop_id| stop_by_id.contains_key(stop_id.as_str()));
    let mut added_parents = true;
    while added_parents {
        added_parents = false;
        let current: Vec<String> = selected.iter().cloned().collect();
        for stop_id in current {
            if let Some(parent) = stop_by_id
                .get(stop_id.as_str())
                .and_then(|stop| stop.parent_station.as_deref())
            {
                if stop_by_id.contains_key(parent) && selected.insert(parent.to_owned()) {
                    added_parents = true;
                }
            }
        }
    }
    if selected.is_empty() {
        bail!("no active stops were found");
    }

    let candidates: Vec<CandidateStop> = selected
        .iter()
        .filter_map(|stop_id| {
            stop_by_id
                .get(stop_id.as_str())
                .map(|stop| (stop_id, *stop))
        })
        .map(|(stop_id, stop)| {
            let point = coordinate_or_zero(stop.stop_lat.as_deref(), stop.stop_lon.as_deref());
            CandidateStop {
                id: stop_id.clone(),
                name: stop.stop_name.clone().unwrap_or_else(|| stop_id.clone()),
                normalized_name: normalize_name(stop.stop_name.as_deref()),
                point,
                has_coordinates: stop
                    .stop_lat
                    .as_deref()
                    .and_then(|value| value.parse::<f64>().ok())
                    .is_some()
                    && stop
                        .stop_lon
                        .as_deref()
                        .and_then(|value| value.parse::<f64>().ok())
                        .is_some(),
                parent_station: stop.parent_station.clone(),
                location_type: stop
                    .location_type
                    .as_deref()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0),
            }
        })
        .collect();
    let candidate_index: HashMap<&str, usize> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.id.as_str(), index))
        .collect();
    let mut union_find = UnionFind::new(candidates.len());
    let mut evidence = Vec::new();

    for (index, candidate) in candidates.iter().enumerate() {
        if let Some(parent) = candidate
            .parent_station
            .as_deref()
            .and_then(|id| candidate_index.get(id).copied())
        {
            union_find.union(index, parent);
            evidence.push(StationMergeEvidence {
                method: StationMergeMethod::ParentStation,
                confidence: 1.0,
                distance_metres: None,
                source_stop_ids: vec![candidate.id.clone(), candidates[parent].id.clone()],
            });
        }
    }

    for pathway in &feed.pathways {
        let Some(&from) = candidate_index.get(pathway.from_stop_id.as_str()) else {
            continue;
        };
        let Some(&to) = candidate_index.get(pathway.to_stop_id.as_str()) else {
            continue;
        };
        if from == to || !candidates[from].has_coordinates || !candidates[to].has_coordinates {
            continue;
        }
        let distance = distance_metres(candidates[from].point, candidates[to].point);
        if distance <= options.station_merge_radius_metres {
            union_find.union(from, to);
            evidence.push(StationMergeEvidence {
                method: StationMergeMethod::Pathway,
                confidence: 1.0,
                distance_metres: Some(distance),
                source_stop_ids: vec![candidates[from].id.clone(), candidates[to].id.clone()],
            });
        }
    }
    for transfer in &feed.transfers {
        if transfer_type(transfer)? == 3 {
            continue;
        }
        let Some(&from) = candidate_index.get(transfer.from_stop_id.as_str()) else {
            continue;
        };
        let Some(&to) = candidate_index.get(transfer.to_stop_id.as_str()) else {
            continue;
        };
        if from == to || !candidates[from].has_coordinates || !candidates[to].has_coordinates {
            continue;
        }
        let distance = distance_metres(candidates[from].point, candidates[to].point);
        let same_name = !candidates[from].normalized_name.is_empty()
            && candidates[from].normalized_name == candidates[to].normalized_name;
        if distance <= options.station_merge_radius_metres && same_name {
            union_find.union(from, to);
            evidence.push(StationMergeEvidence {
                method: StationMergeMethod::ExplicitTransfer,
                confidence: 0.95,
                distance_metres: Some(distance),
                source_stop_ids: vec![candidates[from].id.clone(), candidates[to].id.clone()],
            });
        }
    }

    let mut grid = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.has_coordinates || candidate.normalized_name.is_empty() {
            continue;
        }
        let cell = coordinate_cell(candidate.point);
        for lat_cell in (cell.0 - 1)..=(cell.0 + 1) {
            for lon_cell in (cell.1 - 1)..=(cell.1 + 1) {
                for other in grid.get(&(lat_cell, lon_cell)).into_iter().flatten() {
                    if candidates[*other].normalized_name != candidate.normalized_name {
                        continue;
                    }
                    let distance = distance_metres(candidate.point, candidates[*other].point);
                    if distance <= options.exact_name_radius_metres {
                        union_find.union(index, *other);
                        evidence.push(StationMergeEvidence {
                            method: StationMergeMethod::ExactNameRadius,
                            confidence: 0.98,
                            distance_metres: Some(distance),
                            source_stop_ids: vec![
                                candidate.id.clone(),
                                candidates[*other].id.clone(),
                            ],
                        });
                    }
                }
            }
        }
        grid.entry(cell).or_default().push(index);
    }
    let mut fuzzy_grid = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.has_coordinates || candidate.normalized_name.is_empty() {
            continue;
        }
        let cell = coordinate_cell(candidate.point);
        for lat_cell in (cell.0 - 1)..=(cell.0 + 1) {
            for lon_cell in (cell.1 - 1)..=(cell.1 + 1) {
                for other in fuzzy_grid.get(&(lat_cell, lon_cell)).into_iter().flatten() {
                    if candidates[*other].normalized_name == candidate.normalized_name {
                        continue;
                    }
                    let distance = distance_metres(candidate.point, candidates[*other].point);
                    let similarity = name_similarity(
                        &candidate.normalized_name,
                        &candidates[*other].normalized_name,
                    );
                    if distance <= options.fuzzy_name_radius_metres
                        && similarity >= options.fuzzy_similarity_threshold
                    {
                        union_find.union(index, *other);
                        evidence.push(StationMergeEvidence {
                            method: StationMergeMethod::FuzzyNameRadius,
                            confidence: similarity * 0.9,
                            distance_metres: Some(distance),
                            source_stop_ids: vec![
                                candidate.id.clone(),
                                candidates[*other].id.clone(),
                            ],
                        });
                    }
                }
            }
        }
        fuzzy_grid.entry(cell).or_default().push(index);
    }

    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..candidates.len() {
        let root = union_find.find(index);
        groups.entry(root).or_default().push(index);
    }
    let mut ordered_groups: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    ordered_groups.sort_by_key(|(root, _)| candidates[*root].id.clone());

    let mut stations = Vec::with_capacity(ordered_groups.len());
    let mut station_by_stop = BTreeMap::new();
    for (station_index, (_, members)) in ordered_groups.into_iter().enumerate() {
        let mut member_ids: Vec<String> = members
            .iter()
            .map(|index| candidates[*index].id.clone())
            .collect();
        member_ids.sort();
        let preferred = members
            .iter()
            .min_by_key(|index| {
                (
                    candidates[**index].location_type != 1,
                    candidates[**index].id.clone(),
                )
            })
            .copied()
            .unwrap_or(members[0]);
        let coordinate_members: Vec<&CandidateStop> = members
            .iter()
            .map(|index| &candidates[*index])
            .filter(|candidate| candidate.has_coordinates)
            .collect();
        let (latitude, longitude) = if coordinate_members.is_empty() {
            (0.0, 0.0)
        } else {
            (
                coordinate_members
                    .iter()
                    .map(|candidate| candidate.point.latitude)
                    .sum::<f64>()
                    / coordinate_members.len() as f64,
                coordinate_members
                    .iter()
                    .map(|candidate| candidate.point.longitude)
                    .sum::<f64>()
                    / coordinate_members.len() as f64,
            )
        };
        let merge_confidence = evidence
            .iter()
            .filter(|item| {
                item.source_stop_ids
                    .iter()
                    .any(|id| member_ids.binary_search(id).is_ok())
            })
            .map(|item| item.confidence)
            .fold(1.0_f32, f32::min);
        let index = StationIndex(station_index as u32);
        for member in &member_ids {
            station_by_stop.insert(member.clone(), index);
        }
        stations.push(CanonicalStation {
            index,
            canonical_id: format!("station:{}", candidates[preferred].id),
            name: candidates[preferred].name.clone(),
            latitude,
            longitude,
            raw_stop_ids: member_ids,
            merge_confidence,
            platform_count: members.len().max(1) as u32,
            line_count: 0,
            pattern_count: 0,
            first_departure: u32::MAX,
            last_departure: 0,
            daily_departures: 0,
            daily_arrivals: 0,
            transfer_degree: 0,
            terminal: false,
        });
    }
    Ok((stations, station_by_stop, evidence))
}

fn coordinate_cell(point: SpatialPoint) -> (i32, i32) {
    (
        (point.latitude * 1_000.0).floor() as i32,
        (point.longitude * 1_000.0).floor() as i32,
    )
}

fn build_transit_edges(
    patterns: &[CanonicalPattern],
    stations: &[CanonicalStation],
) -> Vec<CanonicalTransitEdge> {
    #[derive(Default)]
    struct EdgeAccumulator {
        runtimes: Vec<u32>,
        departures_by_bin: Vec<f32>,
        runtimes_by_bin: Vec<Vec<u32>>,
        relative_position: f32,
    }
    let mut accumulators = HashMap::<(u32, u32, u32), EdgeAccumulator>::new();
    for pattern in patterns {
        for trip in &pattern.trips {
            for position in 0..pattern.signature.stops.len().saturating_sub(1) {
                let from = pattern.signature.stops[position];
                let to = pattern.signature.stops[position + 1];
                let departure = trip.stop_times[position].departure;
                let arrival = trip.stop_times[position + 1].arrival;
                if arrival < departure || from == to {
                    continue;
                }
                let entry = accumulators
                    .entry((from.0, to.0, pattern.signature.line.0))
                    .or_insert_with(|| EdgeAccumulator {
                        departures_by_bin: vec![0.0; SERVICE_DAY_BINS],
                        runtimes_by_bin: vec![Vec::new(); SERVICE_DAY_BINS],
                        relative_position: position as f32
                            / pattern.signature.stops.len().saturating_sub(1).max(1) as f32,
                        ..EdgeAccumulator::default()
                    });
                entry.runtimes.push(arrival - departure);
                let bin = ((departure / (15 * 60)) as usize).min(SERVICE_DAY_BINS - 1);
                entry.departures_by_bin[bin] += 1.0;
                entry.runtimes_by_bin[bin].push(arrival - departure);
            }
        }
    }
    let mut edges = Vec::with_capacity(accumulators.len());
    let mut keys: Vec<_> = accumulators.keys().copied().collect();
    keys.sort_unstable();
    for (from, to, line) in keys {
        let entry = accumulators
            .remove(&(from, to, line))
            .expect("edge accumulator exists");
        let bearing = bearing_radians(
            station_point(&stations[from as usize]),
            station_point(&stations[to as usize]),
        );
        let median_runtime_by_bin: Vec<f32> = entry
            .runtimes_by_bin
            .into_iter()
            .map(|mut values| median_u32(&mut values) as f32)
            .collect();
        let mut runtimes = entry.runtimes.clone();
        edges.push(CanonicalTransitEdge {
            from: StationIndex(from),
            to: StationIndex(to),
            line: LineIndex(line),
            distance_metres: station_distance(&stations[from as usize], &stations[to as usize])
                .unwrap_or(0.0),
            median_travel_seconds: median_u32(&mut runtimes),
            minimum_travel_seconds: entry.runtimes.iter().copied().min().unwrap_or(0),
            active_trip_count: entry.runtimes.len() as u32,
            relative_position: entry.relative_position,
            bearing_sin: bearing.sin() as f32,
            bearing_cos: bearing.cos() as f32,
            departures_by_bin: entry.departures_by_bin,
            median_runtime_by_bin,
        });
    }
    edges
}

fn build_transfers(
    feed: &GtfsFeed,
    station_by_stop: &BTreeMap<String, StationIndex>,
    stations: &[CanonicalStation],
) -> Result<Vec<CanonicalTransfer>> {
    let mut transfers = BTreeMap::<(u32, u32), CanonicalTransfer>::new();
    for row in &feed.transfers {
        let Some(&from) = station_by_stop.get(&row.from_stop_id) else {
            continue;
        };
        let Some(&to) = station_by_stop.get(&row.to_stop_id) else {
            continue;
        };
        if from == to {
            continue;
        }
        if transfer_type(row)? == 3 {
            continue;
        }
        let seconds = row
            .min_transfer_time
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.parse::<u32>())
            .transpose()
            .with_context(|| {
                format!(
                    "invalid transfer time for {} -> {}",
                    row.from_stop_id, row.to_stop_id
                )
            })?
            .unwrap_or(0);
        let candidate = CanonicalTransfer {
            from,
            to,
            minimum_transfer_seconds: seconds,
            walking_distance_metres: station_distance(
                &stations[from.0 as usize],
                &stations[to.0 as usize],
            ),
            explicit: true,
            confidence: 1.0,
            same_physical_station: false,
        };
        transfers
            .entry((from.0, to.0))
            .and_modify(|current| {
                if candidate.minimum_transfer_seconds < current.minimum_transfer_seconds {
                    *current = candidate.clone();
                }
            })
            .or_insert(candidate);
    }
    for pathway in &feed.pathways {
        add_pathway_transfer(&mut transfers, pathway, station_by_stop, stations)?;
        if pathway
            .is_bidirectional
            .as_deref()
            .map(|value| value.trim() == "1")
            .unwrap_or(false)
        {
            let reverse = PathwayRecord {
                pathway_id: pathway.pathway_id.clone(),
                from_stop_id: pathway.to_stop_id.clone(),
                to_stop_id: pathway.from_stop_id.clone(),
                pathway_mode: pathway.pathway_mode.clone(),
                is_bidirectional: pathway.is_bidirectional.clone(),
                traversal_time: pathway.traversal_time.clone(),
                length: pathway.length.clone(),
            };
            add_pathway_transfer(&mut transfers, &reverse, station_by_stop, stations)?;
        }
    }
    Ok(transfers.into_values().collect())
}

fn transfer_type(row: &gtfs_ingest::TransferRecord) -> Result<u8> {
    let value = row
        .transfer_type
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<u8>())
        .transpose()
        .with_context(|| {
            format!(
                "invalid transfer type for {} -> {}",
                row.from_stop_id, row.to_stop_id
            )
        })
        .map(|value| value.unwrap_or(0))?;
    if value > 5 {
        bail!(
            "invalid transfer type {} for {} -> {}",
            value,
            row.from_stop_id,
            row.to_stop_id
        );
    }
    Ok(value)
}

fn add_pathway_transfer(
    transfers: &mut BTreeMap<(u32, u32), CanonicalTransfer>,
    pathway: &PathwayRecord,
    station_by_stop: &BTreeMap<String, StationIndex>,
    stations: &[CanonicalStation],
) -> Result<()> {
    let Some(&from) = station_by_stop.get(&pathway.from_stop_id) else {
        return Ok(());
    };
    let Some(&to) = station_by_stop.get(&pathway.to_stop_id) else {
        return Ok(());
    };
    if from == to {
        return Ok(());
    }
    let seconds = pathway
        .traversal_time
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<u32>())
        .transpose()
        .with_context(|| format!("invalid pathway time {}", pathway.pathway_id))?
        .unwrap_or(0);
    let walking_distance = pathway
        .length
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.parse::<f32>().ok())
        .or_else(|| station_distance(&stations[from.0 as usize], &stations[to.0 as usize]));
    transfers
        .entry((from.0, to.0))
        .or_insert(CanonicalTransfer {
            from,
            to,
            minimum_transfer_seconds: seconds,
            walking_distance_metres: walking_distance,
            explicit: true,
            confidence: 1.0,
            same_physical_station: false,
        });
    Ok(())
}

fn build_interchanges(station_lines: &[BTreeSet<u32>]) -> Vec<LineInterchange> {
    let mut counts = BTreeMap::<(u32, u32), u32>::new();
    for lines in station_lines {
        let values: Vec<u32> = lines.iter().copied().collect();
        for (position, from) in values.iter().enumerate() {
            for to in values.iter().skip(position + 1) {
                *counts.entry((*from, *to)).or_default() += 1;
                *counts.entry((*to, *from)).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .map(|((from, to), shared_station_count)| LineInterchange {
            from: LineIndex(from),
            to: LineIndex(to),
            shared_station_count,
        })
        .collect()
}

fn line_segments(patterns: &[CanonicalPattern], line_count: usize) -> Vec<BTreeSet<(u32, u32)>> {
    let mut output = vec![BTreeSet::new(); line_count];
    for pattern in patterns {
        for pair in pattern.signature.stops.windows(2) {
            output[pattern.signature.line.0 as usize].insert((pair[0].0, pair[1].0));
        }
    }
    output
}

fn segment_line_members(segments: &[BTreeSet<(u32, u32)>]) -> HashMap<(u32, u32), BTreeSet<usize>> {
    let mut output = HashMap::new();
    for (line, line_segments) in segments.iter().enumerate() {
        for segment in line_segments {
            output
                .entry(*segment)
                .or_insert_with(BTreeSet::new)
                .insert(line);
        }
    }
    output
}

fn station_point(station: &CanonicalStation) -> SpatialPoint {
    SpatialPoint {
        latitude: station.latitude,
        longitude: station.longitude,
    }
}

fn station_distance(from: &CanonicalStation, to: &CanonicalStation) -> Option<f32> {
    if (from.latitude == 0.0 && from.longitude == 0.0)
        || (to.latitude == 0.0 && to.longitude == 0.0)
    {
        None
    } else {
        Some(distance_metres(station_point(from), station_point(to)))
    }
}

fn in_peak(seconds: u32) -> bool {
    let hour = (seconds / 3600) % 24;
    (7..=10).contains(&hour) || (16..=19).contains(&hour)
}

fn median_headway(values: &mut [u32]) -> f32 {
    headway_from_slice(values)
}

fn headway_from_slice(values: &[u32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let mut gaps: Vec<u32> = values
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    median_u32(&mut gaps) as f32
}

fn median_u32(values: &mut [u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

pub fn save_snapshot(network: &CompiledNetwork, directory: &Path) -> Result<()> {
    network.validate_indices()?;
    if network.snapshot_id != network.manifest.snapshot_id
        || network.manifest.descriptor.snapshot_id() != network.snapshot_id
    {
        bail!("cannot save a snapshot with inconsistent identity");
    }
    fs::create_dir_all(directory).with_context(|| format!("creating {}", directory.display()))?;
    let network_json = serde_json::to_vec_pretty(network).context("encoding compiled network")?;
    let manifest_json =
        serde_json::to_vec_pretty(&network.manifest).context("encoding manifest")?;
    ensure_immutable_compatible(&directory.join("network.json"), &network_json)?;
    ensure_immutable_compatible(&directory.join("manifest.json"), &manifest_json)?;
    write_immutable(&directory.join("network.json"), &network_json)?;
    write_immutable(&directory.join("manifest.json"), &manifest_json)?;
    Ok(())
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            if let Err(error) = file.write_all(bytes) {
                let _ = fs::remove_file(path);
                return Err(error).with_context(|| format!("writing {}", path.display()));
            }
            file.flush()
                .with_context(|| format!("flushing {}", path.display()))?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            if existing == bytes {
                Ok(())
            } else {
                bail!(
                    "refusing to overwrite immutable snapshot file {}",
                    path.display()
                );
            }
        }
        Err(error) => Err(error).with_context(|| format!("creating {}", path.display())),
    }
}

fn ensure_immutable_compatible(path: &Path, bytes: &[u8]) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let existing = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if existing == bytes {
        Ok(())
    } else {
        bail!(
            "refusing to overwrite immutable snapshot file {}",
            path.display()
        );
    }
}

pub fn load_snapshot(directory: &Path) -> Result<CompiledNetwork> {
    let bytes = fs::read(directory.join("network.json"))
        .with_context(|| format!("reading {}/network.json", directory.display()))?;
    let network: CompiledNetwork =
        serde_json::from_slice(&bytes).context("decoding compiled network")?;
    let manifest_bytes = fs::read(directory.join("manifest.json"))
        .with_context(|| format!("reading {}/manifest.json", directory.display()))?;
    let disk_manifest: SnapshotManifest =
        serde_json::from_slice(&manifest_bytes).context("decoding snapshot manifest")?;
    if serde_json::to_vec(&disk_manifest)? != serde_json::to_vec(&network.manifest)? {
        bail!("snapshot manifest does not match network.json");
    }
    if network.snapshot_id != network.manifest.snapshot_id
        || network.manifest.descriptor.snapshot_id() != network.snapshot_id
    {
        bail!("snapshot identity does not match its manifest descriptor");
    }
    network.validate_indices()?;
    Ok(network)
}

pub fn validation_summary(feed: &GtfsFeed) -> serde_json::Value {
    serde_json::json!({
        "valid": feed.validation.is_valid(),
        "errors": feed.validation.errors,
        "warnings": feed.validation.warnings,
        "checked_files": feed.validation.checked_files,
        "row_counts": feed.validation.row_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use gtfs_ingest::{GtfsFeed, StopTimeRecord, TripRecord};

    fn fixture() -> GtfsFeed {
        GtfsFeed::from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/synthetic-feeds/basic"
        ))
        .expect("synthetic GTFS fixture loads")
    }

    #[test]
    fn compiles_canonical_entities_and_preserves_late_times() {
        let feed = fixture();
        assert!(feed.validation.is_valid());
        let blue_late = feed
            .stop_times
            .iter()
            .find(|row| row.trip_id == "blue-2")
            .expect("late trip exists");
        assert_eq!(GtfsFeed::parse_stop_time(blue_late).unwrap().0, 90_900);

        let options = CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap())
            .with_scope("synthetic fixture")
            .with_source_name("fixture");
        let network = compile(&feed, &options).expect("fixture compiles");
        assert_eq!(network.stations.len(), 5);
        assert_eq!(network.lines.len(), 3);
        assert_eq!(network.patterns.len(), 3);
        assert!(network.transit_edges.len() >= 6);
        assert!(network.lines.iter().any(|line| line.daily_trip_count == 2));
        network.validate_indices().unwrap();
    }

    #[test]
    fn snapshot_round_trip_keeps_identity() {
        let feed = fixture();
        let options = CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap())
            .with_scope("synthetic fixture")
            .with_source_name("fixture");
        let network = compile(&feed, &options).unwrap();
        let directory = tempfile::tempdir().unwrap();
        save_snapshot(&network, directory.path()).unwrap();
        let loaded = load_snapshot(directory.path()).unwrap();
        assert_eq!(loaded.snapshot_id, network.snapshot_id);
        assert_eq!(loaded.patterns.len(), network.patterns.len());
        assert_eq!(loaded.stop_to_station, network.stop_to_station);
    }

    #[test]
    fn excludes_stops_used_only_by_inactive_trips() {
        let mut feed = fixture();
        feed.stops.push(StopRecord {
            stop_id: "inactive-stop".into(),
            stop_name: Some("Inactive".into()),
            stop_lat: Some("48.2".into()),
            stop_lon: Some("16.2".into()),
            location_type: Some("0".into()),
            ..StopRecord::default()
        });
        feed.routes.push(RouteRecord {
            route_id: "inactive-route".into(),
            route_short_name: Some("Inactive".into()),
            route_type: Some("3".into()),
            ..RouteRecord::default()
        });
        feed.trips.push(TripRecord {
            route_id: "inactive-route".into(),
            service_id: "never".into(),
            trip_id: "inactive-trip".into(),
            ..TripRecord::default()
        });
        feed.stop_times.extend([
            StopTimeRecord {
                trip_id: "inactive-trip".into(),
                arrival_time: "08:00:00".into(),
                departure_time: "08:00:00".into(),
                stop_id: "inactive-stop".into(),
                stop_sequence: "0".into(),
                ..StopTimeRecord::default()
            },
            StopTimeRecord {
                trip_id: "inactive-trip".into(),
                arrival_time: "08:05:00".into(),
                departure_time: "08:05:00".into(),
                stop_id: "a".into(),
                stop_sequence: "1".into(),
                ..StopTimeRecord::default()
            },
        ]);
        let options = CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap());
        let network = compile(&feed, &options).unwrap();
        assert_eq!(network.stations.len(), 5);
        assert!(!network.stop_to_station.contains_key("inactive-stop"));
    }

    #[test]
    fn snapshot_writes_are_idempotent_but_do_not_overwrite_changes() {
        let feed = fixture();
        let network = compile(
            &feed,
            &CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap()),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        save_snapshot(&network, directory.path()).unwrap();
        save_snapshot(&network, directory.path()).unwrap();

        let mut changed = network.clone();
        changed.manifest.source_name = "changed metadata".into();
        assert!(save_snapshot(&changed, directory.path()).is_err());
    }

    #[test]
    fn loads_documented_merge_and_split_line_rules() {
        let directory = tempfile::tempdir().unwrap();
        let policy_path = directory.path().join("line-policy.yaml");
        std::fs::write(
            &policy_path,
            "version: test-v1\nmerge:\n  - canonical_line: U1\n    route_ids: [raw-a, raw-b]\nsplit:\n  - route_id: raw-c\n    by_pattern:\n      - canonical_line: X1\n        patterns: [shape-x]\n",
        )
        .unwrap();
        let policy = LineGroupingPolicy::from_path(&policy_path).unwrap();
        assert_eq!(
            policy.key_for_route(&RouteRecord {
                route_id: "raw-a".into(),
                ..RouteRecord::default()
            }),
            "manual:U1"
        );
        let route = RouteRecord {
            route_id: "raw-c".into(),
            ..RouteRecord::default()
        };
        let trip = TripRecord {
            trip_id: "trip-c".into(),
            shape_id: Some("shape-x".into()),
            ..TripRecord::default()
        };
        assert_eq!(policy.key_for_trip(&route, &trip), "manual:X1");
    }

    #[test]
    fn scope_trims_regional_trips_and_keeps_parent_stations() {
        let mut feed = fixture();
        feed.routes.push(RouteRecord {
            route_id: "regional".into(),
            route_short_name: Some("Regional".into()),
            route_type: Some("2".into()),
            ..RouteRecord::default()
        });
        feed.trips.push(TripRecord {
            route_id: "regional".into(),
            service_id: "weekday".into(),
            trip_id: "regional-trip".into(),
            ..TripRecord::default()
        });
        for (sequence, stop_id) in [
            "platform_a1",
            "platform_b",
            "platform_c",
            "platform_d",
            "platform_e",
        ]
        .into_iter()
        .enumerate()
        {
            feed.stop_times.push(StopTimeRecord {
                trip_id: "regional-trip".into(),
                arrival_time: format!("08:{:02}:00", sequence * 5),
                departure_time: format!("08:{:02}:00", sequence * 5),
                stop_id: stop_id.into(),
                stop_sequence: sequence.to_string(),
                ..StopTimeRecord::default()
            });
        }

        let directory = tempfile::tempdir().unwrap();
        let boundary = directory.path().join("city.geojson");
        std::fs::write(
            &boundary,
            serde_json::json!({
                "type": "Polygon",
                "coordinates": [[[16.099, 48.099], [16.1025, 48.099], [16.1025, 48.1025], [16.099, 48.1025], [16.099, 48.099]]]
            })
            .to_string(),
        )
        .unwrap();
        let scope_path = directory.path().join("scope.json");
        std::fs::write(
            &scope_path,
            serde_json::json!({
                "name": "city",
                "boundary": "city.geojson",
                "minimum_stops_inside": 2
            })
            .to_string(),
        )
        .unwrap();
        let scope = ScopeDefinition::from_path(&scope_path).unwrap();
        let scoped = scope.apply(&feed).unwrap();

        assert!(scoped
            .trips
            .iter()
            .any(|trip| trip.trip_id == "regional-trip"));
        assert!(!scoped.trips.iter().any(|trip| trip.route_id == "red"));
        assert!(scoped.stop_times.iter().all(|row| {
            ["platform_a1", "platform_a2", "platform_b", "platform_c"]
                .contains(&row.stop_id.as_str())
        }));
        assert_eq!(
            scoped
                .stop_times
                .iter()
                .filter(|row| row.trip_id == "regional-trip")
                .count(),
            3
        );
        assert!(scoped.stops.iter().any(|stop| stop.stop_id == "station_a"));
        assert!(!scoped.stops.iter().any(|stop| stop.stop_id == "station_d"));
        assert!(!scoped.stops.iter().any(|stop| stop.stop_id == "station_e"));
    }
}
