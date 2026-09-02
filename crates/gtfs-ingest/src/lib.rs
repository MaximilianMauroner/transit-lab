//! Streaming-oriented GTFS table ingestion.
//!
//! The loader keeps canonical row types deliberately close to the GTFS
//! spelling. Compilation is the boundary where strings become compact integer
//! indices and feed quirks are resolved.

use anyhow::{bail, Context, Result};
use chrono::{Datelike, NaiveDate, Weekday};
use csv::Trim;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use transit_domain::{parse_gtfs_date, parse_gtfs_time, ValidationReport};
use zip::ZipArchive;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StopRecord {
    pub stop_id: String,
    #[serde(default)]
    pub stop_name: Option<String>,
    #[serde(default)]
    pub stop_lat: Option<String>,
    #[serde(default)]
    pub stop_lon: Option<String>,
    #[serde(default)]
    pub location_type: Option<String>,
    #[serde(default)]
    pub parent_station: Option<String>,
    #[serde(default)]
    pub wheelchair_boarding: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RouteRecord {
    pub route_id: String,
    #[serde(default)]
    pub agency_id: Option<String>,
    #[serde(default)]
    pub route_short_name: Option<String>,
    #[serde(default)]
    pub route_long_name: Option<String>,
    #[serde(default)]
    pub route_type: Option<String>,
    #[serde(default)]
    pub route_color: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TripRecord {
    pub route_id: String,
    pub service_id: String,
    pub trip_id: String,
    #[serde(default)]
    pub trip_headsign: Option<String>,
    #[serde(default)]
    pub direction_id: Option<String>,
    #[serde(default)]
    pub shape_id: Option<String>,
    #[serde(default)]
    pub wheelchair_accessible: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StopTimeRecord {
    pub trip_id: String,
    pub arrival_time: String,
    pub departure_time: String,
    pub stop_id: String,
    pub stop_sequence: String,
    #[serde(default)]
    pub pickup_type: Option<String>,
    #[serde(default)]
    pub drop_off_type: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CalendarRecord {
    pub service_id: String,
    #[serde(default)]
    pub monday: String,
    #[serde(default)]
    pub tuesday: String,
    #[serde(default)]
    pub wednesday: String,
    #[serde(default)]
    pub thursday: String,
    #[serde(default)]
    pub friday: String,
    #[serde(default)]
    pub saturday: String,
    #[serde(default)]
    pub sunday: String,
    pub start_date: String,
    pub end_date: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CalendarDateRecord {
    pub service_id: String,
    pub date: String,
    pub exception_type: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TransferRecord {
    pub from_stop_id: String,
    pub to_stop_id: String,
    #[serde(default)]
    pub transfer_type: Option<String>,
    #[serde(default)]
    pub min_transfer_time: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PathwayRecord {
    pub pathway_id: String,
    pub from_stop_id: String,
    pub to_stop_id: String,
    #[serde(default)]
    pub pathway_mode: Option<String>,
    #[serde(default)]
    pub is_bidirectional: Option<String>,
    #[serde(default)]
    pub traversal_time: Option<String>,
    #[serde(default)]
    pub length: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GtfsFeed {
    pub source_path: PathBuf,
    pub source_hash: [u8; 32],
    pub stops: Vec<StopRecord>,
    pub routes: Vec<RouteRecord>,
    pub trips: Vec<TripRecord>,
    pub stop_times: Vec<StopTimeRecord>,
    pub calendars: Vec<CalendarRecord>,
    pub calendar_dates: Vec<CalendarDateRecord>,
    pub transfers: Vec<TransferRecord>,
    pub pathways: Vec<PathwayRecord>,
    pub validation: ValidationReport,
}

impl GtfsFeed {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut source = FeedSource::open(&path)?;
        let mut validation = ValidationReport::default();

        let stops = read_required(&mut source, "stops.txt", &mut validation)?;
        let routes = read_required(&mut source, "routes.txt", &mut validation)?;
        let trips = read_required(&mut source, "trips.txt", &mut validation)?;
        let stop_times = read_required(&mut source, "stop_times.txt", &mut validation)?;
        let calendars = read_optional(&mut source, "calendar.txt", &mut validation)?;
        let calendar_dates = read_optional(&mut source, "calendar_dates.txt", &mut validation)?;
        let transfers = read_optional(&mut source, "transfers.txt", &mut validation)?;
        let pathways = read_optional(&mut source, "pathways.txt", &mut validation)?;

        let digest = if path.is_file() {
            let digest = Sha256::digest(
                fs::read(&path).with_context(|| format!("hashing {}", path.display()))?,
            );
            let mut output = [0_u8; 32];
            output.copy_from_slice(&digest);
            output
        } else {
            hash_directory(&path)?
        };
        let source_hash = digest;

        validate_references(&stops, &routes, &trips, &stop_times, &mut validation);
        for row in &stop_times {
            if let Err(error) = Self::parse_stop_time(row) {
                validation.errors.push(error.to_string());
            }
        }

        Ok(Self {
            source_path: path,
            source_hash,
            stops,
            routes,
            trips,
            stop_times,
            calendars,
            calendar_dates,
            transfers,
            pathways,
            validation,
        })
    }

    pub fn active_service_ids(&self, date: NaiveDate) -> Result<HashSet<String>> {
        let mut active = HashSet::new();
        for calendar in &self.calendars {
            let start = parse_gtfs_date(&calendar.start_date)?;
            let end = parse_gtfs_date(&calendar.end_date)?;
            if date < start || date > end || !calendar_runs_on(calendar, date.weekday()) {
                continue;
            }
            active.insert(calendar.service_id.clone());
        }

        for exception in &self.calendar_dates {
            if parse_gtfs_date(&exception.date)? != date {
                continue;
            }
            match exception.exception_type.trim() {
                "1" => {
                    active.insert(exception.service_id.clone());
                }
                "2" => {
                    active.remove(&exception.service_id);
                }
                other => bail!("invalid calendar_dates exception_type: {other}"),
            }
        }

        // Some small feeds omit calendar.txt and use a single service_id. In
        // that case retaining all trip services is less surprising than
        // silently producing an empty network.
        if self.calendars.is_empty() && self.calendar_dates.is_empty() {
            active.extend(self.trips.iter().map(|trip| trip.service_id.clone()));
        }
        Ok(active)
    }

    pub fn active_trips(&self, date: NaiveDate) -> Result<Vec<&TripRecord>> {
        let active_services = self.active_service_ids(date)?;
        Ok(self
            .trips
            .iter()
            .filter(|trip| active_services.contains(&trip.service_id))
            .collect())
    }

    pub fn stop_times_for_trip(&self) -> HashMap<&str, Vec<&StopTimeRecord>> {
        let mut grouped: HashMap<&str, Vec<&StopTimeRecord>> = HashMap::new();
        for row in &self.stop_times {
            grouped.entry(row.trip_id.as_str()).or_default().push(row);
        }
        for rows in grouped.values_mut() {
            rows.sort_by_key(|row| row.stop_sequence.parse::<u32>().unwrap_or(u32::MAX));
        }
        grouped
    }

    pub fn parse_stop_time(row: &StopTimeRecord) -> Result<(u32, u32, u32, u8, u8)> {
        let arrival = parse_gtfs_time(&row.arrival_time)
            .with_context(|| format!("trip {} arrival", row.trip_id))?;
        let departure = parse_gtfs_time(&row.departure_time)
            .with_context(|| format!("trip {} departure", row.trip_id))?;
        let sequence: u32 = row.stop_sequence.parse().with_context(|| {
            format!(
                "invalid stop_sequence {} on trip {}",
                row.stop_sequence, row.trip_id
            )
        })?;
        let pickup = optional_type(row.pickup_type.as_deref(), "pickup_type", &row.trip_id)?;
        let dropoff = optional_type(row.drop_off_type.as_deref(), "drop_off_type", &row.trip_id)?;
        if departure < arrival {
            bail!(
                "departure precedes arrival on trip {} at stop {}",
                row.trip_id,
                row.stop_id
            );
        }
        Ok((arrival, departure, sequence, pickup, dropoff))
    }
}

fn optional_type(value: Option<&str>, field: &str, trip_id: &str) -> Result<u8> {
    let Some(value) = value.filter(|value| !value.trim().is_empty()) else {
        return Ok(0);
    };
    let parsed: u8 = value
        .parse()
        .with_context(|| format!("invalid {field} on trip {trip_id}"))?;
    if parsed > 3 {
        bail!("invalid {field} value {parsed} on trip {trip_id}");
    }
    Ok(parsed)
}

fn calendar_runs_on(calendar: &CalendarRecord, weekday: Weekday) -> bool {
    let flag = match weekday {
        Weekday::Mon => &calendar.monday,
        Weekday::Tue => &calendar.tuesday,
        Weekday::Wed => &calendar.wednesday,
        Weekday::Thu => &calendar.thursday,
        Weekday::Fri => &calendar.friday,
        Weekday::Sat => &calendar.saturday,
        Weekday::Sun => &calendar.sunday,
    };
    flag.trim() == "1"
}

fn validate_references(
    stops: &[StopRecord],
    routes: &[RouteRecord],
    trips: &[TripRecord],
    stop_times: &[StopTimeRecord],
    validation: &mut ValidationReport,
) {
    validate_unique_ids(
        stops.iter().map(|row| row.stop_id.as_str()),
        "stop_id",
        validation,
    );
    validate_unique_ids(
        routes.iter().map(|row| row.route_id.as_str()),
        "route_id",
        validation,
    );
    validate_unique_ids(
        trips.iter().map(|row| row.trip_id.as_str()),
        "trip_id",
        validation,
    );
    let stop_ids: HashSet<&str> = stops.iter().map(|row| row.stop_id.as_str()).collect();
    let route_ids: HashSet<&str> = routes.iter().map(|row| row.route_id.as_str()).collect();
    let trip_ids: HashSet<&str> = trips.iter().map(|row| row.trip_id.as_str()).collect();

    for trip in trips {
        if !route_ids.contains(trip.route_id.as_str()) {
            validation.errors.push(format!(
                "trip {} references missing route {}",
                trip.trip_id, trip.route_id
            ));
        }
    }
    for row in stop_times {
        if !trip_ids.contains(row.trip_id.as_str()) {
            validation
                .errors
                .push(format!("stop_time references missing trip {}", row.trip_id));
        }
        if !stop_ids.contains(row.stop_id.as_str()) {
            validation
                .errors
                .push(format!("stop_time references missing stop {}", row.stop_id));
        }
    }
}

fn validate_unique_ids<'a>(
    values: impl IntoIterator<Item = &'a str>,
    field: &str,
    validation: &mut ValidationReport,
) {
    let mut seen = HashSet::new();
    for value in values {
        if value.trim().is_empty() {
            validation.errors.push(format!("{field} must not be empty"));
        } else if !seen.insert(value) {
            validation
                .errors
                .push(format!("duplicate {field}: {value}"));
        }
    }
}

enum FeedSource {
    Directory(PathBuf),
    Zip(ZipArchive<File>),
}

impl FeedSource {
    fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            return Ok(Self::Directory(path.to_path_buf()));
        }
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Ok(Self::Zip(ZipArchive::new(file).with_context(|| {
            format!("opening GTFS ZIP {}", path.display())
        })?))
    }

    fn read(&mut self, name: &str) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Directory(root) => {
                let path = root.join(name);
                if !path.exists() {
                    return Ok(None);
                }
                Ok(Some(
                    fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
                ))
            }
            Self::Zip(archive) => {
                let index = archive
                    .file_names()
                    .position(|entry| entry == name)
                    .or_else(|| {
                        archive
                            .file_names()
                            .position(|entry| entry.rsplit('/').next() == Some(name))
                    });
                let Some(index) = index else {
                    return Ok(None);
                };
                let mut file = archive
                    .by_index(index)
                    .context("reading file from GTFS ZIP")?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)
                    .context("reading GTFS ZIP member")?;
                Ok(Some(bytes))
            }
        }
    }
}

fn read_required<T: DeserializeOwned>(
    source: &mut FeedSource,
    name: &str,
    validation: &mut ValidationReport,
) -> Result<Vec<T>> {
    let Some(bytes) = source.read(name)? else {
        validation
            .errors
            .push(format!("missing required file {name}"));
        bail!("missing required GTFS file {name}");
    };
    let rows = decode_csv(&bytes, name)?;
    validation.checked_files.push(name.to_owned());
    validation.row_counts.insert(name.to_owned(), rows.len());
    Ok(rows)
}

fn read_optional<T: DeserializeOwned>(
    source: &mut FeedSource,
    name: &str,
    validation: &mut ValidationReport,
) -> Result<Vec<T>> {
    let Some(bytes) = source.read(name)? else {
        return Ok(Vec::new());
    };
    let rows = decode_csv(&bytes, name)?;
    validation.checked_files.push(name.to_owned());
    validation.row_counts.insert(name.to_owned(), rows.len());
    Ok(rows)
}

fn hash_directory(path: &Path) -> Result<[u8; 32]> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for file in files {
        let relative = file
            .strip_prefix(path)
            .expect("collected file is below the feed directory")
            .to_string_lossy();
        let bytes = fs::read(&file).with_context(|| format!("hashing {}", file.display()))?;
        hasher.update(relative.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    let digest = hasher.finalize();
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    Ok(output)
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<PathBuf> = fs::read_dir(directory)
        .with_context(|| format!("reading {}", directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            collect_files(&entry, files)?;
        } else if entry.is_file() {
            files.push(entry);
        }
    }
    Ok(())
}

fn decode_csv<T: DeserializeOwned>(bytes: &[u8], name: &str) -> Result<Vec<T>> {
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(Trim::All)
        .from_reader(bytes);
    let mut rows = Vec::new();
    for (row_index, result) in reader.deserialize().enumerate() {
        let row = result.with_context(|| format!("decoding {name} row {}", row_index + 2))?;
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::{write::SimpleFileOptions, ZipWriter};

    #[test]
    fn applies_calendar_date_exceptions() {
        let feed = GtfsFeed {
            source_path: PathBuf::from("synthetic"),
            source_hash: [0; 32],
            stops: Vec::new(),
            routes: Vec::new(),
            trips: vec![TripRecord {
                service_id: "weekday".into(),
                trip_id: "trip".into(),
                ..TripRecord::default()
            }],
            stop_times: Vec::new(),
            calendars: vec![CalendarRecord {
                service_id: "weekday".into(),
                monday: "1".into(),
                start_date: "20260101".into(),
                end_date: "20261231".into(),
                ..CalendarRecord::default()
            }],
            calendar_dates: vec![CalendarDateRecord {
                service_id: "weekday".into(),
                date: "20260907".into(),
                exception_type: "2".into(),
            }],
            transfers: Vec::new(),
            pathways: Vec::new(),
            validation: ValidationReport::default(),
        };
        let date = NaiveDate::from_ymd_opt(2026, 9, 7).unwrap();
        assert!(feed.active_service_ids(date).unwrap().is_empty());
    }

    #[test]
    fn reads_gtfs_tables_from_a_zip_with_nested_paths() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/synthetic-feeds/basic");
        let temporary = tempfile::tempdir().unwrap();
        let zip_path = temporary.path().join("feed.zip");
        let file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(file);
        for name in [
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
        ] {
            writer
                .start_file(format!("feed/{name}"), SimpleFileOptions::default())
                .unwrap();
            writer
                .write_all(&fs::read(fixture.join(name)).unwrap())
                .unwrap();
        }
        writer.finish().unwrap();

        let feed = GtfsFeed::from_path(&zip_path).unwrap();
        assert!(feed.validation.is_valid());
        assert_eq!(feed.stops.len(), 11);
        assert_eq!(
            feed.active_trips(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap())
                .unwrap()
                .len(),
            4
        );
        assert_ne!(feed.source_hash, [0; 32]);
    }
}
