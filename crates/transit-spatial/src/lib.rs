//! Small, deterministic spatial helpers used during station canonicalization.

use transit_domain::CanonicalStation;

#[derive(Clone, Copy, Debug)]
pub struct SpatialPoint {
    pub latitude: f64,
    pub longitude: f64,
}

pub fn distance_metres(a: SpatialPoint, b: SpatialPoint) -> f32 {
    let lat1 = a.latitude.to_radians();
    let lat2 = b.latitude.to_radians();
    let dlat = (b.latitude - a.latitude).to_radians();
    let dlon = (b.longitude - a.longitude).to_radians();
    let haversine =
        (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    (6_371_000.0 * 2.0 * haversine.sqrt().asin()) as f32
}

pub fn bearing_radians(a: SpatialPoint, b: SpatialPoint) -> f64 {
    let lat1 = a.latitude.to_radians();
    let lat2 = b.latitude.to_radians();
    let dlon = (b.longitude - a.longitude).to_radians();
    let y = dlon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    y.atan2(x)
}

pub fn normalize_name(value: Option<&str>) -> String {
    let mut output = String::new();
    for character in value.unwrap_or_default().chars() {
        if character.is_alphanumeric() {
            for lower in character.to_lowercase() {
                output.push(lower);
            }
        }
    }
    output
}

pub fn coordinate_or_zero(latitude: Option<&str>, longitude: Option<&str>) -> SpatialPoint {
    SpatialPoint {
        latitude: latitude.and_then(|value| value.parse().ok()).unwrap_or(0.0),
        longitude: longitude
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.0),
    }
}

pub fn normalized_coordinates(stations: &[CanonicalStation]) -> Vec<[f32; 2]> {
    if stations.is_empty() {
        return Vec::new();
    }
    let min_lat = stations
        .iter()
        .map(|station| station.latitude)
        .fold(f64::INFINITY, f64::min);
    let max_lat = stations
        .iter()
        .map(|station| station.latitude)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lon = stations
        .iter()
        .map(|station| station.longitude)
        .fold(f64::INFINITY, f64::min);
    let max_lon = stations
        .iter()
        .map(|station| station.longitude)
        .fold(f64::NEG_INFINITY, f64::max);
    let lat_span = (max_lat - min_lat).max(f64::EPSILON);
    let lon_span = (max_lon - min_lon).max(f64::EPSILON);
    stations
        .iter()
        .map(|station| {
            [
                ((station.longitude - min_lon) / lon_span) as f32,
                ((station.latitude - min_lat) / lat_span) as f32,
            ]
        })
        .collect()
}

/// A bounded normalized Levenshtein similarity for conservative fuzzy merges.
pub fn name_similarity(left: &str, right: &str) -> f32 {
    if left == right {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let right_chars: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right_chars.len()).collect();
    for (i, left_char) in left.chars().enumerate() {
        let mut current = vec![i + 1; right_chars.len() + 1];
        for (j, right_char) in right_chars.iter().enumerate() {
            let substitution = previous[j] + usize::from(left_char != *right_char);
            current[j + 1] = (substitution.min(previous[j + 1] + 1)).min(current[j] + 1);
        }
        previous = current;
    }
    let distance = *previous.last().unwrap_or(&0) as f32;
    1.0 - distance / left.chars().count().max(right.chars().count()) as f32
}
