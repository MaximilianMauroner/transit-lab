//! Feed registry and immutable raw-feed acquisition.

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeedSpec {
    pub id: String,
    pub display_name: String,
    pub landing_page: String,
    #[serde(alias = "direct_download_url")]
    pub download_url: Option<String>,
    pub geographical_scope: String,
    pub licence: Option<String>,
}

/// Built-in metadata deliberately keeps landing pages separate from direct
/// download URLs. Both agencies rotate download URLs, while the source page
/// remains the auditable reference. Callers may provide a current URL.
pub fn builtin_feeds() -> Vec<FeedSpec> {
    vec![
        FeedSpec {
            id: "berlin".into(),
            display_name: "Berlin / VBB GTFS".into(),
            landing_page: "https://unternehmen.vbb.de/en/digital-services/datasets/".into(),
            download_url: Some("https://unternehmen.vbb.de/gtfs".into()),
            geographical_scope: "Berlin and Brandenburg (VBB region)".into(),
            licence: Some("See the VBB dataset licence".into()),
        },
        FeedSpec {
            id: "paris".into(),
            display_name: "Paris / Île-de-France Mobilités GTFS".into(),
            landing_page: "https://transport.data.gouv.fr/resources/80921".into(),
            download_url: Some("https://eu.ftp.opendatasoft.com/stif/GTFS/IDFM-gtfs.zip".into()),
            geographical_scope: "Île-de-France region".into(),
            licence: Some("Licence Mobilité".into()),
        },
        FeedSpec {
            id: "vienna".into(),
            display_name: "Vienna GTFS".into(),
            landing_page: "https://www.data.gv.at/datasets/ab4a73b6-1c2d-42e1-b4d9-049e04889cf0"
                .into(),
            download_url: Some(
                "https://www.wienerlinien.at/ogd_realtime/doku/ogd/gtfs/gtfs.zip".into(),
            ),
            geographical_scope: "Vienna feed scope".into(),
            licence: Some("See the data.gv.at dataset licence".into()),
        },
        FeedSpec {
            id: "boston".into(),
            display_name: "Boston / MBTA GTFS".into(),
            landing_page: "https://www.mbta.com/developers/gtfs".into(),
            download_url: Some("https://cdn.mbta.com/MBTA_GTFS.zip".into()),
            geographical_scope: "MBTA service area".into(),
            licence: Some("See the MBTA open data terms".into()),
        },
        FeedSpec {
            id: "new-york".into(),
            display_name: "New York City Subway GTFS".into(),
            landing_page: "https://www.mta.info/developers".into(),
            download_url: Some("https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip".into()),
            geographical_scope: "New York City subway network".into(),
            licence: Some("See the MTA developer data terms".into()),
        },
        FeedSpec {
            id: "vancouver".into(),
            display_name: "Vancouver / TransLink GTFS".into(),
            landing_page:
                "https://www.translink.ca/about-us/doing-business-with-translink/open-data".into(),
            download_url: Some("https://gtfs-static.translink.ca/gtfs/google_transit.zip".into()),
            geographical_scope: "Metro Vancouver".into(),
            licence: Some("See the TransLink open data terms".into()),
        },
        FeedSpec {
            id: "vbb".into(),
            display_name: "VBB region GTFS".into(),
            landing_page: "https://unternehmen.vbb.de/en/digital-services/datasets/".into(),
            download_url: None,
            geographical_scope: "VBB region".into(),
            licence: Some("See the VBB dataset licence".into()),
        },
    ]
}

pub fn feed_by_id(id: &str) -> Option<FeedSpec> {
    builtin_feeds().into_iter().find(|feed| feed.id == id)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub feed_id: String,
    pub display_name: String,
    pub source_url: String,
    pub landing_page: String,
    pub downloaded_at: String,
    pub sha256: String,
    pub byte_count: u64,
    pub licence: Option<String>,
    pub geographical_scope: String,
}

pub fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let digest = Sha256::digest(bytes);
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    Ok(output)
}

pub fn fetch_feed(spec: &FeedSpec, url: &str, output_dir: &Path) -> Result<SourceMetadata> {
    if url.trim().is_empty() {
        bail!("a direct GTFS ZIP URL is required for {}", spec.id);
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("transit-lab/0.1")
        .build()
        .context("building GTFS download client")?;
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("downloading {url}"))?;
    let status = response.status();
    if !status.is_success() {
        bail!("GTFS download returned HTTP {status} for {url}");
    }
    let bytes = response.bytes().context("reading GTFS response body")?;
    if !looks_like_zip(&bytes) {
        bail!("GTFS download from {url} did not return a ZIP archive");
    }
    fs::create_dir_all(output_dir).with_context(|| format!("creating {}", output_dir.display()))?;
    let zip_path = output_dir.join("gtfs.zip");
    fs::write(&zip_path, &bytes).with_context(|| format!("writing {}", zip_path.display()))?;

    let digest = Sha256::digest(&bytes);
    let metadata = SourceMetadata {
        feed_id: spec.id.clone(),
        display_name: spec.display_name.clone(),
        source_url: url.to_owned(),
        landing_page: spec.landing_page.clone(),
        downloaded_at: Utc::now().to_rfc3339(),
        sha256: hex_digest(&digest),
        byte_count: bytes.len() as u64,
        licence: spec.licence.clone(),
        geographical_scope: spec.geographical_scope.clone(),
    };
    let metadata_path = output_dir.join("source.json");
    let json = serde_json::to_vec_pretty(&metadata)?;
    fs::write(&metadata_path, json)
        .with_context(|| format!("writing {}", metadata_path.display()))?;
    Ok(metadata)
}

pub fn load_source_metadata(path: &Path) -> Result<SourceMetadata> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).context("decoding source metadata")
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
}

pub fn raw_feed_directory(root: &Path, feed_id: &str, sha256: &str) -> PathBuf {
    root.join(feed_id).join(sha256)
}

#[cfg(test)]
mod tests {
    use super::looks_like_zip;

    #[test]
    fn recognizes_zip_signatures_and_rejects_html() {
        assert!(looks_like_zip(b"PK\x03\x04payload"));
        assert!(looks_like_zip(b"PK\x05\x06"));
        assert!(looks_like_zip(b"PK\x07\x08"));
        assert!(!looks_like_zip(b"<html>not a feed</html>"));
    }
}
