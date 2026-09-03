//! Immutable reference-backend embedding caches.
//!
//! Encoding a city graph is the expensive part of most head experiments. This
//! artifact stores the frozen encoder output together with both graph and
//! encoder fingerprints, so a cache can never be silently reused for a
//! different model or snapshot.

use crate::ReferenceRelationalAutoencoder;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use transit_domain::{hex_digest, sha256_bytes};
use transit_graph::GraphTensor;
use transit_model::{Embeddings, MaskSelection};

pub const EMBEDDING_CACHE_SCHEMA_VERSION: &str = "embedding-cache-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedGraphEmbeddings {
    pub snapshot_id: String,
    pub graph_fingerprint: String,
    pub embeddings: Embeddings,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingCache {
    pub schema_version: String,
    pub encoder_fingerprint: String,
    pub entries: Vec<CachedGraphEmbeddings>,
    pub fingerprint: String,
}

impl EmbeddingCache {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != EMBEDDING_CACHE_SCHEMA_VERSION {
            bail!(
                "unsupported embedding cache schema {}; expected {}",
                self.schema_version,
                EMBEDDING_CACHE_SCHEMA_VERSION
            );
        }
        if self.encoder_fingerprint.trim().is_empty() || self.entries.is_empty() {
            bail!("embedding cache has no encoder or entries");
        }
        let mut snapshots = std::collections::BTreeSet::new();
        for entry in &self.entries {
            if entry.snapshot_id.trim().is_empty()
                || entry.graph_fingerprint.trim().is_empty()
                || entry.embeddings.city.is_empty()
                || !snapshots.insert(&entry.snapshot_id)
            {
                bail!("embedding cache contains an invalid or duplicate graph entry");
            }
            if entry.embeddings.station.iter().any(|row| row.is_empty())
                || entry.embeddings.line.iter().any(|row| row.is_empty())
            {
                bail!("embedding cache contains an empty embedding row");
            }
        }
        if self.fingerprint != embedding_cache_fingerprint(self) {
            bail!("embedding cache fingerprint does not match its contents");
        }
        Ok(())
    }

    pub fn entry(&self, snapshot_id: &str) -> Option<&CachedGraphEmbeddings> {
        self.entries
            .iter()
            .find(|entry| entry.snapshot_id == snapshot_id)
    }
}

pub fn encoder_fingerprint(encoder: &ReferenceRelationalAutoencoder) -> Result<String> {
    let encoded = serde_json::to_vec(encoder).context("encoding encoder fingerprint")?;
    Ok(hex_digest(&sha256_bytes(&encoded)))
}

/// Fingerprint graph content that can affect a reference encoder output.
pub fn graph_fingerprint(graph: &GraphTensor) -> Result<String> {
    let value = serde_json::json!({
        "manifest": graph.manifest,
        "lineNames": graph.line_names,
        "lineIdentities": graph.line_identities,
        "stationFeatures": graph.station_features,
        "stationTemporal": graph.station_temporal,
        "lineFeatures": graph.line_features,
        "lineTemporal": graph.line_temporal,
        "servesSrc": graph.serves_src,
        "servesDst": graph.serves_dst,
        "transitSrc": graph.transit_src,
        "transitDst": graph.transit_dst,
        "transitLine": graph.transit_line,
        "transitFeatures": graph.transit_features,
        "transferSrc": graph.transfer_src,
        "transferDst": graph.transfer_dst,
        "transferFeatures": graph.transfer_features,
        "interchangeSrc": graph.interchange_src,
        "interchangeDst": graph.interchange_dst,
        "patternOffsets": graph.pattern_offsets,
        "patternStops": graph.pattern_stops,
        "patternLines": graph.pattern_lines,
        "patternDirections": graph.pattern_directions,
        "patternTripCounts": graph.pattern_trip_counts,
        "patternStopFeatures": graph.pattern_stop_features,
        "patternSegmentFeatures": graph.pattern_segment_features
    });
    Ok(hex_digest(&sha256_bytes(&serde_json::to_vec(&value)?)))
}

pub fn build_embedding_cache(
    encoder: &ReferenceRelationalAutoencoder,
    graphs: &[&GraphTensor],
) -> Result<EmbeddingCache> {
    if graphs.is_empty() {
        bail!("cannot build an embedding cache without graphs");
    }
    let encoder_fingerprint = encoder_fingerprint(encoder)?;
    let mut entries = Vec::with_capacity(graphs.len());
    for graph in graphs {
        let snapshot_id = graph.manifest.snapshot_id.clone();
        if entries
            .iter()
            .any(|entry: &CachedGraphEmbeddings| entry.snapshot_id == snapshot_id)
        {
            bail!("embedding cache graph snapshots must be unique");
        }
        let embeddings = encoder.encode(graph, &MaskSelection::all_unmasked(graph))?;
        entries.push(CachedGraphEmbeddings {
            snapshot_id,
            graph_fingerprint: graph_fingerprint(graph)?,
            embeddings,
        });
    }
    let mut cache = EmbeddingCache {
        schema_version: EMBEDDING_CACHE_SCHEMA_VERSION.into(),
        encoder_fingerprint,
        entries,
        fingerprint: String::new(),
    };
    cache.fingerprint = embedding_cache_fingerprint(&cache);
    Ok(cache)
}

pub fn embedding_cache_fingerprint(cache: &EmbeddingCache) -> String {
    let mut value = cache.clone();
    value.fingerprint.clear();
    let encoded = serde_json::to_vec(&value).expect("embedding cache is serializable");
    hex_digest(&sha256_bytes(&encoded))
}

pub fn save_embedding_cache(path: &Path, cache: &EmbeddingCache) -> Result<()> {
    cache.validate()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let temporary = path.with_extension(format!("cache.tmp-{}", std::process::id()));
    let encoded = serde_json::to_vec_pretty(cache).context("encoding embedding cache")?;
    let mut file =
        File::create(&temporary).with_context(|| format!("creating {}", temporary.display()))?;
    file.write_all(&encoded)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)
        .with_context(|| format!("committing embedding cache {}", path.display()))?;
    Ok(())
}

pub fn load_embedding_cache(path: &Path) -> Result<EmbeddingCache> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let cache: EmbeddingCache = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding embedding cache {}", path.display()))?;
    cache.validate()?;
    Ok(cache)
}

pub fn validate_cache_for_graphs(
    cache: &EmbeddingCache,
    encoder: &ReferenceRelationalAutoencoder,
    graphs: &[&GraphTensor],
) -> Result<()> {
    let expected_encoder = encoder_fingerprint(encoder)?;
    if cache.encoder_fingerprint != expected_encoder {
        bail!("embedding cache was produced by a different encoder");
    }
    for graph in graphs {
        let entry = cache
            .entry(&graph.manifest.snapshot_id)
            .with_context(|| format!("embedding cache has no {}", graph.manifest.snapshot_id))?;
        if entry.graph_fingerprint != graph_fingerprint(graph)? {
            bail!(
                "embedding cache graph fingerprint mismatch for {}",
                graph.manifest.snapshot_id
            );
        }
        if entry.embeddings.station.len() != graph.manifest.station_count
            || entry.embeddings.line.len() != graph.manifest.line_count
        {
            bail!(
                "embedding cache shape mismatch for {}",
                graph.manifest.snapshot_id
            );
        }
    }
    Ok(())
}

pub fn cache_path_for_output(output: &Path) -> PathBuf {
    output.with_extension("embeddings.json")
}
