use anyhow::{bail, Context, Result};
use chrono::{NaiveDate, SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand};
use gtfs_compile::{compile, load_snapshot, save_snapshot, CompileOptions, LineGroupingPolicy};
use gtfs_ingest::{CalendarRecord, GtfsFeed, RouteRecord, StopRecord, StopTimeRecord, TripRecord};
use gtfs_source::{feed_by_id, fetch_feed, load_source_metadata, raw_feed_directory};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use transit_domain::{hex_digest, parse_departure_time, sha256_bytes, ValidationReport};
use transit_graph::GraphTensor;
use transit_inference::{
    load_predictions, predict_reference, rank_by_accessibility, save_predictions,
};
use transit_labels::{
    generate_line_removal_labels, generate_selected_line_removal_labels, load_jsonl,
    sample_origins, save_jsonl, save_label_manifest, LabelGenerationConfig, OriginCandidate,
};
use transit_model::{
    MaskConfig, MaskSelection, ModelConfig, ReferenceLineRepresentationEncoder,
    ReferenceRelationalAutoencoder,
};
use transit_router::{Router, RouterConfig};
use transit_search::{rank_similar_lines, SimilarityProfile};
use transit_training::{
    load_checkpoint, load_config, save_checkpoint, train_criticality_head,
    train_reference_autoencoder, train_reference_multitask, CriticalityTrainingConfig,
    MultiTaskTrainingConfig, PretrainingConfig, ReferenceCheckpoint,
};

#[derive(Debug, Parser)]
#[command(name = "transit", version, about = "Rust GTFS graph-learning pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Fetch(FetchArgs),
    Validate(InputArgs),
    Compile(CompileArgs),
    Graph {
        #[command(subcommand)]
        command: GraphCommand,
    },
    Labels {
        #[command(subcommand)]
        command: LabelsCommand,
    },
    Train {
        #[command(subcommand)]
        command: TrainCommand,
    },
    Infer {
        #[command(subcommand)]
        command: InferCommand,
    },
    Verify {
        #[command(subcommand)]
        command: VerifyCommand,
    },
    SimilarLines(SimilarLinesArgs),
    Demo(DemoArgs),
}

#[derive(Debug, Subcommand)]
enum GraphCommand {
    Build(GraphBuildArgs),
}

#[derive(Debug, Subcommand)]
enum LabelsCommand {
    LineRemoval(LabelsArgs),
}

#[derive(Debug, Subcommand)]
enum InferCommand {
    Criticality(InferArgs),
}

#[derive(Debug, Args)]
struct FetchArgs {
    feed: String,
    #[arg(long)]
    url: Option<String>,
    #[arg(long, default_value = "data/raw")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct InputArgs {
    #[arg(long)]
    input: PathBuf,
}

#[derive(Debug, Args)]
struct CompileArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    service_date: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "unspecified feed scope")]
    scope: String,
    #[arg(long)]
    line_policy: Option<PathBuf>,
    #[arg(long, default_value = "unknown GTFS feed")]
    source_name: String,
}

#[derive(Debug, Args)]
struct GraphBuildArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct LabelsArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 256)]
    origins: usize,
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        "07:30".to_owned(),
        "08:30".to_owned(),
        "12:00".to_owned(),
        "16:30".to_owned(),
        "17:30".to_owned(),
        "22:00".to_owned()
    ])]
    departure_times: Vec<String>,
    #[arg(long, default_value_t = 4)]
    maximum_transfers: u8,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

#[derive(Debug, Subcommand)]
enum TrainCommand {
    Pretrain(PretrainArgs),
    Criticality(CriticalityArgs),
    #[command(name = "multitask", alias = "representation")]
    MultiTask(MultiTaskArgs),
}

#[derive(Debug, Args)]
struct PretrainArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct CriticalityArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    labels: PathBuf,
    #[arg(long)]
    encoder: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct MultiTaskArgs {
    /// Repeat once per compiled graph. All graphs must use the same graph
    /// feature schema; the encoder is shared across them.
    #[arg(long = "graph", required = true)]
    graphs: Vec<PathBuf>,
    /// Optional labels in the same order as --graph. Missing entries are
    /// treated as unsupervised snapshots.
    #[arg(long = "labels")]
    labels: Vec<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct InferArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "unknown-model")]
    model_id: String,
}

#[derive(Debug, Subcommand)]
enum VerifyCommand {
    TopLines(VerifyTopLinesArgs),
}

#[derive(Debug, Args)]
struct VerifyTopLinesArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    predictions: PathBuf,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, value_delimiter = ',', default_values_t = vec!["07:30".to_owned(), "08:30".to_owned()])]
    departure_times: Vec<String>,
}

#[derive(Debug, Args)]
struct SimilarLinesArgs {
    #[arg(long)]
    query_graph: PathBuf,
    #[arg(long)]
    query_line: String,
    #[arg(long)]
    candidate_graph: PathBuf,
    #[arg(long, default_value = "network-role")]
    profile: String,
    #[arg(long, help = "Override the role weight in a weighted profile")]
    role_weight: Option<f32>,
    #[arg(long)]
    service_weight: Option<f32>,
    #[arg(long)]
    geometry_weight: Option<f32>,
    #[arg(long)]
    resilience_weight: Option<f32>,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    encoder: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DemoArgs {
    #[arg(long, default_value = "data/demo")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let command = Cli::parse().command;
    emit_runtime_event("run.started", json!({}))?;
    match run(command) {
        Ok(()) => {
            emit_runtime_event("run.completed", json!({}))?;
            Ok(())
        }
        Err(error) => {
            let message = error.to_string();
            let _ = emit_runtime_event(
                "run.failed",
                json!({"code": "rust-command-failed", "message": message}),
            );
            Err(error)
        }
    }
}

fn run(command: Command) -> Result<()> {
    match command {
        Command::Fetch(args) => command_fetch(args),
        Command::Validate(args) => command_validate(args),
        Command::Compile(args) => command_compile(args),
        Command::Graph { command } => match command {
            GraphCommand::Build(args) => command_graph_build(args),
        },
        Command::Labels { command } => match command {
            LabelsCommand::LineRemoval(args) => command_labels(args),
        },
        Command::Train { command } => match command {
            TrainCommand::Pretrain(args) => command_pretrain(args),
            TrainCommand::Criticality(args) => command_criticality(args),
            TrainCommand::MultiTask(args) => command_multitask(args),
        },
        Command::Infer { command } => match command {
            InferCommand::Criticality(args) => command_infer(args),
        },
        Command::Verify { command } => match command {
            VerifyCommand::TopLines(args) => command_verify_top_lines(args),
        },
        Command::SimilarLines(args) => command_similar_lines(args),
        Command::Demo(args) => command_demo(args),
    }
}

fn command_fetch(args: FetchArgs) -> Result<()> {
    let spec = feed_by_id(&args.feed).with_context(|| format!("unknown feed {}", args.feed))?;
    let url = args.url.or(spec.download_url.clone()).ok_or_else(|| {
        anyhow::anyhow!(
            "{} has no stable direct URL in the registry; pass --url with the current official ZIP URL",
            spec.id
        )
    })?;
    let staging = args.output.join(&spec.id).join("download");
    let metadata = fetch_feed(&spec, &url, &staging)?;
    let final_directory = raw_feed_directory(&args.output, &spec.id, &metadata.sha256);
    if final_directory.exists() {
        bail!(
            "raw feed snapshot already exists at {}; refusing to overwrite it",
            final_directory.display()
        );
    }
    std::fs::create_dir_all(&final_directory)?;
    std::fs::rename(staging.join("gtfs.zip"), final_directory.join("gtfs.zip"))
        .context("moving immutable GTFS ZIP")?;
    std::fs::rename(
        staging.join("source.json"),
        final_directory.join("source.json"),
    )
    .context("moving immutable source metadata")?;
    std::fs::remove_dir_all(args.output.join(&spec.id).join("download"))?;
    write_artifact_manifest(&final_directory, "raw-gtfs-feed")?;
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

fn command_validate(args: InputArgs) -> Result<()> {
    let feed = GtfsFeed::from_path(&args.input)?;
    println!("{}", serde_json::to_string_pretty(&feed.validation)?);
    if !feed.validation.is_valid() {
        bail!(
            "GTFS validation found {} errors",
            feed.validation.errors.len()
        );
    }
    Ok(())
}

fn command_compile(args: CompileArgs) -> Result<()> {
    let service_date = NaiveDate::parse_from_str(&args.service_date, "%Y-%m-%d")
        .with_context(|| format!("invalid service date {}", args.service_date))?;
    let feed = GtfsFeed::from_path(&args.input)?;
    if !feed.validation.is_valid() {
        bail!(
            "GTFS validation found {} errors; run transit validate for details",
            feed.validation.errors.len()
        );
    }
    let mut options = CompileOptions::for_date(service_date)
        .with_scope(args.scope)
        .with_source_name(args.source_name);
    if let Some(metadata_path) = source_metadata_path(&args.input) {
        let metadata = load_source_metadata(&metadata_path)?;
        if options.source_name == "unknown GTFS feed" {
            options.source_name = metadata.display_name;
        }
        if options.geographical_scope == "unspecified feed scope" {
            options = options.with_scope(metadata.geographical_scope);
        }
        options.licence = metadata.licence;
        options.downloaded_at = Some(metadata.downloaded_at);
    }
    if let Some(path) = args.line_policy {
        options.line_grouping_policy = LineGroupingPolicy::from_path(&path)?;
    }
    let network = compile(&feed, &options)?;
    save_snapshot(&network, &args.output)?;
    write_artifact_manifest(&args.output, "compiled-snapshot")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "snapshot_id": network.snapshot_id,
            "stations": network.stations.len(),
            "lines": network.lines.len(),
            "patterns": network.patterns.len(),
            "transit_edges": network.transit_edges.len(),
            "transfers": network.transfers.len(),
        }))?
    );
    Ok(())
}

fn command_graph_build(args: GraphBuildArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let graph = GraphTensor::from_network(&network)?;
    graph.save(&args.output, &network)?;
    write_artifact_manifest(&args.output, "compiled-graph")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "snapshot_id": graph.manifest.snapshot_id,
            "stations": graph.manifest.station_count,
            "lines": graph.manifest.line_count,
            "transit_edges": graph.manifest.transit_edge_count,
        }))?
    );
    Ok(())
}

fn command_labels(args: LabelsArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let router = Router::from_network(
        &network,
        RouterConfig {
            maximum_transfers: args.maximum_transfers,
            ..RouterConfig::default()
        },
    )?;
    let departures = args
        .departure_times
        .iter()
        .map(|value| parse_departure_time(value))
        .collect::<Result<Vec<_>>>()?;
    let mut label_config = LabelGenerationConfig {
        maximum_origins: args.origins,
        ..LabelGenerationConfig::default()
    };
    label_config.origin_sampling.seed = args.seed;
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(&candidates, args.origins, &label_config.origin_sampling);
    let labels = generate_line_removal_labels(
        &router,
        network.snapshot_id.clone(),
        &origins,
        &departures,
        &label_config,
    );
    save_jsonl(&args.output, &labels)?;
    save_label_manifest(&args.output, &label_config, origins.len())?;
    write_artifact_manifest(&args.output, "criticality-labels")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"labels": labels.len(), "output": args.output}))?
    );
    Ok(())
}

fn command_pretrain(args: PretrainArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let config = args
        .config
        .as_deref()
        .map(load_config::<PretrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    let (encoder, report) = train_reference_autoencoder(&graph, &config)?;
    save_checkpoint(
        &args.output,
        &ReferenceCheckpoint {
            encoder,
            head: None,
            report: Some(report.clone()),
            representation: None,
        },
    )?;
    write_artifact_manifest(&args.output, "model-checkpoint")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn command_criticality(args: CriticalityArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let labels = transit_labels::load_jsonl(&args.labels)?;
    let checkpoint = load_checkpoint(&args.encoder)?;
    let config = args
        .config
        .as_deref()
        .map(load_config::<CriticalityTrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    let (head, report) = train_criticality_head(&checkpoint.encoder, &graph, &labels, &config)?;
    save_checkpoint(
        &args.output,
        &ReferenceCheckpoint {
            encoder: checkpoint.encoder,
            head: Some(head),
            report: Some(report.clone()),
            representation: checkpoint.representation,
        },
    )?;
    write_artifact_manifest(&args.output, "model-checkpoint")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn command_multitask(args: MultiTaskArgs) -> Result<()> {
    if args.labels.len() > args.graphs.len() {
        bail!("provide at most one --labels file for each --graph");
    }
    let config = args
        .config
        .as_deref()
        .map(load_config::<MultiTaskTrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    let graphs: Vec<GraphTensor> = args
        .graphs
        .iter()
        .map(|path| {
            GraphTensor::load(path).with_context(|| format!("loading graph {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut labels = Vec::with_capacity(graphs.len());
    for index in 0..graphs.len() {
        labels.push(if let Some(path) = args.labels.get(index) {
            load_jsonl(path).with_context(|| format!("loading labels {}", path.display()))?
        } else {
            Vec::new()
        });
    }
    let datasets: Vec<(&GraphTensor, &[transit_labels::LineImpactLabel])> = graphs
        .iter()
        .zip(&labels)
        .map(|(graph, labels)| (graph, labels.as_slice()))
        .collect();
    let (checkpoint, report) = train_reference_multitask(&datasets, &config)?;
    save_checkpoint(&args.output, &checkpoint)?;
    write_artifact_manifest(&args.output, "model-checkpoint")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "output": args.output,
            "report": report,
        }))?
    );
    Ok(())
}

fn command_infer(args: InferArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let checkpoint = load_checkpoint(&args.model)?;
    let mut predictions = predict_reference(&checkpoint, &graph)?;
    predictions.model_id = Some(args.model_id);
    rank_by_accessibility(&mut predictions);
    save_predictions(&args.output, &predictions)?;
    write_artifact_manifest(&args.output, "inference-result")?;
    println!("{}", serde_json::to_string_pretty(&predictions)?);
    Ok(())
}

fn command_verify_top_lines(args: VerifyTopLinesArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let prediction_path = args.predictions;
    let predictions = load_predictions(&prediction_path)?;
    let router = Router::from_network(&network, RouterConfig::default())?;
    let label_config = LabelGenerationConfig::default();
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(
        &candidates,
        label_config.maximum_origins,
        &label_config.origin_sampling,
    );
    let departures = args
        .departure_times
        .iter()
        .map(|value| parse_departure_time(value))
        .collect::<Result<Vec<_>>>()?;
    let selected_lines = predictions
        .predictions
        .iter()
        .take(args.top_k)
        .map(|prediction| transit_domain::LineIndex(prediction.line))
        .collect::<Vec<_>>();
    let labels = generate_selected_line_removal_labels(
        &router,
        network.snapshot_id.clone(),
        &origins,
        &departures,
        &label_config,
        &selected_lines,
    );
    let verified: Vec<_> = labels
        .into_iter()
        .map(|label| {
            let display_name = network
                .lines
                .get(label.line.0 as usize)
                .map(|line| line.display_name.clone())
                .unwrap_or_else(|| label.line.to_string());
            json!({
                "line": label.line.0,
                "display_name": display_name,
                "exact_accessibility_auc_loss": label.accessibility_auc_loss,
                "exact_unreachable_share": label.unreachable_share,
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&verified)?);
    Ok(())
}

fn command_similar_lines(args: SimilarLinesArgs) -> Result<()> {
    let query_graph = GraphTensor::load(&args.query_graph)?;
    let candidate_graph = GraphTensor::load(&args.candidate_graph)?;
    let query_line = resolve_line_index(&query_graph, &args.query_line)?;
    let profile: SimilarityProfile = if args.role_weight.is_some()
        || args.service_weight.is_some()
        || args.geometry_weight.is_some()
        || args.resilience_weight.is_some()
    {
        SimilarityProfile::Weighted {
            role: args.role_weight.unwrap_or(0.0),
            service: args.service_weight.unwrap_or(0.0),
            geometry: args.geometry_weight.unwrap_or(0.0),
            resilience: args.resilience_weight.unwrap_or(0.0),
        }
    } else {
        args.profile
            .parse()
            .map_err(|error: String| anyhow::anyhow!(error))?
    };
    let checkpoint = args.encoder.as_deref().map(load_checkpoint).transpose()?;
    let fallback_encoder;
    let encoder = if let Some(checkpoint) = checkpoint.as_ref() {
        &checkpoint.encoder
    } else {
        fallback_encoder = ReferenceRelationalAutoencoder::new(ModelConfig::default());
        &fallback_encoder
    };
    let query_mask = MaskSelection::all_unmasked(&query_graph);
    let query_embeddings = encoder.encode(&query_graph, &query_mask)?;
    let candidate_mask = MaskSelection::all_unmasked(&candidate_graph);
    let candidate_embeddings = encoder.encode(&candidate_graph, &candidate_mask)?;
    let fallback_representation_encoder;
    let (query_representations, candidate_representations) = if let Some(representation) =
        checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.representation.as_ref())
    {
        (
            representation.encode(&query_graph, &query_embeddings)?,
            representation.encode(&candidate_graph, &candidate_embeddings)?,
        )
    } else {
        fallback_representation_encoder = ReferenceLineRepresentationEncoder::default();
        (
            fallback_representation_encoder.encode(&query_graph, &query_embeddings)?,
            fallback_representation_encoder.encode(&candidate_graph, &candidate_embeddings)?,
        )
    };
    let matches = rank_similar_lines(
        &query_graph,
        &query_representations,
        query_line,
        &candidate_graph,
        &candidate_representations,
        &profile,
        args.top_k,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "query": {
                "snapshot": query_graph.manifest.snapshot_id,
                "line": query_line,
                "line_name": query_graph.line_names.get(query_line),
            },
            "profile": profile.to_string(),
            "matches": matches,
        }))?
    );
    Ok(())
}

fn resolve_line_index(graph: &GraphTensor, query: &str) -> Result<usize> {
    if let Ok(index) = query.trim().parse::<usize>() {
        if index < graph.manifest.line_count {
            return Ok(index);
        }
    }
    let needle = query.trim().to_ascii_lowercase();
    if needle.is_empty() {
        bail!("query line cannot be blank");
    }
    let exact: Vec<usize> = graph
        .line_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| (name.to_ascii_lowercase() == needle).then_some(index))
        .collect();
    if exact.len() == 1 {
        return Ok(exact[0]);
    }
    let partial: Vec<usize> = graph
        .line_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| name.to_ascii_lowercase().contains(&needle).then_some(index))
        .collect();
    if partial.len() == 1 {
        return Ok(partial[0]);
    }
    if exact.len() > 1 || partial.len() > 1 {
        bail!("query line {query:?} is ambiguous; use a numeric line index");
    }
    bail!("could not find query line {query:?}")
}

fn command_demo(args: DemoArgs) -> Result<()> {
    let feed = demo_feed();
    let service_date = NaiveDate::from_ymd_opt(2026, 9, 7).expect("valid demo date");
    let options = CompileOptions::for_date(service_date)
        .with_scope("synthetic demo")
        .with_source_name("synthetic-demo");
    let network = compile(&feed, &options)?;
    let snapshot_dir = args.output.join("snapshot");
    save_snapshot(&network, &snapshot_dir)?;
    let graph = GraphTensor::from_network(&network)?;
    let graph_dir = args.output.join("graph");
    graph.save(&graph_dir, &network)?;
    let router = Router::from_network(&network, RouterConfig::default())?;
    let mut label_config = LabelGenerationConfig {
        maximum_origins: network.stations.len(),
        ..LabelGenerationConfig::default()
    };
    label_config.origin_sampling.seed = 7;
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(
        &candidates,
        label_config.maximum_origins,
        &label_config.origin_sampling,
    );
    let departures = vec![
        parse_departure_time("07:30")?,
        parse_departure_time("08:30")?,
    ];
    let labels = generate_line_removal_labels(
        &router,
        network.snapshot_id.clone(),
        &origins,
        &departures,
        &label_config,
    );
    let labels_path = args.output.join("labels.jsonl");
    save_jsonl(&labels_path, &labels)?;
    save_label_manifest(&labels_path, &label_config, origins.len())?;
    write_artifact_manifest(&labels_path, "criticality-labels")?;
    let pretraining = PretrainingConfig {
        model: ModelConfig {
            hidden_dimension: 32,
            temporal_dimension: 16,
            graph_layers: 2,
            dropout: 0.0,
        },
        mask: MaskConfig {
            station_feature_probability: 0.5,
            line_feature_probability: 0.5,
            ..MaskConfig::default()
        },
        steps: 25,
        ..PretrainingConfig::default()
    };
    let (encoder, pretrain_report) = train_reference_autoencoder(&graph, &pretraining)?;
    let (head, head_report) = train_criticality_head(
        &encoder,
        &graph,
        &labels,
        &CriticalityTrainingConfig {
            epochs: 20,
            ..CriticalityTrainingConfig::default()
        },
    )?;
    let checkpoint_path = args.output.join("model.json");
    save_checkpoint(
        &checkpoint_path,
        &ReferenceCheckpoint {
            encoder,
            head: Some(head),
            report: Some(head_report.clone()),
            representation: None,
        },
    )?;
    write_artifact_manifest(&checkpoint_path, "model-checkpoint")?;
    let mut predictions = predict_reference(&load_checkpoint(&checkpoint_path)?, &graph)?;
    predictions.model_id = Some("model-demo-reference".into());
    rank_by_accessibility(&mut predictions);
    let predictions_path = args.output.join("predictions.json");
    save_predictions(&predictions_path, &predictions)?;
    write_artifact_manifest(&predictions_path, "inference-result")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "snapshot_id": network.snapshot_id,
            "stations": network.stations.len(),
            "lines": network.lines.len(),
            "labels": labels.len(),
            "pretraining": pretrain_report,
            "criticality": head_report,
            "output": args.output,
        }))?
    );
    Ok(())
}

fn source_metadata_path(input: &std::path::Path) -> Option<PathBuf> {
    let candidate = if input.is_dir() {
        input.join("source.json")
    } else {
        input.parent()?.join("source.json")
    };
    candidate.is_file().then_some(candidate)
}

static RUNTIME_EVENT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Emit machine-readable events only when the Studio worker opts into the
/// protocol. Human stdout remains a separate diagnostic stream.
fn emit_runtime_event(event_type: &str, payload: Value) -> Result<()> {
    let Some(event_path) = std::env::var_os("TRANSIT_EVENT_FILE") else {
        return Ok(());
    };
    let Some(run_id) = std::env::var_os("TRANSIT_RUN_ID") else {
        return Ok(());
    };
    let run_id = run_id.to_string_lossy().into_owned();
    let mut event = serde_json::Map::new();
    event.insert("schemaVersion".into(), json!(1));
    event.insert(
        "seq".into(),
        json!(RUNTIME_EVENT_SEQUENCE.fetch_add(1, Ordering::Relaxed)),
    );
    event.insert("runId".into(), json!(run_id));
    event.insert(
        "timestamp".into(),
        json!(Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)),
    );
    event.insert("type".into(), json!(event_type));
    if let Value::Object(fields) = payload {
        event.extend(fields);
    }
    let path = PathBuf::from(event_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    serde_json::to_writer(&mut file, &Value::Object(event))?;
    file.write_all(b"\n")?;
    Ok(())
}

fn artifact_manifest_path(output: &Path) -> PathBuf {
    if output.is_dir() {
        output.join("artifact-manifest.json")
    } else {
        let name = output
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("output");
        output
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{name}.artifact-manifest.json"))
    }
}

fn artifact_file_entries(path: &Path, root: &Path, entries: &mut Vec<Value>) -> Result<()> {
    let path = fs::canonicalize(path)
        .with_context(|| format!("resolving artifact output {}", path.display()))?;
    if path.is_file() {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        if name == "events.jsonl"
            || name == "artifact-manifest.json"
            || name.ends_with(".artifact-manifest.json")
        {
            return Ok(());
        }
        let bytes = fs::read(&path)?;
        let relative = path
            .strip_prefix(root)
            .with_context(|| {
                format!(
                    "artifact output {} is outside repository root {}",
                    path.display(),
                    root.display()
                )
            })?
            .to_string_lossy()
            .replace('\\', "/");
        entries.push(json!({
            "path": relative,
            "sizeBytes": bytes.len(),
            "sha256": hex_digest(&sha256_bytes(&bytes)),
        }));
        return Ok(());
    }
    if path.is_dir() {
        for child in fs::read_dir(&path)? {
            artifact_file_entries(&child?.path(), root, entries)?;
        }
    }
    Ok(())
}

/// Write the shared artifact-manifest.v1 contract next to a Rust output.
/// Existing manifests are immutable and must describe the same output.
fn write_artifact_manifest(output: &Path, kind: &str) -> Result<()> {
    if !output.exists() {
        bail!("cannot manifest missing Rust output {}", output.display());
    }
    let artifact_root = std::env::var_os("TRANSIT_LAB_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    let root = fs::canonicalize(&artifact_root)
        .with_context(|| format!("resolving artifact root {}", artifact_root.display()))?;
    let mut files = Vec::new();
    artifact_file_entries(output, &root, &mut files)?;
    files.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    if files.is_empty() {
        bail!("cannot manifest empty Rust output {}", output.display());
    }
    let file_bytes = serde_json::to_vec(&files)?;
    let digest = hex_digest(&sha256_bytes(&file_bytes));
    let manifest = json!({
        "schemaVersion": 1,
        "artifactId": format!("artifact-{kind}-{}", &digest[..24]),
        "kind": kind,
        "fingerprint": digest,
        "sha256": if files.len() == 1 { files[0]["sha256"].clone() } else { json!(hex_digest(&sha256_bytes(&file_bytes))) },
        "createdAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        "producingRunId": std::env::var("TRANSIT_RUN_ID").ok(),
        "inputs": [],
        "gitCommit": std::env::var("TRANSIT_LAB_GIT_COMMIT").unwrap_or_else(|_| "working-tree".into()),
        "configuration": {"producer": "transit-cli", "kind": kind},
        "files": files,
        "metadata": {}
    });
    let manifest_path = artifact_manifest_path(output);
    if let Ok(existing) = fs::read(&manifest_path) {
        let existing: Value =
            serde_json::from_slice(&existing).context("decoding existing artifact manifest")?;
        if existing["fingerprint"] != manifest["fingerprint"]
            || existing["files"] != manifest["files"]
            || existing["kind"] != manifest["kind"]
        {
            bail!(
                "refusing to overwrite immutable artifact manifest {}",
                manifest_path.display()
            );
        }
        return Ok(());
    }
    let encoded = serde_json::to_vec_pretty(&manifest)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&manifest_path)
        .with_context(|| format!("creating {}", manifest_path.display()))?;
    file.write_all(&encoded)?;
    file.write_all(b"\n")?;
    let uri = fs::canonicalize(&manifest_path)
        .ok()
        .and_then(|path| {
            path.strip_prefix(&root)
                .ok()
                .map(|value| value.to_string_lossy().replace('\\', "/"))
        })
        .unwrap_or_else(|| manifest_path.to_string_lossy().replace('\\', "/"));
    emit_runtime_event(
        "artifact.created",
        json!({
            "artifactId": manifest["artifactId"],
            "artifactKind": kind,
            "uri": uri,
            "sha256": manifest["sha256"],
        }),
    )?;
    Ok(())
}

fn demo_feed() -> GtfsFeed {
    let stops: Vec<StopRecord> = [
        ("a", "Alpha", 48.1000, 16.1000),
        ("b", "Bravo", 48.1010, 16.1010),
        ("c", "Central", 48.1020, 16.1020),
        ("d", "Delta", 48.1030, 16.1030),
        ("e", "Echo", 48.1040, 16.1040),
    ]
    .into_iter()
    .map(|(id, name, lat, lon)| StopRecord {
        stop_id: id.into(),
        stop_name: Some(name.into()),
        stop_lat: Some(lat.to_string()),
        stop_lon: Some(lon.to_string()),
        location_type: Some("0".into()),
        ..StopRecord::default()
    })
    .collect();
    let routes = vec![
        RouteRecord {
            route_id: "blue".into(),
            agency_id: Some("demo".into()),
            route_short_name: Some("Blue".into()),
            route_type: Some("1".into()),
            ..RouteRecord::default()
        },
        RouteRecord {
            route_id: "red".into(),
            agency_id: Some("demo".into()),
            route_short_name: Some("Red".into()),
            route_type: Some("3".into()),
            ..RouteRecord::default()
        },
        RouteRecord {
            route_id: "green".into(),
            agency_id: Some("demo".into()),
            route_short_name: Some("Green".into()),
            route_type: Some("3".into()),
            ..RouteRecord::default()
        },
    ];
    let mut trips = Vec::new();
    let mut stop_times = Vec::new();
    let mut add_trip = |trip_id: &str, route_id: &str, stops: &[&str], start: u32| {
        trips.push(TripRecord {
            route_id: route_id.into(),
            service_id: "weekday".into(),
            trip_id: trip_id.into(),
            direction_id: Some("0".into()),
            ..TripRecord::default()
        });
        for (position, stop_id) in stops.iter().enumerate() {
            let arrival = start + position as u32 * 300;
            let departure = arrival + if position + 1 == stops.len() { 0 } else { 30 };
            stop_times.push(StopTimeRecord {
                trip_id: trip_id.into(),
                arrival_time: format_gtfs_time(arrival),
                departure_time: format_gtfs_time(departure),
                stop_id: (*stop_id).into(),
                stop_sequence: position.to_string(),
                ..StopTimeRecord::default()
            });
        }
    };
    for (index, start) in [27_000, 30_600, 43_200].into_iter().enumerate() {
        add_trip(&format!("blue-{index}"), "blue", &["a", "b", "c"], start);
        add_trip(
            &format!("green-{index}"),
            "green",
            &["a", "b", "c"],
            start + 600,
        );
        add_trip(
            &format!("red-{index}"),
            "red",
            &["c", "d", "e"],
            start + 240,
        );
    }
    let calendars = vec![CalendarRecord {
        service_id: "weekday".into(),
        monday: "1".into(),
        tuesday: "1".into(),
        wednesday: "1".into(),
        thursday: "1".into(),
        friday: "1".into(),
        saturday: "0".into(),
        sunday: "0".into(),
        start_date: "20260101".into(),
        end_date: "20261231".into(),
    }];
    let mut row_counts = BTreeMap::new();
    row_counts.insert("stops.txt".into(), stops.len());
    row_counts.insert("routes.txt".into(), routes.len());
    row_counts.insert("trips.txt".into(), trips.len());
    row_counts.insert("stop_times.txt".into(), stop_times.len());
    GtfsFeed {
        source_path: PathBuf::from("synthetic://demo"),
        source_hash: sha256_bytes(b"transit-lab-demo-feed-v1"),
        stops,
        routes,
        trips,
        stop_times,
        calendars,
        calendar_dates: Vec::new(),
        transfers: Vec::new(),
        pathways: Vec::new(),
        validation: ValidationReport {
            checked_files: vec![
                "stops.txt".into(),
                "routes.txt".into(),
                "trips.txt".into(),
                "stop_times.txt".into(),
                "calendar.txt".into(),
            ],
            row_counts,
            ..ValidationReport::default()
        },
    }
}

fn format_gtfs_time(seconds: u32) -> String {
    format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}
