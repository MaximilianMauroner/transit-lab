use chrono::NaiveDate;
use gtfs_compile::{compile, CompileOptions};
use gtfs_ingest::GtfsFeed;
use tempfile::tempdir;
use transit_graph::GraphTensor;
use transit_labels::{generate_line_removal_labels, LabelGenerationConfig};
use transit_model::{MaskConfig, ModelConfig, RepresentationConfig};
use transit_router::{Router, RouterConfig};
use transit_search::{rank_similar_lines, SimilarityProfile};
use transit_training::{
    save_checkpoint, train_reference_multitask, CriticalityTrainingConfig, MultiTaskTrainingConfig,
    PretrainingConfig,
};

fn fixture_feed() -> GtfsFeed {
    GtfsFeed::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/synthetic-feeds/basic"
    ))
    .expect("fixture feed should load")
}

fn graph_and_labels(date: NaiveDate) -> (GraphTensor, Vec<transit_labels::LineImpactLabel>) {
    let network = compile(
        &fixture_feed(),
        &CompileOptions::for_date(date)
            .with_scope("multitask integration")
            .with_source_name("synthetic-multitask"),
    )
    .expect("fixture feed should compile");
    let router = Router::from_network(&network, RouterConfig::default())
        .expect("compiled network should be routable");
    let origins = (0..network.stations.len())
        .map(|station| transit_domain::StationIndex(station as u32))
        .collect::<Vec<_>>();
    let labels = generate_line_removal_labels(
        &router,
        network.snapshot_id.clone(),
        &origins,
        &[7 * 3_600 + 30 * 60],
        &LabelGenerationConfig {
            maximum_origins: origins.len(),
            ..LabelGenerationConfig::default()
        },
    );
    (
        GraphTensor::from_network(&network).expect("network should become a graph"),
        labels,
    )
}

#[test]
fn trains_checkpoint_and_retrieves_across_snapshots() {
    let (first_graph, first_labels) =
        graph_and_labels(NaiveDate::from_ymd_opt(2026, 9, 7).expect("valid fixture date"));
    let (second_graph, second_labels) =
        graph_and_labels(NaiveDate::from_ymd_opt(2026, 9, 8).expect("valid fixture date"));
    assert_ne!(
        first_graph.manifest.snapshot_id,
        second_graph.manifest.snapshot_id
    );
    assert!(!first_labels.is_empty());
    assert!(!second_labels.is_empty());

    let config = MultiTaskTrainingConfig {
        pretraining: PretrainingConfig {
            model: ModelConfig {
                hidden_dimension: 8,
                temporal_dimension: 4,
                graph_layers: 1,
                dropout: 0.0,
            },
            mask: MaskConfig {
                station_feature_probability: 0.5,
                line_feature_probability: 0.5,
                ..MaskConfig::default()
            },
            steps: 2,
            ..PretrainingConfig::default()
        },
        representation: RepresentationConfig {
            base_dimension: 12,
            city_dimension: 6,
            general_dimension: 5,
            role_dimension: 4,
            service_dimension: 4,
            geometry_dimension: 4,
            resilience_dimension: 4,
            seed: 11,
        },
        metric_epochs: 1,
        metric_learning_rate: 0.002,
        max_triplets: 8,
        criticality: CriticalityTrainingConfig {
            epochs: 2,
            max_ranking_pairs: 8,
            ..CriticalityTrainingConfig::default()
        },
        ..MultiTaskTrainingConfig::default()
    };
    let datasets = vec![
        (&first_graph, first_labels.as_slice()),
        (&second_graph, second_labels.as_slice()),
    ];
    let (checkpoint, report) = train_reference_multitask(&datasets, &config)
        .expect("multitask training should complete on the fixture");

    assert_eq!(report.backend, "reference-cpu-multitask");
    assert_eq!(report.dataset_count, 2);
    assert_eq!(report.line_count, first_graph.manifest.line_count * 2);
    assert_eq!(report.pretraining.steps, 2);
    assert!(report.metric_triplets > 0);
    assert!(report.metric_final_loss.is_finite());
    assert!(report.criticality.is_some());

    let representation = checkpoint
        .representation
        .as_ref()
        .expect("multitask checkpoint should contain representation heads");
    let first_embeddings = checkpoint
        .encoder
        .encode(
            &first_graph,
            &transit_model::MaskSelection::all_unmasked(&first_graph),
        )
        .expect("first graph should encode");
    let second_embeddings = checkpoint
        .encoder
        .encode(
            &second_graph,
            &transit_model::MaskSelection::all_unmasked(&second_graph),
        )
        .expect("second graph should encode");
    let first_representations = representation
        .encode(&first_graph, &first_embeddings)
        .expect("first representations should encode");
    let second_representations = representation
        .encode(&second_graph, &second_embeddings)
        .expect("second representations should encode");
    let matches = rank_similar_lines(
        &first_graph,
        &first_representations,
        0,
        &second_graph,
        &second_representations,
        &SimilarityProfile::NetworkRole,
        2,
    )
    .expect("cross-snapshot retrieval should complete");
    assert_eq!(matches.len(), 2);
    assert!(matches[0].similarity.is_finite());
    assert!(matches[0].facet_scores.role.is_finite());

    let directory = tempdir().expect("temporary checkpoint directory");
    let checkpoint_path = directory.path().join("multitask.json");
    save_checkpoint(&checkpoint_path, &checkpoint).expect("checkpoint should serialize");
    assert!(checkpoint_path.is_file());
}
