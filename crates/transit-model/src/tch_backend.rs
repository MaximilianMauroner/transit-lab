//! LibTorch implementation of the masked relational encoder.
//!
//! This module is intentionally feature-gated. Building the default workspace
//! does not require a system LibTorch installation, while Linux/NVIDIA users
//! can enable `tch-backend` and train the same graph schema on a GPU.

use crate::{MaskSelection, ModelConfig};
use anyhow::{Context, Result};
use tch::{nn, Device, Kind, Tensor};
use transit_domain::SERVICE_DAY_BINS;
use transit_graph::{FeatureMatrix, GraphTensor, EDGE_FEATURES};

#[derive(Debug)]
struct RelationalLayer {
    station_self: nn::Linear,
    line_self: nn::Linear,
    station_to_line: nn::Linear,
    line_to_station: nn::Linear,
    transfer: nn::Linear,
    transit_station: nn::Linear,
    transit_line: nn::Linear,
    transit_edge: nn::Linear,
    interchange: nn::Linear,
    station_norm: nn::LayerNorm,
    line_norm: nn::LayerNorm,
}

impl RelationalLayer {
    fn new(path: nn::Path<'_>, hidden: i64, edge_features: i64) -> Self {
        Self {
            station_self: nn::linear(&path / "station_self", hidden, hidden, Default::default()),
            line_self: nn::linear(&path / "line_self", hidden, hidden, Default::default()),
            station_to_line: nn::linear(
                &path / "station_to_line",
                hidden,
                hidden,
                Default::default(),
            ),
            line_to_station: nn::linear(
                &path / "line_to_station",
                hidden,
                hidden,
                Default::default(),
            ),
            transfer: nn::linear(&path / "transfer", hidden, hidden, Default::default()),
            transit_station: nn::linear(
                &path / "transit_station",
                hidden,
                hidden,
                Default::default(),
            ),
            transit_line: nn::linear(&path / "transit_line", hidden, hidden, Default::default()),
            transit_edge: nn::linear(
                &path / "transit_edge",
                edge_features,
                hidden,
                Default::default(),
            ),
            interchange: nn::linear(&path / "interchange", hidden, hidden, Default::default()),
            station_norm: nn::layer_norm(&path / "station_norm", vec![hidden], Default::default()),
            line_norm: nn::layer_norm(&path / "line_norm", vec![hidden], Default::default()),
        }
    }
}

#[derive(Debug)]
struct GraphIndices {
    serves_src: Tensor,
    serves_dst: Tensor,
    transit_src: Tensor,
    transit_dst: Tensor,
    transit_line: Tensor,
    transfer_src: Tensor,
    transfer_dst: Tensor,
    interchange_src: Tensor,
    interchange_dst: Tensor,
}

impl GraphIndices {
    fn from_graph(graph: &GraphTensor, device: Device) -> Self {
        Self {
            serves_src: index_tensor(&graph.serves_src, device),
            serves_dst: index_tensor(&graph.serves_dst, device),
            transit_src: index_tensor(&graph.transit_src, device),
            transit_dst: index_tensor(&graph.transit_dst, device),
            transit_line: index_tensor(&graph.transit_line, device),
            transfer_src: index_tensor(&graph.transfer_src, device),
            transfer_dst: index_tensor(&graph.transfer_dst, device),
            interchange_src: index_tensor(&graph.interchange_src, device),
            interchange_dst: index_tensor(&graph.interchange_dst, device),
        }
    }
}

#[derive(Debug)]
pub struct TchEmbeddings {
    pub station: Tensor,
    pub line: Tensor,
    pub city: Tensor,
}

#[derive(Debug)]
pub struct TchReconstruction {
    pub embeddings: TchEmbeddings,
    pub station_features: Tensor,
    pub line_features: Tensor,
    pub served_by_logits: Tensor,
    pub transfer_logits: Tensor,
}

#[derive(Debug)]
pub struct TchRelationalAutoencoder {
    pub var_store: nn::VarStore,
    station_temporal_conv1: nn::Conv1D,
    station_temporal_conv2: nn::Conv1D,
    line_temporal_conv1: nn::Conv1D,
    line_temporal_conv2: nn::Conv1D,
    station_input: nn::Linear,
    line_input: nn::Linear,
    pattern_token: nn::Linear,
    pattern_update: nn::Linear,
    layers: Vec<RelationalLayer>,
    station_decoder: nn::Linear,
    line_decoder: nn::Linear,
    indices: GraphIndices,
    device: Device,
}

impl TchRelationalAutoencoder {
    pub fn new(device: Device, graph: &GraphTensor, config: &ModelConfig) -> Self {
        let var_store = nn::VarStore::new(device);
        let root = var_store.root();
        let temporal_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let temporal_width = config.temporal_dimension as i64;
        let hidden = config.hidden_dimension as i64;
        let station_temporal_conv1 = nn::conv1d(
            &root / "station_temporal_conv1",
            4,
            temporal_width,
            3,
            temporal_config,
        );
        let station_temporal_conv2 = nn::conv1d(
            &root / "station_temporal_conv2",
            temporal_width,
            temporal_width,
            3,
            temporal_config,
        );
        let line_temporal_conv1 = nn::conv1d(
            &root / "line_temporal_conv1",
            4,
            temporal_width,
            3,
            temporal_config,
        );
        let line_temporal_conv2 = nn::conv1d(
            &root / "line_temporal_conv2",
            temporal_width,
            temporal_width,
            3,
            temporal_config,
        );
        let station_input = nn::linear(
            &root / "station_input",
            graph.station_features.cols as i64 + temporal_width * 2,
            hidden,
            Default::default(),
        );
        let line_input = nn::linear(
            &root / "line_input",
            graph.line_features.cols as i64 + temporal_width * 2,
            hidden,
            Default::default(),
        );
        let pattern_token = nn::linear(
            &root / "pattern_token",
            hidden + 3 + EDGE_FEATURES as i64,
            hidden,
            Default::default(),
        );
        let pattern_update = nn::linear(
            &root / "pattern_update",
            hidden * 2,
            hidden,
            Default::default(),
        );
        let layers = (0..config.graph_layers)
            .map(|index| {
                RelationalLayer::new(
                    &root / format!("graph_layer_{index}"),
                    hidden,
                    EDGE_FEATURES as i64,
                )
            })
            .collect();
        let station_decoder = nn::linear(
            &root / "station_decoder",
            hidden,
            graph.station_features.cols as i64,
            Default::default(),
        );
        let line_decoder = nn::linear(
            &root / "line_decoder",
            hidden,
            graph.line_features.cols as i64,
            Default::default(),
        );
        Self {
            var_store,
            station_temporal_conv1,
            station_temporal_conv2,
            line_temporal_conv1,
            line_temporal_conv2,
            station_input,
            line_input,
            pattern_token,
            pattern_update,
            layers,
            station_decoder,
            line_decoder,
            indices: GraphIndices::from_graph(graph, device),
            device,
        }
    }

    pub fn forward(
        &self,
        graph: &GraphTensor,
        mask: &MaskSelection,
        train: bool,
    ) -> Result<TchReconstruction> {
        graph.validate()?;
        validate_mask(graph, mask)?;
        let station_static = matrix_tensor(&graph.station_features, self.device)?
            * visible_rows(&mask.station_rows, self.device);
        let line_static = matrix_tensor(&graph.line_features, self.device)?
            * visible_rows(&mask.line_rows, self.device);
        let station_temporal = temporal_tensor(
            &graph.station_temporal,
            graph.manifest.station_count,
            self.device,
        )? * visible_temporal(
            &mask.station_temporal_blocks,
            graph.manifest.station_count,
            self.device,
        );
        let line_temporal =
            temporal_tensor(&graph.line_temporal, graph.manifest.line_count, self.device)?
                * visible_temporal(
                    &mask.line_temporal_blocks,
                    graph.manifest.line_count,
                    self.device,
                );
        let station_temporal = station_temporal
            .apply_t(&self.station_temporal_conv1, train)
            .gelu("none")
            .apply_t(&self.station_temporal_conv2, train)
            .gelu("none");
        let line_temporal = line_temporal
            .apply_t(&self.line_temporal_conv1, train)
            .gelu("none")
            .apply_t(&self.line_temporal_conv2, train)
            .gelu("none");
        let mut station = Tensor::cat(&[station_static, pool_temporal(&station_temporal)], 1)
            .apply(&self.station_input)
            .gelu("none");
        let mut line = Tensor::cat(&[line_static, pool_temporal(&line_temporal)], 1)
            .apply(&self.line_input)
            .gelu("none");
        let pattern_context = pattern_sequence_context(
            graph,
            &station,
            &self.pattern_token,
            &self.pattern_update,
            self.device,
        )?;
        line = (line + pattern_context * 0.22).gelu("none");
        let served_mask = edge_visibility(&mask.served_edges, self.device);
        let transfer_mask = edge_visibility(&mask.transfer_edges, self.device);
        let transit_mask =
            Tensor::ones([graph.transit_src.len() as i64], (Kind::Float, self.device));
        let interchange_mask = Tensor::ones(
            [graph.interchange_src.len() as i64],
            (Kind::Float, self.device),
        );

        for layer in &self.layers {
            let station_to_line = mean_aggregate(
                &station,
                &self.indices.serves_src,
                &self.indices.serves_dst,
                graph.manifest.line_count,
                &layer.station_to_line,
                Some(&served_mask),
            );
            let line_to_station = mean_aggregate(
                &line,
                &self.indices.serves_dst,
                &self.indices.serves_src,
                graph.manifest.station_count,
                &layer.line_to_station,
                Some(&served_mask),
            );
            let transfer = mean_aggregate(
                &station,
                &self.indices.transfer_src,
                &self.indices.transfer_dst,
                graph.manifest.station_count,
                &layer.transfer,
                Some(&transfer_mask),
            );
            let transit = transit_aggregate(TransitAggregation {
                station: &station,
                line: &line,
                edge_features: &graph.transit_features,
                source_indices: &self.indices.transit_src,
                destination_indices: &self.indices.transit_dst,
                line_indices: &self.indices.transit_line,
                destination_count: graph.manifest.station_count,
                station_projection: &layer.transit_station,
                line_projection: &layer.transit_line,
                edge_projection: &layer.transit_edge,
                visibility: Some(&transit_mask),
                device: self.device,
            })?;
            let interchange = mean_aggregate(
                &line,
                &self.indices.interchange_src,
                &self.indices.interchange_dst,
                graph.manifest.line_count,
                &layer.interchange,
                Some(&interchange_mask),
            );
            station = (station.apply(&layer.station_self)
                + line_to_station
                + transfer * 0.12
                + transit * 0.16)
                .gelu("none")
                .apply(&layer.station_norm);
            line = (line.apply(&layer.line_self) + station_to_line + interchange * 0.14)
                .gelu("none")
                .apply(&layer.line_norm);
        }
        let city = city_pool(&station, &line);
        let station_features = station.apply(&self.station_decoder);
        let line_features = line.apply(&self.line_decoder);
        let served_by_logits = dot_rows(
            &station,
            &line,
            &self.indices.serves_src,
            &self.indices.serves_dst,
        );
        let transfer_logits = dot_rows(
            &station,
            &station,
            &self.indices.transfer_src,
            &self.indices.transfer_dst,
        );
        Ok(TchReconstruction {
            embeddings: TchEmbeddings {
                station,
                line,
                city,
            },
            station_features,
            line_features,
            served_by_logits,
            transfer_logits,
        })
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.var_store
            .save(path)
            .with_context(|| format!("saving LibTorch checkpoint {}", path.display()))?;
        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        self.var_store
            .load(path)
            .with_context(|| format!("loading LibTorch checkpoint {}", path.display()))?;
        Ok(())
    }
}

fn matrix_tensor(matrix: &FeatureMatrix, device: Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(&matrix.values)
        .to_device(device)
        .reshape([matrix.rows as i64, matrix.cols as i64]))
}

fn pattern_sequence_context(
    graph: &GraphTensor,
    station: &Tensor,
    token_projection: &nn::Linear,
    recurrent_update: &nn::Linear,
    device: Device,
) -> Result<Tensor> {
    let hidden = station.size().get(1).copied().unwrap_or(0);
    let mut output = Tensor::zeros(
        [graph.manifest.line_count as i64, hidden],
        (Kind::Float, device),
    );
    let mut counts = Tensor::zeros([graph.manifest.line_count as i64, 1], (Kind::Float, device));
    let stop_features = matrix_tensor(&graph.pattern_stop_features, device)?;
    let segment_features = matrix_tensor(&graph.pattern_segment_features, device)?;
    let mut segment_offset = 0_i64;
    for pattern in 0..graph.manifest.pattern_count {
        let start = graph.pattern_offsets[pattern] as i64;
        let end = graph.pattern_offsets[pattern + 1] as i64;
        if end <= start {
            continue;
        }
        let line = graph.pattern_lines[pattern] as i64;
        let mut state = Tensor::zeros([1, hidden], (Kind::Float, device));
        for position in start..end {
            let station_index =
                Tensor::from_slice(&[graph.pattern_stops[position as usize] as i64])
                    .to_device(device);
            let station_value = station.index_select(0, &station_index);
            let stop_value = stop_features.narrow(0, position, 1);
            let local_position = position - start;
            let segment_value = if local_position < end - start - 1 {
                segment_features.narrow(0, segment_offset + local_position, 1)
            } else {
                Tensor::zeros([1, EDGE_FEATURES as i64], (Kind::Float, device))
            };
            let token = Tensor::cat(&[station_value, stop_value, segment_value], 1)
                .apply(token_projection)
                .gelu("none");
            state = Tensor::cat(&[state, token], 1)
                .apply(recurrent_update)
                .gelu("none");
        }
        let line_index = Tensor::from_slice(&[line]).to_device(device);
        let weight = graph
            .pattern_trip_counts
            .get(pattern)
            .copied()
            .unwrap_or(1)
            .max(1) as f32;
        output = output.index_add(0, &line_index, &(state.shallow_clone() * f64::from(weight)));
        counts = counts.index_add(
            0,
            &line_index,
            &Tensor::from_slice(&[weight])
                .to_device(device)
                .reshape([1, 1]),
        );
        segment_offset += (end - start - 1).max(0);
    }
    Ok(output / counts.clamp_min(1.0))
}

fn temporal_tensor(matrix: &FeatureMatrix, rows: usize, device: Device) -> Result<Tensor> {
    if matrix.cols != SERVICE_DAY_BINS * 4 {
        anyhow::bail!(
            "temporal matrix has {} columns, expected {}",
            matrix.cols,
            SERVICE_DAY_BINS * 4
        );
    }
    Ok(Tensor::from_slice(&matrix.values)
        .to_device(device)
        .reshape([rows as i64, 4, SERVICE_DAY_BINS as i64]))
}

fn pool_temporal(value: &Tensor) -> Tensor {
    let mean = value.mean_dim(&[2_i64][..], false, Kind::Float);
    let max = value.max_dim(2, false).0;
    Tensor::cat(&[mean, max], 1)
}

fn index_tensor(values: &[u32], device: Device) -> Tensor {
    Tensor::from_slice(
        &values
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}

fn visible_rows(rows: &[bool], device: Device) -> Tensor {
    Tensor::from_slice(
        &rows
            .iter()
            .map(|masked| if *masked { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
    .unsqueeze(1)
}

fn visible_temporal(blocks: &[bool], rows: usize, device: Device) -> Tensor {
    Tensor::from_slice(
        &blocks
            .iter()
            .map(|masked| if *masked { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
    .reshape([rows as i64, SERVICE_DAY_BINS as i64])
    .unsqueeze(1)
}

fn edge_visibility(mask: &[bool], device: Device) -> Tensor {
    Tensor::from_slice(
        &mask
            .iter()
            .map(|masked| if *masked { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}

fn mean_aggregate(
    source: &Tensor,
    source_indices: &Tensor,
    destination_indices: &Tensor,
    destination_count: usize,
    projection: &nn::Linear,
    visibility: Option<&Tensor>,
) -> Tensor {
    let mut messages = source.index_select(0, source_indices).apply(projection);
    let mut weights = Tensor::ones(
        [destination_indices.size()[0]],
        (Kind::Float, source.device()),
    );
    if let Some(visibility) = visibility {
        messages *= visibility.unsqueeze(1);
        weights = visibility.shallow_clone();
    }
    let hidden = messages.size()[1];
    let sums = Tensor::zeros(
        [destination_count as i64, hidden],
        (Kind::Float, source.device()),
    )
    .index_add(0, destination_indices, &messages);
    let counts = Tensor::zeros(
        [destination_count as i64, 1],
        (Kind::Float, source.device()),
    )
    .index_add(0, destination_indices, &weights.unsqueeze(1))
    .clamp_min(1.0);
    sums / counts
}

struct TransitAggregation<'a> {
    station: &'a Tensor,
    line: &'a Tensor,
    edge_features: &'a FeatureMatrix,
    source_indices: &'a Tensor,
    destination_indices: &'a Tensor,
    line_indices: &'a Tensor,
    destination_count: usize,
    station_projection: &'a nn::Linear,
    line_projection: &'a nn::Linear,
    edge_projection: &'a nn::Linear,
    visibility: Option<&'a Tensor>,
    device: Device,
}

fn transit_aggregate(args: TransitAggregation<'_>) -> Result<Tensor> {
    let edge_values = matrix_tensor(args.edge_features, args.device)?;
    let mut messages = args
        .station
        .index_select(0, args.source_indices)
        .apply(args.station_projection)
        + args
            .line
            .index_select(0, args.line_indices)
            .apply(args.line_projection)
        + edge_values.apply(args.edge_projection);
    let mut weights = Tensor::ones(
        [args.destination_indices.size()[0]],
        (Kind::Float, args.device),
    );
    if let Some(visibility) = args.visibility {
        messages *= visibility.unsqueeze(1);
        weights = visibility.shallow_clone();
    }
    let hidden = messages.size()[1];
    let sums = Tensor::zeros(
        [args.destination_count as i64, hidden],
        (Kind::Float, args.device),
    )
    .index_add(0, args.destination_indices, &messages);
    let counts = Tensor::zeros(
        [args.destination_count as i64, 1],
        (Kind::Float, args.device),
    )
    .index_add(0, args.destination_indices, &weights.unsqueeze(1))
    .clamp_min(1.0);
    Ok(sums / counts)
}

fn dot_rows(
    left: &Tensor,
    right: &Tensor,
    left_indices: &Tensor,
    right_indices: &Tensor,
) -> Tensor {
    (left.index_select(0, left_indices) * right.index_select(0, right_indices)).sum_dim_intlist(
        &[1_i64][..],
        false,
        Kind::Float,
    )
}

fn city_pool(station: &Tensor, line: &Tensor) -> Tensor {
    let station_mean = station.mean_dim(&[0_i64][..], false, Kind::Float);
    let station_max = station.max_dim(0, false).0;
    let line_mean = line.mean_dim(&[0_i64][..], false, Kind::Float);
    let line_max = line.max_dim(0, false).0;
    (station_mean + station_max + line_mean + line_max) / 4.0
}

fn validate_mask(graph: &GraphTensor, mask: &MaskSelection) -> Result<()> {
    if mask.station_rows.len() != graph.manifest.station_count
        || mask.line_rows.len() != graph.manifest.line_count
        || mask.station_temporal_blocks.len() != graph.manifest.station_count * SERVICE_DAY_BINS
        || mask.line_temporal_blocks.len() != graph.manifest.line_count * SERVICE_DAY_BINS
        || mask.served_edges.len() != graph.serves_src.len()
        || mask.transfer_edges.len() != graph.transfer_src.len()
    {
        anyhow::bail!("mask shape does not match graph shape");
    }
    Ok(())
}
