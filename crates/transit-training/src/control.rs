use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The small file-based control protocol keeps the training engine independent
/// from SQLite, Bun, and an HTTP server. The worker owns the file and the
/// trainer only reads it at safe optimizer-step boundaries.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DesiredTrainingState {
    #[default]
    Running,
    Paused,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingControlFile {
    #[serde(rename = "schemaVersion", default = "control_schema_version")]
    pub schema_version: u32,
    #[serde(rename = "desiredState", default)]
    pub desired_state: DesiredTrainingState,
    #[serde(rename = "checkpointRequested", default)]
    pub checkpoint_requested: bool,
    #[serde(rename = "requestedAt", default)]
    pub requested_at: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

fn control_schema_version() -> u32 {
    1
}

impl Default for TrainingControlFile {
    fn default() -> Self {
        Self {
            schema_version: control_schema_version(),
            desired_state: DesiredTrainingState::Running,
            checkpoint_requested: false,
            requested_at: None,
            reason: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlDirective {
    Continue,
    Checkpoint,
    Pause,
    Cancel,
}

/// Reads a control file on demand and optionally enforces an attempt deadline.
/// A missing file means continue, which makes the API safe for legacy CLI use.
pub struct TrainingControl {
    path: Option<PathBuf>,
    started_at: Instant,
    max_wall_time: Option<Duration>,
    checkpoint_grace: Duration,
}

impl TrainingControl {
    pub fn new(path: Option<PathBuf>, max_wall_time: Option<Duration>) -> Self {
        Self {
            path,
            started_at: Instant::now(),
            max_wall_time,
            checkpoint_grace: Duration::ZERO,
        }
    }

    /// Configure an attempt deadline so the trainer yields early enough to
    /// finish an atomic checkpoint before an external scheduler kills it.
    pub fn with_policy(
        path: Option<PathBuf>,
        max_wall_time: Option<Duration>,
        checkpoint_grace: Option<Duration>,
    ) -> Self {
        let mut control = Self::new(path, max_wall_time);
        control.checkpoint_grace = checkpoint_grace.unwrap_or_default();
        control
    }

    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self::new(Some(path.as_ref().to_path_buf()), None)
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn deadline_expired(&self) -> bool {
        self.max_wall_time.is_some_and(|deadline| {
            let yield_after = deadline.saturating_sub(self.checkpoint_grace);
            self.started_at.elapsed() >= yield_after
        })
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn directive(&self) -> Result<ControlDirective> {
        // Read an explicit user directive before applying the attempt
        // deadline.  Cancellation is terminal and must win over a scheduler
        // deadline; otherwise a cancelled run could be reported as a clean
        // time slice and be requeued.  A requested pause also remains a user
        // pause, even when the worker happens to be at the end of its window.
        let explicit = if let Some(path) = &self.path {
            let bytes = match fs::read(path) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("reading control file {}", path.display()))
                }
            };
            bytes
                .map(|bytes| {
                    serde_json::from_slice::<TrainingControlFile>(&bytes)
                        .with_context(|| format!("decoding control file {}", path.display()))
                })
                .transpose()?
        } else {
            None
        };

        if let Some(control) = explicit {
            if control.schema_version != control_schema_version() {
                anyhow::bail!(
                    "unsupported training control schema {}; expected {}",
                    control.schema_version,
                    control_schema_version()
                );
            }
            return Ok(match control.desired_state {
                DesiredTrainingState::Cancelled => ControlDirective::Cancel,
                DesiredTrainingState::Paused => ControlDirective::Pause,
                DesiredTrainingState::Running if control.checkpoint_requested => {
                    ControlDirective::Checkpoint
                }
                DesiredTrainingState::Running if self.deadline_expired() => {
                    ControlDirective::Checkpoint
                }
                DesiredTrainingState::Running => ControlDirective::Continue,
            });
        }

        if self.deadline_expired() {
            Ok(ControlDirective::Checkpoint)
        } else {
            Ok(ControlDirective::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_control(path: &Path, desired_state: DesiredTrainingState) {
        fs::write(
            path,
            serde_json::to_vec(&TrainingControlFile {
                desired_state,
                ..TrainingControlFile::default()
            })
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn terminal_cancel_wins_over_an_expired_attempt_deadline() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("control.json");
        write_control(&path, DesiredTrainingState::Cancelled);
        let control =
            TrainingControl::with_policy(Some(path), Some(Duration::ZERO), Some(Duration::ZERO));
        assert_eq!(control.directive().unwrap(), ControlDirective::Cancel);
    }

    #[test]
    fn an_expired_deadline_requests_checkpoint_only_for_a_running_attempt() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("control.json");
        write_control(&path, DesiredTrainingState::Running);
        let control =
            TrainingControl::with_policy(Some(path), Some(Duration::ZERO), Some(Duration::ZERO));
        assert_eq!(control.directive().unwrap(), ControlDirective::Checkpoint);
    }
}
