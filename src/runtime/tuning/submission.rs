use super::super::{BinaryData, Gpu, Session, UnaryParams, record_groups};
use crate::tune::{
    TuneDecision, TuneError, TuneOptions, TuneOutcome, TuneSubmissionOptions, TuneSubmissionReport,
    decide, measure_pairs,
};
use blade_graphics::{self as bg, ShaderData};
use std::time::{Duration, Instant};

struct Commands<'a> {
    gpu: &'a Gpu,
    encoder: bg::CommandEncoder,
    sync: Option<bg::SyncPoint>,
}

impl<'a> Commands<'a> {
    fn new(gpu: &'a Gpu, chunks: usize) -> Self {
        Self {
            gpu,
            encoder: gpu.create_command_encoder(bg::CommandEncoderDesc {
                name: "submission_tune",
                buffer_count: chunks as u32 + 1,
                manual_barriers: false,
            }),
            sync: None,
        }
    }

    fn wait(&mut self) -> Result<(), TuneError> {
        if let Some(ref sync) = self.sync {
            if !self
                .gpu
                .wait_for(sync, !0)
                .map_err(|_| TuneError("GPU wait failed"))?
            {
                return Err(TuneError("GPU wait did not complete"));
            }
            self.sync = None;
        }
        Ok(())
    }

    fn submit_wait(&mut self) -> Result<(), TuneError> {
        self.sync = Some(self.gpu.submit(&mut self.encoder));
        self.wait()
    }
}

impl Drop for Commands<'_> {
    fn drop(&mut self) {
        let _ = self.wait();
        self.gpu.destroy_command_encoder(&mut self.encoder);
    }
}

struct Image {
    source: bg::Buffer,
    trial: bg::Buffer,
    reference: bg::Buffer,
    size: u64,
}

struct Probe<'a> {
    commands: Commands<'a>,
    buffers: Vec<bg::Buffer>,
    images: Vec<Image>,
    flag: bg::Buffer,
    compare: bg::ComputePipeline,
}

impl<'a> Probe<'a> {
    fn new(session: &'a Session, written: &[bool]) -> Self {
        let gpu = &session.gpu;
        let shader = gpu.create_shader(bg::ShaderDesc {
            source: include_str!("../../shaders/compare_words.wgsl"),
            naga_module: None,
        });
        let compare = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: "compare_words",
            data_layouts: &[&BinaryData::layout()],
            compute: shader.at("main"),
        });
        let flag = gpu.create_buffer(bg::BufferDesc {
            name: "submission_tune_flag",
            size: 4,
            memory: bg::Memory::Shared,
        });
        let mut probe = Self {
            commands: Commands::new(gpu, 1),
            buffers: session.buffers.clone(),
            images: Vec::new(),
            flag,
            compare,
        };
        for (physical, &write) in written.iter().enumerate() {
            if !write {
                continue;
            }
            let size = session.alias.sizes[physical].max(4) as u64;
            let trial = gpu.create_buffer(bg::BufferDesc {
                name: "submission_tune_image",
                size,
                memory: if session.alias.device_local[physical] {
                    bg::Memory::DeviceTransient
                } else {
                    bg::Memory::Shared
                },
            });
            let reference = gpu.create_buffer(bg::BufferDesc {
                name: "submission_tune_reference",
                size,
                memory: bg::Memory::DeviceTransient,
            });
            probe.images.push(Image {
                source: session.physical_buffers[physical].handle,
                trial,
                reference,
                size,
            });
            for (logical, &p) in session.alias.map.iter().enumerate() {
                if p == physical {
                    probe.buffers[logical] = trial;
                }
            }
        }
        probe
    }

    fn copy_images(&mut self, save: bool) -> Result<(), TuneError> {
        self.commands.encoder.start();
        {
            let mut pass = self.commands.encoder.transfer("submission_tune_copy");
            for image in &self.images {
                let (src, dst) = if save {
                    (image.trial, image.reference)
                } else {
                    (image.source, image.trial)
                };
                pass.copy_buffer_to_buffer(src.at(0), dst.at(0), image.size);
            }
        }
        self.commands.submit_wait()
    }

    fn run(
        &mut self,
        session: &Session,
        trial: &mut Commands<'_>,
        chunks: usize,
    ) -> Result<f64, TuneError> {
        self.copy_images(false)?;
        let start = Instant::now();
        trial.encoder.start();
        record_groups(
            &session.gpu,
            &mut trial.encoder,
            &mut trial.sync,
            &session.plan,
            &session.groups,
            &session.pipelines,
            &self.buffers,
            chunks,
        );
        trial.submit_wait()?;
        Ok(start.elapsed().as_secs_f64() * 1e3)
    }

    fn matches(&mut self) -> Result<bool, TuneError> {
        self.commands.encoder.start();
        self.commands
            .encoder
            .transfer("submission_tune_zero")
            .fill_buffer(self.flag.at(0), 4, 0);
        {
            let mut pass = self.commands.encoder.compute("submission_tune_check");
            for image in &self.images {
                let len = (image.size / 4) as u32;
                let mut pc = pass.with(&self.compare);
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: image.trial.at(0),
                        src_b: image.reference.at(0),
                        dst: self.flag.at(0),
                        params: UnaryParams {
                            len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
                pc.dispatch([len.div_ceil(256).min(0xFFFF), 1, 1]);
            }
        }
        self.commands.submit_wait()?;
        Ok(unsafe { self.flag.data().cast::<u32>().read_volatile() } == 0)
    }

    fn search(
        &mut self,
        session: &Session,
        policy: &TuneOptions,
        start: Instant,
        report: &mut TuneSubmissionReport,
    ) -> Result<(), TuneError> {
        let initial = report.selected;
        let expired = || start.elapsed() >= policy.max_time;
        let mut incumbent = Commands::new(&session.gpu, initial);
        for repeat in 0..2 {
            if expired() {
                report.skipped = Some(TuneDecision::TimeBudget);
                return Ok(());
            }
            self.run(session, &mut incumbent, initial)?;
            if repeat == 0 {
                self.copy_images(true)?;
            } else if !self.matches()? {
                report.skipped = Some(TuneDecision::InvalidOutput);
                return Ok(());
            }
        }
        let mut candidates = vec![1];
        while candidates.last().unwrap() * 2 <= report.max_chunks {
            candidates.push(candidates.last().unwrap() * 2);
        }
        if candidates.last() != Some(&report.max_chunks) {
            candidates.push(report.max_chunks);
        }
        for chunks in candidates.into_iter().filter(|&n| n != initial) {
            if expired() {
                report.skipped = Some(TuneDecision::TimeBudget);
                break;
            }
            let began = Instant::now();
            let mut outcome =
                TuneOutcome::new((), session.plan.dispatches.len(), report.selected, chunks);
            outcome.phase_times = None;
            let mut challenger = Commands::new(&session.gpu, chunks);
            self.run(session, &mut challenger, chunks)?;
            outcome.qualified = self.matches()?;
            if outcome.qualified {
                for _ in 0..policy.warmup_runs {
                    if expired() {
                        break;
                    }
                    self.run(session, &mut incumbent, report.selected)?;
                    if expired() {
                        break;
                    }
                    self.run(session, &mut challenger, chunks)?;
                }
                let mut failure = None;
                (outcome.baseline_ms, outcome.candidate_ms) =
                    measure_pairs(policy.sample_pairs, |alternative| {
                        if expired() {
                            return None;
                        }
                        let (commands, count) = if alternative {
                            (&mut challenger, chunks)
                        } else {
                            (&mut incumbent, report.selected)
                        };
                        match self
                            .run(session, commands, count)
                            .and_then(|ms| self.matches().map(|valid| (ms, valid)))
                        {
                            Ok((ms, true)) => (!expired()).then_some(ms),
                            Ok((_, false)) => {
                                outcome.qualified = false;
                                None
                            }
                            Err(error) => {
                                failure = Some(error);
                                None
                            }
                        }
                    });
                if let Some(error) = failure {
                    return Err(error);
                }
                if outcome.qualified {
                    decide(&mut outcome, policy);
                } else {
                    outcome.decision = TuneDecision::InvalidOutput;
                    outcome.failure =
                        Some("timed writable image differs from original schedule".to_owned());
                }
            } else {
                outcome.decision = TuneDecision::InvalidOutput;
                outcome.failure = Some("writable image differs from original schedule".to_owned());
            }
            if outcome.selected != report.selected {
                report.selected = outcome.selected;
                std::mem::swap(&mut incumbent, &mut challenger);
            }
            if outcome.decision == TuneDecision::TimeBudget {
                report.skipped = Some(TuneDecision::TimeBudget);
            }
            outcome.elapsed = began.elapsed();
            report.outcomes.push(outcome);
        }
        Ok(())
    }
}

impl Drop for Probe<'_> {
    fn drop(&mut self) {
        let _ = self.commands.wait();
        let gpu = self.commands.gpu;
        gpu.destroy_compute_pipeline(&mut self.compare);
        gpu.destroy_buffer(self.flag);
        for image in self.images.drain(..) {
            gpu.destroy_buffer(image.trial);
            gpu.destroy_buffer(image.reference);
        }
    }
}

impl Session {
    /// Measure fresh-command submission counts on the current graph inputs.
    ///
    /// Initialize representative inputs, weights and caches first. Trials borrow
    /// readonly allocations and copy writable ones, preserving aliasing and
    /// placement without modifying live state. Keep shared inputs unchanged and
    /// avoid competing workloads. Runtime-appended optimizer passes and output
    /// readback are not measured.
    ///
    /// Complete writable images must match the original schedule bitwise. A
    /// non-repeatable baseline keeps the incumbent. The soft deadline is checked
    /// between probes; in-flight work cannot be preempted. Only the selected
    /// submission count is installed, replacing the caller's chunk policy.
    pub fn tune_submissions(
        &mut self,
        options: TuneSubmissionOptions,
    ) -> Result<TuneSubmissionReport, TuneError> {
        let policy = options.policy()?;
        if self.profile_window.is_some() {
            return Err(TuneError(
                "disable per-dispatch profiling before scheduling probes",
            ));
        }
        let initial = self.submission_chunks.min(self.groups.len().max(1));
        let max_chunks = options.max_chunks.min(self.groups.len().max(1));
        if initial > max_chunks {
            return Err(TuneError("current submission count exceeds search bound"));
        }
        let start = Instant::now();
        let mut report = TuneSubmissionReport {
            selected: initial,
            max_chunks,
            options,
            outcomes: Vec::new(),
            scratch_bytes: 0,
            elapsed: Duration::ZERO,
            skipped: None,
        };
        if max_chunks == 1 || start.elapsed() >= policy.max_time {
            report.skipped = (max_chunks != 1).then_some(TuneDecision::TimeBudget);
            report.elapsed = start.elapsed();
            return Ok(report);
        }
        let mut written = vec![false; self.alias.sizes.len()];
        for d in &self.plan.dispatches {
            for b in std::iter::once(&d.output_buffer).chain(&d.extra_outputs) {
                written[self.alias.map[b.0 as usize]] = true;
            }
        }
        for (p, &write) in written.iter().enumerate() {
            if write {
                let size = self.alias.sizes[p].max(4);
                if !size.is_multiple_of(4) || size / 4 > u32::MAX as usize {
                    return Err(TuneError(
                        "submission probe requires word-aligned writable images",
                    ));
                }
                report.scratch_bytes = report
                    .scratch_bytes
                    .checked_add(
                        size.checked_mul(2)
                            .ok_or(TuneError("submission scratch size overflow"))?,
                    )
                    .ok_or(TuneError("submission scratch size overflow"))?;
            }
        }
        report.scratch_bytes = report
            .scratch_bytes
            .checked_add(4)
            .ok_or(TuneError("submission scratch size overflow"))?;
        let memory = self.gpu.memory_stats();
        if report.scratch_bytes > policy.max_scratch_bytes {
            report.skipped = Some(TuneDecision::ScratchLimit);
        } else if memory.budget != 0
            && report.scratch_bytes as u64
                > super::super::safe_device_memory_remaining(memory.usage, memory.budget)
        {
            report.skipped = Some(TuneDecision::DeviceMemoryBudget);
        }
        if report.skipped.is_some() {
            report.elapsed = start.elapsed();
            return Ok(report);
        }
        self.wait();
        if start.elapsed() >= policy.max_time {
            report.skipped = Some(TuneDecision::TimeBudget);
        } else {
            let mut probe = Probe::new(self, &written);
            probe.search(self, &policy, start, &mut report)?;
        }
        if report.selected != initial {
            self.set_submission_chunks(report.selected);
        }
        report.elapsed = start.elapsed();
        log::info!(
            "submission tune: {initial} -> {}, {} comparisons in {:.3}s, skipped {:?}",
            report.selected,
            report.outcomes.len(),
            report.elapsed.as_secs_f64(),
            report.skipped
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::{Commands, Duration, Probe, TuneDecision, TuneSubmissionOptions};

    #[test]
    fn writable_image_comparison_checks_changed_tail_bits() {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[8, 16]);
        let w = graph.parameter("w", &[16, 16]);
        let y = graph.matmul(x, w);
        let z = graph.matmul(y, w);
        graph.set_outputs(vec![z]);
        let mut config = crate::SessionConfig::inference_from_env();
        config.tune = false;
        config.runtime.coop = crate::CoopPolicy::Disabled;
        let mut session = crate::build(&graph, config).0;
        session.set_input("x", &[0.5; 128]);
        session.set_parameter("w", &[0.125; 256]);
        session.step();
        session.wait();
        let mut before = vec![0.0; 128];
        session.read_output_by_index(0, &mut before);
        assert!(before.iter().all(|&v| v == 2.0));
        assert!(
            session
                .tune_submissions(TuneSubmissionOptions {
                    max_chunks: 0,
                    ..Default::default()
                })
                .is_err()
        );
        for (options, skipped) in [
            (
                TuneSubmissionOptions {
                    max_time: Duration::ZERO,
                    ..Default::default()
                },
                Some(TuneDecision::TimeBudget),
            ),
            (
                TuneSubmissionOptions {
                    max_scratch_bytes: 0,
                    ..Default::default()
                },
                Some(TuneDecision::ScratchLimit),
            ),
            (
                TuneSubmissionOptions {
                    max_chunks: 1,
                    ..Default::default()
                },
                None,
            ),
        ] {
            let report = session.tune_submissions(options).unwrap();
            assert_eq!(report.selected, 1);
            assert_eq!(report.skipped, skipped);
            assert!(report.outcomes.is_empty());
        }
        session.set_profiling(true);
        assert!(
            session
                .tune_submissions(TuneSubmissionOptions::default())
                .is_err()
        );
        session.set_profiling(false);
        let mut written = vec![false; session.alias.sizes.len()];
        for d in &session.plan.dispatches {
            written[session.alias.map[d.output_buffer.0 as usize]] = true;
        }
        let mut probe = Probe::new(&session, &written);
        let mut commands = Commands::new(&session.gpu, 2);
        probe.run(&session, &mut commands, 1).unwrap();
        probe.copy_images(true).unwrap();
        probe.run(&session, &mut commands, 2).unwrap();
        assert!(probe.matches().unwrap());
        for i in 0..probe.images.len() {
            probe.copy_images(true).unwrap();
            let image = &probe.images[i];
            probe.commands.encoder.start();
            probe.commands.encoder.transfer("corrupt_tail").fill_buffer(
                image.reference.at(image.size - 4),
                4,
                u8::MAX,
            );
            probe.commands.submit_wait().unwrap();
            assert!(!probe.matches().unwrap());
        }
        let mut after = vec![0.0; 128];
        session.read_output_by_index(0, &mut after);
        assert_eq!(after, before);
    }
}
