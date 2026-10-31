//! Class races: a class's incumbent and challengers timed together.
//!
//! Every candidate's pipelines compile in one parallel batch, then all of them
//! run in shared forward/reverse rounds on one scratch. With GPU timestamps
//! a round is one submission with a timed pass per candidate; otherwise each
//! candidate is a host-timed submission. Challengers that can no longer pass
//! the decision rule are dropped without qualification, and only a finalist
//! that would be installed is qualified. See `docs/search-convergence.md`.

use super::{
    ENCODER_SCRATCH_BYTES, Scratch, SearchClass, Staging, outputs_agree, prepared_inputs,
    qualify_output, reordered_outputs_agree, scratch_bytes, small_pattern_scale, typed_inputs,
};
use crate::compile::{BufferRef, Dispatch};
use crate::runtime::{Pipelines, Session, Variant};
use crate::tune::{
    MachineProfile, MatmulTile, TuneDecision, TuneOptions, TuneOutcome, TunePreparationTimes,
    TuneQualificationTimes, TuneRace, decide, median,
};
use blade_graphics as bg;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Rounds every challenger gets before elimination starts.
const ELIMINATE_AFTER: usize = 4;
/// Challengers kept after the first elimination: successive halving's width.
const FINALISTS: usize = 4;
/// Target share of a timed pass for pass overhead (subject to the repeat cap).
const PASS_OVERHEAD_SHARE: f64 = 0.02;

pub(super) struct RaceResult {
    pub(super) selected: MatmulTile,
    pub(super) outcomes: Vec<TuneOutcome>,
    pub(super) race: TuneRace,
}

type Run<'a> = Vec<(&'a bg::ComputePipeline, &'a Dispatch)>;

/// Blade reclaims descriptor caches unused in the previous recording. A
/// shared host encoder cycling through many layouts would repeatedly measure
/// allocation/destruction instead of steady-state execution. Keep one encoder
/// per candidate (still one shared scratch); timestamp rounds already record
/// every live layout together and need only the scratch encoder.
struct HostEncoders<'a> {
    gpu: &'a bg::Context,
    encoders: Vec<Option<bg::CommandEncoder>>,
}

impl<'a> HostEncoders<'a> {
    fn new(gpu: &'a bg::Context, alive: &[bool], timestamps: bool) -> Self {
        Self {
            gpu,
            encoders: if timestamps {
                Vec::new()
            } else {
                alive
                    .iter()
                    .map(|&live| {
                        live.then(|| {
                            gpu.create_command_encoder(bg::CommandEncoderDesc {
                                name: "tune_race_host",
                                buffer_count: 1,
                                manual_barriers: false,
                            })
                        })
                    })
                    .collect()
            },
        }
    }
}

impl Drop for HostEncoders<'_> {
    fn drop(&mut self) {
        for encoder in self.encoders.iter_mut().flatten() {
            self.gpu.destroy_command_encoder(encoder);
        }
    }
}

impl Session {
    /// Time `incumbent` and every other challenger of `class` together, and
    /// return the qualified winner, if one passes the decision rule.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn race_class(
        &mut self,
        class: &SearchClass,
        incumbent: MatmulTile,
        options: &TuneOptions,
        start: Instant,
        staging: &mut Staging<'_>,
        profile: Option<&MachineProfile>,
        predicted_ms: Option<f64>,
    ) -> RaceResult {
        let race_start = Instant::now();
        let mut tiles = vec![incumbent];
        for &tile in &class.challengers {
            if !tiles.contains(&tile) {
                tiles.push(tile);
            }
        }
        let mut outcomes: Vec<TuneOutcome> = tiles[1..]
            .iter()
            .map(|&tile| {
                let mut outcome =
                    TuneOutcome::new(class.key.clone(), class.members.len(), incumbent, tile);
                // Phases are shared by the whole race; see `TuneRace`.
                outcome.phase_times = None;
                outcome
            })
            .collect();
        let mut race = TuneRace {
            class: class.key.clone(),
            dispatches: class.members.len(),
            initial: incumbent,
            selected: incumbent,
            candidates: 0,
            rounds: 0,
            gpu_timestamps: false,
            predicted_ms,
            incumbent_ms: None,
            complete: true,
            compile_time: Duration::ZERO,
            preparation: Duration::ZERO,
            racing: Duration::ZERO,
            qualification: Duration::ZERO,
            elapsed: Duration::ZERO,
        };
        let conclude = |mut race: TuneRace, outcomes: Vec<TuneOutcome>| {
            race.complete &= outcomes.iter().all(decided_outcome);
            race.elapsed = race_start.elapsed();
            RaceResult {
                selected: race.selected,
                outcomes,
                race,
            }
        };

        // Bindings sized for the largest candidate, as one comparison would.
        let preparation = Instant::now();
        let PreparedRace {
            logical_sizes,
            sizes,
            output_index,
            sequences,
        } = prepare_race(class, &self.plan.dispatches[class.members[0]], &tiles);
        let Some(bytes) = scratch_bytes(&sizes)
            .and_then(|bytes| {
                bytes.checked_add(sizes[1].checked_mul((class.input_b_copies - 1) as usize)?)
            })
            .filter(|&bytes| bytes <= options.max_scratch_bytes)
        else {
            fail_remaining(&mut outcomes, TuneDecision::ScratchLimit);
            return conclude(race, outcomes);
        };
        staging.discard_unmatched(*sizes.iter().max().unwrap());
        staging.discard_input_b(&class.key, sizes[1]);
        let memory = self.gpu.memory_stats();
        if memory.budget != 0
            && (bytes - staging.retained_bytes()) as u64
                > crate::runtime::safe_device_memory_remaining(memory.usage, memory.budget)
        {
            fail_remaining(&mut outcomes, TuneDecision::DeviceMemoryBudget);
            return conclude(race, outcomes);
        }

        let Some(mut alive) = self.compile_race(
            &sequences,
            &tiles,
            incumbent,
            options,
            start,
            &mut outcomes,
            &mut race,
        ) else {
            return conclude(race, outcomes);
        };
        race.candidates = alive.iter().filter(|&&alive| alive).count();
        if race.candidates == 1 {
            return conclude(race, outcomes);
        }
        if start.elapsed() >= options.max_time {
            for (outcome, &alive) in outcomes.iter_mut().zip(&alive[1..]) {
                if alive {
                    outcome.decision = TuneDecision::TimeBudget;
                }
            }
            return conclude(race, outcomes);
        }

        let timestamps = self.gpu_timing && self.gpu.capabilities().timing;
        race.gpu_timestamps = timestamps;
        let Some(bytes) =
            race_scratch_bytes(bytes, alive.iter().filter(|&&v| v).count(), timestamps)
                .filter(|&bytes| bytes <= options.max_scratch_bytes)
        else {
            fail_remaining(&mut outcomes, TuneDecision::ScratchLimit);
            return conclude(race, outcomes);
        };
        let memory = self.gpu.memory_stats();
        if memory.budget != 0
            && (bytes - staging.retained_bytes()) as u64
                > crate::runtime::safe_device_memory_remaining(memory.usage, memory.budget)
        {
            fail_remaining(&mut outcomes, TuneDecision::DeviceMemoryBudget);
            return conclude(race, outcomes);
        }

        let device_local: Vec<bool> = (0..sizes.len())
            .map(|i| {
                i > output_index || class.key.device_local[if i == output_index { 3 } else { i }]
            })
            .collect();
        let mut preparation_times = TunePreparationTimes::default();
        let mut cleanup = None;
        let mut scratch = Scratch::new(
            &device_local,
            &sizes,
            (&class.key, output_index),
            bytes,
            staging,
            &mut preparation_times,
            &mut cleanup,
        );
        // Ordinary-magnitude inputs, the bytes qualification ends on as well.
        let inputs = race_inputs(class, &logical_sizes, &sizes, 0);
        for (index, data) in inputs.iter().enumerate() {
            if start.elapsed() >= options.max_time {
                fail_remaining(&mut outcomes, TuneDecision::TimeBudget);
                return conclude(race, outcomes);
            }
            if index != 1 || !scratch.input_b_reused {
                scratch.upload(index, data, None);
            }
        }
        scratch.input_b_uploaded = true;
        drop(inputs);
        {
            let _timer = super::PhaseTimer::new(&mut preparation_times.buffers);
            scratch.replicate_input_b(class.input_b_copies, device_local[1]);
        }
        let runs: Vec<Option<Run<'_>>> = sequences
            .iter()
            .zip(&alive)
            .map(|(sequence, &alive)| {
                alive.then(|| {
                    sequence
                        .iter()
                        .map(|dispatch| {
                            (&self.pipelines.map[&Pipelines::key(dispatch)].raw, dispatch)
                        })
                        .collect()
                })
            })
            .collect();
        let mut host = HostEncoders::new(&self.gpu, &alive, timestamps);
        race.preparation = preparation.elapsed().saturating_sub(race.compile_time);

        let racing = Instant::now();
        race_rounds(
            &mut scratch,
            &mut host,
            &runs,
            &mut alive,
            options,
            profile,
            start,
            &mut outcomes,
            &mut race,
        );
        drop(host);
        race.racing = racing.elapsed();

        let qualification = Instant::now();
        qualify_finalists(
            &mut scratch,
            class,
            &logical_sizes,
            &sizes,
            &runs,
            &tiles,
            options,
            start,
            &mut outcomes,
            &mut race,
        );
        race.qualification = qualification.elapsed();
        drop(runs);
        drop(scratch);
        conclude(race, outcomes)
    }

    fn compile_race(
        &mut self,
        sequences: &[Vec<Dispatch>],
        tiles: &[MatmulTile],
        incumbent: MatmulTile,
        options: &TuneOptions,
        start: Instant,
        outcomes: &mut [TuneOutcome],
        race: &mut TuneRace,
    ) -> Option<Vec<bool>> {
        // Compile every missing pipeline in one parallel batch.
        let compile = Instant::now();
        let mut wanted: Vec<(Variant, &Dispatch, Option<crate::codegen::CoopConfig>)> = Vec::new();
        for (sequence, &tile) in sequences.iter().zip(tiles) {
            for (position, dispatch) in sequence.iter().enumerate() {
                let key = Pipelines::key(dispatch);
                if self.pipelines.map.contains_key(&key) || wanted.iter().any(|w| w.0 == key) {
                    continue;
                }
                let coop = if position == 0 {
                    tile.coop_config()
                } else {
                    None
                };
                wanted.push((key, dispatch, coop));
            }
        }
        let jobs: Vec<_> = wanted
            .iter()
            .map(|&(ref key, dispatch, ref coop)| (key.clone(), dispatch, coop.as_ref()))
            .collect();
        let compiled = Pipelines::compile_all(
            &self.gpu,
            self.pipelines.knobs,
            self.pipelines.dump_dir.as_deref(),
            &jobs,
            Some((start, options.max_time)),
        );
        let mut rejected: HashMap<Variant, String> = HashMap::new();
        let mut interrupted = false;
        for ((key, ..), result) in jobs.into_iter().zip(compiled) {
            match result {
                Some(Ok(pipeline)) => {
                    self.pipelines.map.insert(key, pipeline);
                }
                Some(Err(error)) => {
                    rejected.insert(key, error);
                }
                None => interrupted = true,
            }
        }
        race.compile_time = compile.elapsed();
        if interrupted {
            fail_remaining(outcomes, TuneDecision::TimeBudget);
            return None;
        }
        let mut alive = vec![true; tiles.len()];
        for (index, sequence) in sequences.iter().enumerate() {
            let Some(error) = sequence
                .iter()
                .find_map(|dispatch| rejected.get(&Pipelines::key(dispatch)))
            else {
                continue;
            };
            if index == 0 {
                race.complete = false;
                for outcome in outcomes.iter_mut() {
                    outcome.decision = TuneDecision::ShaderRejected;
                    outcome.failure = Some(format!("incumbent {incumbent:?}: {error}"));
                }
                return None;
            }
            alive[index] = false;
            let outcome = &mut outcomes[index - 1];
            outcome.decision = TuneDecision::ShaderRejected;
            outcome.failure = Some(format!("{:?}: {error}", tiles[index]));
        }
        Some(alive)
    }
}

/// Bound the modeled weight working set by the caller's sample and memory
/// limits. Reserve every potential host encoder before choosing a copy count,
/// so compilation failures cannot change a memo's sampling contract.
pub(super) fn weight_copies(
    class: &SearchClass,
    options: &TuneOptions,
    distinct: usize,
    timestamps: bool,
) -> u32 {
    let sizes = class
        .key
        .buffer_sizes()
        .expect("class collection checked extents");
    let Some(bytes) = scratch_bytes(&sizes)
        .and_then(|bytes| race_scratch_bytes(bytes, class.challengers.len() + 1, timestamps))
    else {
        return 1;
    };
    let extra = options.max_scratch_bytes.saturating_sub(bytes) / sizes[1].max(1);
    distinct
        .min(options.dispatches_per_sample as usize)
        .min(extra.saturating_add(1))
        .max(1) as u32
}

struct PreparedRace {
    logical_sizes: Vec<usize>,
    sizes: Vec<usize>,
    output_index: usize,
    sequences: Vec<Vec<Dispatch>>,
}

/// Private bindings cover the largest tile, including split-product reductions.
fn prepare_race(class: &SearchClass, template: &Dispatch, tiles: &[MatmulTile]) -> PreparedRace {
    let logical_sizes = class
        .key
        .buffer_sizes()
        .expect("class collection checked extents");
    let mut sizes: Vec<usize> = Vec::new();
    for &tile in tiles {
        let tile_sizes = tile.buffer_sizes(&class.key).expect("legal candidate");
        sizes.resize(tile_sizes.len(), 0);
        for (size, tile_size) in sizes.iter_mut().zip(tile_sizes) {
            *size = (*size).max(tile_size.next_multiple_of(4));
        }
    }
    let output_index = sizes.len() - 1;
    let mut template = template.clone();
    template.input_buffers = (0..output_index).map(|i| BufferRef(i as u32)).collect();
    template.output_buffer = BufferRef(output_index as u32);
    if let Some(ref mut norm) = template.gemv_rmsnorm {
        template.input_buffers = vec![BufferRef(0), BufferRef(1)];
        norm.weight = BufferRef(2);
        template.output_buffer = BufferRef(3);
    }
    let mut sequences: Vec<Vec<Dispatch>> = tiles
        .iter()
        .map(|&tile| {
            let mut dispatch = template.clone();
            tile.apply(&mut dispatch, &class.key);
            vec![dispatch]
        })
        .collect();
    if class.key.split_k {
        // Every candidate writes partitions into a partial buffer of the
        // live size, and a SumRows adds them into the output.
        let partial = BufferRef(sizes.len() as u32);
        let columns = class.key.m * class.key.n;
        sizes[output_index] = logical_sizes[output_index].next_multiple_of(4);
        sizes.push(class.key.binding_bytes[output_index]);
        for (sequence, tile) in sequences.iter_mut().zip(tiles) {
            let splits = tile
                .partitions()
                .expect("split products only take partitioned candidates");
            let sum = sequence[0].partial_sum(partial, sequence[0].output_buffer, columns, splits);
            sequence[0].output_buffer = partial;
            sequence.push(sum);
        }
    }
    PreparedRace {
        logical_sizes,
        sizes,
        output_index,
        sequences,
    }
}

/// Warm all compiled candidates, then collect balanced rounds and eliminate.
fn race_rounds(
    scratch: &mut Scratch<'_, '_>,
    host: &mut HostEncoders<'_>,
    runs: &[Option<Run<'_>>],
    alive: &mut [bool],
    options: &TuneOptions,
    profile: Option<&MachineProfile>,
    start: Instant,
    outcomes: &mut [TuneOutcome],
    race: &mut TuneRace,
) {
    let timestamps = race.gpu_timestamps;
    let mut fastest = f64::INFINITY;
    for round in 0..options.warmup_runs {
        let order = round_order(alive, round as usize);
        let warm = run_round(
            scratch,
            host,
            runs,
            &order,
            (scratch.input_b_copies.len() + 1) as u32,
            timestamps,
            start,
            options.max_time,
        );
        let Some(warm) = warm else {
            fail_remaining(
                outcomes,
                if start.elapsed() >= options.max_time {
                    TuneDecision::TimeBudget
                } else {
                    TuneDecision::InvalidTiming
                },
            );
            return;
        };
        fastest = warm.into_iter().fold(fastest, f64::min);
    }
    let reps = repetitions(
        options,
        profile,
        timestamps,
        runs[0].as_ref().unwrap().len(),
        fastest,
    )
    .max((scratch.input_b_copies.len() + 1) as u32);
    // samples[candidate][round]: a dropped candidate has no later samples.
    let mut samples: Vec<Vec<Option<f64>>> = vec![Vec::new(); alive.len()];
    let max_rounds = options.sample_pairs;
    let mut invalid_timing = false;
    while race.rounds < max_rounds && start.elapsed() < options.max_time {
        let order = round_order(alive, race.rounds);
        let Some(times) = run_round(
            scratch,
            host,
            runs,
            &order,
            reps,
            timestamps,
            start,
            options.max_time,
        ) else {
            invalid_timing = start.elapsed() < options.max_time;
            break;
        };
        // A round that only finished after the deadline is not evidence.
        if start.elapsed() >= options.max_time {
            break;
        }
        for column in &mut samples {
            column.push(None);
        }
        for (&candidate, ms) in order.iter().zip(times) {
            *samples[candidate].last_mut().unwrap() = Some(ms);
        }
        race.rounds += 1;
        // Keep the same candidates for both halves of a forward/reverse pair.
        if race.rounds >= ELIMINATE_AFTER && race.rounds.is_multiple_of(2) {
            eliminate(&samples, alive, outcomes, options.min_improvement);
        }
        let challengers: Vec<usize> = (1..alive.len()).filter(|&i| alive[i]).collect();
        if challengers.is_empty()
            || (race.rounds >= ELIMINATE_AFTER
                && challengers
                    .iter()
                    .all(|&i| pairs(&samples, i).len() >= options.sample_pairs))
        {
            break;
        }
    }
    let incumbent_samples: Vec<f64> = samples[0].iter().flatten().copied().collect();
    race.incumbent_ms = (!incumbent_samples.is_empty()).then(|| median(&incumbent_samples));

    decide_race(&samples, reps, invalid_timing, options, outcomes);
}

fn decide_race(
    samples: &[Vec<Option<f64>>],
    reps: u32,
    invalid_timing: bool,
    options: &TuneOptions,
    outcomes: &mut [TuneOutcome],
) {
    // Decide every challenger from the rounds it shared with the incumbent.
    for candidate in 1..samples.len() {
        let outcome = &mut outcomes[candidate - 1];
        if outcome.decision == TuneDecision::ShaderRejected {
            continue;
        }
        outcome.sample_repetitions = Some(reps);
        let paired = pairs(samples, candidate);
        if outcome.decision == TuneDecision::Eliminated {
            outcome.baseline_ms = paired.iter().map(|pair| pair.0).collect();
            outcome.candidate_ms = paired.iter().map(|pair| pair.1).collect();
            outcome.baseline_median_ms = Some(median(&outcome.baseline_ms));
            outcome.candidate_median_ms = Some(median(&outcome.candidate_ms));
            continue;
        }
        if invalid_timing {
            outcome.decision = TuneDecision::InvalidTiming;
            continue;
        }
        let recent = &paired[paired.len().saturating_sub(options.sample_pairs)..];
        outcome.baseline_ms = recent.iter().map(|pair| pair.0).collect();
        outcome.candidate_ms = recent.iter().map(|pair| pair.1).collect();
        // Provisionally qualified: only a timing winner is checked below.
        outcome.qualified = true;
        decide(outcome, options);
        outcome.qualified = false;
    }
}

fn qualify_finalists(
    scratch: &mut Scratch<'_, '_>,
    class: &SearchClass,
    logical_sizes: &[usize],
    sizes: &[usize],
    runs: &[Option<Run<'_>>],
    tiles: &[MatmulTile],
    options: &TuneOptions,
    start: Instant,
    outcomes: &mut [TuneOutcome],
    race: &mut TuneRace,
) {
    let incumbent = race.initial;
    // Qualify winners in measured preference order until one passes.
    // Tiny/noisy differences between finalists must not override the same
    // improvement guard that protected the original incumbent.
    let winners = finalist_order(outcomes, options);
    let mut reference: [Option<Vec<f32>>; 2] = Default::default();
    let mut incumbent_failed = false;
    for winner in winners {
        let outcome = &mut outcomes[winner];
        if race.selected != incumbent || incumbent_failed {
            // Another finalist was faster, or nothing can be compared.
            outcome.decision = TuneDecision::Eliminated;
            outcome.selected = incumbent;
            continue;
        }
        if start.elapsed() >= options.max_time {
            outcome.decision = TuneDecision::TimeBudget;
            outcome.selected = incumbent;
            continue;
        }
        let checked = qualify(
            scratch,
            class,
            logical_sizes,
            sizes,
            [
                runs[0].as_ref().unwrap(),
                runs[winner + 1].as_ref().unwrap(),
            ],
            &mut reference,
            start,
            options.max_time,
        );
        match checked {
            Ok(()) => {
                outcome.qualified = true;
                race.selected = tiles[winner + 1];
            }
            Err(QualificationFailure::Output {
                by_incumbent,
                failure,
            }) => {
                incumbent_failed = by_incumbent;
                race.complete &= !by_incumbent;
                outcome.decision = TuneDecision::InvalidOutput;
                outcome.failure = Some(failure);
                outcome.selected = incumbent;
            }
            Err(QualificationFailure::TimeBudget) => {
                outcome.decision = TuneDecision::TimeBudget;
                outcome.selected = incumbent;
            }
        }
    }
}

/// A resource failure is retryable; keep any independent shader rejection.
fn fail_remaining(outcomes: &mut [TuneOutcome], decision: TuneDecision) {
    for outcome in outcomes {
        if outcome.decision != TuneDecision::ShaderRejected {
            outcome.decision = decision;
        }
    }
}

/// `scratch` already includes the shared scratch encoder. Timestamp rounds
/// reuse it; host rounds additionally retain one encoder per compiled candidate.
fn race_scratch_bytes(scratch: usize, candidates: usize, timestamps: bool) -> Option<usize> {
    let encoders = if timestamps { 0 } else { candidates };
    scratch.checked_add(encoders.checked_mul(ENCODER_SCRATCH_BYTES)?)
}

impl Scratch<'_, '_> {
    /// One submission with one timed pass per entry, repeating the entry's
    /// sequence `reps` times inside it. Explicit barriers order EVERY dispatch:
    /// repeated output writes need synchronization too, even for identical data.
    /// Returns milliseconds per sequence, or `None` without complete timestamps.
    fn run_timed(
        &mut self,
        entries: &[(&[(&bg::ComputePipeline, &Dispatch)], u32)],
    ) -> Option<Vec<f64>> {
        let mut pieces: Vec<bg::BufferPiece> = self.buffers.iter().map(|&b| b.into()).collect();
        self.encoder.start();
        for &(sequence, reps) in entries {
            let mut pass = self.encoder.compute("tune_race");
            for repetition in 0..reps {
                pieces[1] =
                    super::timed_input_b(self.buffers[1], &self.input_b_copies, repetition).into();
                for (position, &(pipeline, dispatch)) in sequence.iter().enumerate() {
                    if repetition != 0 || position != 0 {
                        pass.barrier();
                    }
                    let mut pc = pass.with(pipeline);
                    Session::bind_dispatch(&pieces, dispatch, &mut pc);
                    pc.dispatch(dispatch.workgroups);
                }
            }
        }
        let sync = self.gpu.submit(&mut self.encoder);
        let timings =
            crate::runtime::wait_for_timed_encoder(self.gpu, &sync, &mut self.encoder, true)
                .ok()
                .flatten()?;
        if timings.passes.len() != entries.len() {
            return None;
        }
        Some(
            timings
                .pass_durations()
                .zip(entries)
                .map(|((_, duration), &(_, reps))| duration.as_secs_f64() * 1e3 / f64::from(reps))
                .collect(),
        )
    }
}

/// Milliseconds per sequence for each candidate in `order`.
#[allow(clippy::too_many_arguments)]
fn run_round(
    scratch: &mut Scratch<'_, '_>,
    host: &mut HostEncoders<'_>,
    runs: &[Option<Run<'_>>],
    order: &[usize],
    reps: u32,
    timestamps: bool,
    start: Instant,
    budget: Duration,
) -> Option<Vec<f64>> {
    if start.elapsed() >= budget {
        return None;
    }
    let run = |candidate: usize| runs[candidate].as_deref().expect("live candidate compiled");
    let times = if timestamps {
        let entries: Vec<_> = order
            .iter()
            .map(|&candidate| (run(candidate), reps))
            .collect();
        scratch.run_timed(&entries)?
    } else {
        order
            .iter()
            .map(|&candidate| {
                (start.elapsed() < budget).then(|| {
                    Scratch::run_on(
                        scratch.gpu,
                        &scratch.buffers,
                        &scratch.input_b_copies,
                        host.encoders[candidate]
                            .as_mut()
                            .expect("live host candidate owns an encoder"),
                        run(candidate),
                        reps,
                    ) / f64::from(reps)
                })
            })
            .collect::<Option<Vec<_>>>()?
    };
    times
        .iter()
        .all(|ms| ms.is_finite() && *ms > 0.0)
        .then_some(times)
}

/// Reverse each rotated round before changing its order. A rotation alone
/// leaves the incumbent at one end for most of a short race, biasing drift.
fn round_order(alive: &[bool], round: usize) -> Vec<usize> {
    let mut order: Vec<_> = (0..alive.len()).filter(|&i| alive[i]).collect();
    let shift = (round / 2) % order.len();
    order.rotate_left(shift);
    if round % 2 == 1 {
        order.reverse();
    }
    order
}

/// Resource pressure and bad clocks are not evidence about a kernel. In
/// particular, a scratch-limited search must not poison a later, larger one.
fn decided_outcome(outcome: &TuneOutcome) -> bool {
    match outcome.decision {
        TuneDecision::FasterCandidate => outcome.qualified,
        TuneDecision::KeepBaseline | TuneDecision::Eliminated => !outcome.candidate_ms.is_empty(),
        TuneDecision::ShaderRejected | TuneDecision::InvalidOutput => true,
        TuneDecision::TimeBudget
        | TuneDecision::ScratchLimit
        | TuneDecision::DeviceMemoryBudget
        | TuneDecision::InvalidTiming => false,
    }
}

/// Repetitions per timed pass (or per host-timed submission), shared by every
/// candidate so that per-pass costs weigh equally on all of them.
fn repetitions(
    options: &TuneOptions,
    profile: Option<&MachineProfile>,
    timestamps: bool,
    sequence_len: usize,
    fastest_ms: f64,
) -> u32 {
    if options.target_sample_time.is_none() || !fastest_ms.is_finite() {
        return options.dispatches_per_sample;
    }
    match profile.and_then(|profile| profile.pass_overhead) {
        // Enough repetitions that pass overhead stays a small share of a pass.
        Some(overhead) if timestamps && sequence_len == 1 => {
            let count = overhead.as_secs_f64() * 1e3 / (PASS_OVERHEAD_SHARE * fastest_ms);
            (count.ceil() as u32).clamp(1, options.dispatches_per_sample)
        }
        // Keep split-sequence samples short; each sequence has internal barriers.
        _ if timestamps && sequence_len > 1 => 1,
        _ => options.sample_repetitions([fastest_ms, fastest_ms]),
    }
}

/// Rounds where both the incumbent and `candidate` were timed.
fn pairs(samples: &[Vec<Option<f64>>], candidate: usize) -> Vec<(f64, f64)> {
    samples[0]
        .iter()
        .zip(&samples[candidate])
        .filter_map(|(&baseline, &candidate)| Some((baseline?, candidate?)))
        .collect()
}

/// Prefer the earliest candidate the measured fastest cannot beat by the
/// configured margin plus paired noise. All these candidates have already
/// beaten the original incumbent. Anchor each tie band to the fastest, not a
/// chain of near-ties, and recompute it for qualification fallbacks.
///
/// An isolated scratch timing is not whole-program latency. In particular,
/// dispatch overlap can reverse tiny kernel gains. Respecting the caller's
/// resolution here avoids selecting on differences we would reject in a duel.
fn finalist_order(outcomes: &[TuneOutcome], options: &TuneOptions) -> Vec<usize> {
    let mut remaining: Vec<_> = (0..outcomes.len())
        .filter(|&i| outcomes[i].decision == TuneDecision::FasterCandidate)
        .collect();
    let mut ordered = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let fastest = *remaining
            .iter()
            .min_by(|&&a, &&b| {
                outcomes[a]
                    .candidate_median_ms
                    .unwrap()
                    .total_cmp(&outcomes[b].candidate_median_ms.unwrap())
            })
            .unwrap();
        let position = remaining
            .iter()
            .position(|&candidate| {
                let mut comparison = TuneOutcome::new((), 1, candidate, fastest);
                comparison
                    .baseline_ms
                    .clone_from(&outcomes[candidate].candidate_ms);
                comparison
                    .candidate_ms
                    .clone_from(&outcomes[fastest].candidate_ms);
                comparison.qualified = true; // timing only; qualify before installation
                decide(&mut comparison, options);
                comparison.decision != TuneDecision::FasterCandidate
            })
            .unwrap();
        ordered.push(remaining.remove(position));
    }
    ordered
}

/// Drop challengers whose fastest observed sample is already slower than the
/// incumbent's median less `min_improvement`. This is a heuristic on current
/// evidence, not a bound on future samples. Keep the fastest `FINALISTS`.
fn eliminate(
    samples: &[Vec<Option<f64>>],
    alive: &mut [bool],
    outcomes: &mut [TuneOutcome],
    min_improvement: f64,
) {
    let incumbent: Vec<f64> = samples[0].iter().flatten().copied().collect();
    let bar = median(&incumbent) * (1.0 - min_improvement);
    let mut contenders = Vec::new();
    for candidate in 1..alive.len() {
        if !alive[candidate] {
            continue;
        }
        let times: Vec<f64> = samples[candidate].iter().flatten().copied().collect();
        let fastest = times.iter().copied().fold(f64::INFINITY, f64::min);
        if fastest > bar {
            alive[candidate] = false;
            outcomes[candidate - 1].decision = TuneDecision::Eliminated;
        } else {
            contenders.push((median(&times), candidate));
        }
    }
    contenders.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(_, candidate) in contenders.iter().skip(FINALISTS) {
        alive[candidate] = false;
        outcomes[candidate - 1].decision = TuneDecision::Eliminated;
    }
}

/// The deterministic inputs for one pattern, in the class's storage types.
fn race_inputs(
    class: &SearchClass,
    logical_sizes: &[usize],
    sizes: &[usize],
    pattern: u32,
) -> Vec<Vec<f32>> {
    match class.key.effective_matrix_types() {
        Some(types) => typed_inputs(&class.key, types, pattern, sizes),
        None => prepared_inputs(logical_sizes, sizes, pattern, class.key.weight_format),
    }
}

enum QualificationFailure {
    TimeBudget,
    Output { by_incumbent: bool, failure: String },
}

/// Qualify both patterns with the sequential comparison's numerical checks.
/// Keep incumbent outputs across finalists; check deadlines between transfers
/// and dispatches, including after the final validation before installation.
fn qualify(
    scratch: &mut Scratch<'_, '_>,
    class: &SearchClass,
    logical_sizes: &[usize],
    sizes: &[usize],
    [incumbent, challenger]: [&Run<'_>; 2],
    reference: &mut [Option<Vec<f32>>; 2],
    start: Instant,
    budget: Duration,
) -> Result<(), QualificationFailure> {
    let check_budget = || {
        if start.elapsed() < budget {
            Ok(())
        } else {
            Err(QualificationFailure::TimeBudget)
        }
    };
    let mut times = TuneQualificationTimes::default();
    let agree = if class.key.split_k {
        reordered_outputs_agree
    } else {
        outputs_agree
    };
    // Finish on ordinary inputs, as timing used.
    for pattern in [1u32, 0] {
        check_budget()?;
        let inputs = race_inputs(class, logical_sizes, sizes, pattern);
        for (index, data) in inputs.iter().enumerate() {
            // B is identical across patterns and already resident.
            if index != 1 {
                check_budget()?;
                scratch.upload(index, data, Some(&mut times));
            }
        }
        let scale = if pattern == 0 {
            1.0
        } else {
            small_pattern_scale(&class.key)
        };
        let slot = pattern as usize;
        if reference[slot].is_none() {
            check_budget()?;
            scratch.poison_outputs(sizes, &mut times);
            check_budget()?;
            scratch.run(incumbent, 1);
            check_budget()?;
            let output = scratch.read_output(&mut times);
            if !qualify_output(&class.key, &inputs, &output, scale) {
                return Err(QualificationFailure::Output {
                    by_incumbent: true,
                    failure: format!("incumbent, input pattern {pattern}: reference mismatch"),
                });
            }
            reference[slot] = Some(output);
        }
        check_budget()?;
        scratch.poison_outputs(sizes, &mut times);
        check_budget()?;
        scratch.run(challenger, 1);
        check_budget()?;
        let output = scratch.read_output(&mut times);
        let expected = reference[slot].as_ref().unwrap();
        let identical =
            bytemuck::cast_slice::<f32, u8>(expected) == bytemuck::cast_slice::<f32, u8>(&output);
        if !(identical
            || (qualify_output(&class.key, &inputs, &output, scale)
                && agree(expected, &output, scale)))
        {
            return Err(QualificationFailure::Output {
                by_incumbent: false,
                failure: format!("input pattern {pattern}: reference or cross-variant mismatch"),
            });
        }
    }
    check_budget()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{MatmulTile, TuneClass};

    fn outcome(candidate: MatmulTile) -> TuneOutcome {
        TuneOutcome::new(TuneClass::default(), 1, MatmulTile::Tile64, candidate)
    }

    #[test]
    fn transient_failures_are_not_memo_decisions() {
        let mut outcome = outcome(MatmulTile::Tile32);
        for decision in [
            TuneDecision::TimeBudget,
            TuneDecision::ScratchLimit,
            TuneDecision::DeviceMemoryBudget,
            TuneDecision::InvalidTiming,
        ] {
            outcome.decision = decision;
            assert!(!decided_outcome(&outcome), "{decision:?}");
        }
        // A default, unmeasured outcome is not evidence either.
        outcome.decision = TuneDecision::KeepBaseline;
        assert!(!decided_outcome(&outcome));
        outcome.candidate_ms = vec![1.0; 4];
        assert!(decided_outcome(&outcome));
        outcome.decision = TuneDecision::FasterCandidate;
        assert!(!decided_outcome(&outcome));
        outcome.qualified = true;
        assert!(decided_outcome(&outcome));
    }

    fn distinct_weight_graph() -> crate::Graph {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[1, 64]);
        let w = graph.parameter("w", &[64, 96]);
        let v = graph.parameter("v", &[64, 96]);
        let a = graph.matmul(x, w);
        let b = graph.matmul(x, v);
        graph.set_outputs(vec![a, b]);
        graph
    }

    #[test]
    fn distinct_weights_respect_scratch_capacity_and_aliases() {
        let gpu = crate::reference::gpu::shared_context();
        let graph = distinct_weight_graph();
        let make = |plan| {
            Session::with_context_opts(
                plan,
                gpu.clone(),
                crate::SessionOptions {
                    coop: crate::CoopPolicy::Disabled,
                    ..Default::default()
                },
            )
        };
        let plan = crate::compile::compile(&graph);
        let session = make(plan.clone());
        let options = TuneOptions::default();
        assert_eq!(session.tune_classes(&options).0[0].input_b_copies, 2);
        assert_eq!(
            session
                .tune_classes(&TuneOptions {
                    max_scratch_bytes: 0,
                    ..options.clone()
                })
                .0[0]
                .input_b_copies,
            1
        );
        let mut repeated = plan;
        repeated.dispatches[1].input_buffers[1] = repeated.dispatches[0].input_buffers[1];
        assert_eq!(make(repeated).tune_classes(&options).0[0].input_b_copies, 1);
    }

    #[test]
    #[ignore = "GPU qualification of distinct-weight sampling on host and timestamp paths"]
    fn timed_samples_read_weight_copies_but_qualification_reads_original() {
        let gpu = std::sync::Arc::new(
            crate::init_gpu_context_with(crate::GpuOptions {
                timing: true,
                ..crate::GpuOptions::from_env()
            })
            .unwrap(),
        );
        let graph = distinct_weight_graph();
        let mut session = Session::with_context_opts(
            crate::compile::compile(&graph),
            gpu.clone(),
            crate::SessionOptions {
                coop: crate::CoopPolicy::Disabled,
                ..Default::default()
            },
        );
        let options = TuneOptions::default();
        let (classes, _) = session.tune_classes(&options);
        assert_eq!(classes.len(), 1);
        let class = &classes[0];
        session
            .pipelines
            .ensure_tune_tile(
                &gpu,
                &session.plan.dispatches[class.members[0]],
                class.initial,
            )
            .unwrap();
        let PreparedRace {
            logical_sizes,
            sizes,
            output_index,
            sequences,
        } = prepare_race(
            class,
            &session.plan.dispatches[class.members[0]],
            &[class.initial],
        );
        let bytes = scratch_bytes(&sizes).unwrap() + sizes[1];
        let placement = vec![true; sizes.len()];
        let mut staging = Staging::new(&gpu, options.staging, options.staging_reuse);
        let mut preparation = TunePreparationTimes::default();
        let mut cleanup = None;
        let mut scratch = Scratch::new(
            &placement,
            &sizes,
            (&class.key, output_index),
            bytes,
            &mut staging,
            &mut preparation,
            &mut cleanup,
        );
        let inputs = race_inputs(class, &logical_sizes, &sizes, 0);
        for (i, data) in inputs.iter().enumerate() {
            scratch.upload(i, data, None);
        }
        scratch.input_b_uploaded = true;
        scratch.replicate_input_b(2, true);
        let sequence: Vec<_> = sequences[0]
            .iter()
            .map(|d| (&session.pipelines.map[&Pipelines::key(d)].raw, d))
            .collect();
        let mut times = TuneQualificationTimes::default();
        scratch.encoder.start();
        scratch
            .encoder
            .transfer("poison weight replica")
            .fill_buffer(scratch.input_b_copies[0].into(), sizes[1] as u64, 0xff);
        scratch.submit_wait();
        for timestamps in [false, true] {
            let ms = if timestamps {
                scratch.run_timed(&[(&sequence, 2)]).unwrap()[0]
            } else {
                Scratch::run_on(
                    &gpu,
                    &scratch.buffers,
                    &scratch.input_b_copies,
                    &mut scratch.encoder,
                    &sequence,
                    2,
                )
            };
            assert!(ms.is_finite() && ms > 0.0);
            assert!(scratch.read_output(&mut times).iter().all(|v| v.is_nan()));
            scratch.run(&sequence, 1);
            assert!(qualify_output(
                &class.key,
                &inputs,
                &scratch.read_output(&mut times),
                1.0
            ));
        }
    }

    #[test]
    fn expired_compilation_budget_does_not_start_shader_jobs() {
        let gpu = crate::reference::gpu::shared_context();
        let dispatch = Dispatch::default();
        let jobs = vec![(Pipelines::key(&dispatch), &dispatch, None); 12];
        let results = Pipelines::compile_all(
            &gpu,
            Default::default(),
            None,
            &jobs,
            Some((Instant::now(), Duration::ZERO)),
        );
        assert_eq!(results.len(), jobs.len());
        assert!(results.iter().all(Option::is_none));
    }

    #[test]
    fn paired_rounds_balance_positions_even_after_elimination() {
        for alive in [vec![true; 25], vec![true, false, true, true, false]] {
            for pair in 0..6 {
                let forward = round_order(&alive, pair * 2);
                let reverse = round_order(&alive, pair * 2 + 1);
                assert_eq!(forward, reverse.into_iter().rev().collect::<Vec<_>>());
                assert!(forward.iter().all(|&i| alive[i]));
                assert_eq!(forward.len(), alive.iter().filter(|&&v| v).count());
            }
        }
    }

    fn timing_winner(samples: &[f64], options: &TuneOptions) -> TuneOutcome {
        let mut outcome = outcome(MatmulTile::Tile32);
        outcome.baseline_ms = vec![100.0; options.sample_pairs];
        outcome.candidate_ms = samples.to_vec();
        outcome.qualified = true;
        decide(&mut outcome, options);
        assert_eq!(outcome.decision, TuneDecision::FasterCandidate);
        outcome.qualified = false;
        outcome
    }

    #[test]
    fn finalists_require_a_meaningful_gain_over_each_other() {
        let options = TuneOptions::default();
        let winner = |ms| timing_winner(&vec![ms; options.sample_pairs], &options);
        // Both beat the original incumbent, but 2% is below the requested 5%.
        assert_eq!(
            finalist_order(&[winner(10.0), winner(9.8)], &options),
            [0, 1]
        );
        assert_eq!(
            finalist_order(&[winner(10.0), winner(8.0)], &options),
            [1, 0]
        );
        // The caller can explicitly ask to resolve smaller differences.
        let precise = TuneOptions {
            min_improvement: 0.0,
            ..options.clone()
        };
        assert_eq!(
            finalist_order(&[winner(10.0), winner(9.8)], &precise),
            [1, 0]
        );
    }

    #[test]
    fn finalist_ties_include_paired_noise_and_do_not_chain() {
        let options = TuneOptions::default();
        let winner = |ms| timing_winner(&vec![ms; options.sample_pairs], &options);
        let noisy = timing_winner(&[8.0, 9.5, 8.0, 9.5, 8.0, 9.5], &options);
        // Median gain is >5%, but not >5% plus the paired noise margin.
        assert_eq!(finalist_order(&[winner(10.0), noisy], &options), [0, 1]);
        // 10.8 ties 10.4, which ties 10.0. Only the last pair is in the
        // fastest's band. If that candidate fails qualification, re-rank.
        assert_eq!(
            finalist_order(&[winner(10.8), winner(10.4), winner(10.0)], &options),
            [1, 2, 0]
        );
        let mut rejected = winner(5.0);
        rejected.decision = TuneDecision::InvalidOutput;
        assert_eq!(finalist_order(&[rejected, winner(10.0)], &options), [1]);
    }

    #[test]
    fn hopeless_challengers_drop_and_only_finalists_remain() {
        // Incumbent at 10 ms; the bar for a 5% win is 9.5 ms.
        let incumbent = vec![Some(10.0); 3];
        let challenger = |ms: f64| vec![Some(ms); 3];
        let samples = vec![
            incumbent,
            challenger(9.6), // fastest sample above the bar: cannot win
            challenger(9.0),
            challenger(8.0),
            challenger(7.0),
            challenger(6.0),
            challenger(5.0), // five contenders: the slowest one goes
        ];
        let mut alive = vec![true; samples.len()];
        let mut outcomes: Vec<_> = (1..samples.len())
            .map(|_| outcome(MatmulTile::Tile32))
            .collect();
        eliminate(&samples, &mut alive, &mut outcomes, 0.05);
        assert_eq!(alive, [true, false, false, true, true, true, true]);
        assert_eq!(outcomes[0].decision, TuneDecision::Eliminated);
        assert_eq!(outcomes[1].decision, TuneDecision::Eliminated);
        assert!(
            outcomes[2..]
                .iter()
                .all(|o| o.decision == TuneDecision::KeepBaseline)
        );
    }

    #[test]
    fn a_single_fast_sample_keeps_a_challenger_in_the_race() {
        let samples = vec![vec![Some(10.0); 3], vec![Some(12.0), Some(9.0), Some(12.0)]];
        let mut alive = vec![true; 2];
        let mut outcomes = vec![outcome(MatmulTile::Tile32)];
        eliminate(&samples, &mut alive, &mut outcomes, 0.05);
        assert_eq!(alive, [true, true]);
    }

    #[test]
    fn pairs_follow_rounds_both_candidates_ran() {
        let samples = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(4.0), None, Some(6.0)],
        ];
        assert_eq!(pairs(&samples, 1), [(1.0, 4.0), (3.0, 6.0)]);
    }

    #[test]
    fn repetitions_bound_pass_overhead_and_keep_sequences_whole() {
        let options = TuneOptions::default();
        let profile = MachineProfile {
            pass_overhead: Some(Duration::from_micros(10)),
            ..Default::default()
        };
        // 10 us of overhead on a 0.1 ms dispatch needs 5 repetitions for 2%.
        assert_eq!(repetitions(&options, Some(&profile), true, 1, 0.1), 5);
        // A long dispatch needs only one; a short one is capped.
        assert_eq!(repetitions(&options, Some(&profile), true, 1, 10.0), 1);
        assert_eq!(
            repetitions(&options, Some(&profile), true, 1, 0.001),
            options.dispatches_per_sample
        );
        // Multi-dispatch sequences already take one pass per dispatch.
        assert_eq!(repetitions(&options, Some(&profile), true, 2, 0.001), 1);
        // Host timing follows the sample-duration target, as sequential search does.
        assert_eq!(
            repetitions(&options, Some(&profile), false, 1, 0.5),
            options.sample_repetitions([0.5, 0.5])
        );
        let fixed = TuneOptions {
            target_sample_time: None,
            ..options
        };
        for sequence_len in [1, 2] {
            assert_eq!(
                repetitions(&fixed, Some(&profile), true, sequence_len, 10.0),
                fixed.dispatches_per_sample
            );
        }
    }
}
