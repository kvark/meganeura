//! Class races: a class's incumbent and challengers timed together.
//!
//! Every candidate's pipelines compile in one parallel batch, then all of them
//! run in shared rounds on one scratch, in rotated order. With GPU timestamps
//! a round is one submission with a timed pass per candidate; otherwise each
//! candidate is a host-timed submission. Challengers that can no longer pass
//! the decision rule are dropped without qualification, and only a finalist
//! that would be installed is qualified. See `docs/search-convergence.md`.

use super::{
    Scratch, SearchClass, Staging, outputs_agree, prepared_inputs, qualify_output,
    reordered_outputs_agree, scratch_bytes, small_pattern_scale, typed_inputs,
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
const ELIMINATE_AFTER: usize = 3;
/// Challengers kept after the first elimination: successive halving's width.
const FINALISTS: usize = 4;
/// Largest share of a timed pass that pass overhead may take.
const PASS_OVERHEAD_SHARE: f64 = 0.02;

pub(super) struct RaceResult {
    pub(super) selected: MatmulTile,
    pub(super) outcomes: Vec<TuneOutcome>,
    pub(super) race: TuneRace,
}

type Run<'a> = Vec<(&'a bg::ComputePipeline, &'a Dispatch)>;

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
            complete: false,
            compile_time: Duration::ZERO,
            preparation: Duration::ZERO,
            racing: Duration::ZERO,
            qualification: Duration::ZERO,
            elapsed: Duration::ZERO,
        };
        let conclude = |mut race: TuneRace, mut outcomes: Vec<TuneOutcome>| {
            for outcome in &mut outcomes {
                outcome.elapsed = Duration::ZERO;
            }
            race.complete = outcomes
                .iter()
                .all(|outcome| outcome.decision != TuneDecision::TimeBudget);
            race.elapsed = race_start.elapsed();
            RaceResult {
                selected: race.selected,
                outcomes,
                race,
            }
        };
        let fail_all = |outcomes: &mut [TuneOutcome], decision: TuneDecision| {
            for outcome in outcomes.iter_mut() {
                outcome.decision = decision;
            }
        };

        // Bindings sized for the largest candidate, as one comparison would.
        let preparation = Instant::now();
        let logical_sizes = class
            .key
            .buffer_sizes()
            .expect("class collection checked extents");
        let mut sizes: Vec<usize> = Vec::new();
        for &tile in &tiles {
            let tile_sizes = tile.buffer_sizes(&class.key).expect("legal candidate");
            sizes.resize(tile_sizes.len(), 0);
            for (size, tile_size) in sizes.iter_mut().zip(tile_sizes) {
                *size = (*size).max(tile_size.next_multiple_of(4));
            }
        }
        let output_index = sizes.len() - 1;
        let mut template = self.plan.dispatches[class.members[0]].clone();
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
            for (sequence, tile) in sequences.iter_mut().zip(&tiles) {
                let splits = tile
                    .partitions()
                    .expect("split products only take partitioned candidates");
                let sum =
                    sequence[0].partial_sum(partial, sequence[0].output_buffer, columns, splits);
                sequence[0].output_buffer = partial;
                sequence.push(sum);
            }
        }
        let Some(bytes) = scratch_bytes(&sizes).filter(|&bytes| bytes <= options.max_scratch_bytes)
        else {
            fail_all(&mut outcomes, TuneDecision::ScratchLimit);
            return conclude(race, outcomes);
        };
        staging.discard_unmatched(*sizes.iter().max().unwrap());
        staging.discard_input_b(&class.key, sizes[1]);
        let memory = self.gpu.memory_stats();
        if memory.budget != 0
            && (bytes - staging.retained_bytes()) as u64
                > crate::runtime::safe_device_memory_remaining(memory.usage, memory.budget)
        {
            fail_all(&mut outcomes, TuneDecision::DeviceMemoryBudget);
            return conclude(race, outcomes);
        }

        // Compile every missing pipeline in one parallel batch.
        let compile = Instant::now();
        let mut wanted: Vec<(Variant, &Dispatch, Option<crate::codegen::CoopConfig>)> = Vec::new();
        for (sequence, &tile) in sequences.iter().zip(&tiles) {
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
        );
        let mut rejected: HashMap<Variant, String> = HashMap::new();
        for ((key, ..), result) in jobs.into_iter().zip(compiled) {
            match result {
                Ok(pipeline) => {
                    self.pipelines.map.insert(key, pipeline);
                }
                Err(error) => {
                    rejected.insert(key, error);
                }
            }
        }
        race.compile_time = compile.elapsed();
        let mut alive = vec![true; tiles.len()];
        for (index, sequence) in sequences.iter().enumerate() {
            let Some(error) = sequence
                .iter()
                .find_map(|dispatch| rejected.get(&Pipelines::key(dispatch)))
            else {
                continue;
            };
            if index == 0 {
                for outcome in &mut outcomes {
                    outcome.decision = TuneDecision::ShaderRejected;
                    outcome.failure = Some(format!("incumbent {incumbent:?}: {error}"));
                }
                return conclude(race, outcomes);
            }
            alive[index] = false;
            let outcome = &mut outcomes[index - 1];
            outcome.decision = TuneDecision::ShaderRejected;
            outcome.failure = Some(format!("{:?}: {error}", tiles[index]));
        }
        race.candidates = alive.iter().filter(|&&alive| alive).count();
        if start.elapsed() >= options.max_time {
            for (outcome, &alive) in outcomes.iter_mut().zip(&alive[1..]) {
                if alive {
                    outcome.decision = TuneDecision::TimeBudget;
                }
            }
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
            if index != 1 || !scratch.input_b_reused {
                scratch.upload(index, data, None);
            }
        }
        scratch.input_b_uploaded = true;
        drop(inputs);
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
        race.preparation = preparation.elapsed().saturating_sub(race.compile_time);

        let racing = Instant::now();
        let timestamps = self.gpu_timing && self.gpu.capabilities().timing;
        race.gpu_timestamps = timestamps;
        let live =
            |alive: &[bool]| -> Vec<usize> { (0..alive.len()).filter(|&i| alive[i]).collect() };
        let warm_order = live(&alive);
        let Some(warm) = run_round(&mut scratch, &runs, &warm_order, 1, timestamps) else {
            fail_all(&mut outcomes, TuneDecision::InvalidTiming);
            return conclude(race, outcomes);
        };
        let fastest = warm
            .iter()
            .copied()
            .filter(|ms| ms.is_finite() && *ms > 0.0)
            .fold(f64::INFINITY, f64::min);
        let reps = repetitions(options, profile, timestamps, sequences[0].len(), fastest);
        // samples[candidate][round]: a dropped candidate has no later samples.
        let mut samples: Vec<Vec<Option<f64>>> = vec![Vec::new(); tiles.len()];
        let max_rounds = ELIMINATE_AFTER + 2 * options.sample_pairs;
        let mut invalid_timing = false;
        while race.rounds < max_rounds && start.elapsed() < options.max_time {
            let mut order = live(&alive);
            let shift = race.rounds % order.len();
            order.rotate_left(shift);
            let Some(times) = run_round(&mut scratch, &runs, &order, reps, timestamps) else {
                invalid_timing = true;
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
            if race.rounds >= ELIMINATE_AFTER {
                eliminate(&samples, &mut alive, &mut outcomes, options.min_improvement);
            }
            let challengers: Vec<usize> = (1..tiles.len()).filter(|&i| alive[i]).collect();
            if challengers.is_empty()
                || (race.rounds >= ELIMINATE_AFTER
                    && challengers
                        .iter()
                        .all(|&i| pairs(&samples, i).len() >= options.sample_pairs))
            {
                break;
            }
        }
        race.racing = racing.elapsed();
        let incumbent_samples: Vec<f64> = samples[0].iter().flatten().copied().collect();
        race.incumbent_ms = (!incumbent_samples.is_empty()).then(|| median(&incumbent_samples));

        // Decide every challenger from the rounds it shared with the incumbent.
        for candidate in 1..tiles.len() {
            let outcome = &mut outcomes[candidate - 1];
            if outcome.decision == TuneDecision::ShaderRejected {
                continue;
            }
            outcome.sample_repetitions = Some(reps);
            let paired = pairs(&samples, candidate);
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

        // Qualify winners, fastest first, until one passes.
        let qualification = Instant::now();
        let mut winners: Vec<usize> = (1..tiles.len())
            .filter(|&i| outcomes[i - 1].decision == TuneDecision::FasterCandidate)
            .collect();
        winners.sort_by(|&a, &b| {
            outcomes[a - 1]
                .candidate_median_ms
                .unwrap_or(f64::INFINITY)
                .total_cmp(&outcomes[b - 1].candidate_median_ms.unwrap_or(f64::INFINITY))
        });
        let mut reference: [Option<Vec<f32>>; 2] = Default::default();
        let mut incumbent_failed = false;
        for winner in winners {
            let outcome = &mut outcomes[winner - 1];
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
                &mut scratch,
                class,
                &logical_sizes,
                &sizes,
                [runs[0].as_ref().unwrap(), runs[winner].as_ref().unwrap()],
                &mut reference,
            );
            match checked {
                Ok(()) => {
                    outcome.qualified = true;
                    race.selected = tiles[winner];
                }
                Err((by_incumbent, failure)) => {
                    incumbent_failed = by_incumbent;
                    outcome.decision = TuneDecision::InvalidOutput;
                    outcome.failure = Some(failure);
                    outcome.selected = incumbent;
                }
            }
        }
        race.qualification = qualification.elapsed();
        drop(runs);
        drop(scratch);
        conclude(race, outcomes)
    }
}

impl Scratch<'_, '_> {
    /// One submission with one timed pass per entry, repeating the entry's
    /// sequence `reps` times inside it. A sequence of several dispatches takes
    /// one pass per dispatch instead, so dependent dispatches stay ordered.
    /// Returns milliseconds per sequence, or `None` without complete timestamps.
    fn run_timed(
        &mut self,
        entries: &[(&[(&bg::ComputePipeline, &Dispatch)], u32)],
    ) -> Option<Vec<f64>> {
        let pieces: Vec<bg::BufferPiece> = self.buffers.iter().map(|&b| b.into()).collect();
        let mut owners = Vec::new();
        self.encoder.start();
        for (entry, &(sequence, reps)) in entries.iter().enumerate() {
            if let &[(pipeline, dispatch)] = sequence {
                let mut pass = self.encoder.compute("tune_race");
                let mut pc = pass.with(pipeline);
                for _ in 0..reps {
                    Session::bind_dispatch(&pieces, dispatch, &mut pc);
                    pc.dispatch(dispatch.workgroups);
                }
                owners.push(entry);
            } else {
                for _ in 0..reps {
                    for &(pipeline, dispatch) in sequence {
                        let mut pass = self.encoder.compute("tune_race");
                        let mut pc = pass.with(pipeline);
                        Session::bind_dispatch(&pieces, dispatch, &mut pc);
                        pc.dispatch(dispatch.workgroups);
                        owners.push(entry);
                    }
                }
            }
        }
        let sync = self.gpu.submit(&mut self.encoder);
        let timings =
            crate::runtime::wait_for_timed_encoder(self.gpu, &sync, &mut self.encoder, true)
                .ok()
                .flatten()?;
        if timings.passes.len() != owners.len() {
            return None;
        }
        let mut totals = vec![0.0; entries.len()];
        for ((_, duration), &owner) in timings.pass_durations().zip(&owners) {
            totals[owner] += duration.as_secs_f64() * 1e3;
        }
        Some(
            totals
                .into_iter()
                .zip(entries)
                .map(|(total, &(_, reps))| total / f64::from(reps))
                .collect(),
        )
    }
}

/// Milliseconds per sequence for each candidate in `order`.
fn run_round(
    scratch: &mut Scratch<'_, '_>,
    runs: &[Option<Run<'_>>],
    order: &[usize],
    reps: u32,
    timestamps: bool,
) -> Option<Vec<f64>> {
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
            .map(|&candidate| scratch.run(run(candidate), reps) / f64::from(reps))
            .collect()
    };
    times
        .iter()
        .all(|ms| ms.is_finite() && *ms > 0.0)
        .then_some(times)
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
    if !fastest_ms.is_finite() {
        return options.dispatches_per_sample;
    }
    match profile.and_then(|profile| profile.pass_overhead) {
        // Enough repetitions that pass overhead stays a small share of a pass.
        Some(overhead) if timestamps && sequence_len == 1 => {
            let count = overhead.as_secs_f64() * 1e3 / (PASS_OVERHEAD_SHARE * fastest_ms);
            (count.ceil() as u32).clamp(1, options.dispatches_per_sample)
        }
        // Sequences already take one pass per dispatch.
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

/// Drop challengers whose fastest sample is already slower than the
/// incumbent's median less `min_improvement`: their median cannot pass the
/// decision rule. Then keep only the fastest `FINALISTS` by median.
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
    match class.key.matrix_types {
        Some(types) => typed_inputs(&class.key, types, pattern, sizes),
        None => prepared_inputs(logical_sizes, sizes, pattern, class.key.weight_format),
    }
}

/// Qualify `[incumbent, challenger]` on both input patterns with the checks of
/// a sequential comparison: finite outputs, sampled f64 reference dots, and
/// agreement with the incumbent. Incumbent outputs are kept in `reference`
/// across challengers. The error says whether the incumbent failed.
fn qualify(
    scratch: &mut Scratch<'_, '_>,
    class: &SearchClass,
    logical_sizes: &[usize],
    sizes: &[usize],
    [incumbent, challenger]: [&Run<'_>; 2],
    reference: &mut [Option<Vec<f32>>; 2],
) -> Result<(), (bool, String)> {
    let mut times = TuneQualificationTimes::default();
    let agree = if class.key.split_k {
        reordered_outputs_agree
    } else {
        outputs_agree
    };
    // Finish on ordinary inputs, as timing used.
    for pattern in [1u32, 0] {
        let inputs = race_inputs(class, logical_sizes, sizes, pattern);
        for (index, data) in inputs.iter().enumerate() {
            // B is identical across patterns and already resident.
            if index != 1 {
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
            scratch.poison_outputs(sizes, &mut times);
            scratch.run(incumbent, 1);
            let output = scratch.read_output(&mut times);
            if !qualify_output(&class.key, &inputs, &output, scale) {
                return Err((
                    true,
                    format!("incumbent, input pattern {pattern}: reference mismatch"),
                ));
            }
            reference[slot] = Some(output);
        }
        scratch.poison_outputs(sizes, &mut times);
        scratch.run(challenger, 1);
        let output = scratch.read_output(&mut times);
        let expected = reference[slot].as_ref().unwrap();
        let identical =
            bytemuck::cast_slice::<f32, u8>(expected) == bytemuck::cast_slice::<f32, u8>(&output);
        if !(identical
            || (qualify_output(&class.key, &inputs, &output, scale)
                && agree(expected, &output, scale)))
        {
            return Err((
                false,
                format!("input pattern {pattern}: reference or cross-variant mismatch"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{MatmulTile, TuneClass};

    fn outcome(candidate: MatmulTile) -> TuneOutcome {
        TuneOutcome::new(TuneClass::default(), 1, MatmulTile::Tile64, candidate)
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
    }
}
