use anyhow::{Context, Result};
use bidking_rs::{
    CalcParams, ComboResult, ValueSample, infer_grid_from_average_for_quality,
    infer_grid_from_average_with_sizes, load_core, load_embedded_core, normalize_calc_params,
    recommended_bid_value,
};
use regex::Regex;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ReferenceCase {
    name: String,
    tier: String,
    map_id: Option<String>,
    nest_id: Option<String>,
    total: i32,
    total_grid: Option<f64>,
    avg_grid_all: Option<f64>,
    high_quality_count: Option<i32>,
    gw_count: Option<i32>,
    gw_min: Option<i32>,
    gw_grid: Option<f64>,
    gw_avg: Option<f64>,
    blue_count: Option<i32>,
    blue_min: Option<i32>,
    blue_grid: Option<f64>,
    blue_avg: Option<f64>,
    purple_count: Option<i32>,
    purple_min: Option<i32>,
    purple_grid: Option<f64>,
    purple_avg: Option<f64>,
    gold_count: Option<i32>,
    gold_min: Option<i32>,
    gold_grid: Option<f64>,
    gold_avg: Option<f64>,
    red_count: Option<i32>,
    red_min: Option<i32>,
    red_grid: Option<f64>,
    red_avg: Option<f64>,
    safety: f64,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    combos: usize,
    raw: usize,
    bid_p25: i64,
    bid_p50: i64,
    bid_p75: i64,
    top: ExpectedTop,
}

#[derive(Debug, Deserialize)]
struct ExpectedTop {
    greenwhite: i32,
    blue: i32,
    purple: i32,
    gold: i32,
    red: i32,
    probability_6dp: f64,
    final_value: i64,
    total_grid_est_1dp: f64,
}

fn sum_grid_mentions(line: &str) -> i32 {
    let re = Regex::new(r"\((\d+)格\)").unwrap();
    re.captures_iter(line)
        .filter_map(|captures| captures.get(1))
        .filter_map(|m| m.as_str().parse::<i32>().ok())
        .sum()
}

#[test]
fn reference_case_2101_total_63() -> Result<()> {
    let data = Path::new("../decompiled_4_12_2/MapBidCalculator.calculator_data_merged.csv");
    let static_data = Path::new("../core_algorithm/static_data.json");
    if !data.exists() || !static_data.exists() {
        eprintln!("reference extraction files are not present; skipping local reference test");
        return Ok(());
    }

    let mut core = load_core(data, static_data)?;
    let cp = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 63,
        safety_factor: 0.85,
        max_show: 1,
        ..Default::default()
    };
    let results = core.run(cp.clone())?;
    let (p25, p50, p75) = core.price_range_for_last_run(&cp);

    assert_eq!(results.len(), 1054);
    assert_eq!(core.raw_results.len(), 766480);
    assert_eq!((p25 * cp.safety_factor).round() as i64, 87_058);
    assert_eq!((p50 * cp.safety_factor).round() as i64, 107_902);
    assert_eq!((p75 * cp.safety_factor).round() as i64, 133_904);
    let top = &results[0];
    assert_eq!(
        (
            top.greenwhite_count,
            top.blue_count,
            top.purple_count,
            top.gold_count,
            top.red_count
        ),
        (39, 18, 5, 1, 0)
    );
    Ok(())
}

#[test]
fn embedded_data_case_2101_total_63() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 63,
        safety_factor: 0.85,
        max_show: 1,
        ..Default::default()
    };
    let results = core.run(cp.clone())?;
    let (p25, p50, p75) = core.price_range_for_last_run(&cp);

    assert_eq!(results.len(), 1054);
    assert_eq!(core.raw_results.len(), 766480);
    assert_eq!((p25 * cp.safety_factor).round() as i64, 87_058);
    assert_eq!((p50 * cp.safety_factor).round() as i64, 107_902);
    assert_eq!((p75 * cp.safety_factor).round() as i64, 133_904);
    let top = &results[0];
    assert_eq!(
        (
            top.greenwhite_count,
            top.blue_count,
            top.purple_count,
            top.gold_count,
            top.red_count
        ),
        (39, 18, 5, 1, 0)
    );
    Ok(())
}

#[test]
fn price_range_uses_explicit_results_not_last_run_cache() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 63,
        safety_factor: 1.0,
        max_show: 1,
        ..Default::default()
    };
    core.run(cp.clone())?;

    let explicit = vec![ComboResult {
        final_value: 1_234_567.0,
        probability: 1.0,
        high_variance: 0.0,
        ..Default::default()
    }];
    let explicit_range = core.price_range(&explicit, &cp);
    let last_run_range = core.price_range_for_last_run(&cp);

    assert_eq!(explicit_range.0.round() as i64, 1_234_567);
    assert_eq!(explicit_range.1.round() as i64, 1_234_567);
    assert_eq!(explicit_range.2.round() as i64, 1_234_567);
    assert_ne!(last_run_range.1.round() as i64, 1_234_567);
    Ok(())
}

#[test]
fn manual_purple_and_gold_prices_affect_value_estimates() -> Result<()> {
    let base = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 63,
        safety_factor: 1.0,
        max_show: 1,
        ..Default::default()
    };

    let mut core = load_embedded_core()?;
    let base_results = core.run(base.clone())?;
    let (base_p25, base_p50, base_p75) = core.price_range_for_last_run(&base);
    let base_top = base_results.first().context("base produced no combos")?;

    let manual = CalcParams {
        manual_purple_per_item: Some(1_000_000.0),
        manual_gold_per_grid: Some(100_000.0),
        ..base
    };
    let mut core = load_embedded_core()?;
    let manual_results = core.run(manual.clone())?;
    let (manual_p25, manual_p50, manual_p75) = core.price_range_for_last_run(&manual);
    let manual_top = manual_results
        .first()
        .context("manual-price case produced no combos")?;

    assert_eq!(base_results.len(), manual_results.len());
    assert_eq!(
        (
            base_top.greenwhite_count,
            base_top.blue_count,
            base_top.purple_count,
            base_top.gold_count,
            base_top.red_count
        ),
        (
            manual_top.greenwhite_count,
            manual_top.blue_count,
            manual_top.purple_count,
            manual_top.gold_count,
            manual_top.red_count
        )
    );
    assert!(manual_top.final_value > base_top.final_value + 4_500_000.0);
    assert!(manual_p25 > base_p25 + 1_500_000.0);
    assert!(manual_p50 > base_p50 + 2_500_000.0);
    assert!(manual_p75 > base_p75 + 4_000_000.0);
    Ok(())
}

#[test]
fn zero_manual_prices_are_ignored() -> Result<()> {
    let base = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 63,
        purple_count: Some(5),
        purple_grid: Some(12.0),
        gold_count: Some(1),
        gold_grid: Some(3.0),
        safety_factor: 1.0,
        max_show: 1,
        ..Default::default()
    };

    let zero_manual = CalcParams {
        manual_purple_per_item: Some(0.0),
        manual_purple_per_grid: Some(0.0),
        manual_gold_per_item: Some(0.0),
        manual_gold_per_grid: Some(0.0),
        ..base.clone()
    };

    let mut core = load_embedded_core()?;
    let base_results = core.run(base.clone())?;
    let (_, base_p50, _) = core.price_range_for_last_run(&base);

    let mut core = load_embedded_core()?;
    let zero_results = core.run(zero_manual.clone())?;
    let (_, zero_p50, _) = core.price_range_for_last_run(&zero_manual);

    assert!((base_results[0].final_value - zero_results[0].final_value).abs() < 1e-6);
    assert!((base_p50 - zero_p50).abs() < 1e-6);
    Ok(())
}

#[test]
fn normalize_params_converts_average_grid_to_total_grid() {
    let cp = normalize_calc_params(CalcParams {
        total_count: 58,
        avg_grid_all: Some(2.35),
        ..Default::default()
    });
    assert_eq!(cp.total_grid_target, Some(136.3));
}

#[test]
fn manual_per_grid_price_still_affects_estimates_when_per_item_is_filled() -> Result<()> {
    let manual_low_grid = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 23,
        purple_count: Some(6),
        purple_grid: Some(12.0),
        purple_avg: Some(2.0),
        gold_avg: Some(1.0),
        manual_purple_per_item: Some(5_266.0),
        manual_purple_per_grid: Some(100.0),
        manual_gold_per_item: Some(7_856.0),
        manual_gold_per_grid: Some(100.0),
        safety_factor: 1.0,
        max_show: 1,
        ..Default::default()
    };

    let manual_high_grid = CalcParams {
        manual_purple_per_grid: Some(10_000.0),
        manual_gold_per_grid: Some(10_000.0),
        ..manual_low_grid.clone()
    };

    let mut core = load_embedded_core()?;
    let low_results = core.run(manual_low_grid.clone())?;
    let low_top = low_results
        .first()
        .context("low per-grid case produced no combos")?;
    let (low_p25, low_p50, low_p75) = core.price_range_for_last_run(&manual_low_grid);

    let mut core = load_embedded_core()?;
    let high_results = core.run(manual_high_grid.clone())?;
    let high_top = high_results
        .first()
        .context("high per-grid case produced no combos")?;
    let (high_p25, high_p50, high_p75) = core.price_range_for_last_run(&manual_high_grid);

    assert_eq!(low_results.len(), high_results.len());
    assert!(high_top.final_value > low_top.final_value + 50_000.0);
    assert!(
        high_top.high_variance > low_top.high_variance * 1.5,
        "manual prices should also update percentile variance, low={}, high={}",
        low_top.high_variance,
        high_top.high_variance
    );
    assert!(high_p25 > low_p25 + 20_000.0);
    assert!(high_p50 > low_p50 + 20_000.0);
    assert!(high_p75 > low_p75 + 20_000.0);
    Ok(())
}

#[test]
fn manual_per_grid_price_uses_total_grid_allocated_target() -> Result<()> {
    let low_total_grid = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 8,
        total_grid_target: Some(16.0),
        gw_count: Some(0),
        blue_count: Some(0),
        purple_count: Some(0),
        gold_count: Some(8),
        red_count: Some(0),
        manual_gold_per_grid: Some(10_000.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let high_total_grid = CalcParams {
        total_grid_target: Some(56.0),
        ..low_total_grid.clone()
    };

    let mut core = load_embedded_core()?;
    let low_results = core.run(low_total_grid.clone())?;
    let low_top = low_results
        .first()
        .context("low total-grid manual per-grid case produced no combos")?;
    let (_, low_p50, _) = core.price_range_for_last_run(&low_total_grid);

    let mut core = load_embedded_core()?;
    let high_results = core.run(high_total_grid.clone())?;
    let high_top = high_results
        .first()
        .context("high total-grid manual per-grid case produced no combos")?;
    let (_, high_p50, _) = core.price_range_for_last_run(&high_total_grid);

    assert!(
        high_top.final_value > low_top.final_value + 300_000.0,
        "manual per-grid pricing should use the allocated target grid, low={}, high={}",
        low_top.final_value,
        high_top.final_value
    );
    assert!(
        high_top.high_variance > low_top.high_variance * 2.0,
        "manual per-grid variance should use the allocated target grid, low={}, high={}",
        low_top.high_variance,
        high_top.high_variance
    );
    assert!(
        high_p50 > low_p50 + 300_000.0,
        "manual per-grid pricing should use the allocated target grid, low={low_p50}, high={high_p50}"
    );
    Ok(())
}

#[test]
fn composition_lines_use_total_grid_allocated_target() -> Result<()> {
    let low_total_grid = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 8,
        total_grid_target: Some(16.0),
        gw_count: Some(0),
        blue_count: Some(0),
        purple_count: Some(0),
        gold_count: Some(8),
        red_count: Some(0),
        manual_gold_per_grid: Some(10_000.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let high_total_grid = CalcParams {
        total_grid_target: Some(56.0),
        ..low_total_grid.clone()
    };

    let mut core = load_embedded_core()?;
    let low_results = core.run(low_total_grid.clone())?;
    let low_lines = core.combo_composition_lines(&low_results[0], &low_total_grid);

    let mut core = load_embedded_core()?;
    let high_results = core.run(high_total_grid.clone())?;
    let high_lines = core.combo_composition_lines(&high_results[0], &high_total_grid);

    let low_gold = low_lines
        .iter()
        .find(|line| line.starts_with("金(Q5):"))
        .context("missing low gold composition line")?;
    let high_gold = high_lines
        .iter()
        .find(|line| line.starts_with("金(Q5):"))
        .context("missing high gold composition line")?;

    assert!(
        sum_grid_mentions(high_gold) > sum_grid_mentions(low_gold),
        "composition lines should follow allocated target grids, low={low_gold}, high={high_gold}"
    );
    Ok(())
}

#[test]
fn composition_lines_normalize_average_grid_params() -> Result<()> {
    let cp = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 8,
        avg_grid_all: Some(7.0),
        gw_count: Some(0),
        blue_count: Some(0),
        purple_count: Some(0),
        gold_count: Some(8),
        red_count: Some(0),
        manual_gold_per_grid: Some(10_000.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let normalized = normalize_calc_params(cp.clone());

    let mut core = load_embedded_core()?;
    let results = core.run(cp.clone())?;
    let raw_lines = core.combo_composition_lines(&results[0], &cp);

    let mut core = load_embedded_core()?;
    let results = core.run(normalized.clone())?;
    let normalized_lines = core.combo_composition_lines(&results[0], &normalized);

    assert_eq!(raw_lines, normalized_lines);
    Ok(())
}

#[test]
fn composition_lines_do_not_panic_on_unknown_tier() -> Result<()> {
    let mut core = load_embedded_core()?;
    let run_cp = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 8,
        gw_count: Some(0),
        blue_count: Some(0),
        purple_count: Some(0),
        gold_count: Some(8),
        red_count: Some(0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let results = core.run(run_cp.clone())?;
    let unknown_tier_cp = CalcParams {
        tier: "unknown".to_string(),
        ..run_cp
    };

    let lines = core.combo_composition_lines(&results[0], &unknown_tier_cp);
    assert!(!lines.is_empty());
    Ok(())
}

#[test]
fn random_value_samples_shift_probability_weighting() -> Result<()> {
    let base = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2005".to_string(),
        total_count: 23,
        total_grid_target: Some(57.0),
        purple_count: Some(6),
        purple_grid: Some(12.0),
        purple_avg: Some(2.0),
        gold_avg: Some(1.0),
        safety_factor: 1.0,
        max_show: 1,
        ..Default::default()
    };

    let sampled = CalcParams {
        value_samples: vec![
            ValueSample {
                count: 3,
                avg_value: 611.33,
            },
            ValueSample {
                count: 6,
                avg_value: 1412.21,
            },
        ],
        ..base.clone()
    };

    let mut core = load_embedded_core()?;
    let base_results = core.run(base.clone())?;
    let (base_p25, base_p50, base_p75) = core.price_range_for_last_run(&base);

    let mut core = load_embedded_core()?;
    let sampled_results = core.run(sampled.clone())?;
    let (sampled_p25, sampled_p50, sampled_p75) = core.price_range_for_last_run(&sampled);

    assert!(!base_results.is_empty());
    assert!(!sampled_results.is_empty());
    assert!(sampled_p75 < base_p75);
    assert!(
        (sampled_p25 - base_p25).abs()
            + (sampled_p50 - base_p50).abs()
            + (sampled_p75 - base_p75).abs()
            > 50.0
    );
    Ok(())
}

#[test]
fn min_value_floor_sets_total_value_lower_bound() -> Result<()> {
    let base = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2005".to_string(),
        total_count: 23,
        total_grid_target: Some(57.0),
        purple_count: Some(6),
        purple_grid: Some(12.0),
        purple_avg: Some(2.0),
        gold_avg: Some(1.0),
        safety_factor: 1.0,
        max_show: 1,
        ..Default::default()
    };
    let floor = 60_000.0;
    let with_floor = CalcParams {
        min_value_floor: Some(floor),
        ..base.clone()
    };

    let mut core = load_embedded_core()?;
    core.run(base.clone())?;
    let (base_p25, base_p50, _) = core.price_range_for_last_run(&base);

    let mut core = load_embedded_core()?;
    core.run(with_floor.clone())?;
    let (floor_p25, floor_p50, floor_p75) = core.price_range_for_last_run(&with_floor);

    assert!(base_p25 < floor);
    assert!(base_p50 < floor);
    assert!(floor_p25 >= floor);
    assert!(floor_p50 >= floor);
    assert!(floor_p75 >= floor);
    Ok(())
}

#[test]
fn min_value_floor_filters_low_value_combos_before_percentiles() -> Result<()> {
    let mut core = load_embedded_core()?;
    let floor = 371_868.0;
    let cp = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 26,
        gw_grid: Some(5.0),
        blue_count: Some(6),
        gold_count: Some(8),
        gold_grid: Some(23.0),
        gold_avg: Some(2.87),
        min_value_floor: Some(floor),
        safety_factor: 0.85,
        max_show: 10,
        ..Default::default()
    };

    let results = core.run(cp.clone())?;
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|combo| combo.final_value + f64::EPSILON >= floor),
        "combos below the game minimum estimate should be removed, not clamped"
    );
    assert!(
        !results.iter().any(|combo| {
            combo.greenwhite_count == 5
                && combo.blue_count == 6
                && combo.purple_count == 7
                && combo.gold_count == 8
                && combo.red_count == 0
        }),
        "known below-floor case2 combo should be filtered out"
    );

    let (p25, p50, p75) = core.price_range_for_last_run(&cp);
    assert!(p25 > floor);
    assert!(p50 >= floor);
    assert!(p75 >= floor);
    let bid_p25 = recommended_bid_value(p25, &cp);
    let bid_p50 = recommended_bid_value(p50, &cp);
    assert!(bid_p25 >= floor);
    assert!(bid_p25 < p25);
    assert!(bid_p50 >= floor);
    assert!(bid_p50 < p50);
    Ok(())
}

#[test]
fn recommended_bid_keeps_game_floor_as_lower_bound() {
    let cp = CalcParams {
        min_value_floor: Some(371_868.0),
        safety_factor: 0.85,
        ..Default::default()
    };

    assert_eq!(
        recommended_bid_value(371_868.0, &cp).round() as i64,
        371_868
    );
    assert_eq!(
        recommended_bid_value(401_856.0, &cp).round() as i64,
        397_358
    );

    let no_floor = CalcParams {
        safety_factor: 0.85,
        ..Default::default()
    };
    assert_eq!(
        recommended_bid_value(401_856.0, &no_floor).round() as i64,
        341_578
    );
}

#[test]
fn million_plus_red_items_expand_single_combo_price_range() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "106".to_string(),
        map_nest_id: "2601".to_string(),
        total_count: 13,
        gw_count: Some(0),
        blue_count: Some(0),
        purple_count: Some(0),
        gold_count: Some(0),
        red_count: Some(13),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };

    let results = core.run(cp.clone())?;
    assert_eq!(results.len(), 1);
    assert!(
        results[0].high_value_price_points.len() > 10,
        "million-plus red items should create an explicit price distribution"
    );

    let (p25, p50, p75) = core.price_range_for_last_run(&cp);
    assert!(p25 > 0.0);
    assert!(p25 < p50);
    assert!(p50 < p75);
    assert!(
        p75 - p25 > 1_000_000.0,
        "jackpot enumeration should preserve a wide high-value tail, p25={p25}, p75={p75}"
    );
    Ok(())
}

#[test]
fn million_plus_enumeration_only_expands_top_three_probability_combos() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "106".to_string(),
        map_nest_id: "2601".to_string(),
        total_count: 20,
        red_count: Some(5),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };

    let results = core.run(cp)?;
    assert!(results.len() > 3);
    let expanded = results
        .iter()
        .filter(|combo| !combo.high_value_price_points.is_empty())
        .count();
    assert_eq!(
        expanded, 3,
        "only the probability top 3 combos should use million-plus item enumeration"
    );
    Ok(())
}

#[test]
fn min_value_floor_conflict_returns_error_instead_of_zero_price() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 26,
        gw_grid: Some(5.0),
        min_value_floor: Some(999_999_999.0),
        safety_factor: 0.85,
        max_show: 10,
        ..Default::default()
    };

    let err = core.run(cp).expect_err("conflicting min floor should fail");
    assert!(
        format!("{err:#}").contains("当前预估最低价格与模型冲突"),
        "{err:#}"
    );
    Ok(())
}

#[test]
fn high_quality_count_constrains_without_dropping_low_tier_value() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2037".to_string(),
        total_count: 110,
        high_quality_count: Some(18),
        min_purple: 8,
        min_gold: 3,
        min_red: 1,
        safety_factor: 0.85,
        max_show: 10,
        ..Default::default()
    };

    let results = core.run(cp.clone())?;
    assert!(!results.is_empty());
    for combo in &results {
        assert_eq!(
            combo.greenwhite_count
                + combo.blue_count
                + combo.purple_count
                + combo.gold_count
                + combo.red_count,
            cp.total_count
        );
        assert_eq!(combo.purple_count + combo.gold_count + combo.red_count, 18);
        assert_eq!(combo.greenwhite_count + combo.blue_count, 92);
    }
    let top = results.first().unwrap();
    assert!(top.greenwhite_count > 0);
    assert!(top.blue_count > 0);
    assert!(
        top.final_value > 700_000.0,
        "low-tier and blue value should be included in the total estimate"
    );
    Ok(())
}

#[test]
fn high_quality_count_equal_total_still_honors_all_constraints() -> Result<()> {
    let mut core = load_embedded_core()?;
    let contradictory = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 18,
        high_quality_count: Some(18),
        blue_count: Some(1),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let err = core
        .run(contradictory)
        .expect_err("all-high evidence should reject nonzero blue count");
    assert!(
        format!("{err:#}").contains("没有找到符合当前条件的组合"),
        "{err:#}"
    );

    let base = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2033".to_string(),
        total_count: 18,
        high_quality_count: Some(18),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let mut core = load_embedded_core()?;
    core.run(base.clone())?;
    let (_, base_p50, _) = core.price_range_for_last_run(&base);

    let with_grid = CalcParams {
        total_grid_target: Some(90.0),
        ..base
    };
    let mut core = load_embedded_core()?;
    core.run(with_grid.clone())?;
    let (_, grid_p50, _) = core.price_range_for_last_run(&with_grid);

    assert!(
        grid_p50 > base_p50 + 50_000.0,
        "global grid target should influence all-high pricing, base={base_p50}, grid={grid_p50}"
    );
    Ok(())
}

#[test]
fn large_color_count_average_grid_constraints_are_allowed() -> Result<()> {
    let mut core = load_embedded_core()?;
    let cp = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2037".to_string(),
        total_count: 110,
        blue_count: Some(52),
        blue_avg: Some(2.5),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };

    let results = core.run(cp)?;
    assert!(!results.is_empty());
    assert!(results.iter().all(|combo| combo.blue_count == 52));
    assert!(
        core.raw_results.iter().all(|combo| combo.blue_count == 52),
        "average-grid constraints for counts above 40 should not be rejected"
    );
    Ok(())
}

#[test]
fn blue_twenty_grid_items_are_valid_but_purple_twenty_is_not() -> Result<()> {
    let mut core = load_embedded_core()?;
    let blue = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 1,
        blue_count: Some(1),
        blue_grid: Some(20.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let blue_results = core.run(blue)?;
    assert_eq!(blue_results.len(), 1);
    assert_eq!(blue_results[0].blue_count, 1);
    assert_eq!(blue_results[0].blue_grid_est, 20.0);

    let mut core = load_embedded_core()?;
    let purple = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 1,
        purple_count: Some(1),
        purple_grid: Some(20.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let err = core
        .run(purple)
        .expect_err("purple quality has no 20-grid item in embedded data");
    assert!(format!("{err:#}").contains("没有找到符合当前条件的组合"));
    Ok(())
}

#[test]
fn exact_grid_constraints_use_current_map_shapes() -> Result<()> {
    let mut core = load_embedded_core()?;
    let impossible_express_red = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 1,
        red_count: Some(1),
        red_grid: Some(4.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let err = core
        .run(impossible_express_red)
        .expect_err("express map has no one-piece 4-grid red item");
    assert!(format!("{err:#}").contains("没有找到符合当前条件的组合"));

    let mut core = load_embedded_core()?;
    let possible_villa_red = CalcParams {
        tier: "104".to_string(),
        map_nest_id: "2037".to_string(),
        total_count: 1,
        red_count: Some(1),
        red_grid: Some(4.0),
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };
    let results = core.run(possible_villa_red)?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].red_grid_est, 4.0);
    Ok(())
}

#[test]
fn average_grid_inference_uses_real_quality_shapes() {
    assert_eq!(
        infer_grid_from_average_for_quality(1, 20.0, Some(3)),
        Some(20)
    );
    assert_eq!(infer_grid_from_average_for_quality(1, 7.0, Some(3)), None);
    assert_eq!(infer_grid_from_average_for_quality(1, 7.0, Some(4)), None);
    assert_eq!(
        infer_grid_from_average_for_quality(11, 2.54, Some(3)),
        Some(28)
    );
}

#[test]
fn average_grid_inference_can_use_current_map_shapes() -> Result<()> {
    let mut core = load_embedded_core()?;
    let express_red_sizes = core.loader.get_map_grid_sizes_by_quality(Some("2001"), 6);
    let villa_red_sizes = core.loader.get_map_grid_sizes_by_quality(Some("2037"), 6);

    assert_eq!(
        infer_grid_from_average_for_quality(1, 4.0, Some(6)),
        Some(4)
    );
    assert_eq!(
        infer_grid_from_average_with_sizes(1, 4.0, &express_red_sizes),
        None
    );
    assert_eq!(
        infer_grid_from_average_with_sizes(1, 4.0, &villa_red_sizes),
        Some(4)
    );
    Ok(())
}

#[test]
fn modified_model_preserves_core_invariants() -> Result<()> {
    let mut core = load_embedded_core()?;
    let floor = 60_000.0;
    let cp = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2005".to_string(),
        total_count: 23,
        total_grid_target: Some(57.0),
        purple_count: Some(6),
        purple_grid: Some(12.0),
        purple_avg: Some(2.0),
        gold_avg: Some(1.0),
        min_value_floor: Some(floor),
        manual_purple_per_item: Some(5_266.0),
        manual_purple_per_grid: Some(586.0),
        manual_gold_per_item: Some(7_856.0),
        manual_gold_per_grid: Some(589.0),
        value_samples: vec![
            ValueSample {
                count: 3,
                avg_value: 611.33,
            },
            ValueSample {
                count: 6,
                avg_value: 1412.21,
            },
        ],
        safety_factor: 1.0,
        max_show: 10,
        ..Default::default()
    };

    let results = core.run(cp.clone())?;
    assert!(!results.is_empty());
    assert!(!core.raw_results.is_empty());

    let raw_probability_sum: f64 = core.raw_results.iter().map(|combo| combo.probability).sum();
    let shown_probability_sum: f64 = results.iter().map(|combo| combo.probability).sum();
    assert!((raw_probability_sum - 1.0).abs() < 1e-9);
    assert!((shown_probability_sum - 1.0).abs() < 1e-9);

    for combo in &results {
        assert_eq!(
            combo.greenwhite_count
                + combo.blue_count
                + combo.purple_count
                + combo.gold_count
                + combo.red_count,
            cp.total_count
        );
        assert_eq!(combo.purple_count, 6);
        assert!((combo.purple_grid_est - 12.0).abs() < 1e-9);
        assert!(combo.probability.is_finite());
        assert!(combo.probability >= 0.0);
        assert!(combo.final_value.is_finite());
        assert!(combo.final_value >= 0.0);
        assert!(combo.total_grid_est.is_finite());
        assert!(combo.total_grid_est >= 0.0);
    }

    let (p25, p50, p75) = core.price_range_for_last_run(&cp);
    assert!(p25.is_finite());
    assert!(p50.is_finite());
    assert!(p75.is_finite());
    assert!(p25 <= p50);
    assert!(p50 <= p75);
    assert!(p25 >= floor);
    assert!(p50 >= floor);
    assert!(p75 >= floor);
    Ok(())
}

#[test]
fn embedded_data_case_2601_hidden_auction_total_63() -> Result<()> {
    let mut core = load_embedded_core()?;

    assert_eq!(
        core.static_data.map_to_nest.get("2601").map(String::as_str),
        Some("2601")
    );
    assert_eq!(
        core.static_data.map_names.get("2601").map(String::as_str),
        Some("隐秘拍卖会")
    );
    assert_eq!(
        core.static_data.drop_weights.get("106"),
        Some(&vec![10.0, 100.0, 200.0, 240.0, 220.0, 200.0])
    );

    let cp = CalcParams {
        tier: "106".to_string(),
        map_nest_id: "2601".to_string(),
        total_count: 63,
        safety_factor: 0.85,
        max_show: 1,
        ..Default::default()
    };
    let results = core.run(cp.clone())?;
    let (p25, p50, p75) = core.price_range_for_last_run(&cp);

    assert_eq!(results.len(), 2396);
    assert_eq!(core.raw_results.len(), 766480);
    assert_eq!((p25 * cp.safety_factor).round() as i64, 1_812_336);
    assert_eq!((p50 * cp.safety_factor).round() as i64, 2_254_686);
    assert_eq!((p75 * cp.safety_factor).round() as i64, 2_697_036);
    let top = &results[0];
    assert_eq!(
        (
            top.greenwhite_count,
            top.blue_count,
            top.purple_count,
            top.gold_count,
            top.red_count
        ),
        (7, 13, 16, 14, 13)
    );
    assert_eq!(top.final_value.round() as i64, 2_664_105);
    assert_eq!((top.total_grid_est * 10.0).round() / 10.0, 177.4);
    Ok(())
}

#[test]
fn grid_constraints_reject_impossible_item_sizes_from_4_12_3() -> Result<()> {
    let mut core = load_embedded_core()?;
    let base = CalcParams {
        tier: "101".to_string(),
        map_nest_id: "2001".to_string(),
        total_count: 1,
        blue_count: Some(1),
        safety_factor: 0.85,
        max_show: 1,
        ..Default::default()
    };

    let exact_impossible = CalcParams {
        blue_grid: Some(7.0),
        ..base.clone()
    };
    let err = core
        .run(exact_impossible)
        .expect_err("impossible exact grid should be reported as no-solution");
    assert!(format!("{err:#}").contains("没有找到符合当前条件的组合"));

    let avg_impossible = CalcParams {
        blue_avg: Some(7.0),
        ..base.clone()
    };
    let err = core
        .run(avg_impossible)
        .expect_err("impossible average grid should be reported as no-solution");
    assert!(format!("{err:#}").contains("没有找到符合当前条件的组合"));

    let exact_possible = CalcParams {
        blue_grid: Some(6.0),
        ..base
    };
    assert_eq!(core.run(exact_possible)?.len(), 1);
    Ok(())
}

#[test]
fn generated_reference_cases_4_12_2() -> Result<()> {
    let data = Path::new("../decompiled_4_12_2/MapBidCalculator.calculator_data_merged.csv");
    let static_data = Path::new("../core_algorithm/static_data.json");
    if !data.exists() || !static_data.exists() {
        eprintln!("reference extraction files are not present; skipping local reference test");
        return Ok(());
    }

    let fixture = Path::new("tests/fixtures/reference_cases_4_12_2.json");
    let cases: Vec<ReferenceCase> = serde_json::from_reader(
        std::fs::File::open(fixture).with_context(|| format!("open {}", fixture.display()))?,
    )
    .context("parse reference fixture")?;

    for case in cases {
        let mut core = load_core(data, static_data)?;
        let nest_id = case
            .nest_id
            .clone()
            .or_else(|| {
                case.map_id
                    .as_ref()
                    .and_then(|map_id| core.static_data.map_to_nest.get(map_id).cloned())
            })
            .unwrap_or_else(|| "2001".to_string());
        let cp = CalcParams {
            tier: case.tier.clone(),
            map_nest_id: nest_id,
            total_count: case.total,
            total_grid_target: case.total_grid,
            avg_grid_all: case.avg_grid_all,
            high_quality_count: case.high_quality_count,
            gw_count: case.gw_count,
            min_gw: case.gw_min.unwrap_or_default(),
            gw_grid: case.gw_grid,
            gw_avg: case.gw_avg,
            blue_count: case.blue_count,
            min_blue: case.blue_min.unwrap_or_default(),
            blue_grid: case.blue_grid,
            blue_avg: case.blue_avg,
            purple_count: case.purple_count,
            min_purple: case.purple_min.unwrap_or_default(),
            purple_grid: case.purple_grid,
            purple_avg: case.purple_avg,
            gold_count: case.gold_count,
            min_gold: case.gold_min.unwrap_or_default(),
            gold_grid: case.gold_grid,
            gold_avg: case.gold_avg,
            red_count: case.red_count,
            min_red: case.red_min.unwrap_or_default(),
            red_grid: case.red_grid,
            red_avg: case.red_avg,
            safety_factor: case.safety,
            max_show: 1,
            ..Default::default()
        };
        let results = core.run(cp.clone()).with_context(|| case.name.clone())?;
        let (p25, p50, p75) = core.price_range_for_last_run(&cp);
        assert_eq!(results.len(), case.expected.combos, "{}", case.name);
        assert_eq!(core.raw_results.len(), case.expected.raw, "{}", case.name);
        assert_close_i64(
            (p25 * cp.safety_factor).round() as i64,
            case.expected.bid_p25,
            &case.name,
        );
        assert_close_i64(
            (p50 * cp.safety_factor).round() as i64,
            case.expected.bid_p50,
            &case.name,
        );
        assert_close_i64(
            (p75 * cp.safety_factor).round() as i64,
            case.expected.bid_p75,
            &case.name,
        );
        let top = results.first().context("case produced no combos")?;
        assert_eq!(
            top.greenwhite_count, case.expected.top.greenwhite,
            "{}",
            case.name
        );
        assert_eq!(top.blue_count, case.expected.top.blue, "{}", case.name);
        assert_eq!(top.purple_count, case.expected.top.purple, "{}", case.name);
        assert_eq!(top.gold_count, case.expected.top.gold, "{}", case.name);
        assert_eq!(top.red_count, case.expected.top.red, "{}", case.name);
        assert_eq!(
            (top.probability * 1_000_000.0).round() / 1_000_000.0,
            case.expected.top.probability_6dp,
            "{}",
            case.name
        );
        assert_eq!(
            top.final_value.round() as i64,
            case.expected.top.final_value,
            "{}",
            case.name
        );
        assert_eq!(
            (top.total_grid_est * 10.0).round() / 10.0,
            case.expected.top.total_grid_est_1dp,
            "{}",
            case.name
        );
    }

    Ok(())
}

fn assert_close_i64(actual: i64, expected: i64, case_name: &str) {
    assert!(
        (actual - expected).abs() <= 1,
        "{case_name}: actual {actual}, expected {expected}"
    );
}
