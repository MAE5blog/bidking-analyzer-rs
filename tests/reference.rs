use anyhow::{Context, Result};
use bidking_rs::{CalcParams, load_core, load_embedded_core};
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
    let (p25, p50, p75) = core.price_range(&results, &cp);

    assert_eq!(results.len(), 1054);
    assert_eq!(core.raw_results.len(), 766480);
    assert_eq!((p25 * cp.safety_factor).round() as i64, 85_913);
    assert_eq!((p50 * cp.safety_factor).round() as i64, 105_758);
    assert_eq!((p75 * cp.safety_factor).round() as i64, 130_632);
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
    let (p25, p50, p75) = core.price_range(&results, &cp);

    assert_eq!(results.len(), 1054);
    assert_eq!(core.raw_results.len(), 766480);
    assert_eq!((p25 * cp.safety_factor).round() as i64, 85_913);
    assert_eq!((p50 * cp.safety_factor).round() as i64, 105_758);
    assert_eq!((p75 * cp.safety_factor).round() as i64, 130_632);
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
    let (p25, p50, p75) = core.price_range(&results, &cp);

    assert_eq!(results.len(), 2396);
    assert_eq!(core.raw_results.len(), 766480);
    assert_eq!((p25 * cp.safety_factor).round() as i64, 1_820_837);
    assert_eq!((p50 * cp.safety_factor).round() as i64, 2_263_426);
    assert_eq!((p75 * cp.safety_factor).round() as i64, 2_706_014);
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
    assert_eq!((top.total_grid_est * 10.0).round() / 10.0, 177.0);
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
    assert!(core.run(exact_impossible)?.is_empty());

    let avg_impossible = CalcParams {
        blue_avg: Some(7.0),
        ..base.clone()
    };
    assert!(core.run(avg_impossible)?.is_empty());

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
        let (p25, p50, p75) = core.price_range(&results, &cp);
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
