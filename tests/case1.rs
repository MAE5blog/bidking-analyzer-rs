use anyhow::Result;
use bidking_rs::{
    CalcParams, ComboResult, infer_grid_from_average_for_quality, load_embedded_core,
};

#[derive(Debug, Clone)]
struct GuiState {
    cp: CalcParams,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            cp: CalcParams {
                tier: "104".to_string(),
                map_nest_id: "2031".to_string(),
                total_count: 36,
                safety_factor: 0.85,
                max_show: 10,
                ..Default::default()
            },
        }
    }
}

impl GuiState {
    fn calculate_and_auto_fill(&mut self) -> Result<Vec<ComboResult>> {
        let mut core = load_embedded_core()?;
        let results = core.run(self.cp.clone())?;
        auto_fill_unique_fields(&mut self.cp, &core.raw_results);
        Ok(results)
    }
}

#[test]
fn case1_late_round_ocr_clues_do_not_collapse_to_zero_results() -> Result<()> {
    let mut state = GuiState::default();

    state.cp.gw_grid = Some(27.0);
    state.cp.min_value_floor = Some(11_218.0);
    let round1 = state.calculate_and_auto_fill()?;
    assert!(!round1.is_empty());

    state.cp.gw_avg = Some(2.25);
    state.cp.gold_avg = Some(5.0);
    let round2 = state.calculate_and_auto_fill()?;
    assert!(!round2.is_empty());

    state.cp.blue_count = Some(11);
    state.cp.purple_avg = Some(2.36);
    state.cp.manual_gold_per_item = Some(43_087.5);
    let round3 = state.calculate_and_auto_fill()?;
    assert!(!round3.is_empty());
    assert_eq!(
        state.cp.blue_grid, None,
        "model-estimated blue grid should not be written back before OCR sees blue average"
    );

    state.cp.blue_avg = Some(2.54);
    let round4 = state.calculate_and_auto_fill()?;
    assert!(
        !round4.is_empty(),
        "round 4 with blue average should still have viable combos"
    );
    assert_eq!(
        state.cp.blue_grid,
        Some(28.0),
        "blue count 11 and average 2.54 uniquely imply 28 grids"
    );

    state.cp.gw_count = Some(12);
    let round5 = state.calculate_and_auto_fill()?;
    assert!(
        !round5.is_empty(),
        "round 5 with green/white count should still have viable combos"
    );

    let core = load_embedded_core()?;
    let (p25, p50, p75) = core.price_range(&round5, &state.cp);
    assert!(p25 > 0.0);
    assert!(p50 > 0.0);
    assert!(p75 > 0.0);
    Ok(())
}

fn auto_fill_unique_fields(cp: &mut CalcParams, results: &[ComboResult]) {
    cp.gw_count = cp
        .gw_count
        .or_else(|| unique_i32(results, |r| r.greenwhite_count));
    set_grid_from_average(&mut cp.gw_grid, cp.gw_count, cp.gw_avg, None);
    cp.blue_count = cp
        .blue_count
        .or_else(|| unique_i32(results, |r| r.blue_count));
    set_grid_from_average(&mut cp.blue_grid, cp.blue_count, cp.blue_avg, Some(3));
    cp.purple_count = cp
        .purple_count
        .or_else(|| unique_i32(results, |r| r.purple_count));
    set_grid_from_average(&mut cp.purple_grid, cp.purple_count, cp.purple_avg, Some(4));
    cp.gold_count = cp
        .gold_count
        .or_else(|| unique_i32(results, |r| r.gold_count));
    set_grid_from_average(&mut cp.gold_grid, cp.gold_count, cp.gold_avg, Some(5));
    cp.red_count = cp
        .red_count
        .or_else(|| unique_i32(results, |r| r.red_count));
    set_grid_from_average(&mut cp.red_grid, cp.red_count, cp.red_avg, Some(6));
}

fn unique_i32(results: &[ComboResult], f: impl Fn(&ComboResult) -> i32) -> Option<i32> {
    let mut iter = results.iter();
    let first = f(iter.next()?);
    iter.all(|result| f(result) == first).then_some(first)
}

fn set_grid_from_average(
    target: &mut Option<f64>,
    count: Option<i32>,
    avg: Option<f64>,
    quality: Option<i32>,
) {
    if let Some(grid) = count
        .zip(avg)
        .and_then(|(count, avg)| infer_grid_from_average_for_quality(count, avg, quality))
    {
        *target = Some(grid as f64);
    }
}
