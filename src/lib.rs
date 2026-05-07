use anyhow::{Context, Result};
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub mod importer;
pub mod ocr;

pub const PROB_CUTOFF: f64 = 0.0001;
pub const EMBEDDED_DATA_VERSION: &str = "auctionanalyzer-4.12.3";
const MAX_ITEM_GRID_SIZE: i32 = 20;
const DEFAULT_ITEM_GRID_SIZES: &[i32] = &[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20];
const BLUE_ITEM_GRID_SIZES: &[i32] = &[1, 2, 3, 4, 5, 6, 8, 9, 15, 16, 20];
const PURPLE_ITEM_GRID_SIZES: &[i32] = &[1, 2, 3, 4, 5, 6, 8, 9, 10, 12];
const GOLD_ITEM_GRID_SIZES: &[i32] = &[1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18];
const RED_ITEM_GRID_SIZES: &[i32] = &[1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16];
const VALUE_SAMPLE_EVIDENCE_WEIGHT: f64 = 0.62;
const PRICE_RANGE_MIN_MASS: f64 = 0.999;
const PRICE_RANGE_MAX_COMBOS: usize = 50_000;
const JACKPOT_VALUE_THRESHOLD: f64 = 1_000_000.0;
const JACKPOT_VALUE_BIN: f64 = 10_000.0;
const JACKPOT_MAX_RED_COUNT: i32 = 20;
const JACKPOT_MAX_PRICE_POINTS: usize = 512;
const JACKPOT_TOP_ENUM_COMBOS: usize = 3;

const EMBEDDED_STATIC_DATA: &str = include_str!("../data/auctionanalyzer-4.12.3/static_data.json");
const EMBEDDED_MERGED_CSV: &[u8] = include_bytes!(
    "../data/auctionanalyzer-4.12.3/resources/MapBidCalculator.calculator_data_merged.csv"
);

#[derive(Debug, Clone, Deserialize)]
pub struct StaticData {
    pub drop_weights: HashMap<String, Vec<f64>>,
    pub quality_p50_default: HashMap<String, f64>,
    pub nest_weighted_prices: HashMap<String, Vec<f64>>,
    pub map_to_nest: HashMap<String, String>,
    pub map_names: HashMap<String, String>,
}

impl StaticData {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("open static data {}", path.as_ref().display()))?;
        serde_json::from_reader(file).context("parse static data json")
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("parse embedded static data json")
    }
}

#[derive(Debug, Clone)]
pub struct ItemRecord {
    pub record_type: String,
    pub item_id: String,
    pub name: String,
    pub quality: i32,
    pub value: f64,
    pub shape: String,
    pub drop_id: String,
    pub ref_id: String,
    pub weight: f64,
    pub ref_type: String,
    pub grid_size: i32,
    pub grid_trusted: bool,
}

#[derive(Debug, Deserialize)]
struct CsvRow {
    #[serde(default)]
    record_type: String,
    #[serde(default)]
    item_id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    quality: String,
    #[serde(default, rename = "base_value")]
    value: String,
    #[serde(default)]
    shape: String,
    #[serde(default)]
    drop_id: String,
    #[serde(default)]
    ref_id: String,
    #[serde(default)]
    weight: String,
    #[serde(default)]
    ref_type: String,
}

#[derive(Debug, Clone)]
struct DropEdge {
    ref_id: String,
    weight: f64,
    _ref_type: String,
    p: f64,
}

#[derive(Debug, Clone)]
pub struct MapQualityProbs {
    pub p_low: f64,
    pub p_high: f64,
    pub pb: f64,
    pub pp: f64,
    pub pg: f64,
    pub pr: f64,
    pub source: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QualityGridStats {
    pub mean: f64,
    pub variance: f64,
    pub count: usize,
    pub effective_count: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QualityPriceStats {
    pub mean: f64,
    pub variance: f64,
    pub p50: f64,
    pub count: usize,
}

#[derive(Debug, Clone)]
struct GridValueDistribution {
    weights: Vec<f64>,
    value_sums: Vec<f64>,
    max_grid: usize,
    mean_grid: f64,
    neutral_value: f64,
}

#[derive(Debug, Clone)]
pub struct CalcParams {
    pub tier: String,
    pub map_nest_id: String,
    pub total_count: i32,
    pub total_grid_target: Option<f64>,
    pub avg_grid_all: Option<f64>,
    pub high_quality_count: Option<i32>,
    pub gw_count: Option<i32>,
    pub min_gw: i32,
    pub gw_grid: Option<f64>,
    pub gw_avg: Option<f64>,
    pub blue_count: Option<i32>,
    pub min_blue: i32,
    pub blue_grid: Option<f64>,
    pub blue_avg: Option<f64>,
    pub purple_count: Option<i32>,
    pub min_purple: i32,
    pub purple_grid: Option<f64>,
    pub purple_avg: Option<f64>,
    pub gold_count: Option<i32>,
    pub min_gold: i32,
    pub gold_grid: Option<f64>,
    pub gold_avg: Option<f64>,
    pub red_count: Option<i32>,
    pub min_red: i32,
    pub red_grid: Option<f64>,
    pub red_avg: Option<f64>,
    pub safety_factor: f64,
    pub max_show: usize,
    pub min_value_floor: Option<f64>,
    pub manual_purple_per_item: Option<f64>,
    pub manual_purple_per_grid: Option<f64>,
    pub manual_gold_per_item: Option<f64>,
    pub manual_gold_per_grid: Option<f64>,
    pub value_samples: Vec<ValueSample>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueSample {
    pub count: i32,
    pub avg_value: f64,
}

impl Default for CalcParams {
    fn default() -> Self {
        Self {
            tier: "101".to_string(),
            map_nest_id: "2001".to_string(),
            total_count: 30,
            total_grid_target: None,
            avg_grid_all: None,
            high_quality_count: None,
            gw_count: None,
            min_gw: 0,
            gw_grid: None,
            gw_avg: None,
            blue_count: None,
            min_blue: 0,
            blue_grid: None,
            blue_avg: None,
            purple_count: None,
            min_purple: 0,
            purple_grid: None,
            purple_avg: None,
            gold_count: None,
            min_gold: 0,
            gold_grid: None,
            gold_avg: None,
            red_count: None,
            min_red: 0,
            red_grid: None,
            red_avg: None,
            safety_factor: 0.85,
            max_show: 10,
            min_value_floor: None,
            manual_purple_per_item: None,
            manual_purple_per_grid: None,
            manual_gold_per_item: None,
            manual_gold_per_grid: None,
            value_samples: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ComboResult {
    pub greenwhite_count: i32,
    pub blue_count: i32,
    pub purple_count: i32,
    pub gold_count: i32,
    pub red_count: i32,
    pub log_w: f64,
    pub probability: f64,
    pub final_value: f64,
    pub high_variance: f64,
    pub high_value_price_points: Vec<PricePoint>,
    pub total_grid_est: f64,
    pub greenwhite_grid_est: f64,
    pub blue_grid_est: f64,
    pub purple_grid_est: f64,
    pub gold_grid_est: f64,
    pub red_grid_est: f64,
    pub greenwhite_grid_value: Option<i32>,
    pub blue_grid_value: Option<i32>,
    pub purple_grid_value: Option<i32>,
    pub gold_grid_value: Option<i32>,
    pub red_grid_value: Option<i32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PricePoint {
    pub value: f64,
    pub probability: f64,
}

#[derive(Debug, Clone, Copy)]
struct PriceModel {
    greenwhite_mean: f64,
    blue_mean: f64,
    purple_mean: f64,
    gold_mean: f64,
    red_mean: f64,
    blue_variance: f64,
    purple_variance: f64,
    gold_variance: f64,
    red_variance: f64,
}

#[derive(Debug, Clone, Copy)]
struct GridStatsModel {
    greenwhite_mean: f64,
    blue_mean: f64,
    purple_mean: f64,
    gold_mean: f64,
    red_mean: f64,
    greenwhite_variance: f64,
    blue_variance: f64,
    purple_variance: f64,
    gold_variance: f64,
    red_variance: f64,
}

#[derive(Debug, Clone, Copy)]
struct HighQualityGridSizes<'a> {
    purple: &'a [i32],
    gold: &'a [i32],
    red: &'a [i32],
}

#[derive(Debug, Clone, Default)]
struct PricingGridTargets {
    gw: Option<i32>,
    blue: Option<i32>,
    purple: Option<i32>,
    gold: Option<i32>,
    red: Option<i32>,
}

#[derive(Debug, Clone, Copy)]
struct ResidualGridSlot {
    index: usize,
    quality: Option<i32>,
    count: i32,
    model_grid: f64,
}

pub struct DataLoader {
    pub static_data: StaticData,
    drop_graph: HashMap<String, Vec<DropEdge>>,
    pub items_by_quality: HashMap<i32, Vec<ItemRecord>>,
    pub items_by_id: HashMap<String, ItemRecord>,
    avg_grid_by_quality: HashMap<i32, f64>,
    grid_impact_by_quality: HashMap<i32, f64>,
    resolve_cache: HashMap<String, HashMap<String, f64>>,
    map_prob_cache: HashMap<String, MapQualityProbs>,
    map_grid_stats_cache: HashMap<String, QualityGridStats>,
    quality_stats_cache: HashMap<String, QualityPriceStats>,
    grid_value_cache: HashMap<String, Option<f64>>,
    grid_dist_cache: HashMap<String, Option<GridValueDistribution>>,
}

impl DataLoader {
    pub fn new(static_data: StaticData) -> Self {
        Self {
            static_data,
            drop_graph: HashMap::new(),
            items_by_quality: [(3, vec![]), (4, vec![]), (5, vec![]), (6, vec![])]
                .into_iter()
                .collect(),
            items_by_id: HashMap::new(),
            avg_grid_by_quality: HashMap::new(),
            grid_impact_by_quality: HashMap::new(),
            resolve_cache: HashMap::new(),
            map_prob_cache: HashMap::new(),
            map_grid_stats_cache: HashMap::new(),
            quality_stats_cache: HashMap::new(),
            grid_value_cache: HashMap::new(),
            grid_dist_cache: HashMap::new(),
        }
    }

    pub fn load_merged_csv(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let source = format!("merged csv {}", path.as_ref().display());
        let file = File::open(path.as_ref())
            .with_context(|| format!("open merged csv {}", path.as_ref().display()))?;
        self.load_merged_csv_reader(file, &source)
    }

    pub fn load_merged_csv_bytes(&mut self, bytes: &[u8], source: &str) -> Result<()> {
        self.load_merged_csv_reader(bytes, source)
    }

    fn load_merged_csv_reader<R: Read>(&mut self, reader: R, source: &str) -> Result<()> {
        self.clear_loaded_data();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(reader);
        let mut edges: HashMap<String, Vec<DropEdge>> = HashMap::new();
        for row in reader.deserialize::<CsvRow>() {
            let row = row.with_context(|| format!("parse csv row in {source}"))?;
            let rec = row.record_type.trim().to_uppercase();
            let mut item = ItemRecord {
                record_type: rec.clone(),
                item_id: row.item_id.trim().to_string(),
                name: row.name.trim().to_string(),
                quality: parse_i32(&row.quality),
                value: parse_f64(&row.value),
                shape: row.shape.trim().to_string(),
                drop_id: row.drop_id.trim().to_string(),
                ref_id: row.ref_id.trim().to_string(),
                weight: parse_f64(&row.weight),
                ref_type: row.ref_type.trim().to_string(),
                grid_size: 0,
                grid_trusted: false,
            };
            if rec == "ITEM" {
                if (3..=6).contains(&item.quality) && item.value > 0.0 {
                    parse_grid_size(&mut item);
                    self.items_by_quality
                        .entry(item.quality)
                        .or_default()
                        .push(item.clone());
                    if !item.item_id.is_empty() {
                        self.items_by_id.insert(item.item_id.clone(), item);
                    }
                }
            } else if rec == "DROP"
                && !item.drop_id.is_empty()
                && !item.ref_id.is_empty()
                && item.weight > 0.0
            {
                edges
                    .entry(item.drop_id.clone())
                    .or_default()
                    .push(DropEdge {
                        ref_id: item.ref_id,
                        weight: item.weight,
                        _ref_type: item.ref_type,
                        p: 0.0,
                    });
            }
        }
        for values in edges.values_mut() {
            let total: f64 = values.iter().map(|e| e.weight).sum();
            if total > 0.0 {
                for edge in values {
                    edge.p = edge.weight / total;
                }
            }
        }
        self.drop_graph = edges;
        for (quality, items) in &self.items_by_quality {
            let trusted: Vec<f64> = items
                .iter()
                .filter(|i| i.grid_trusted && i.grid_size > 0)
                .map(|i| i.grid_size as f64)
                .collect();
            if !trusted.is_empty() {
                self.avg_grid_by_quality
                    .insert(*quality, trusted.iter().sum::<f64>() / trusted.len() as f64);
            }
        }
        self.compute_grid_impact_by_quality();
        Ok(())
    }

    fn clear_loaded_data(&mut self) {
        self.drop_graph.clear();
        self.items_by_quality.clear();
        self.items_by_id.clear();
        self.avg_grid_by_quality.clear();
        self.grid_impact_by_quality.clear();
        self.resolve_cache.clear();
        self.map_prob_cache.clear();
        self.map_grid_stats_cache.clear();
        self.quality_stats_cache.clear();
        self.grid_value_cache.clear();
        self.grid_dist_cache.clear();
    }

    pub fn drop_graph_loaded(&self) -> bool {
        !self.drop_graph.is_empty()
    }

    fn compute_grid_impact_by_quality(&mut self) {
        self.grid_impact_by_quality.clear();
        for quality in 3..=6 {
            let items = self
                .items_by_quality
                .get(&quality)
                .cloned()
                .unwrap_or_default();
            let max_quality_grid = max_grid_for_quality(Some(quality));
            let mut usable: Vec<ItemRecord> = items
                .iter()
                .filter(|i| {
                    i.grid_trusted
                        && i.grid_size > 0
                        && i.grid_size <= max_quality_grid
                        && i.value > 0.0
                })
                .cloned()
                .collect();
            if usable.len() < 4 {
                self.grid_impact_by_quality.insert(quality, 0.0);
                continue;
            }
            let mut sorted_by_value: Vec<ItemRecord> = items
                .iter()
                .filter(|i| i.value > 0.0 && i.value.is_finite())
                .cloned()
                .collect();
            sorted_by_value.sort_by(|a, b| fcmp(b.value, a.value));
            let high_take = ((sorted_by_value.len() as f64) * 0.3).floor() as usize;
            let high_ids: HashSet<String> = sorted_by_value
                .into_iter()
                .take(high_take)
                .map(|i| i.item_id)
                .collect();
            usable.retain(|i| !high_ids.contains(&i.item_id));
            if usable.len() < 4 {
                self.grid_impact_by_quality.insert(quality, 0.0);
                continue;
            }
            let mut grids: Vec<i32> = usable.iter().map(|i| i.grid_size).collect();
            grids.sort_unstable();
            grids.dedup();
            let mut buckets = Vec::new();
            for grid in grids {
                let vals: Vec<f64> = usable
                    .iter()
                    .filter(|i| i.grid_size == grid)
                    .map(|i| i.value)
                    .collect();
                buckets.push((grid, median(vals)));
            }
            let mut slopes = Vec::new();
            for a in 0..buckets.len() {
                for b in (a + 1)..buckets.len() {
                    let delta_grid = buckets[b].0 - buckets[a].0;
                    if delta_grid > 0 {
                        let slope = ((buckets[b].1 - buckets[a].1) / delta_grid as f64).abs();
                        if slope.is_finite() && slope > 0.0 {
                            slopes.push(slope);
                        }
                    }
                }
            }
            let med_value = median(usable.iter().map(|i| i.value).collect::<Vec<_>>());
            let val = if slopes.is_empty() {
                med_value * 0.08
            } else {
                median(slopes)
            };
            self.grid_impact_by_quality
                .insert(quality, (med_value * 0.02).max((med_value * 0.25).min(val)));
        }
    }

    pub fn resolve_drop_to_items(&mut self, drop_id: Option<&str>) -> Option<HashMap<String, f64>> {
        let drop_id = drop_id?;
        if drop_id.is_empty() || self.drop_graph.is_empty() {
            return None;
        }
        if let Some(value) = self.resolve_cache.get(drop_id) {
            return Some(value.clone());
        }
        let mut out = HashMap::new();
        let mut path = HashSet::new();
        path.insert(drop_id.to_string());
        self.dfs_resolve(drop_id, 1.0, &mut out, &mut path);
        let total: f64 = out.values().sum();
        if total <= 0.0 {
            return None;
        }
        for value in out.values_mut() {
            *value /= total;
        }
        self.resolve_cache.insert(drop_id.to_string(), out.clone());
        Some(out)
    }

    fn dfs_resolve(
        &self,
        drop_id: &str,
        scale: f64,
        out: &mut HashMap<String, f64>,
        path: &mut HashSet<String>,
    ) {
        if let Some(edges) = self.drop_graph.get(drop_id) {
            for edge in edges {
                let p = scale * edge.p;
                let ref_id = edge.ref_id.as_str();
                let is_item = self.items_by_id.contains_key(ref_id);
                let is_drop = self.drop_graph.contains_key(ref_id);
                if is_item {
                    *out.entry(edge.ref_id.clone()).or_insert(0.0) += p;
                }
                if is_drop && !is_item && !path.contains(ref_id) {
                    path.insert(edge.ref_id.clone());
                    self.dfs_resolve(ref_id, p, out, path);
                    path.remove(ref_id);
                }
            }
        }
    }

    pub fn get_map_quality_probs(
        &mut self,
        nest_id: Option<&str>,
        tier_weights: &[f64],
    ) -> MapQualityProbs {
        let key = format!(
            "{}|{}",
            nest_id.unwrap_or("*"),
            tier_weights
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        if let Some(value) = self.map_prob_cache.get(&key) {
            return value.clone();
        }
        let total = tier_weights.iter().sum::<f64>().max(1.0);
        let p_low = (tier_weights[0] + tier_weights[1]) / total;
        let p_high = 1.0 - p_low;
        let high_total = tier_weights[2..6].iter().sum::<f64>().max(1.0);
        let fallback = apply_prob_floor(
            MapQualityProbs {
                p_low,
                p_high,
                pb: tier_weights[2] / high_total,
                pp: tier_weights[3] / high_total,
                pg: tier_weights[4] / high_total,
                pr: tier_weights[5] / high_total,
                source: "tier".to_string(),
            },
            None,
        );
        if self.drop_graph.is_empty() || nest_id.is_none() {
            self.map_prob_cache.insert(key, fallback.clone());
            return fallback;
        }
        let resolved = self.resolve_drop_to_items(nest_id);
        let Some(resolved) = resolved else {
            self.map_prob_cache.insert(key, fallback.clone());
            return fallback;
        };
        let mut by_quality = HashMap::from([(3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)]);
        let mut mass = 0.0;
        for (item_id, weight) in resolved {
            if let Some(item) = self.items_by_id.get(&item_id)
                && (3..=6).contains(&item.quality)
            {
                *by_quality.entry(item.quality).or_insert(0.0) += weight;
                mass += weight;
            }
        }
        if mass <= 0.0 {
            self.map_prob_cache.insert(key, fallback.clone());
            return fallback;
        }
        let available = [
            by_quality[&3] > 0.0,
            by_quality[&4] > 0.0,
            by_quality[&5] > 0.0,
            by_quality[&6] > 0.0,
        ];
        let raw = apply_prob_floor_with_availability(
            MapQualityProbs {
                p_low,
                p_high,
                pb: by_quality[&3] / mass,
                pp: by_quality[&4] / mass,
                pg: by_quality[&5] / mass,
                pr: by_quality[&6] / mass,
                source: "map".to_string(),
            },
            Some(&fallback),
            available,
        );
        self.map_prob_cache.insert(key, raw.clone());
        raw
    }

    fn get_map_grid_stats_by_quality(
        &mut self,
        nest_id: Option<&str>,
        quality: i32,
        fallback_mean: f64,
        fallback_variance: f64,
    ) -> QualityGridStats {
        let key = format!(
            "{}|{}|{:.4}|{:.4}",
            nest_id.unwrap_or("*"),
            quality,
            fallback_mean,
            fallback_variance
        );
        if let Some(value) = self.map_grid_stats_cache.get(&key) {
            return *value;
        }
        let max_quality_grid = max_grid_for_quality(Some(quality)) as f64;
        let default = QualityGridStats {
            mean: fallback_mean.clamp(1.0, max_quality_grid),
            variance: fallback_variance.max(0.25),
            count: 0,
            effective_count: 0.0,
        };
        let items: Vec<ItemRecord> = self
            .items_by_quality
            .get(&quality)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|i| {
                i.grid_trusted
                    && i.grid_size > 0
                    && i.grid_size <= max_grid_for_quality(Some(quality))
            })
            .collect();
        if items.is_empty() {
            self.map_grid_stats_cache.insert(key, default);
            return default;
        }
        let resolved = if self.drop_graph_loaded() {
            self.resolve_drop_to_items(nest_id)
        } else {
            None
        };
        let mut pairs = Vec::new();
        let mut weight_sum = 0.0;
        let mut sum_sq = 0.0;
        for item in &items {
            let w = match resolved.as_ref() {
                Some(r) => r.get(&item.item_id).copied().unwrap_or(0.0),
                None => 1.0,
            };
            if w.is_finite() && w > 0.0 {
                pairs.push((item.grid_size as f64, w));
                weight_sum += w;
                sum_sq += w * w;
            }
        }
        if weight_sum <= 0.0 {
            if resolved.is_some() {
                self.map_grid_stats_cache.insert(key, default);
                return default;
            }
            pairs = items.iter().map(|i| (i.grid_size as f64, 1.0)).collect();
            weight_sum = pairs.len() as f64;
            sum_sq = pairs.len() as f64;
        }
        let mean = pairs
            .iter()
            .map(|(grid, w)| grid * (w / weight_sum))
            .sum::<f64>();
        let mut variance = pairs
            .iter()
            .map(|(grid, w)| (grid - mean).powi(2) * (w / weight_sum))
            .sum::<f64>();
        let effective = if sum_sq > 0.0 {
            weight_sum * weight_sum / sum_sq
        } else {
            pairs.len() as f64
        };
        let shrink = ((effective - 1.0) / 5.0).clamp(0.0, 1.0);
        variance = variance * shrink + default.variance * (1.0 - shrink);
        let result = QualityGridStats {
            mean: mean.clamp(1.0, max_quality_grid),
            variance: variance.clamp(0.25, 36.0),
            count: pairs.len(),
            effective_count: effective,
        };
        self.map_grid_stats_cache.insert(key, result);
        result
    }

    pub fn get_map_grid_sizes_by_quality(
        &mut self,
        nest_id: Option<&str>,
        quality: i32,
    ) -> Vec<i32> {
        let items = self
            .items_by_quality
            .get(&quality)
            .cloned()
            .unwrap_or_default();
        let resolved = if self.drop_graph_loaded() {
            self.resolve_drop_to_items(nest_id)
        } else {
            None
        };
        let mut sizes = items
            .into_iter()
            .filter(|item| {
                item.grid_trusted
                    && item.grid_size > 0
                    && item.grid_size <= max_grid_for_quality(Some(quality))
                    && resolved
                        .as_ref()
                        .map(|resolved| item_map_weight(Some(resolved), item) > 0.0)
                        .unwrap_or(true)
            })
            .map(|item| item.grid_size)
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        sizes.dedup();
        if sizes.is_empty() {
            if resolved.is_some() {
                vec![]
            } else {
                grid_sizes_for_quality(Some(quality)).to_vec()
            }
        } else {
            sizes
        }
    }

    fn get_quality_stats(&mut self, quality: i32, drop_id: Option<&str>) -> QualityPriceStats {
        let key = format!("{}|{}", drop_id.unwrap_or("*"), quality);
        if let Some(value) = self.quality_stats_cache.get(&key) {
            return *value;
        }
        let items = self
            .items_by_quality
            .get(&quality)
            .cloned()
            .unwrap_or_default();
        if items.is_empty() {
            self.quality_stats_cache
                .insert(key, QualityPriceStats::default());
            return QualityPriceStats::default();
        }
        let resolved = if self.drop_graph_loaded() {
            self.resolve_drop_to_items(drop_id)
        } else {
            None
        };
        let mut values = Vec::new();
        let mut weight_sum = 0.0;
        for item in &items {
            let mut w = match resolved.as_ref() {
                Some(r) => r.get(&item.item_id).copied().unwrap_or(0.0),
                None => 1.0,
            };
            if !w.is_finite() || w < 0.0 {
                w = 0.0;
            }
            values.push((item.value, w));
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            if resolved.is_some() {
                self.quality_stats_cache
                    .insert(key, QualityPriceStats::default());
                return QualityPriceStats::default();
            }
            values = values.into_iter().map(|(value, _)| (value, 1.0)).collect();
            weight_sum = values.len() as f64;
        }
        let mean = values
            .iter()
            .map(|(value, w)| value * (w / weight_sum))
            .sum::<f64>();
        let variance = values
            .iter()
            .map(|(value, w)| (value - mean).powi(2) * (w / weight_sum))
            .sum::<f64>();
        let result = QualityPriceStats {
            mean,
            variance,
            p50: weighted_quantile(&values, 0.5),
            count: values.len(),
        };
        self.quality_stats_cache.insert(key, result);
        result
    }

    fn get_grid_conditioned_value(
        &mut self,
        quality: i32,
        count: i32,
        target_grid: Option<i32>,
        drop_id: Option<&str>,
    ) -> Option<f64> {
        let target_grid = target_grid?;
        if count <= 0
            || target_grid < count
            || target_grid > max_grid_for_quality(Some(quality)) * count
        {
            return None;
        }
        let key = format!(
            "{}|{}|{}|{}",
            drop_id.unwrap_or("*"),
            quality,
            count,
            target_grid
        );
        if let Some(value) = self.grid_value_cache.get(&key) {
            return *value;
        }
        let dist = self.get_grid_value_distribution(quality, count, drop_id);
        let Some(dist) = dist else {
            self.grid_value_cache.insert(key, None);
            return None;
        };
        let tg = target_grid as usize;
        if tg > dist.max_grid {
            self.grid_value_cache.insert(key, None);
            return None;
        }
        let w = dist.weights[tg];
        if w > 1e-12 {
            let base_value = dist.value_sums[tg] / w;
            let value = self.apply_grid_impact_calibration(
                quality,
                count,
                target_grid,
                dist.mean_grid,
                base_value,
                dist.neutral_value,
            );
            self.grid_value_cache.insert(key, Some(value));
            return Some(value);
        }
        let sigma = 0.75_f64.max((count as f64).sqrt() * 0.45);
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        for grid in count as usize..=dist.max_grid {
            if dist.weights[grid] > 0.0 {
                let delta = grid as f64 - target_grid as f64;
                let kernel = (-0.5 * delta * delta / (sigma * sigma)).exp();
                weight_sum += dist.weights[grid] * kernel;
                value_sum += dist.value_sums[grid] * kernel;
            }
        }
        let value = if weight_sum > 0.0 {
            Some(self.apply_grid_impact_calibration(
                quality,
                count,
                target_grid,
                dist.mean_grid,
                value_sum / weight_sum,
                dist.neutral_value,
            ))
        } else {
            None
        };
        self.grid_value_cache.insert(key, value);
        value
    }

    fn get_grid_value_distribution(
        &mut self,
        quality: i32,
        count: i32,
        drop_id: Option<&str>,
    ) -> Option<GridValueDistribution> {
        let key = format!("{}|{}|{}", drop_id.unwrap_or("*"), quality, count);
        if let Some(value) = self.grid_dist_cache.get(&key) {
            return value.clone();
        }
        let items: Vec<ItemRecord> = self
            .items_by_quality
            .get(&quality)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|i| {
                i.grid_trusted
                    && i.grid_size > 0
                    && i.grid_size <= max_grid_for_quality(Some(quality))
                    && i.value > 0.0
            })
            .collect();
        if items.is_empty() {
            self.grid_dist_cache.insert(key, None);
            return None;
        }
        let resolved = if self.drop_graph_loaded() {
            self.resolve_drop_to_items(drop_id)
        } else {
            None
        };
        let mut pool = Vec::new();
        let mut total_w = 0.0;
        for item in &items {
            let mut w = match resolved.as_ref() {
                Some(r) => r.get(&item.item_id).copied().unwrap_or(0.0),
                None => 1.0,
            };
            if !w.is_finite() || w < 0.0 {
                w = 0.0;
            }
            if w > 0.0 {
                pool.push((item.grid_size as usize, item.value, w));
                total_w += w;
            }
        }
        if total_w <= 0.0 {
            if resolved.is_some() {
                self.grid_dist_cache.insert(key, None);
                return None;
            }
            pool = items
                .iter()
                .map(|i| (i.grid_size as usize, i.value, 1.0))
                .collect();
        }
        let Some(max_item_grid) = pool.iter().map(|(grid, _, _)| *grid).max() else {
            self.grid_dist_cache.insert(key, None);
            return None;
        };
        let max_grid = max_item_grid * count as usize;
        let mut weights = vec![0.0; max_grid + 1];
        let mut value_sums = vec![0.0; max_grid + 1];
        weights[0] = 1.0;
        for _ in 0..count {
            let mut next_weights = vec![0.0; max_grid + 1];
            let mut next_values = vec![0.0; max_grid + 1];
            for grid in 0..weights.len() {
                let prev_w = weights[grid];
                if prev_w <= 0.0 {
                    continue;
                }
                let prev_value_sum = value_sums[grid];
                for (item_grid, item_value, item_w) in &pool {
                    let new_grid = grid + item_grid;
                    if new_grid <= max_grid {
                        next_weights[new_grid] += prev_w * item_w;
                        next_values[new_grid] +=
                            item_w * prev_value_sum + item_w * item_value * prev_w;
                    }
                }
            }
            let norm: f64 = next_weights.iter().sum();
            if norm <= 0.0 {
                self.grid_dist_cache.insert(key, None);
                return None;
            }
            weights = next_weights.into_iter().map(|x| x / norm).collect();
            value_sums = next_values.into_iter().map(|x| x / norm).collect();
        }
        let mean_grid = weighted_mean_grid(&weights);
        let result = GridValueDistribution {
            weights,
            value_sums,
            max_grid,
            mean_grid,
            neutral_value: count as f64 * self.quality_mean_fallback(quality, drop_id),
        };
        self.grid_dist_cache.insert(key, Some(result.clone()));
        Some(result)
    }

    fn quality_mean_fallback(&mut self, quality: i32, drop_id: Option<&str>) -> f64 {
        let stats = self.get_quality_stats(quality, drop_id);
        if stats.count > 0 && stats.mean > 0.0 {
            stats.mean
        } else {
            0.0
        }
    }

    fn apply_grid_impact_calibration(
        &self,
        quality: i32,
        count: i32,
        target_grid: i32,
        mean_grid: f64,
        base_value: f64,
        neutral_value: f64,
    ) -> f64 {
        let impact = self
            .grid_impact_by_quality
            .get(&quality)
            .copied()
            .unwrap_or(0.0);
        if impact <= 0.0 || count <= 0 || base_value <= 0.0 {
            return base_value;
        }
        let delta = target_grid as f64 - mean_grid;
        let cap = (base_value * 0.35).max(impact * count.max(1) as f64);
        let adjustment = (delta * impact).clamp(-cap, cap);
        let mut value = (base_value + adjustment).max(1.0);
        if delta > 0.0 {
            value = value.max(neutral_value + delta * impact * 0.65);
        }
        value
    }
}

pub struct BidKingCore {
    pub loader: DataLoader,
    pub static_data: StaticData,
    pub raw_results: Vec<ComboResult>,
    price_range_results: Vec<ComboResult>,
    high_value_cache: HashMap<String, f64>,
    jackpot_cache: HashMap<String, Option<Vec<PricePoint>>>,
}

impl BidKingCore {
    pub fn new(loader: DataLoader, static_data: StaticData) -> Self {
        Self {
            loader,
            static_data,
            raw_results: Vec::new(),
            price_range_results: Vec::new(),
            high_value_cache: HashMap::new(),
            jackpot_cache: HashMap::new(),
        }
    }

    pub fn run(&mut self, cp: CalcParams) -> Result<Vec<ComboResult>> {
        self.high_value_cache.clear();
        self.raw_results.clear();
        self.price_range_results.clear();
        let cp = normalize_calc_params(cp);
        validate_calc_params(&cp)?;
        let mut results = Vec::new();
        let tier_weights = self.tier_weights(&cp.tier)?;
        let probs = self
            .loader
            .get_map_quality_probs(Some(&cp.map_nest_id), &tier_weights);
        let grid_stats = self.build_grid_stats_model(Some(&cp.map_nest_id));
        let price_model = self.build_price_model(&cp, &tier_weights);
        let blue_grid_sizes = self
            .loader
            .get_map_grid_sizes_by_quality(Some(&cp.map_nest_id), 3);
        let purple_grid_sizes = self
            .loader
            .get_map_grid_sizes_by_quality(Some(&cp.map_nest_id), 4);
        let gold_grid_sizes = self
            .loader
            .get_map_grid_sizes_by_quality(Some(&cp.map_nest_id), 5);
        let red_grid_sizes = self
            .loader
            .get_map_grid_sizes_by_quality(Some(&cp.map_nest_id), 6);
        let gw_grid_value = round_opt(cp.gw_grid);
        let blue_grid_value = round_opt(cp.blue_grid);
        let purple_grid_value = round_opt(cp.purple_grid);
        let gold_grid_value = round_opt(cp.gold_grid);
        let red_grid_value = round_opt(cp.red_grid);

        if high_quality_only_mode(&cp) {
            return self.run_high_quality_only(
                &cp,
                &probs,
                grid_stats,
                price_model,
                HighQualityGridSizes {
                    purple: &purple_grid_sizes,
                    gold: &gold_grid_sizes,
                    red: &red_grid_sizes,
                },
            );
        }

        let valid_gw = build_valid_color_counts(
            DEFAULT_ITEM_GRID_SIZES,
            cp.total_count,
            cp.gw_count,
            cp.min_gw,
            cp.gw_grid,
            cp.gw_avg,
        );
        let valid_b = build_valid_color_counts(
            &blue_grid_sizes,
            cp.total_count,
            cp.blue_count,
            cp.min_blue,
            cp.blue_grid,
            cp.blue_avg,
        );
        let valid_p = build_valid_color_counts(
            &purple_grid_sizes,
            cp.total_count,
            cp.purple_count,
            cp.min_purple,
            cp.purple_grid,
            cp.purple_avg,
        );
        let valid_g = build_valid_color_counts(
            &gold_grid_sizes,
            cp.total_count,
            cp.gold_count,
            cp.min_gold,
            cp.gold_grid,
            cp.gold_avg,
        );
        let valid_r = build_valid_color_counts(
            &red_grid_sizes,
            cp.total_count,
            cp.red_count,
            cp.min_red,
            cp.red_grid,
            cp.red_avg,
        );

        let req_b = required_count(cp.blue_count, cp.min_blue);
        let req_p = required_count(cp.purple_count, cp.min_purple);
        let req_g = required_count(cp.gold_count, cp.min_gold);
        let req_r = required_count(cp.red_count, cp.min_red);
        let log_pb = probs.pb.ln();
        let log_pp = probs.pp.ln();
        let log_pg = probs.pg.ln();
        let log_pr = probs.pr.ln();

        for gw in inclusive_range(count_range(
            cp.total_count,
            cp.gw_count,
            cp.min_gw,
            req_b + req_p + req_g + req_r,
        )) {
            if !valid_at(&valid_gw, gw) {
                continue;
            }
            let high_count = cp.total_count - gw;
            let log_w_gw = if cp.gw_count.is_some() {
                0.0
            } else {
                log_binom_p(cp.total_count, gw, probs.p_low)
            };
            let b_range = if let Some(high_quality_count) = valid_high_quality_count(&cp) {
                let b = high_count - high_quality_count;
                if b < 0 || b > high_count {
                    (1, 0)
                } else {
                    (b, b)
                }
            } else {
                count_range(
                    high_count,
                    cp.blue_count,
                    cp.min_blue,
                    req_p + req_g + req_r,
                )
            };
            for b in inclusive_range(b_range) {
                if !valid_at(&valid_b, b) {
                    continue;
                }
                for p in inclusive_range(count_range(
                    high_count - b,
                    cp.purple_count,
                    cp.min_purple,
                    req_g + req_r,
                )) {
                    if !valid_at(&valid_p, p) {
                        continue;
                    }
                    for g in inclusive_range(count_range(
                        high_count - b - p,
                        cp.gold_count,
                        cp.min_gold,
                        req_r,
                    )) {
                        if !valid_at(&valid_g, g) {
                            continue;
                        }
                        let r = high_count - b - p - g;
                        if r < 0 || r > cp.total_count || !valid_at(&valid_r, r) {
                            continue;
                        }
                        let mut log_w = log_w_gw + log_fact(high_count)
                            - log_fact(b)
                            - log_fact(p)
                            - log_fact(g)
                            - log_fact(r);
                        if b > 0 {
                            log_w += b as f64 * log_pb;
                        }
                        if p > 0 {
                            log_w += p as f64 * log_pp;
                        }
                        if g > 0 {
                            log_w += g as f64 * log_pg;
                        }
                        if r > 0 {
                            log_w += r as f64 * log_pr;
                        }
                        log_w +=
                            red_count_caution_log(&cp, r, cp.total_count, probs.p_high * probs.pr);
                        let grid_gw = cp
                            .gw_grid
                            .unwrap_or(gw as f64 * cp.gw_avg.unwrap_or(grid_stats.greenwhite_mean));
                        let grid_b = cp
                            .blue_grid
                            .unwrap_or(b as f64 * cp.blue_avg.unwrap_or(grid_stats.blue_mean));
                        let grid_p = cp
                            .purple_grid
                            .unwrap_or(p as f64 * cp.purple_avg.unwrap_or(grid_stats.purple_mean));
                        let grid_g = cp
                            .gold_grid
                            .unwrap_or(g as f64 * cp.gold_avg.unwrap_or(grid_stats.gold_mean));
                        let grid_r = cp
                            .red_grid
                            .unwrap_or(r as f64 * cp.red_avg.unwrap_or(grid_stats.red_mean));
                        let total_grid = grid_gw + grid_b + grid_p + grid_g + grid_r;
                        if let Some(target) = cp.total_grid_target {
                            let sigma = sigma_from_unknowns(&cp, &grid_stats, b, p, g, r, gw);
                            log_w += total_grid_prior_log(total_grid - target, sigma);
                        }
                        log_w += avg_prior_log(b, grid_b, cp.blue_avg, 0.35);
                        log_w += avg_prior_log(p, grid_p, cp.purple_avg, 0.45);
                        log_w += avg_prior_log(g, grid_g, cp.gold_avg, 0.55);
                        log_w += avg_prior_log(r, grid_r, cp.red_avg, 0.7);
                        if cp.gw_grid.is_none() {
                            log_w += avg_prior_log(gw, grid_gw, cp.gw_avg, 0.35);
                        }
                        log_w += value_sample_prior_log(
                            &cp,
                            cp.total_count,
                            gw,
                            b,
                            p,
                            g,
                            r,
                            grid_gw,
                            grid_b,
                            grid_p,
                            grid_g,
                            grid_r,
                            &price_model,
                        );
                        results.push(ComboResult {
                            greenwhite_count: gw,
                            blue_count: b,
                            purple_count: p,
                            gold_count: g,
                            red_count: r,
                            log_w,
                            total_grid_est: total_grid,
                            greenwhite_grid_est: grid_gw,
                            blue_grid_est: grid_b,
                            purple_grid_est: grid_p,
                            gold_grid_est: grid_g,
                            red_grid_est: grid_r,
                            greenwhite_grid_value: gw_grid_value,
                            blue_grid_value,
                            purple_grid_value,
                            gold_grid_value,
                            red_grid_value,
                            ..Default::default()
                        });
                    }
                }
            }
        }
        ensure_candidate_combos(&results)?;
        if valid_min_value_floor(&cp).is_some() {
            self.populate_combo_values(&mut results, &cp, price_model, grid_stats, true, true);
            let results = filter_combos_below_min_value_floor(results, &cp)?;
            let mut filtered = self.finalize_combos(results, true)?;
            self.populate_combo_values(&mut filtered, &cp, price_model, grid_stats, true, true);
            let mut price_range_results = std::mem::take(&mut self.price_range_results);
            self.populate_combo_values(
                &mut price_range_results,
                &cp,
                price_model,
                grid_stats,
                true,
                true,
            );
            self.price_range_results = price_range_results;
            Ok(filtered)
        } else {
            let mut filtered = self.finalize_combos(results, false)?;
            self.populate_combo_values(&mut filtered, &cp, price_model, grid_stats, true, true);
            let mut price_range_results = std::mem::take(&mut self.price_range_results);
            self.populate_combo_values(
                &mut price_range_results,
                &cp,
                price_model,
                grid_stats,
                true,
                true,
            );
            self.price_range_results = price_range_results;
            Ok(filtered)
        }
    }

    fn run_high_quality_only(
        &mut self,
        cp: &CalcParams,
        probs: &MapQualityProbs,
        grid_stats: GridStatsModel,
        price_model: PriceModel,
        grid_sizes: HighQualityGridSizes<'_>,
    ) -> Result<Vec<ComboResult>> {
        let high_count = cp.high_quality_count.unwrap_or_default();
        let purple_grid_value = round_opt(cp.purple_grid);
        let gold_grid_value = round_opt(cp.gold_grid);
        let red_grid_value = round_opt(cp.red_grid);
        let valid_p = build_valid_color_counts(
            grid_sizes.purple,
            high_count,
            cp.purple_count,
            cp.min_purple,
            cp.purple_grid,
            cp.purple_avg,
        );
        let valid_g = build_valid_color_counts(
            grid_sizes.gold,
            high_count,
            cp.gold_count,
            cp.min_gold,
            cp.gold_grid,
            cp.gold_avg,
        );
        let valid_r = build_valid_color_counts(
            grid_sizes.red,
            high_count,
            cp.red_count,
            cp.min_red,
            cp.red_grid,
            cp.red_avg,
        );
        let req_g = required_count(cp.gold_count, cp.min_gold);
        let req_r = required_count(cp.red_count, cp.min_red);
        let log_pp = probs.pp.ln();
        let log_pg = probs.pg.ln();
        let log_pr = probs.pr.ln();
        let mut results = Vec::new();

        for p in inclusive_range(count_range(
            high_count,
            cp.purple_count,
            cp.min_purple,
            req_g + req_r,
        )) {
            if !valid_at(&valid_p, p) {
                continue;
            }
            for g in inclusive_range(count_range(
                high_count - p,
                cp.gold_count,
                cp.min_gold,
                req_r,
            )) {
                if !valid_at(&valid_g, g) {
                    continue;
                }
                let r = high_count - p - g;
                if r < 0 || r > high_count || !valid_at(&valid_r, r) {
                    continue;
                }
                let mut log_w = log_fact(high_count) - log_fact(p) - log_fact(g) - log_fact(r);
                if p > 0 {
                    log_w += p as f64 * log_pp;
                }
                if g > 0 {
                    log_w += g as f64 * log_pg;
                }
                if r > 0 {
                    log_w += r as f64 * log_pr;
                }
                log_w += red_count_caution_log(cp, r, high_count, probs.pr);
                let grid_p = cp
                    .purple_grid
                    .unwrap_or(p as f64 * cp.purple_avg.unwrap_or(grid_stats.purple_mean));
                let grid_g = cp
                    .gold_grid
                    .unwrap_or(g as f64 * cp.gold_avg.unwrap_or(grid_stats.gold_mean));
                let grid_r = cp
                    .red_grid
                    .unwrap_or(r as f64 * cp.red_avg.unwrap_or(grid_stats.red_mean));
                let total_grid = grid_p + grid_g + grid_r;
                log_w += avg_prior_log(p, grid_p, cp.purple_avg, 0.45);
                log_w += avg_prior_log(g, grid_g, cp.gold_avg, 0.55);
                log_w += avg_prior_log(r, grid_r, cp.red_avg, 0.7);
                results.push(ComboResult {
                    greenwhite_count: 0,
                    blue_count: 0,
                    purple_count: p,
                    gold_count: g,
                    red_count: r,
                    log_w,
                    total_grid_est: total_grid,
                    greenwhite_grid_est: 0.0,
                    blue_grid_est: 0.0,
                    purple_grid_est: grid_p,
                    gold_grid_est: grid_g,
                    red_grid_est: grid_r,
                    purple_grid_value,
                    gold_grid_value,
                    red_grid_value,
                    ..Default::default()
                });
            }
        }

        ensure_candidate_combos(&results)?;
        if valid_min_value_floor(cp).is_some() {
            self.populate_combo_values(&mut results, cp, price_model, grid_stats, false, false);
            let results = filter_combos_below_min_value_floor(results, cp)?;
            let mut filtered = self.finalize_combos(results, true)?;
            self.populate_combo_values(&mut filtered, cp, price_model, grid_stats, false, false);
            let mut price_range_results = std::mem::take(&mut self.price_range_results);
            self.populate_combo_values(
                &mut price_range_results,
                cp,
                price_model,
                grid_stats,
                false,
                false,
            );
            self.price_range_results = price_range_results;
            Ok(filtered)
        } else {
            let mut filtered = self.finalize_combos(results, false)?;
            self.populate_combo_values(&mut filtered, cp, price_model, grid_stats, false, false);
            let mut price_range_results = std::mem::take(&mut self.price_range_results);
            self.populate_combo_values(
                &mut price_range_results,
                cp,
                price_model,
                grid_stats,
                false,
                false,
            );
            self.price_range_results = price_range_results;
            Ok(filtered)
        }
    }

    fn finalize_combos(
        &mut self,
        mut combos: Vec<ComboResult>,
        keep_all_for_price_range: bool,
    ) -> Result<Vec<ComboResult>> {
        self.price_range_results.clear();
        combos.retain(|combo| combo.log_w.is_finite());
        if combos.is_empty() {
            self.raw_results.clear();
            anyhow::bail!("没有找到概率有效的组合，请检查件数、格数、品质概率或 OCR 识别结果");
        }
        let max_l = combos
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = combos.iter().map(|c| (c.log_w - max_l).exp()).sum();
        if !denom.is_finite() || denom <= 0.0 {
            self.raw_results.clear();
            anyhow::bail!("组合概率归一化失败，请检查件数、格数、品质概率或 OCR 识别结果");
        }
        for combo in &mut combos {
            combo.probability = (combo.log_w - max_l).exp() / denom;
        }
        self.raw_results = combos.clone();
        self.raw_results.sort_by(combo_probability_order);
        self.price_range_results = price_range_source(&self.raw_results, keep_all_for_price_range);
        let mut filtered: Vec<ComboResult> = combos
            .into_iter()
            .filter(|c| c.probability >= PROB_CUTOFF)
            .collect();
        if filtered.is_empty() {
            filtered = vec![self.raw_results[0].clone()];
        }
        let prob_sum: f64 = filtered.iter().map(|c| c.probability).sum();
        if prob_sum > 0.0 {
            for combo in &mut filtered {
                combo.probability /= prob_sum;
            }
        }
        filtered.sort_by(combo_probability_order);
        Ok(filtered)
    }

    fn populate_combo_values(
        &mut self,
        combos: &mut [ComboResult],
        cp: &CalcParams,
        price: PriceModel,
        grid: GridStatsModel,
        include_low_tiers: bool,
        use_total_grid_target: bool,
    ) {
        let jackpot_indices = top_probability_indices(combos, JACKPOT_TOP_ENUM_COMBOS);
        for (index, combo) in combos.iter_mut().enumerate() {
            let gw = if include_low_tiers {
                combo.greenwhite_count
            } else {
                0
            };
            let b = if include_low_tiers {
                combo.blue_count
            } else {
                0
            };
            let p = combo.purple_count;
            let g = combo.gold_count;
            let r = combo.red_count;
            let grid_gw = if include_low_tiers {
                cp.gw_grid
                    .unwrap_or(gw as f64 * cp.gw_avg.unwrap_or(grid.greenwhite_mean))
            } else {
                0.0
            };
            let grid_b = if include_low_tiers {
                cp.blue_grid
                    .unwrap_or(b as f64 * cp.blue_avg.unwrap_or(grid.blue_mean))
            } else {
                0.0
            };
            let grid_p = cp
                .purple_grid
                .unwrap_or(p as f64 * cp.purple_avg.unwrap_or(grid.purple_mean));
            let grid_g = cp
                .gold_grid
                .unwrap_or(g as f64 * cp.gold_avg.unwrap_or(grid.gold_mean));
            let grid_r = cp
                .red_grid
                .unwrap_or(r as f64 * cp.red_avg.unwrap_or(grid.red_mean));
            let targets = build_pricing_grid_targets(
                cp,
                gw,
                b,
                p,
                g,
                r,
                grid_gw,
                grid_b,
                grid_p,
                grid_g,
                grid_r,
                use_total_grid_target,
            );
            let mut value = 0.0;
            if include_low_tiers {
                value += self.quality_value(
                    cp,
                    3,
                    b,
                    price.blue_mean,
                    grid_b,
                    targets.blue,
                    Some(&cp.map_nest_id),
                );
            }
            value += self.quality_value(
                cp,
                4,
                p,
                price.purple_mean,
                grid_p,
                targets.purple,
                Some(&cp.map_nest_id),
            );
            value += self.quality_value(
                cp,
                5,
                g,
                price.gold_mean,
                grid_g,
                targets.gold,
                Some(&cp.map_nest_id),
            );
            let red_value = self.quality_value(
                cp,
                6,
                r,
                price.red_mean,
                grid_r,
                targets.red,
                Some(&cp.map_nest_id),
            );
            value += red_value;
            if include_low_tiers {
                value += gw as f64 * price.greenwhite_mean;
            }
            combo.final_value = value;
            combo.high_value_price_points = if jackpot_indices.contains(&index) {
                self.combo_high_value_price_points(value, red_value, r, Some(&cp.map_nest_id))
            } else {
                Vec::new()
            };
            let purple_variance =
                quality_combo_variance(cp, 4, p, grid_p, targets.purple, price.purple_variance);
            let gold_variance =
                quality_combo_variance(cp, 5, g, grid_g, targets.gold, price.gold_variance);
            combo.high_variance = if include_low_tiers {
                b as f64 * price.blue_variance
            } else {
                0.0
            } + purple_variance
                + gold_variance
                + r as f64 * price.red_variance;
            combo.greenwhite_grid_est = grid_gw;
            combo.blue_grid_est = grid_b;
            combo.purple_grid_est = grid_p;
            combo.gold_grid_est = grid_g;
            combo.red_grid_est = grid_r;
            combo.total_grid_est = grid_gw + grid_b + grid_p + grid_g + grid_r;
        }
    }

    fn combo_high_value_price_points(
        &mut self,
        combo_value: f64,
        red_value: f64,
        red_count: i32,
        drop_id: Option<&str>,
    ) -> Vec<PricePoint> {
        if red_count <= 0 || !combo_value.is_finite() || !red_value.is_finite() {
            return Vec::new();
        }
        let Some(red_points) = self.red_high_value_distribution(red_count, drop_id) else {
            return Vec::new();
        };
        if red_points.len() <= 1 {
            return Vec::new();
        }
        let expected_red = red_points
            .iter()
            .map(|point| point.value * point.probability)
            .sum::<f64>();
        if !expected_red.is_finite() {
            return Vec::new();
        }
        let base_without_red = combo_value - red_value;
        let shift = red_value - expected_red;
        let mut points = red_points
            .into_iter()
            .map(|point| PricePoint {
                value: base_without_red + point.value + shift,
                probability: point.probability,
            })
            .collect::<Vec<_>>();
        normalize_price_points(&mut points);
        if price_point_spread(&points) >= JACKPOT_VALUE_BIN {
            points
        } else {
            Vec::new()
        }
    }

    fn red_high_value_distribution(
        &mut self,
        count: i32,
        drop_id: Option<&str>,
    ) -> Option<Vec<PricePoint>> {
        if count <= 0 || count > JACKPOT_MAX_RED_COUNT {
            return None;
        }
        let key = format!(
            "{}|{}|{:.0}",
            drop_id.unwrap_or("*"),
            count,
            JACKPOT_VALUE_THRESHOLD
        );
        if let Some(value) = self.jackpot_cache.get(&key) {
            return value.clone();
        }

        let mut categories = self.red_high_value_categories(drop_id);
        normalize_price_points(&mut categories);
        if categories
            .iter()
            .all(|point| point.value < JACKPOT_VALUE_THRESHOLD)
        {
            self.jackpot_cache.insert(key, None);
            return None;
        }

        let mut dist = vec![PricePoint {
            value: 0.0,
            probability: 1.0,
        }];
        for _ in 0..count {
            let mut next = Vec::with_capacity(
                dist.len()
                    .saturating_mul(categories.len())
                    .min(JACKPOT_MAX_PRICE_POINTS * categories.len().max(1)),
            );
            for prev in &dist {
                for item in &categories {
                    next.push(PricePoint {
                        value: prev.value + item.value,
                        probability: prev.probability * item.probability,
                    });
                }
            }
            dist = compress_price_points(next);
            if dist.is_empty() {
                self.jackpot_cache.insert(key, None);
                return None;
            }
        }
        let value = if price_point_spread(&dist) >= JACKPOT_VALUE_BIN {
            Some(dist)
        } else {
            None
        };
        self.jackpot_cache.insert(key, value.clone());
        value
    }

    fn red_high_value_categories(&mut self, drop_id: Option<&str>) -> Vec<PricePoint> {
        let items = self
            .loader
            .items_by_quality
            .get(&6)
            .cloned()
            .unwrap_or_default();
        if items.is_empty() {
            return Vec::new();
        }
        let resolved = if self.loader.drop_graph_loaded() {
            self.loader.resolve_drop_to_items(drop_id)
        } else {
            None
        };
        let mut high = Vec::new();
        let mut ordinary_weight = 0.0;
        let mut ordinary_value_sum = 0.0;
        let mut total_weight = 0.0;
        for item in &items {
            let w = item_map_weight(resolved.as_ref(), item);
            if !w.is_finite() || w <= 0.0 || !item.value.is_finite() || item.value <= 0.0 {
                continue;
            }
            total_weight += w;
            if item.value >= JACKPOT_VALUE_THRESHOLD {
                high.push(PricePoint {
                    value: item.value,
                    probability: w,
                });
            } else {
                ordinary_weight += w;
                ordinary_value_sum += item.value * w;
            }
        }
        if total_weight <= 0.0 || high.is_empty() {
            return Vec::new();
        }
        let mut categories = high;
        if ordinary_weight > 0.0 {
            categories.push(PricePoint {
                value: ordinary_value_sum / ordinary_weight,
                probability: ordinary_weight,
            });
        }
        categories
    }

    #[allow(clippy::too_many_arguments)]
    fn quality_value(
        &mut self,
        cp: &CalcParams,
        quality: i32,
        count: i32,
        price_mean: f64,
        grid_est: f64,
        target_grid: Option<i32>,
        drop_id: Option<&str>,
    ) -> f64 {
        let manual_grid_est = target_grid
            .map(|target_grid| target_grid as f64)
            .unwrap_or(grid_est);
        if let Some(value) = manual_quality_total_value(cp, quality, count, manual_grid_est) {
            value
        } else {
            self.high_value_for_quality(quality, count, price_mean, target_grid, drop_id)
        }
    }

    fn high_value_for_quality(
        &mut self,
        quality: i32,
        count: i32,
        price_mean: f64,
        target_grid: Option<i32>,
        drop_id: Option<&str>,
    ) -> f64 {
        if count <= 0 {
            return 0.0;
        }
        if self
            .loader
            .items_by_quality
            .get(&quality)
            .map(|v| v.is_empty())
            .unwrap_or(true)
        {
            return count as f64 * price_mean;
        }
        let key = format!(
            "{quality},{count},{}",
            target_grid.map_or("-".to_string(), |v| v.to_string())
        );
        if let Some(value) = self.high_value_cache.get(&key) {
            return *value;
        }
        let mut value = count as f64 * price_mean;
        let mut conditioned = None;
        if let Some(target_grid) = target_grid {
            conditioned =
                self.loader
                    .get_grid_conditioned_value(quality, count, Some(target_grid), drop_id);
            if let Some(v) = conditioned
                && v.is_finite()
                && v > 0.0
            {
                value = v;
            }
        }
        let composed = self.compose_items_for_quality(quality, count, value, target_grid, drop_id);
        if let Some(composed) = composed
            && composed.is_finite()
        {
            value = if let Some(conditioned) = conditioned {
                conditioned * 0.4 + composed * 0.6
            } else {
                composed
            };
        }
        self.high_value_cache.insert(key, value);
        value
    }

    fn compose_items_for_quality(
        &mut self,
        quality: i32,
        count: i32,
        target_sum: f64,
        target_grid: Option<i32>,
        drop_id: Option<&str>,
    ) -> Option<f64> {
        if count <= 0 {
            return None;
        }
        let pool = self.choose_pool(
            quality,
            target_sum / count as f64,
            drop_id,
            target_grid,
            count,
        );
        if pool.is_empty() {
            return None;
        }
        let resolved = if self.loader.drop_graph_loaded() {
            self.loader.resolve_drop_to_items(drop_id)
        } else {
            None
        };
        let mut states = vec![BeamState::default()];
        let has_target = target_grid.is_some();
        let beam = if count <= 2 {
            24
        } else if count <= 4 {
            48
        } else if count <= 8 {
            64
        } else {
            80
        };
        for _ in 0..count {
            let mut expanded = Vec::with_capacity(states.len() * pool.len());
            for state in &states {
                for item in &pool {
                    let w = item_map_weight(resolved.as_ref(), item);
                    expanded.push(BeamState {
                        sum: state.sum + item.value,
                        grid: state.grid + if item.grid_trusted { item.grid_size } else { 0 },
                        unknown_grid: state.unknown_grid + if item.grid_trusted { 0 } else { 1 },
                        log_map_weight: state.log_map_weight + w.max(1e-9).ln(),
                    });
                }
            }
            expanded.sort_by(|a, b| {
                fcmp(
                    beam_score(a, target_sum, target_grid),
                    beam_score(b, target_sum, target_grid),
                )
            });
            expanded.truncate(beam.min(expanded.len()));
            states = expanded;
        }
        if states.is_empty() {
            return None;
        }
        if has_target {
            let target_grid = target_grid.unwrap();
            let mut exact: Vec<BeamState> = states
                .iter()
                .filter(|s| s.unknown_grid == 0 && s.grid == target_grid)
                .cloned()
                .collect();
            if !exact.is_empty() {
                exact.sort_by(|a, b| {
                    fcmp(
                        beam_score(a, target_sum, Some(target_grid)),
                        beam_score(b, target_sum, Some(target_grid)),
                    )
                });
                return Some(exact[0].sum);
            }
            if states.iter().any(|s| s.unknown_grid > 0) {
                return None;
            }
            if count == 2 {
                return self.try_exhaustive_pair_exact_grid(
                    quality,
                    target_grid,
                    target_sum,
                    resolved.as_ref(),
                );
            }
            return None;
        }
        states.sort_by(|a, b| {
            fcmp(
                beam_score(a, target_sum, None),
                beam_score(b, target_sum, None),
            )
        });
        Some(states[0].sum)
    }

    fn choose_pool(
        &mut self,
        quality: i32,
        unit_target: f64,
        drop_id: Option<&str>,
        target_grid: Option<i32>,
        count: i32,
    ) -> Vec<ItemRecord> {
        let mut pool = self
            .loader
            .items_by_quality
            .get(&quality)
            .cloned()
            .unwrap_or_default();
        if pool.is_empty() {
            return vec![];
        }
        let has_target = target_grid.is_some() && count > 0;
        if has_target {
            let trusted: Vec<ItemRecord> =
                pool.iter().filter(|i| i.grid_trusted).cloned().collect();
            if !trusted.is_empty() {
                pool = trusted;
            }
        }
        let resolved = if self.loader.drop_graph_loaded() {
            self.loader.resolve_drop_to_items(drop_id)
        } else {
            None
        };
        if let Some(resolved) = resolved.as_ref() {
            let in_map: Vec<ItemRecord> = pool
                .iter()
                .filter(|item| item_map_weight(Some(resolved), item) > 0.0)
                .cloned()
                .collect();
            if !in_map.is_empty() {
                pool = in_map;
            }
        }
        let mut by_value = pool.clone();
        by_value.sort_by(|a, b| {
            fcmp(abs(a.value - unit_target), abs(b.value - unit_target)).then_with(|| {
                fcmp(
                    item_map_weight(resolved.as_ref(), b),
                    item_map_weight(resolved.as_ref(), a),
                )
            })
        });
        if !has_target {
            by_value.truncate(18);
            return by_value;
        }
        let unit_grid = target_grid.unwrap() as f64 / (count as f64).max(1.0);
        let mut by_grid = pool.clone();
        by_grid.sort_by(|a, b| {
            fcmp(
                abs(a.grid_size as f64 - unit_grid),
                abs(b.grid_size as f64 - unit_grid),
            )
            .then_with(|| fcmp(abs(a.value - unit_target), abs(b.value - unit_target)))
        });
        let mut by_weight = pool.clone();
        by_weight.sort_by(|a, b| {
            fcmp(
                item_map_weight(resolved.as_ref(), b),
                item_map_weight(resolved.as_ref(), a),
            )
        });
        let limit = pool.len().min(60);
        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        add_top(&mut selected, &mut seen, &by_grid, limit.min(30));
        add_top(&mut selected, &mut seen, &by_value, limit.min(30));
        add_top(&mut selected, &mut seen, &by_weight, limit.min(20));
        if selected.len() < pool.len().min(12) {
            add_top(&mut selected, &mut seen, &by_value, pool.len().min(12));
        }
        selected.truncate(limit);
        selected
    }

    fn try_exhaustive_pair_exact_grid(
        &self,
        quality: i32,
        target_grid: i32,
        target_sum: f64,
        resolved: Option<&HashMap<String, f64>>,
    ) -> Option<f64> {
        let items: Vec<ItemRecord> = self
            .loader
            .items_by_quality
            .get(&quality)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|i| {
                i.grid_trusted
                    && resolved
                        .map(|resolved| item_map_weight(Some(resolved), i) > 0.0)
                        .unwrap_or(true)
            })
            .collect();
        let mut best = None;
        let mut best_score = f64::INFINITY;
        let mut best_weight = f64::NEG_INFINITY;
        for a in &items {
            for b in &items {
                if a.grid_size + b.grid_size != target_grid {
                    continue;
                }
                let total = a.value + b.value;
                let log_weight = item_map_weight(resolved, a).max(1e-9).ln()
                    + item_map_weight(resolved, b).max(1e-9).ln();
                let score = abs(total - target_sum) / target_sum.max(1.0) - 0.15 * log_weight;
                if best.is_none()
                    || score < best_score - 1e-12
                    || ((score - best_score).abs() <= 1e-12 && log_weight > best_weight + 1e-9)
                {
                    best = Some(total);
                    best_score = score;
                    best_weight = log_weight;
                }
            }
        }
        best
    }

    fn build_grid_stats_model(&mut self, nest_id: Option<&str>) -> GridStatsModel {
        let blue = self
            .loader
            .get_map_grid_stats_by_quality(nest_id, 3, fallback_avg_grid(3), 3.0);
        let purple =
            self.loader
                .get_map_grid_stats_by_quality(nest_id, 4, fallback_avg_grid(4), 5.0);
        let gold = self
            .loader
            .get_map_grid_stats_by_quality(nest_id, 5, fallback_avg_grid(5), 9.0);
        let red = self
            .loader
            .get_map_grid_stats_by_quality(nest_id, 6, fallback_avg_grid(6), 12.0);
        GridStatsModel {
            greenwhite_mean: 2.2,
            blue_mean: blue.mean,
            purple_mean: purple.mean,
            gold_mean: gold.mean,
            red_mean: red.mean,
            greenwhite_variance: 2.3,
            blue_variance: blue.variance,
            purple_variance: purple.variance,
            gold_variance: gold.variance,
            red_variance: red.variance,
        }
    }

    fn build_price_model(&mut self, cp: &CalcParams, tier_weights: &[f64]) -> PriceModel {
        let blue = self.loader.get_quality_stats(3, Some(&cp.map_nest_id));
        let purple = self.loader.get_quality_stats(4, Some(&cp.map_nest_id));
        let gold = self.loader.get_quality_stats(5, Some(&cp.map_nest_id));
        let red = self.loader.get_quality_stats(6, Some(&cp.map_nest_id));
        let grid = self.build_grid_stats_model(Some(&cp.map_nest_id));
        let manual_purple_mean = manual_quality_unit_value(cp, 4, grid.purple_mean);
        let manual_gold_mean = manual_quality_unit_value(cp, 5, grid.gold_mean);
        let purple_mean =
            manual_purple_mean.unwrap_or_else(|| self.quality_mean_or_fallback(purple, 4));
        let gold_mean = manual_gold_mean.unwrap_or_else(|| self.quality_mean_or_fallback(gold, 5));
        PriceModel {
            greenwhite_mean: self.greenwhite_mean(Some(&cp.map_nest_id), tier_weights),
            blue_mean: self.quality_mean_or_fallback(blue, 3),
            purple_mean,
            gold_mean,
            red_mean: self.quality_mean_or_fallback(red, 6),
            blue_variance: blue.variance,
            purple_variance: manual_purple_mean
                .map(fallback_value_variance)
                .unwrap_or(purple.variance),
            gold_variance: manual_gold_mean
                .map(fallback_value_variance)
                .unwrap_or(gold.variance),
            red_variance: red.variance,
        }
    }

    fn greenwhite_mean(&self, nest_id: Option<&str>, tier_weights: &[f64]) -> f64 {
        if let Some(prices) = self
            .static_data
            .nest_weighted_prices
            .get(nest_id.unwrap_or(""))
            && prices.len() >= 2
        {
            let tier0 = tier_weights.first().copied().unwrap_or(0.0);
            let tier1 = tier_weights.get(1).copied().unwrap_or(0.0);
            let weight = tier0 + tier1;
            if weight > 0.0 {
                return (prices[0] * tier0 + prices[1] * tier1) / weight;
            }
        }
        400.0
    }

    fn quality_mean_or_fallback(&self, stats: QualityPriceStats, quality: i32) -> f64 {
        if stats.count > 0 && stats.mean > 0.0 {
            stats.mean
        } else {
            self.static_data
                .quality_p50_default
                .get(&quality.to_string())
                .copied()
                .unwrap_or(0.0)
        }
    }

    pub fn tier_weights(&self, tier: &str) -> Result<Vec<f64>> {
        self.static_data
            .drop_weights
            .get(tier)
            .cloned()
            .with_context(|| format!("unknown tier {tier}"))
    }

    pub fn price_range(&self, results: &[ComboResult], cp: &CalcParams) -> (f64, f64, f64) {
        Self::price_range_from_source(results, cp)
    }

    pub fn price_range_for_last_run(&self, cp: &CalcParams) -> (f64, f64, f64) {
        let source = if !self.price_range_results.is_empty() {
            &self.price_range_results
        } else {
            &self.raw_results
        };
        Self::price_range_from_source(source, cp)
    }

    fn price_range_from_source(source: &[ComboResult], cp: &CalcParams) -> (f64, f64, f64) {
        if source.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted_values: Vec<(f64, f64)> = source
            .iter()
            .flat_map(|r| combo_price_values_for_range(r, cp))
            .collect();
        normalize_value_probabilities(&mut sorted_values);
        if sorted_values.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        sorted_values.sort_by(|a, b| fcmp(a.0, b.0));
        let p50 = weighted_probability_quantile(&sorted_values, 0.5);
        let mut p25 = weighted_probability_quantile(&sorted_values, 0.25);
        let mut p75 = weighted_probability_quantile(&sorted_values, 0.75);
        let min_value_floor = valid_min_value_floor(cp);
        let variance_probability_sum: f64 = source
            .iter()
            .filter(|r| combo_matches_min_value_floor(r, cp))
            .map(|r| r.probability)
            .sum();
        let variance: f64 = if variance_probability_sum > 0.0 {
            source
                .iter()
                .filter(|r| combo_matches_min_value_floor(r, cp))
                .map(|r| r.high_variance * (r.probability / variance_probability_sum))
                .sum::<f64>()
        } else {
            0.0
        };
        let sd = variance.max(0.0).sqrt();
        if let Some(floor) = min_value_floor {
            if sd > 0.0 && (p75 - p25).abs() <= 1.0 {
                p25 = (p50 - 0.3373 * sd).max(floor).min(p50);
                p75 = p75.max(p50 + 0.3373 * sd);
            }
            return (p25, p50, p75);
        }
        let lower = p50 - 0.3373 * sd;
        p25 = (p25 * 0.65).max(p25.min(lower));
        p75 = p75.max(p50 + 0.3373 * sd);
        (p25, p50, p75)
    }

    pub fn combo_composition_lines(&mut self, combo: &ComboResult, cp: &CalcParams) -> Vec<String> {
        let cp = normalize_calc_params(cp.clone());
        let tier_weights = self.tier_weights_or_default(&cp.tier);
        let price = self.build_price_model(&cp, &tier_weights);
        let targets = build_pricing_grid_targets(
            &cp,
            combo.greenwhite_count,
            combo.blue_count,
            combo.purple_count,
            combo.gold_count,
            combo.red_count,
            combo.greenwhite_grid_est,
            combo.blue_grid_est,
            combo.purple_grid_est,
            combo.gold_grid_est,
            combo.red_grid_est,
            true,
        );
        let qualities = [
            (
                3,
                "蓝(Q3)",
                combo.blue_count,
                price.blue_mean,
                combo.blue_grid_est,
                targets.blue,
            ),
            (
                4,
                "紫(Q4)",
                combo.purple_count,
                price.purple_mean,
                combo.purple_grid_est,
                targets.purple,
            ),
            (
                5,
                "金(Q5)",
                combo.gold_count,
                price.gold_mean,
                combo.gold_grid_est,
                targets.gold,
            ),
            (
                6,
                "红(Q6)",
                combo.red_count,
                price.red_mean,
                combo.red_grid_est,
                targets.red,
            ),
        ];
        let mut lines = Vec::new();
        for (quality, label, count, mean, grid_est, target_grid) in qualities {
            if count <= 0 {
                lines.push(format!("{label}: --"));
                continue;
            }
            let manual_grid_est = target_grid
                .map(|target_grid| target_grid as f64)
                .unwrap_or(grid_est);
            let target_sum = manual_quality_total_value(&cp, quality, count, manual_grid_est)
                .unwrap_or_else(|| {
                    self.high_value_for_quality(
                        quality,
                        count,
                        mean,
                        target_grid,
                        Some(&cp.map_nest_id),
                    )
                });
            let pool = self.choose_pool(
                quality,
                target_sum / count as f64,
                Some(&cp.map_nest_id),
                target_grid,
                count,
            );
            let items = pool
                .into_iter()
                .take(count.max(0) as usize)
                .map(|item| {
                    if item.grid_trusted {
                        format!("{}×1({}格)", item.name, item.grid_size)
                    } else {
                        format!("{}×1", item.name)
                    }
                })
                .collect::<Vec<_>>();
            if items.is_empty() {
                lines.push(format!("{label}: 约 {} 件", count));
            } else {
                lines.push(format!("{label}: {}", items.join("、")));
            }
        }
        lines
    }

    fn tier_weights_or_default(&self, tier: &str) -> Vec<f64> {
        let mut weights = self
            .tier_weights(tier)
            .ok()
            .or_else(|| self.static_data.drop_weights.values().next().cloned())
            .unwrap_or_else(|| vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        if weights.len() < 6 {
            weights.resize(6, 0.0);
        }
        weights
    }
}

#[derive(Debug, Clone, Default)]
struct BeamState {
    sum: f64,
    grid: i32,
    unknown_grid: i32,
    log_map_weight: f64,
}

fn parse_i32(text: &str) -> i32 {
    text.trim().parse::<f64>().unwrap_or(0.0) as i32
}

fn parse_f64(text: &str) -> f64 {
    text.trim().parse::<f64>().unwrap_or(0.0)
}

fn parse_grid_size(item: &mut ItemRecord) {
    let Ok(shape) = item.shape.parse::<i32>() else {
        return;
    };
    if shape <= 0 {
        return;
    }
    let rows = shape / 10;
    let cols = shape % 10;
    if rows > 0 && cols > 0 {
        let grid = rows * cols;
        if (1..=MAX_ITEM_GRID_SIZE).contains(&grid) {
            item.grid_size = grid;
            item.grid_trusted = true;
        }
    }
}

fn apply_prob_floor(
    raw: MapQualityProbs,
    tier_fallback: Option<&MapQualityProbs>,
) -> MapQualityProbs {
    apply_prob_floor_with_availability(raw, tier_fallback, [true, true, true, true])
}

fn apply_prob_floor_with_availability(
    raw: MapQualityProbs,
    tier_fallback: Option<&MapQualityProbs>,
    available: [bool; 4],
) -> MapQualityProbs {
    let fallback = tier_fallback.unwrap_or(&raw);
    let floors = [
        0.001_f64.max(fallback.pb * 0.1),
        0.001_f64.max(fallback.pp * 0.1),
        0.001_f64.max(fallback.pg * 0.1),
        0.001_f64.max(fallback.pr * 0.1),
    ];
    let mut vals = [
        if available[0] {
            raw.pb.max(floors[0])
        } else {
            0.0
        },
        if available[1] {
            raw.pp.max(floors[1])
        } else {
            0.0
        },
        if available[2] {
            raw.pg.max(floors[2])
        } else {
            0.0
        },
        if available[3] {
            raw.pr.max(floors[3])
        } else {
            0.0
        },
    ];
    let total: f64 = vals.iter().sum();
    if total <= 0.0 {
        vals = [0.25, 0.25, 0.25, 0.25];
    } else {
        for value in &mut vals {
            *value /= total;
        }
    }
    MapQualityProbs {
        p_low: raw.p_low,
        p_high: raw.p_high,
        pb: vals[0],
        pp: vals[1],
        pg: vals[2],
        pr: vals[3],
        source: raw.source,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_pricing_grid_targets(
    cp: &CalcParams,
    gw: i32,
    b: i32,
    p: i32,
    g: i32,
    r: i32,
    grid_gw: f64,
    grid_b: f64,
    grid_p: f64,
    grid_g: f64,
    grid_r: f64,
    use_total_grid_target: bool,
) -> PricingGridTargets {
    let mut targets = PricingGridTargets {
        gw: implied_target_grid(gw, cp.gw_grid, cp.gw_avg, None),
        blue: implied_target_grid(b, cp.blue_grid, cp.blue_avg, Some(3)),
        purple: implied_target_grid(p, cp.purple_grid, cp.purple_avg, Some(4)),
        gold: implied_target_grid(g, cp.gold_grid, cp.gold_avg, Some(5)),
        red: implied_target_grid(r, cp.red_grid, cp.red_avg, Some(6)),
    };
    if use_total_grid_target && let Some(total_grid_target) = cp.total_grid_target {
        let slots = [
            ("gw", gw, grid_gw, targets.gw),
            ("blue", b, grid_b, targets.blue),
            ("purple", p, grid_p, targets.purple),
            ("gold", g, grid_g, targets.gold),
            ("red", r, grid_r, targets.red),
        ];
        let mut known = 0.0;
        let mut unknown_count = 0;
        for (_, count, _, target) in slots {
            if let Some(target) = target {
                known += target as f64;
            } else if count > 0 {
                unknown_count += 1;
            }
        }
        let residual = total_grid_target - known;
        if unknown_count > 0 && residual >= 0.0 {
            let mut unknowns = Vec::new();
            if targets.gw.is_none() {
                unknowns.push(ResidualGridSlot {
                    index: 0,
                    quality: None,
                    count: gw,
                    model_grid: grid_gw,
                });
            }
            if targets.blue.is_none() {
                unknowns.push(ResidualGridSlot {
                    index: 1,
                    quality: Some(3),
                    count: b,
                    model_grid: grid_b,
                });
            }
            if targets.purple.is_none() {
                unknowns.push(ResidualGridSlot {
                    index: 2,
                    quality: Some(4),
                    count: p,
                    model_grid: grid_p,
                });
            }
            if targets.gold.is_none() {
                unknowns.push(ResidualGridSlot {
                    index: 3,
                    quality: Some(5),
                    count: g,
                    model_grid: grid_g,
                });
            }
            if targets.red.is_none() {
                unknowns.push(ResidualGridSlot {
                    index: 4,
                    quality: Some(6),
                    count: r,
                    model_grid: grid_r,
                });
            }
            for (index, grid) in allocate_residual_grid_targets(residual, &unknowns) {
                match index {
                    0 => targets.gw = Some(grid),
                    1 => targets.blue = Some(grid),
                    2 => targets.purple = Some(grid),
                    3 => targets.gold = Some(grid),
                    4 => targets.red = Some(grid),
                    _ => {}
                }
            }
        }
    }
    targets
}

fn allocate_residual_grid_targets(residual: f64, slots: &[ResidualGridSlot]) -> Vec<(usize, i32)> {
    #[derive(Debug, Clone)]
    struct Allocation {
        index: usize,
        min: i32,
        cap: i32,
        extra: i32,
        weight: f64,
        remainder: f64,
    }

    let mut allocations = slots
        .iter()
        .filter(|slot| slot.count > 0)
        .map(|slot| {
            let min = slot.count;
            let max = max_grid_for_quality(slot.quality) * slot.count;
            let cap = (max - min).max(0);
            let weight = if slot.model_grid.is_finite() && slot.model_grid > 0.0 {
                slot.model_grid
            } else {
                slot.count as f64
            };
            Allocation {
                index: slot.index,
                min,
                cap,
                extra: 0,
                weight: weight.max(0.0001),
                remainder: 0.0,
            }
        })
        .collect::<Vec<_>>();
    if allocations.is_empty() || !residual.is_finite() {
        return Vec::new();
    }
    let min_sum: i32 = allocations.iter().map(|slot| slot.min).sum();
    let cap_sum: i32 = allocations.iter().map(|slot| slot.cap).sum();
    let desired = round_to_i32(residual).clamp(min_sum, min_sum + cap_sum);
    let mut remaining = desired - min_sum;
    if remaining <= 0 {
        return allocations
            .into_iter()
            .map(|slot| (slot.index, slot.min))
            .collect();
    }
    let weight_sum: f64 = allocations
        .iter()
        .filter(|slot| slot.cap > 0)
        .map(|slot| slot.weight)
        .sum();
    if weight_sum > 0.0 {
        for slot in &mut allocations {
            if slot.cap <= 0 {
                continue;
            }
            let raw = remaining as f64 * slot.weight / weight_sum;
            let extra = raw.floor() as i32;
            slot.extra = extra.clamp(0, slot.cap);
            slot.remainder = raw - extra as f64;
        }
        remaining -= allocations.iter().map(|slot| slot.extra).sum::<i32>();
    }
    while remaining > 0 {
        let mut order = (0..allocations.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| {
            fcmp(allocations[*b].remainder, allocations[*a].remainder)
                .then_with(|| fcmp(allocations[*b].weight, allocations[*a].weight))
        });
        let mut progressed = false;
        for idx in order {
            if remaining <= 0 {
                break;
            }
            let slot = &mut allocations[idx];
            if slot.extra < slot.cap {
                slot.extra += 1;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    allocations
        .into_iter()
        .map(|slot| (slot.index, slot.min + slot.extra))
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| fcmp(*a, *b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn log_fact(n: i32) -> f64 {
    if n < 0 {
        return f64::NEG_INFINITY;
    }
    (2..=n).map(|i| (i as f64).ln()).sum()
}

fn log_binom_p(n: i32, k: i32, p: f64) -> f64 {
    if p <= 0.0 {
        return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if p >= 1.0 {
        return if k == n { 0.0 } else { f64::NEG_INFINITY };
    }
    log_fact(n) - log_fact(k) - log_fact(n - k)
        + k as f64 * p.ln()
        + (n - k) as f64 * (1.0 - p).ln()
}

pub fn normalize_calc_params(mut cp: CalcParams) -> CalcParams {
    if high_quality_only_mode(&cp) {
        cp.avg_grid_all = None;
        return cp;
    }
    if cp.total_grid_target.is_none() && cp.avg_grid_all.is_some() && cp.total_count > 0 {
        cp.total_grid_target = cp.avg_grid_all.map(|avg| avg * cp.total_count as f64);
    }
    cp
}

fn validate_calc_params(cp: &CalcParams) -> Result<()> {
    if high_quality_only_mode(cp) {
        return validate_high_quality_only_calc_params(cp);
    }
    if cp.total_count <= 0 {
        anyhow::bail!("总件数必须大于 0；如果只知道高品质数量，请填写紫金红总数");
    }
    validate_optional_count("紫金红总数", cp.high_quality_count, cp.total_count)?;
    validate_count_field("绿白件数", cp.gw_count, cp.min_gw, cp.total_count)?;
    validate_count_field("蓝件数", cp.blue_count, cp.min_blue, cp.total_count)?;
    validate_count_field("紫件数", cp.purple_count, cp.min_purple, cp.total_count)?;
    validate_count_field("金件数", cp.gold_count, cp.min_gold, cp.total_count)?;
    validate_count_field("红件数", cp.red_count, cp.min_red, cp.total_count)?;

    let lower_bound_sum = [
        cp.gw_count.unwrap_or(cp.min_gw.max(0)),
        cp.blue_count.unwrap_or(cp.min_blue.max(0)),
        cp.purple_count.unwrap_or(cp.min_purple.max(0)),
        cp.gold_count.unwrap_or(cp.min_gold.max(0)),
        cp.red_count.unwrap_or(cp.min_red.max(0)),
    ]
    .into_iter()
    .sum::<i32>();
    if lower_bound_sum > cp.total_count {
        anyhow::bail!("各品质件数/至少件数之和超过总件数");
    }

    validate_optional_nonnegative_f64("总格数", cp.total_grid_target)?;
    validate_optional_nonnegative_f64("全部品均格", cp.avg_grid_all)?;
    validate_global_grid_bounds("总格数", cp.total_grid_target, cp.total_count)?;
    validate_global_avg_grid_bounds(cp.avg_grid_all)?;
    validate_optional_nonnegative_f64("绿白总格", cp.gw_grid)?;
    validate_optional_nonnegative_f64("绿白均格", cp.gw_avg)?;
    validate_optional_nonnegative_f64("蓝总格", cp.blue_grid)?;
    validate_optional_nonnegative_f64("蓝均格", cp.blue_avg)?;
    validate_optional_nonnegative_f64("紫总格", cp.purple_grid)?;
    validate_optional_nonnegative_f64("紫均格", cp.purple_avg)?;
    validate_optional_nonnegative_f64("金总格", cp.gold_grid)?;
    validate_optional_nonnegative_f64("金均格", cp.gold_avg)?;
    validate_optional_nonnegative_f64("红总格", cp.red_grid)?;
    validate_optional_nonnegative_f64("红均格", cp.red_avg)?;
    validate_optional_nonnegative_f64("当前预估最低价格", cp.min_value_floor)?;
    validate_optional_nonnegative_f64("紫色每件均价", cp.manual_purple_per_item)?;
    validate_optional_nonnegative_f64("紫色每格均价", cp.manual_purple_per_grid)?;
    validate_optional_nonnegative_f64("金色每件均价", cp.manual_gold_per_item)?;
    validate_optional_nonnegative_f64("金色每格均价", cp.manual_gold_per_grid)?;

    if !cp.safety_factor.is_finite() || cp.safety_factor <= 0.0 {
        anyhow::bail!("安全系数必须是大于 0 的有效数字");
    }
    for (index, sample) in cp.value_samples.iter().enumerate() {
        if sample.count <= 0 || sample.count > cp.total_count {
            anyhow::bail!("随机样本第 {} 行件数必须在 1 到总件数之间", index + 1);
        }
        if !sample.avg_value.is_finite() || sample.avg_value < 0.0 {
            anyhow::bail!("随机样本第 {} 行均价不能为负数", index + 1);
        }
    }
    Ok(())
}

fn validate_high_quality_only_calc_params(cp: &CalcParams) -> Result<()> {
    let high_count = cp.high_quality_count.unwrap_or_default();
    if high_count <= 0 {
        anyhow::bail!("紫金红总数必须大于 0");
    }
    validate_high_quality_only_absent("总格数", cp.total_grid_target.is_some())?;
    validate_high_quality_only_absent("全部品均格", cp.avg_grid_all.is_some())?;
    validate_high_quality_only_absent("绿白件数", cp.gw_count.is_some())?;
    validate_high_quality_only_absent("绿白至少件数", cp.min_gw > 0)?;
    validate_high_quality_only_absent("绿白总格", cp.gw_grid.is_some())?;
    validate_high_quality_only_absent("绿白均格", cp.gw_avg.is_some())?;
    validate_high_quality_only_absent("蓝件数", cp.blue_count.is_some())?;
    validate_high_quality_only_absent("蓝至少件数", cp.min_blue > 0)?;
    validate_high_quality_only_absent("蓝总格", cp.blue_grid.is_some())?;
    validate_high_quality_only_absent("蓝均格", cp.blue_avg.is_some())?;
    if !cp.value_samples.is_empty() {
        anyhow::bail!("随机样本信息需要总件数，当前仅填写了紫金红总数");
    }

    validate_count_field_with_limit(
        "紫件数",
        cp.purple_count,
        cp.min_purple,
        high_count,
        "紫金红总数",
    )?;
    validate_count_field_with_limit(
        "金件数",
        cp.gold_count,
        cp.min_gold,
        high_count,
        "紫金红总数",
    )?;
    validate_count_field_with_limit("红件数", cp.red_count, cp.min_red, high_count, "紫金红总数")?;
    let lower_bound_sum = [
        cp.purple_count.unwrap_or(cp.min_purple.max(0)),
        cp.gold_count.unwrap_or(cp.min_gold.max(0)),
        cp.red_count.unwrap_or(cp.min_red.max(0)),
    ]
    .into_iter()
    .sum::<i32>();
    if lower_bound_sum > high_count {
        anyhow::bail!("紫/金/红件数或至少件数之和超过紫金红总数");
    }

    validate_optional_nonnegative_f64("紫总格", cp.purple_grid)?;
    validate_optional_nonnegative_f64("紫均格", cp.purple_avg)?;
    validate_optional_nonnegative_f64("金总格", cp.gold_grid)?;
    validate_optional_nonnegative_f64("金均格", cp.gold_avg)?;
    validate_optional_nonnegative_f64("红总格", cp.red_grid)?;
    validate_optional_nonnegative_f64("红均格", cp.red_avg)?;
    validate_optional_nonnegative_f64("当前预估最低价格", cp.min_value_floor)?;
    validate_optional_nonnegative_f64("紫色每件均价", cp.manual_purple_per_item)?;
    validate_optional_nonnegative_f64("紫色每格均价", cp.manual_purple_per_grid)?;
    validate_optional_nonnegative_f64("金色每件均价", cp.manual_gold_per_item)?;
    validate_optional_nonnegative_f64("金色每格均价", cp.manual_gold_per_grid)?;
    if !cp.safety_factor.is_finite() || cp.safety_factor <= 0.0 {
        anyhow::bail!("安全系数必须是大于 0 的有效数字");
    }
    Ok(())
}

fn validate_high_quality_only_absent(name: &str, present: bool) -> Result<()> {
    if present {
        anyhow::bail!("{name}需要总件数，当前仅填写了紫金红总数");
    }
    Ok(())
}

fn validate_count_field(
    name: &str,
    fixed_count: Option<i32>,
    min_count: i32,
    total_count: i32,
) -> Result<()> {
    validate_count_field_with_limit(name, fixed_count, min_count, total_count, "总件数")
}

fn validate_count_field_with_limit(
    name: &str,
    fixed_count: Option<i32>,
    min_count: i32,
    max_count: i32,
    max_name: &str,
) -> Result<()> {
    validate_optional_count_with_limit(name, fixed_count, max_count, max_name)?;
    if min_count < 0 || min_count > max_count {
        anyhow::bail!("{name}至少件数必须在 0 到{max_name}之间");
    }
    if let Some(fixed_count) = fixed_count
        && fixed_count < min_count
    {
        anyhow::bail!("{name}不能小于该品质的至少件数");
    }
    Ok(())
}

fn validate_optional_count(name: &str, value: Option<i32>, total_count: i32) -> Result<()> {
    validate_optional_count_with_limit(name, value, total_count, "总件数")
}

fn validate_optional_count_with_limit(
    name: &str,
    value: Option<i32>,
    max_count: i32,
    max_name: &str,
) -> Result<()> {
    if let Some(value) = value
        && (value < 0 || value > max_count)
    {
        anyhow::bail!("{name}必须在 0 到{max_name}之间：当前为 {value}，{max_name}为 {max_count}");
    }
    Ok(())
}

fn validate_optional_nonnegative_f64(name: &str, value: Option<f64>) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        anyhow::bail!("{name}必须是非负有效数字");
    }
    Ok(())
}

fn validate_global_grid_bounds(name: &str, value: Option<f64>, total_count: i32) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if total_count <= 0 {
        return Ok(());
    }
    let min = total_count as f64;
    let max = (MAX_ITEM_GRID_SIZE * total_count) as f64;
    if value < min || value > max {
        anyhow::bail!("{name}必须在 {:.0} 到 {:.0} 之间", min, max);
    }
    Ok(())
}

fn validate_global_avg_grid_bounds(value: Option<f64>) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value <= 0.0 || value > MAX_ITEM_GRID_SIZE as f64 {
        anyhow::bail!("全部品均格必须大于 0 且不超过 {MAX_ITEM_GRID_SIZE}");
    }
    Ok(())
}

fn ensure_candidate_combos(results: &[ComboResult]) -> Result<()> {
    if results.is_empty() {
        anyhow::bail!("没有找到符合当前条件的组合，请检查件数、格数、均格或 OCR 识别结果");
    }
    Ok(())
}

fn required_count(fixed: Option<i32>, min_count: i32) -> i32 {
    match fixed {
        None => min_count.max(0),
        Some(v) if v < 0 => 536_870_911,
        Some(v) => v.min(536_870_911),
    }
}

fn count_range(max_inclusive: i32, fixed: Option<i32>, min_count: i32, reserve: i32) -> (i32, i32) {
    let end = max_inclusive - reserve;
    if end < 0 {
        return (1, 0);
    }
    if let Some(value) = fixed {
        if value < 0 || value > end {
            return (1, 0);
        }
        return (value, value);
    }
    let start = min_count.max(0);
    if start > end { (1, 0) } else { (start, end) }
}

fn inclusive_range(range: (i32, i32)) -> std::ops::RangeInclusive<i32> {
    let (start, end) = range;
    start..=end
}

fn valid_at(valid: &[bool], idx: i32) -> bool {
    idx >= 0 && valid.get(idx as usize).copied().unwrap_or(false)
}

fn build_valid_color_counts(
    grid_sizes: &[i32],
    max_count: i32,
    fixed_count: Option<i32>,
    min_count: i32,
    grid: Option<f64>,
    avg: Option<f64>,
) -> Vec<bool> {
    if max_count < 0 {
        return vec![];
    }
    let mut valid = vec![false; max_count as usize + 1];
    if let Some(fixed_count) = fixed_count {
        if fixed_count >= 0 && fixed_count <= max_count && fixed_count >= min_count {
            valid[fixed_count as usize] = is_valid_color_count(fixed_count, grid, avg, grid_sizes);
        }
        return valid;
    }
    for count in min_count.max(0)..=max_count {
        valid[count as usize] = is_valid_color_count(count, grid, avg, grid_sizes);
    }
    valid
}

fn is_valid_color_count(
    count: i32,
    grid: Option<f64>,
    avg: Option<f64>,
    grid_sizes: &[i32],
) -> bool {
    if count > 0 && grid_sizes.is_empty() {
        return false;
    }
    let rounded_grid = grid.map(round_to_i32);
    if let Some(avg) = avg
        && avg <= 0.0
    {
        if count == 0 {
            return rounded_grid.map(|grid| grid == 0).unwrap_or(true);
        }
        return false;
    }
    if let Some(rounded_grid) = rounded_grid {
        if count == 0 {
            if rounded_grid == 0 {
                return avg.map(|v| v.abs() < 1e-9).unwrap_or(true);
            }
            return false;
        }
        if rounded_grid < count || rounded_grid > max_grid_for_count(count, grid_sizes) {
            return false;
        }
        if !can_compose_grid_total(count, rounded_grid, grid_sizes) {
            return false;
        }
        if avg.is_some() && !avg_match(rounded_grid, count, avg, grid_sizes) {
            return false;
        }
    } else if let Some(avg) = avg {
        if count == 0 {
            return avg == 0.0;
        }
        if !avg_count_match(count, avg) {
            return false;
        }
        if !avg_can_map_to_composable_grid(count, avg, grid_sizes) {
            return false;
        }
    }
    true
}

fn avg_match(grid: i32, count: i32, avg: Option<f64>, grid_sizes: &[i32]) -> bool {
    let Some(avg) = avg else {
        return true;
    };
    if count == 0 {
        return true;
    }
    if !avg_count_match(count, avg) {
        return false;
    }
    if !can_compose_grid_total(count, grid, grid_sizes) {
        return false;
    }
    let target = (avg * 100.0 + 1e-7).floor() as i32;
    let got = (grid as f64 * 100.0 / count as f64 + 1e-7).floor() as i32;
    got == target
}

fn avg_can_map_to_composable_grid(count: i32, avg: f64, grid_sizes: &[i32]) -> bool {
    if count <= 0 || !avg.is_finite() || avg <= 0.0 {
        return false;
    }
    let target = (avg * 100.0 + 1e-7).floor() as i32;
    let max_grid = max_grid_for_count(count, grid_sizes);
    let reachable = grid_reachability(count, max_grid, grid_sizes);
    (count..=max_grid).any(|grid| {
        reachable.get(grid as usize).copied().unwrap_or(false)
            && (grid as f64 * 100.0 / count as f64 + 1e-7).floor() as i32 == target
    })
}

pub fn infer_grid_from_average_for_quality(
    count: i32,
    avg: f64,
    quality: Option<i32>,
) -> Option<i32> {
    infer_grid_from_average_with_sizes(count, avg, grid_sizes_for_quality(quality))
}

pub fn infer_grid_from_average_with_sizes(count: i32, avg: f64, grid_sizes: &[i32]) -> Option<i32> {
    if count <= 0 || !avg.is_finite() || avg <= 0.0 {
        return None;
    }
    let target = (avg * 100.0 + 1e-7).floor() as i32;
    let max_grid = max_grid_for_count(count, grid_sizes);
    let reachable = grid_reachability(count, max_grid, grid_sizes);
    let mut matches = (count..=max_grid).filter(|grid| {
        reachable.get(*grid as usize).copied().unwrap_or(false)
            && (*grid as f64 * 100.0 / count as f64 + 1e-7).floor() as i32 == target
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

pub fn grid_matches_average_for_quality(
    count: i32,
    grid: i32,
    avg: f64,
    quality: Option<i32>,
) -> bool {
    grid_matches_average_with_sizes(count, grid, avg, grid_sizes_for_quality(quality))
}

pub fn grid_matches_average_with_sizes(
    count: i32,
    grid: i32,
    avg: f64,
    grid_sizes: &[i32],
) -> bool {
    avg_match(grid, count, Some(avg), grid_sizes)
}

fn can_compose_grid_total(count: i32, grid: i32, grid_sizes: &[i32]) -> bool {
    if count == 0 {
        return grid == 0;
    }
    if count < 0 || grid < count || grid > max_grid_for_count(count, grid_sizes) {
        return false;
    }
    grid_reachability(count, grid, grid_sizes)
        .get(grid as usize)
        .copied()
        .unwrap_or(false)
}

fn grid_reachability(count: i32, max_grid: i32, grid_sizes: &[i32]) -> Vec<bool> {
    if count < 0 || max_grid < 0 {
        return vec![];
    }
    let max_grid = max_grid as usize;
    let mut current = vec![false; max_grid + 1];
    current[0] = true;
    for _ in 0..count {
        let mut next = vec![false; max_grid + 1];
        for (grid, reachable) in current.iter().enumerate().take(max_grid + 1) {
            if !*reachable {
                continue;
            }
            for item_grid in grid_sizes {
                let new_grid = grid + *item_grid as usize;
                if new_grid <= max_grid {
                    next[new_grid] = true;
                }
            }
        }
        current = next;
    }
    current
}

fn grid_sizes_for_quality(quality: Option<i32>) -> &'static [i32] {
    match quality {
        Some(3) => BLUE_ITEM_GRID_SIZES,
        Some(4) => PURPLE_ITEM_GRID_SIZES,
        Some(5) => GOLD_ITEM_GRID_SIZES,
        Some(6) => RED_ITEM_GRID_SIZES,
        _ => DEFAULT_ITEM_GRID_SIZES,
    }
}

fn max_grid_for_quality(quality: Option<i32>) -> i32 {
    grid_sizes_for_quality(quality)
        .iter()
        .copied()
        .max()
        .unwrap_or(MAX_ITEM_GRID_SIZE)
}

fn max_grid_for_count(count: i32, grid_sizes: &[i32]) -> i32 {
    count.max(0)
        * grid_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(MAX_ITEM_GRID_SIZE)
}

fn avg_count_match(count: i32, avg: f64) -> bool {
    if count <= 0 || !avg.is_finite() || avg < 0.0 {
        return false;
    }
    let key = avg_fraction_key(avg);
    (0..count).any(|k| ((k as f64 * 100.0 / count as f64) + 1e-7).floor() as i32 == key)
}

fn avg_fraction_key(avg: f64) -> i32 {
    let frac = avg - avg.floor();
    if frac < 1e-9 || 1.0 - frac < 1e-9 {
        0
    } else {
        (frac * 100.0 + 1e-7).floor() as i32
    }
}

fn round_opt(value: Option<f64>) -> Option<i32> {
    value.map(round_to_i32)
}

fn round_to_i32(value: f64) -> i32 {
    // .NET Math.Round and Python round both use banker rounding for .5.
    let floor = value.floor();
    let frac = value - floor;
    if (frac - 0.5).abs() < 1e-12 {
        let floor_i = floor as i32;
        if floor_i % 2 == 0 {
            floor_i
        } else {
            floor_i + 1
        }
    } else {
        value.round() as i32
    }
}

fn fallback_avg_grid(quality: i32) -> f64 {
    match quality {
        3 => 2.2,
        4 => 2.4,
        5 => 2.8,
        6 => 3.2,
        _ => 2.5,
    }
}

fn sigma_from_unknowns(
    cp: &CalcParams,
    grid: &GridStatsModel,
    b: i32,
    p: i32,
    g: i32,
    r: i32,
    gw: i32,
) -> f64 {
    let mut var = 0.0;
    if cp.gw_grid.is_none() {
        var += gw as f64 * grid.greenwhite_variance;
    }
    if cp.blue_grid.is_none() {
        var += b as f64 * grid.blue_variance;
    }
    if cp.purple_grid.is_none() {
        var += p as f64 * grid.purple_variance;
    }
    if cp.gold_grid.is_none() {
        var += g as f64 * grid.gold_variance;
    }
    if cp.red_grid.is_none() {
        var += r as f64 * grid.red_variance;
    }
    1.4_f64.max(var.max(1.0).sqrt())
}

fn total_grid_prior_log(diff: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1.0);
    let z = diff.abs() / sigma;
    let clipped = z.min(3.5);
    let mut value = -0.5 * clipped * clipped;
    if z > 3.5 {
        value -= 0.15 * (z - 3.5);
    }
    value
}

fn red_count_caution_log(
    cp: &CalcParams,
    red_count: i32,
    total_count: i32,
    red_probability: f64,
) -> f64 {
    if cp.red_count.is_some() || red_count <= 0 || total_count <= 0 {
        return 0.0;
    }
    let red_probability = red_probability.clamp(0.0, 1.0);
    if red_probability <= 0.0 || red_probability >= 1.0 {
        return 0.0;
    }
    let mean = total_count as f64 * red_probability;
    let sd = (0.25_f64.max(total_count as f64 * red_probability * (1.0 - red_probability))).sqrt();
    let limit = 8.0_f64.max(mean + 3.0 * sd);
    if red_count as f64 <= limit {
        return 0.0;
    }
    let scale = 3.0_f64.max(sd * 2.5);
    let z = (red_count as f64 - limit) / scale;
    -0.04 * z * z
}

fn avg_prior_log(count: i32, grid_est: f64, avg: Option<f64>, base_sigma: f64) -> f64 {
    let Some(avg) = avg else {
        return 0.0;
    };
    if count <= 0 || avg <= 0.0 {
        return 0.0;
    }
    let diff = grid_est / count as f64 - avg;
    let sigma = base_sigma / (count as f64).sqrt().max(1.0);
    -0.5 * diff * diff / (sigma * sigma)
}

fn implied_target_grid(
    count: i32,
    grid_in: Option<f64>,
    avg_in: Option<f64>,
    quality: Option<i32>,
) -> Option<i32> {
    if count <= 0 {
        return None;
    }
    if let Some(grid) = grid_in
        && grid.is_finite()
    {
        return Some(clamp_grid_target(count, round_to_i32(grid), quality));
    }
    if let Some(avg) = avg_in
        && avg.is_finite()
        && avg > 0.0
    {
        return Some(clamp_grid_target(
            count,
            round_to_i32(count as f64 * avg),
            quality,
        ));
    }
    None
}

fn clamp_grid_target(count: i32, grid: i32, quality: Option<i32>) -> i32 {
    if count <= 0 {
        0
    } else {
        grid.clamp(count, max_grid_for_quality(quality) * count)
    }
}

fn item_map_weight(resolved: Option<&HashMap<String, f64>>, item: &ItemRecord) -> f64 {
    resolved
        .and_then(|r| r.get(&item.item_id).copied())
        .unwrap_or(0.0)
}

fn weighted_quantile(values: &[(f64, f64)], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| fcmp(a.0, b.0));
    let total: f64 = sorted.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 {
        return sorted[sorted.len() / 2].0;
    }
    let threshold = total * quantile;
    let mut acc = 0.0;
    for (value, weight) in &sorted {
        acc += weight;
        if acc >= threshold {
            return *value;
        }
    }
    sorted.last().unwrap().0
}

fn weighted_probability_quantile(sorted_values: &[(f64, f64)], quantile: f64) -> f64 {
    let mut acc = 0.0;
    for (value, probability) in sorted_values {
        acc += probability;
        if acc >= quantile {
            return *value;
        }
    }
    sorted_values.last().map(|v| v.0).unwrap_or(0.0)
}

fn weighted_mean_grid(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .enumerate()
        .map(|(i, w)| i as f64 * w)
        .sum::<f64>()
        / total
}

fn top_probability_indices(combos: &[ComboResult], limit: usize) -> HashSet<usize> {
    if limit == 0 {
        return HashSet::new();
    }
    let mut indexed = combos
        .iter()
        .enumerate()
        .filter(|(_, combo)| combo.probability.is_finite() && combo.probability > 0.0)
        .map(|(index, combo)| (index, combo.probability))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| fcmp(b.1, a.1).then_with(|| a.0.cmp(&b.0)));
    indexed
        .into_iter()
        .take(limit)
        .map(|(index, _)| index)
        .collect()
}

fn price_range_source(raw_results: &[ComboResult], keep_all: bool) -> Vec<ComboResult> {
    if keep_all {
        return raw_results.to_vec();
    }
    let mut out = Vec::new();
    let mut mass = 0.0;
    for combo in raw_results {
        if !combo.probability.is_finite() || combo.probability <= 0.0 {
            continue;
        }
        if combo.probability >= PROB_CUTOFF || mass < PRICE_RANGE_MIN_MASS || out.is_empty() {
            out.push(combo.clone());
            mass += combo.probability;
        }
        if mass >= PRICE_RANGE_MIN_MASS && combo.probability < PROB_CUTOFF {
            break;
        }
        if out.len() >= PRICE_RANGE_MAX_COMBOS {
            break;
        }
    }
    if out.is_empty() {
        raw_results.first().cloned().into_iter().collect()
    } else {
        out
    }
}

fn filter_combos_below_min_value_floor(
    mut combos: Vec<ComboResult>,
    cp: &CalcParams,
) -> Result<Vec<ComboResult>> {
    if let Some(floor) = valid_min_value_floor(cp) {
        let before = combos.len();
        for combo in &mut combos {
            restrict_combo_price_points_to_floor(combo, floor);
        }
        combos.retain(|combo| combo_matches_min_value_floor(combo, cp));
        if before > 0 && combos.is_empty() {
            anyhow::bail!(
                "当前预估最低价格与模型冲突：全部 {before} 个组合估值都低于该价格，请检查 OCR 识别或清空该字段后重算"
            );
        }
    }
    Ok(combos)
}

fn combo_matches_min_value_floor(combo: &ComboResult, cp: &CalcParams) -> bool {
    if !combo.final_value.is_finite() {
        return false;
    }
    let Some(floor) = valid_min_value_floor(cp) else {
        return true;
    };
    if combo.final_value >= floor {
        return true;
    }
    combo
        .high_value_price_points
        .iter()
        .any(|point| point.value.is_finite() && point.probability > 0.0 && point.value >= floor)
}

fn combo_price_values_for_range(combo: &ComboResult, cp: &CalcParams) -> Vec<(f64, f64)> {
    let floor = valid_min_value_floor(cp);
    if combo.high_value_price_points.is_empty() {
        if combo_matches_min_value_floor(combo, cp) {
            return vec![(combo.final_value, combo.probability)];
        }
        return Vec::new();
    }
    combo
        .high_value_price_points
        .iter()
        .filter(|point| {
            floor.map(|floor| point.value >= floor).unwrap_or(true)
                && point.value.is_finite()
                && point.probability.is_finite()
                && point.probability > 0.0
        })
        .map(|point| (point.value, combo.probability * point.probability))
        .collect()
}

fn restrict_combo_price_points_to_floor(combo: &mut ComboResult, floor: f64) {
    if combo.high_value_price_points.is_empty() {
        return;
    }
    combo
        .high_value_price_points
        .retain(|point| point.value.is_finite() && point.probability > 0.0 && point.value >= floor);
    normalize_price_points(&mut combo.high_value_price_points);
    if combo.high_value_price_points.is_empty() {
        return;
    }
    combo.final_value = combo
        .high_value_price_points
        .iter()
        .map(|point| point.value * point.probability)
        .sum();
}

fn normalize_value_probabilities(values: &mut Vec<(f64, f64)>) {
    values.retain(|(value, probability)| {
        value.is_finite() && probability.is_finite() && *probability > 0.0
    });
    let probability_sum: f64 = values.iter().map(|(_, probability)| probability).sum();
    if probability_sum <= 0.0 {
        values.clear();
        return;
    }
    for (_, probability) in values {
        *probability /= probability_sum;
    }
}

fn normalize_price_points(points: &mut Vec<PricePoint>) {
    points.retain(|point| {
        point.value.is_finite() && point.probability.is_finite() && point.probability > 0.0
    });
    let probability_sum: f64 = points.iter().map(|point| point.probability).sum();
    if probability_sum <= 0.0 {
        points.clear();
        return;
    }
    for point in points {
        point.probability /= probability_sum;
    }
}

fn compress_price_points(points: Vec<PricePoint>) -> Vec<PricePoint> {
    let mut buckets: HashMap<i64, (f64, f64)> = HashMap::new();
    for point in points {
        if !point.value.is_finite() || !point.probability.is_finite() || point.probability <= 0.0 {
            continue;
        }
        let bucket = (point.value / JACKPOT_VALUE_BIN).round() as i64;
        let entry = buckets.entry(bucket).or_insert((0.0, 0.0));
        entry.0 += point.probability;
        entry.1 += point.value * point.probability;
    }
    let mut out = buckets
        .into_values()
        .filter_map(|(probability, weighted_value)| {
            (probability > 0.0).then_some(PricePoint {
                value: weighted_value / probability,
                probability,
            })
        })
        .collect::<Vec<_>>();
    normalize_price_points(&mut out);
    if out.len() <= JACKPOT_MAX_PRICE_POINTS {
        out.sort_by(|a, b| fcmp(a.value, b.value));
        return out;
    }

    let min_point = out.iter().min_by(|a, b| fcmp(a.value, b.value)).copied();
    let max_point = out.iter().max_by(|a, b| fcmp(a.value, b.value)).copied();
    out.sort_by(|a, b| fcmp(b.probability, a.probability));
    out.truncate(JACKPOT_MAX_PRICE_POINTS.saturating_sub(2).max(1));
    if let Some(point) = min_point
        && !out
            .iter()
            .any(|existing| (existing.value - point.value).abs() < 1e-9)
    {
        out.push(point);
    }
    if let Some(point) = max_point
        && !out
            .iter()
            .any(|existing| (existing.value - point.value).abs() < 1e-9)
    {
        out.push(point);
    }
    normalize_price_points(&mut out);
    out.sort_by(|a, b| fcmp(a.value, b.value));
    out
}

fn price_point_spread(points: &[PricePoint]) -> f64 {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for point in points {
        if point.probability > 0.0 && point.value.is_finite() {
            min_value = min_value.min(point.value);
            max_value = max_value.max(point.value);
        }
    }
    if min_value.is_finite() && max_value.is_finite() {
        max_value - min_value
    } else {
        0.0
    }
}

pub fn recommended_bid_value(value: f64, cp: &CalcParams) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    let safety_factor = if cp.safety_factor.is_finite() && cp.safety_factor > 0.0 {
        cp.safety_factor
    } else {
        1.0
    };
    if let Some(floor) = valid_min_value_floor(cp) {
        floor + (value - floor).max(0.0) * safety_factor
    } else {
        value * safety_factor
    }
}

fn valid_min_value_floor(cp: &CalcParams) -> Option<f64> {
    cp.min_value_floor
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn valid_high_quality_count(cp: &CalcParams) -> Option<i32> {
    cp.high_quality_count
        .filter(|value| *value >= 0 && *value <= cp.total_count)
}

fn high_quality_only_mode(cp: &CalcParams) -> bool {
    cp.total_count <= 0 && cp.high_quality_count.is_some_and(|value| value > 0)
}

#[allow(clippy::too_many_arguments)]
fn value_sample_prior_log(
    cp: &CalcParams,
    modeled_count: i32,
    gw: i32,
    blue: i32,
    purple: i32,
    gold: i32,
    red: i32,
    greenwhite_grid_est: f64,
    blue_grid_est: f64,
    purple_grid_est: f64,
    gold_grid_est: f64,
    red_grid_est: f64,
    price: &PriceModel,
) -> f64 {
    if cp.value_samples.is_empty() {
        return 0.0;
    }
    let population_count = (gw + blue + purple + gold + red).max(0);
    if population_count <= 0 || modeled_count <= 0 {
        return 0.0;
    }
    let targets = build_pricing_grid_targets(
        cp,
        gw,
        blue,
        purple,
        gold,
        red,
        greenwhite_grid_est,
        blue_grid_est,
        purple_grid_est,
        gold_grid_est,
        red_grid_est,
        true,
    );
    let purple_manual_grid_est = targets
        .purple
        .map(|grid| grid as f64)
        .unwrap_or(purple_grid_est);
    let gold_manual_grid_est = targets
        .gold
        .map(|grid| grid as f64)
        .unwrap_or(gold_grid_est);
    let (purple_mean, purple_variance) = quality_mean_and_variance_for_prior(
        cp,
        4,
        purple,
        purple_manual_grid_est,
        price.purple_mean,
        price.purple_variance,
    );
    let (gold_mean, gold_variance) = quality_mean_and_variance_for_prior(
        cp,
        5,
        gold,
        gold_manual_grid_est,
        price.gold_mean,
        price.gold_variance,
    );
    let qualities = [
        (
            gw,
            price.greenwhite_mean,
            fallback_value_variance(price.greenwhite_mean),
        ),
        (
            blue,
            price.blue_mean,
            quality_variance(price.blue_mean, price.blue_variance),
        ),
        (purple, purple_mean, purple_variance),
        (gold, gold_mean, gold_variance),
        (
            red,
            price.red_mean,
            quality_variance(price.red_mean, price.red_variance),
        ),
    ];
    let n = population_count as f64;
    let mut total_mean = 0.0;
    let mut total_second = 0.0;
    for (count, mean, variance) in qualities {
        if count <= 0 || !mean.is_finite() {
            continue;
        }
        total_mean += count as f64 * mean;
        total_second += count as f64 * (variance.max(0.0) + mean * mean);
    }
    let population_mean = total_mean / n;
    if !population_mean.is_finite() {
        return 0.0;
    }
    let population_variance = (total_second / n - population_mean * population_mean).max(0.0);
    let mut log_w = 0.0;
    for sample in &cp.value_samples {
        if sample.count <= 0 || sample.count > modeled_count || !sample.avg_value.is_finite() {
            continue;
        }
        let sample_count = sample.count.min(population_count) as f64;
        if sample_count <= 0.0 {
            continue;
        }
        let finite_population_correction = if population_count > 1 {
            ((population_count - sample.count.min(population_count)) as f64
                / (population_count - 1) as f64)
                .max(0.15)
        } else {
            1.0
        };
        let model_sigma = (population_variance / sample_count * finite_population_correction)
            .max(0.0)
            .sqrt();
        let noise_floor = population_mean
            .abs()
            .max(sample.avg_value.abs())
            .mul_add(0.18, 0.0)
            .max(300.0);
        let sigma = model_sigma.max(noise_floor);
        let z = (sample.avg_value - population_mean) / sigma;
        log_w += -0.5 * z * z * VALUE_SAMPLE_EVIDENCE_WEIGHT;
    }
    log_w
}

fn quality_variance(mean: f64, variance: f64) -> f64 {
    if variance.is_finite() && variance > 0.0 {
        variance
    } else {
        fallback_value_variance(mean)
    }
}

fn fallback_value_variance(mean: f64) -> f64 {
    let sigma = mean.abs().mul_add(0.7, 250.0).max(350.0);
    sigma * sigma
}

fn quality_mean_and_variance_for_prior(
    cp: &CalcParams,
    quality: i32,
    count: i32,
    grid_est: f64,
    fallback_mean: f64,
    fallback_variance: f64,
) -> (f64, f64) {
    if count > 0
        && let Some(total) = manual_quality_total_value(cp, quality, count, grid_est)
    {
        let mean = total / count as f64;
        return (mean, fallback_value_variance(mean));
    }
    let mean = finite_nonnegative(fallback_mean);
    (mean, quality_variance(mean, fallback_variance))
}

fn quality_combo_variance(
    cp: &CalcParams,
    quality: i32,
    count: i32,
    grid_est: f64,
    target_grid: Option<i32>,
    fallback_variance: f64,
) -> f64 {
    if count <= 0 {
        return 0.0;
    }
    let manual_grid_est = target_grid
        .map(|target_grid| target_grid as f64)
        .unwrap_or(grid_est);
    if let Some(total) = manual_quality_total_value(cp, quality, count, manual_grid_est) {
        let mean = total / count as f64;
        return count as f64 * fallback_value_variance(mean);
    }
    count as f64 * fallback_variance.max(0.0)
}

fn manual_quality_unit_value(cp: &CalcParams, quality: i32, average_grid_est: f64) -> Option<f64> {
    let grid_est = if average_grid_est.is_finite() && average_grid_est > 0.0 {
        average_grid_est
    } else {
        fallback_avg_grid(quality)
    };
    manual_quality_total_value(cp, quality, 1, grid_est)
}

fn manual_quality_total_value(
    cp: &CalcParams,
    quality: i32,
    count: i32,
    grid_est: f64,
) -> Option<f64> {
    if count <= 0 {
        return None;
    }
    let (per_item, per_grid) = match quality {
        4 => (cp.manual_purple_per_item, cp.manual_purple_per_grid),
        5 => (cp.manual_gold_per_item, cp.manual_gold_per_grid),
        _ => return None,
    };
    let item_total = valid_manual_value(per_item).map(|value| count as f64 * value);
    let grid_total = valid_manual_value(per_grid)
        .and_then(|value| grid_est.is_finite().then_some(grid_est.max(0.0) * value));
    match (item_total, grid_total) {
        (Some(item), Some(grid)) => Some((item + grid) / 2.0),
        (Some(item), None) => Some(item),
        (None, Some(grid)) => Some(grid),
        (None, None) => None,
    }
}

fn valid_manual_value(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v > 0.0)
}

fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn beam_score(state: &BeamState, target_sum: f64, target_grid: Option<i32>) -> f64 {
    let value_error = abs(state.sum - target_sum) / target_sum.max(1.0);
    let grid_error = target_grid
        .map(|g| abs(state.grid as f64 - g as f64) / (g.max(1) as f64))
        .unwrap_or(0.0);
    if target_grid.is_some() {
        grid_error * 3.0 + value_error * 0.6 - 0.15 * state.log_map_weight
    } else {
        value_error + 0.4 * grid_error - 0.15 * state.log_map_weight
    }
}

fn add_top(
    selected: &mut Vec<ItemRecord>,
    seen: &mut HashSet<String>,
    seq: &[ItemRecord],
    n: usize,
) {
    for item in seq.iter().take(n.min(seq.len())) {
        if seen.insert(item.item_id.clone()) {
            selected.push(item.clone());
        }
    }
}

fn abs(value: f64) -> f64 {
    value.abs()
}

fn fcmp(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn combo_probability_order(a: &ComboResult, b: &ComboResult) -> Ordering {
    fcmp(b.probability, a.probability)
        .then_with(|| a.greenwhite_count.cmp(&b.greenwhite_count))
        .then_with(|| a.blue_count.cmp(&b.blue_count))
        .then_with(|| a.purple_count.cmp(&b.purple_count))
        .then_with(|| a.gold_count.cmp(&b.gold_count))
        .then_with(|| a.red_count.cmp(&b.red_count))
        .then_with(|| fcmp(a.total_grid_est, b.total_grid_est))
        .then_with(|| fcmp(a.final_value, b.final_value))
}

pub fn load_core(
    data_path: impl AsRef<Path>,
    static_path: impl AsRef<Path>,
) -> Result<BidKingCore> {
    let static_data = StaticData::load(static_path)?;
    let mut loader = DataLoader::new(static_data.clone());
    loader.load_merged_csv(data_path)?;
    Ok(BidKingCore::new(loader, static_data))
}

pub fn load_embedded_static_data() -> Result<StaticData> {
    StaticData::from_json_str(EMBEDDED_STATIC_DATA)
}

pub fn load_embedded_core() -> Result<BidKingCore> {
    let static_data = load_embedded_static_data()?;
    let mut loader = DataLoader::new(static_data.clone());
    loader.load_merged_csv_bytes(EMBEDDED_MERGED_CSV, "embedded calculator data")?;
    Ok(BidKingCore::new(loader, static_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_static_data() -> StaticData {
        StaticData {
            drop_weights: HashMap::new(),
            quality_p50_default: HashMap::new(),
            nest_weighted_prices: HashMap::new(),
            map_to_nest: HashMap::new(),
            map_names: HashMap::new(),
        }
    }

    fn static_data_with_tier_weights(weights: Vec<f64>) -> StaticData {
        let mut data = empty_static_data();
        data.drop_weights.insert("test".to_string(), weights);
        data
    }

    fn item(id: &str, quality: i32, grid_size: i32, value: f64) -> ItemRecord {
        ItemRecord {
            record_type: "ITEM".to_string(),
            item_id: id.to_string(),
            name: id.to_string(),
            quality,
            value,
            shape: String::new(),
            drop_id: String::new(),
            ref_id: String::new(),
            weight: 0.0,
            ref_type: String::new(),
            grid_size,
            grid_trusted: true,
        }
    }

    fn add_item(loader: &mut DataLoader, item: ItemRecord) {
        loader
            .items_by_quality
            .entry(item.quality)
            .or_default()
            .push(item.clone());
        loader.items_by_id.insert(item.item_id.clone(), item);
    }

    #[test]
    fn map_grid_stats_keep_quality_specific_twenty_grid_mean() {
        let mut loader = DataLoader::new(empty_static_data());
        add_item(&mut loader, item("blue20", 3, 20, 10_000.0));
        loader.drop_graph.insert(
            "map".to_string(),
            vec![DropEdge {
                ref_id: "blue20".to_string(),
                weight: 1.0,
                _ref_type: "ITEM".to_string(),
                p: 1.0,
            }],
        );

        let stats = loader.get_map_grid_stats_by_quality(Some("map"), 3, 3.0, 3.0);

        assert_eq!(stats.mean, 20.0);
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn map_absent_quality_does_not_fall_back_to_global_pool() {
        let mut loader = DataLoader::new(empty_static_data());
        add_item(&mut loader, item("blue20", 3, 20, 10_000.0));
        add_item(&mut loader, item("purple4", 4, 4, 20_000.0));
        add_item(&mut loader, item("red2", 6, 2, 50_000.0));
        loader.drop_graph.insert(
            "map".to_string(),
            vec![
                DropEdge {
                    ref_id: "blue20".to_string(),
                    weight: 1.0,
                    _ref_type: "ITEM".to_string(),
                    p: 1.0,
                },
                DropEdge {
                    ref_id: "red2".to_string(),
                    weight: 1.0,
                    _ref_type: "ITEM".to_string(),
                    p: 1.0,
                },
            ],
        );

        let purple_sizes = loader.get_map_grid_sizes_by_quality(Some("map"), 4);
        let valid_purple = build_valid_color_counts(&purple_sizes, 1, None, 0, None, None);
        let purple_grid_stats = loader.get_map_grid_stats_by_quality(Some("map"), 4, 4.0, 3.0);
        let purple_stats = loader.get_quality_stats(4, Some("map"));
        let purple_dist = loader.get_grid_value_distribution(4, 1, Some("map"));
        let probs = loader.get_map_quality_probs(Some("map"), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let impossible_red_value = loader.get_grid_conditioned_value(6, 1, Some(4), Some("map"));
        let possible_red_value = loader.get_grid_conditioned_value(6, 1, Some(2), Some("map"));

        assert!(purple_sizes.is_empty());
        assert!(valid_at(&valid_purple, 0));
        assert!(!valid_at(&valid_purple, 1));
        assert_eq!(purple_grid_stats.count, 0);
        assert_eq!(purple_stats.count, 0);
        assert!(purple_dist.is_none());
        assert!(probs.pb > 0.0);
        assert_eq!(probs.pp, 0.0);
        assert_eq!(probs.pg, 0.0);
        assert!(probs.pr > 0.0);
        assert!(impossible_red_value.is_none());
        assert_eq!(possible_red_value, Some(50_000.0));
    }

    #[test]
    fn unknown_drop_node_falls_back_to_global_pool() {
        let mut loader = DataLoader::new(empty_static_data());
        add_item(&mut loader, item("purple4", 4, 4, 20_000.0));
        loader.drop_graph.insert(
            "known".to_string(),
            vec![DropEdge {
                ref_id: "purple4".to_string(),
                weight: 1.0,
                _ref_type: "ITEM".to_string(),
                p: 1.0,
            }],
        );

        let resolved = loader.resolve_drop_to_items(Some("missing"));
        let purple_sizes = loader.get_map_grid_sizes_by_quality(Some("missing"), 4);
        let purple_stats = loader.get_quality_stats(4, Some("missing"));

        assert!(resolved.is_none());
        assert_eq!(purple_sizes, vec![4]);
        assert_eq!(purple_stats.count, 1);
    }

    #[test]
    fn loading_merged_csv_replaces_previous_loaded_data() {
        let mut loader = DataLoader::new(empty_static_data());
        let csv = b"record_type,item_id,name,quality,base_value,shape,drop_id,ref_id,weight,ref_type\nITEM,i1,one,4,20000,11,,,,\n";

        loader.load_merged_csv_bytes(csv, "first").unwrap();
        loader.load_merged_csv_bytes(csv, "second").unwrap();

        assert_eq!(loader.items_by_quality.get(&4).unwrap().len(), 1);
        assert_eq!(loader.items_by_id.len(), 1);
    }

    #[test]
    fn value_sample_prior_uses_total_grid_allocated_manual_value() {
        let low_total_grid = CalcParams {
            total_count: 8,
            total_grid_target: Some(16.0),
            manual_gold_per_grid: Some(10_000.0),
            value_samples: vec![ValueSample {
                count: 8,
                avg_value: 70_000.0,
            }],
            ..Default::default()
        };
        let high_total_grid = CalcParams {
            total_grid_target: Some(56.0),
            ..low_total_grid.clone()
        };
        let price = PriceModel {
            greenwhite_mean: 400.0,
            blue_mean: 4_000.0,
            purple_mean: 20_000.0,
            gold_mean: 25_000.0,
            red_mean: 50_000.0,
            blue_variance: 1.0,
            purple_variance: 1.0,
            gold_variance: 1.0,
            red_variance: 1.0,
        };

        let low_log = value_sample_prior_log(
            &low_total_grid,
            8,
            0,
            0,
            0,
            8,
            0,
            0.0,
            0.0,
            0.0,
            22.4,
            0.0,
            &price,
        );
        let high_log = value_sample_prior_log(
            &high_total_grid,
            8,
            0,
            0,
            0,
            8,
            0,
            0.0,
            0.0,
            0.0,
            22.4,
            0.0,
            &price,
        );

        assert!(
            high_log > low_log + 0.2,
            "sample prior should follow total-grid allocated manual value, low={low_log}, high={high_log}"
        );
    }

    #[test]
    fn total_grid_allocation_preserves_exact_residual_sum() {
        let cp = CalcParams {
            total_count: 7,
            total_grid_target: Some(23.0),
            ..Default::default()
        };

        let targets = build_pricing_grid_targets(&cp, 3, 2, 2, 0, 0, 6.0, 4.0, 4.0, 0.0, 0.0, true);

        assert_eq!(
            targets.gw.unwrap() + targets.blue.unwrap() + targets.purple.unwrap(),
            23
        );
        assert!(targets.gw.unwrap() >= 3);
        assert!(targets.blue.unwrap() >= 2);
        assert!(targets.purple.unwrap() >= 2);
    }

    #[test]
    fn total_grid_allocation_clamps_to_feasible_residual_bounds() {
        let slots = [
            ResidualGridSlot {
                index: 0,
                quality: Some(4),
                count: 2,
                model_grid: 4.0,
            },
            ResidualGridSlot {
                index: 1,
                quality: Some(5),
                count: 1,
                model_grid: 2.0,
            },
        ];

        let low = allocate_residual_grid_targets(1.0, &slots);
        let high = allocate_residual_grid_targets(1_000.0, &slots);

        assert_eq!(low.iter().map(|(_, grid)| *grid).sum::<i32>(), 3);
        assert_eq!(high.iter().map(|(_, grid)| *grid).sum::<i32>(), 42);
    }

    #[test]
    fn validation_rejects_impossible_global_grid_fields() {
        let too_small_total = CalcParams {
            total_count: 10,
            total_grid_target: Some(9.0),
            ..Default::default()
        };
        let zero_average = CalcParams {
            total_count: 10,
            avg_grid_all: Some(0.0),
            ..Default::default()
        };

        assert!(
            format!("{:#}", validate_calc_params(&too_small_total).unwrap_err()).contains("总格数")
        );
        assert!(
            format!("{:#}", validate_calc_params(&zero_average).unwrap_err())
                .contains("全部品均格")
        );
    }

    #[test]
    fn high_quality_total_can_run_without_total_count() {
        let static_data = static_data_with_tier_weights(vec![0.0, 0.0, 1.0, 2.0, 2.0, 1.0]);
        let mut loader = DataLoader::new(static_data.clone());
        add_item(&mut loader, item("purple", 4, 2, 20_000.0));
        add_item(&mut loader, item("gold", 5, 3, 40_000.0));
        add_item(&mut loader, item("red", 6, 4, 120_000.0));
        let mut core = BidKingCore::new(loader, static_data);
        let cp = CalcParams {
            tier: "test".to_string(),
            map_nest_id: "map".to_string(),
            total_count: 0,
            high_quality_count: Some(6),
            safety_factor: 1.0,
            max_show: 10,
            ..Default::default()
        };

        let results = core
            .run(cp.clone())
            .expect("紫金红总数 alone should be enough for high-quality mode");
        let (p25, p50, p75) = core.price_range_for_last_run(&cp);

        assert!(!results.is_empty());
        assert!(
            results
                .iter()
                .all(|r| r.greenwhite_count == 0 && r.blue_count == 0)
        );
        assert!(
            results
                .iter()
                .all(|r| r.purple_count + r.gold_count + r.red_count == 6)
        );
        assert!(p25 > 0.0 && p50 >= p25 && p75 >= p50);
    }

    #[test]
    fn high_quality_only_ignores_global_average_without_total_count() {
        let cp = normalize_calc_params(CalcParams {
            total_count: 0,
            high_quality_count: Some(3),
            avg_grid_all: Some(2.4),
            ..Default::default()
        });

        assert_eq!(cp.avg_grid_all, None);
        validate_calc_params(&cp).unwrap();
    }

    #[test]
    fn high_quality_only_still_rejects_global_total_grid_without_total_count() {
        let cp = normalize_calc_params(CalcParams {
            total_count: 0,
            high_quality_count: Some(3),
            total_grid_target: Some(8.0),
            ..Default::default()
        });

        let err = validate_calc_params(&cp).unwrap_err();
        assert!(format!("{err:#}").contains("总格数需要总件数"));
    }

    #[test]
    fn high_quality_only_rejects_low_tier_constraints_without_total_count() {
        let cp = CalcParams {
            total_count: 0,
            high_quality_count: Some(6),
            blue_count: Some(1),
            ..Default::default()
        };

        let err = validate_calc_params(&cp).unwrap_err();
        assert!(format!("{err:#}").contains("蓝件数需要总件数"));
    }

    #[test]
    fn run_errors_when_all_candidate_weights_are_non_finite() {
        let static_data = static_data_with_tier_weights(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let loader = DataLoader::new(static_data.clone());
        let mut core = BidKingCore::new(loader, static_data);
        let cp = CalcParams {
            tier: "test".to_string(),
            total_count: 1,
            min_gw: 1,
            safety_factor: 1.0,
            max_show: 10,
            ..Default::default()
        };

        let err = core
            .run(cp)
            .expect_err("zero low-tier probability should not produce a fake uniform result");
        assert!(
            format!("{err:#}").contains("没有找到概率有效的组合"),
            "{err:#}"
        );
    }
}
