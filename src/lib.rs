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
pub const EMBEDDED_DATA_VERSION: &str = "auctionanalyzer-4.12.2";

const EMBEDDED_STATIC_DATA: &str = include_str!("../data/auctionanalyzer-4.12.2/static_data.json");
const EMBEDDED_MERGED_CSV: &[u8] = include_bytes!(
    "../data/auctionanalyzer-4.12.2/resources/MapBidCalculator.calculator_data_merged.csv"
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
    pub revealed_count: Option<i32>,
    pub revealed_total_value: Option<f64>,
    pub manual_purple_per_item: Option<f64>,
    pub manual_purple_per_grid: Option<f64>,
    pub manual_gold_per_item: Option<f64>,
    pub manual_gold_per_grid: Option<f64>,
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
            revealed_count: None,
            revealed_total_value: None,
            manual_purple_per_item: None,
            manual_purple_per_grid: None,
            manual_gold_per_item: None,
            manual_gold_per_grid: None,
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

#[derive(Debug, Clone, Default)]
struct PricingGridTargets {
    gw: Option<i32>,
    blue: Option<i32>,
    purple: Option<i32>,
    gold: Option<i32>,
    red: Option<i32>,
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
            let mut usable: Vec<ItemRecord> = items
                .iter()
                .filter(|i| i.grid_trusted && i.grid_size > 0 && i.grid_size <= 18 && i.value > 0.0)
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
        if total > 0.0 {
            for value in out.values_mut() {
                *value /= total;
            }
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
            if let Some(item) = self.items_by_id.get(&item_id) {
                if (3..=6).contains(&item.quality) {
                    *by_quality.entry(item.quality).or_insert(0.0) += weight;
                    mass += weight;
                }
            }
        }
        if mass <= 0.0 {
            self.map_prob_cache.insert(key, fallback.clone());
            return fallback;
        }
        let raw = apply_prob_floor(
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
        let default = QualityGridStats {
            mean: fallback_mean.clamp(1.0, 18.0),
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
            .filter(|i| i.grid_trusted && i.grid_size > 0 && i.grid_size <= 18)
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
            mean: mean.clamp(1.0, 18.0),
            variance: variance.clamp(0.25, 36.0),
            count: pairs.len(),
            effective_count: effective,
        };
        self.map_grid_stats_cache.insert(key, result);
        result
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
        if count <= 0 || target_grid < count || target_grid > 18 * count {
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
            .filter(|i| i.grid_trusted && i.grid_size > 0 && i.grid_size <= 18 && i.value > 0.0)
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
            pool = items
                .iter()
                .map(|i| (i.grid_size as usize, i.value, 1.0))
                .collect();
        }
        let max_grid = (18 * count) as usize;
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
    high_value_cache: HashMap<String, f64>,
}

impl BidKingCore {
    pub fn new(loader: DataLoader, static_data: StaticData) -> Self {
        Self {
            loader,
            static_data,
            raw_results: Vec::new(),
            high_value_cache: HashMap::new(),
        }
    }

    pub fn run(&mut self, mut cp: CalcParams) -> Result<Vec<ComboResult>> {
        self.high_value_cache.clear();
        if cp.total_grid_target.is_none() && cp.avg_grid_all.is_some() && cp.total_count > 0 {
            cp.total_grid_target = cp.avg_grid_all.map(|avg| avg * cp.total_count as f64);
        }
        if cp.high_quality_count.unwrap_or(0) > 0 {
            return self.run_high_quality_only(&cp);
        }
        let mut results = Vec::new();
        let tier_weights = self.tier_weights(&cp.tier)?;
        let probs = self
            .loader
            .get_map_quality_probs(Some(&cp.map_nest_id), &tier_weights);
        let grid_stats = self.build_grid_stats_model(Some(&cp.map_nest_id));
        let price_model = self.build_price_model(&cp, &tier_weights);
        let gw_grid_value = round_opt(cp.gw_grid);
        let blue_grid_value = round_opt(cp.blue_grid);
        let purple_grid_value = round_opt(cp.purple_grid);
        let gold_grid_value = round_opt(cp.gold_grid);
        let red_grid_value = round_opt(cp.red_grid);

        let valid_gw = build_valid_color_counts(
            cp.total_count,
            cp.gw_count,
            cp.min_gw,
            cp.gw_grid,
            cp.gw_avg,
        );
        let valid_b = build_valid_color_counts(
            cp.total_count,
            cp.blue_count,
            cp.min_blue,
            cp.blue_grid,
            cp.blue_avg,
        );
        let valid_p = build_valid_color_counts(
            cp.total_count,
            cp.purple_count,
            cp.min_purple,
            cp.purple_grid,
            cp.purple_avg,
        );
        let valid_g = build_valid_color_counts(
            cp.total_count,
            cp.gold_count,
            cp.min_gold,
            cp.gold_grid,
            cp.gold_avg,
        );
        let valid_r = build_valid_color_counts(
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
            for b in inclusive_range(count_range(
                high_count,
                cp.blue_count,
                cp.min_blue,
                req_p + req_g + req_r,
            )) {
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
        let mut filtered = self.finalize_combos(results);
        self.populate_combo_values(&mut filtered, &cp, price_model, grid_stats, true, true);
        Ok(filtered)
    }

    fn run_high_quality_only(&mut self, cp: &CalcParams) -> Result<Vec<ComboResult>> {
        let total = cp.high_quality_count.unwrap_or(0);
        let mut results = Vec::new();
        let tier_weights = self.tier_weights(&cp.tier)?;
        let probs = self
            .loader
            .get_map_quality_probs(Some(&cp.map_nest_id), &tier_weights);
        let denom = probs.pp + probs.pg + probs.pr;
        let (pp, pg, pr) = (probs.pp / denom, probs.pg / denom, probs.pr / denom);
        let grid_stats = self.build_grid_stats_model(Some(&cp.map_nest_id));
        let price_model = self.build_price_model(cp, &tier_weights);
        let valid_p = build_valid_color_counts(
            total,
            cp.purple_count,
            cp.min_purple,
            cp.purple_grid,
            cp.purple_avg,
        );
        let valid_g =
            build_valid_color_counts(total, cp.gold_count, cp.min_gold, cp.gold_grid, cp.gold_avg);
        let valid_r =
            build_valid_color_counts(total, cp.red_count, cp.min_red, cp.red_grid, cp.red_avg);
        let req_g = required_count(cp.gold_count, cp.min_gold);
        let req_r = required_count(cp.red_count, cp.min_red);
        for p in inclusive_range(count_range(
            total,
            cp.purple_count,
            cp.min_purple,
            req_g + req_r,
        )) {
            if !valid_at(&valid_p, p) {
                continue;
            }
            for g in inclusive_range(count_range(total - p, cp.gold_count, cp.min_gold, req_r)) {
                if !valid_at(&valid_g, g) {
                    continue;
                }
                let r = total - p - g;
                if r < 0 || r > total || !valid_at(&valid_r, r) {
                    continue;
                }
                let mut log_w = log_fact(total) - log_fact(p) - log_fact(g) - log_fact(r);
                if p > 0 {
                    log_w += p as f64 * pp.ln();
                }
                if g > 0 {
                    log_w += g as f64 * pg.ln();
                }
                if r > 0 {
                    log_w += r as f64 * pr.ln();
                }
                log_w += red_count_caution_log(cp, r, total, pr);
                let grid_p = cp
                    .purple_grid
                    .unwrap_or(p as f64 * cp.purple_avg.unwrap_or(grid_stats.purple_mean));
                let grid_g = cp
                    .gold_grid
                    .unwrap_or(g as f64 * cp.gold_avg.unwrap_or(grid_stats.gold_mean));
                let grid_r = cp
                    .red_grid
                    .unwrap_or(r as f64 * cp.red_avg.unwrap_or(grid_stats.red_mean));
                log_w += avg_prior_log(p, grid_p, cp.purple_avg, 0.45);
                log_w += avg_prior_log(g, grid_g, cp.gold_avg, 0.55);
                log_w += avg_prior_log(r, grid_r, cp.red_avg, 0.7);
                results.push(ComboResult {
                    purple_count: p,
                    gold_count: g,
                    red_count: r,
                    log_w,
                    total_grid_est: grid_p + grid_g + grid_r,
                    greenwhite_grid_est: 0.0,
                    blue_grid_est: 0.0,
                    purple_grid_est: grid_p,
                    gold_grid_est: grid_g,
                    red_grid_est: grid_r,
                    purple_grid_value: round_opt(cp.purple_grid),
                    gold_grid_value: round_opt(cp.gold_grid),
                    red_grid_value: round_opt(cp.red_grid),
                    ..Default::default()
                });
            }
        }
        let mut filtered = self.finalize_combos(results);
        self.populate_combo_values(&mut filtered, cp, price_model, grid_stats, false, false);
        Ok(filtered)
    }

    fn finalize_combos(&mut self, mut combos: Vec<ComboResult>) -> Vec<ComboResult> {
        if combos.is_empty() {
            self.raw_results.clear();
            return vec![];
        }
        let max_l = combos
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = combos.iter().map(|c| (c.log_w - max_l).exp()).sum();
        if denom <= 0.0 {
            combos.sort_by(|a, b| fcmp(b.log_w, a.log_w));
            self.raw_results = combos.clone();
            return combos;
        }
        for combo in &mut combos {
            combo.probability = (combo.log_w - max_l).exp() / denom;
        }
        self.raw_results = combos.clone();
        self.raw_results
            .sort_by(|a, b| fcmp(b.probability, a.probability));
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
        filtered.sort_by(|a, b| fcmp(b.probability, a.probability));
        filtered
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
        for combo in combos {
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
                value += self.high_value_for_quality(
                    3,
                    b,
                    price.blue_mean,
                    targets.blue,
                    Some(&cp.map_nest_id),
                );
            }
            value += self.high_value_for_quality(
                4,
                p,
                price.purple_mean,
                targets.purple,
                Some(&cp.map_nest_id),
            );
            value += self.high_value_for_quality(
                5,
                g,
                price.gold_mean,
                targets.gold,
                Some(&cp.map_nest_id),
            );
            value += self.high_value_for_quality(
                6,
                r,
                price.red_mean,
                targets.red,
                Some(&cp.map_nest_id),
            );
            if include_low_tiers {
                value += gw as f64 * price.greenwhite_mean;
            }
            combo.final_value = value;
            combo.high_variance = if include_low_tiers {
                b as f64 * price.blue_variance
            } else {
                0.0
            } + p as f64 * price.purple_variance
                + g as f64 * price.gold_variance
                + r as f64 * price.red_variance;
            combo.greenwhite_grid_est = grid_gw;
            combo.blue_grid_est = grid_b;
            combo.purple_grid_est = grid_p;
            combo.gold_grid_est = grid_g;
            combo.red_grid_est = grid_r;
            combo.total_grid_est = grid_gw + grid_b + grid_p + grid_g + grid_r;
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
            if let Some(v) = conditioned {
                if v.is_finite() && v > 0.0 {
                    value = v;
                }
            }
        }
        let composed = self.compose_items_for_quality(quality, count, value, target_grid, drop_id);
        if let Some(composed) = composed {
            if composed.is_finite() {
                value = if let Some(conditioned) = conditioned {
                    conditioned * 0.4 + composed * 0.6
                } else {
                    composed
                };
            }
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
            .filter(|i| i.grid_trusted)
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
        let purple_mean = cp
            .manual_purple_per_item
            .or_else(|| cp.manual_purple_per_grid.map(|v| v * grid.purple_mean))
            .unwrap_or_else(|| self.quality_mean_or_fallback(purple, 4));
        let gold_mean = cp
            .manual_gold_per_item
            .or_else(|| cp.manual_gold_per_grid.map(|v| v * grid.gold_mean))
            .unwrap_or_else(|| self.quality_mean_or_fallback(gold, 5));
        PriceModel {
            greenwhite_mean: self.greenwhite_mean(Some(&cp.map_nest_id), tier_weights),
            blue_mean: self.quality_mean_or_fallback(blue, 3),
            purple_mean,
            gold_mean,
            red_mean: self.quality_mean_or_fallback(red, 6),
            blue_variance: blue.variance,
            purple_variance: purple.variance,
            gold_variance: gold.variance,
            red_variance: red.variance,
        }
    }

    fn greenwhite_mean(&self, nest_id: Option<&str>, tier_weights: &[f64]) -> f64 {
        if let Some(prices) = self
            .static_data
            .nest_weighted_prices
            .get(nest_id.unwrap_or(""))
        {
            if prices.len() >= 2 {
                let weight = tier_weights[0] + tier_weights[1];
                if weight > 0.0 {
                    return (prices[0] * tier_weights[0] + prices[1] * tier_weights[1]) / weight;
                }
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
        if results.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted_values: Vec<(f64, f64)> = results
            .iter()
            .map(|r| {
                (
                    apply_revealed_value_adjustment(r.final_value, cp),
                    r.probability,
                )
            })
            .collect();
        sorted_values.sort_by(|a, b| fcmp(a.0, b.0));
        let p50 = weighted_probability_quantile(&sorted_values, 0.5);
        let mut p25 = weighted_probability_quantile(&sorted_values, 0.25);
        let mut p75 = weighted_probability_quantile(&sorted_values, 0.75);
        let ratio = unrevealed_ratio(cp);
        let variance: f64 = results
            .iter()
            .map(|r| r.high_variance * r.probability)
            .sum::<f64>()
            * ratio
            * ratio;
        let sd = variance.max(0.0).sqrt();
        let lower = p50 - 0.3373 * sd;
        p25 = (p25 * 0.65).max(p25.min(lower));
        p75 = p75.max(p50 + 0.3373 * sd);
        (p25, p50, p75)
    }

    pub fn combo_composition_lines(&mut self, combo: &ComboResult, cp: &CalcParams) -> Vec<String> {
        let tier_weights = self.tier_weights(&cp.tier).unwrap_or_default();
        let price = self.build_price_model(cp, &tier_weights);
        let qualities = [
            (
                3,
                "蓝(Q3)",
                combo.blue_count,
                price.blue_mean,
                combo.blue_grid_value,
            ),
            (
                4,
                "紫(Q4)",
                combo.purple_count,
                price.purple_mean,
                combo.purple_grid_value,
            ),
            (
                5,
                "金(Q5)",
                combo.gold_count,
                price.gold_mean,
                combo.gold_grid_value,
            ),
            (
                6,
                "红(Q6)",
                combo.red_count,
                price.red_mean,
                combo.red_grid_value,
            ),
        ];
        let mut lines = Vec::new();
        for (quality, label, count, mean, grid) in qualities {
            if count <= 0 {
                lines.push(format!("{label}: --"));
                continue;
            }
            let target_sum =
                self.high_value_for_quality(quality, count, mean, grid, Some(&cp.map_nest_id));
            let pool = self.choose_pool(
                quality,
                target_sum / count as f64,
                Some(&cp.map_nest_id),
                grid,
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
        if (1..=18).contains(&grid) {
            item.grid_size = grid;
            item.grid_trusted = true;
        }
    }
}

fn apply_prob_floor(
    raw: MapQualityProbs,
    tier_fallback: Option<&MapQualityProbs>,
) -> MapQualityProbs {
    let fallback = tier_fallback.unwrap_or(&raw);
    let floors = [
        0.001_f64.max(fallback.pb * 0.1),
        0.001_f64.max(fallback.pp * 0.1),
        0.001_f64.max(fallback.pg * 0.1),
        0.001_f64.max(fallback.pr * 0.1),
    ];
    let mut vals = [
        raw.pb.max(floors[0]),
        raw.pp.max(floors[1]),
        raw.pg.max(floors[2]),
        raw.pr.max(floors[3]),
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
        gw: implied_target_grid(gw, cp.gw_grid, cp.gw_avg),
        blue: implied_target_grid(b, cp.blue_grid, cp.blue_avg),
        purple: implied_target_grid(p, cp.purple_grid, cp.purple_avg),
        gold: implied_target_grid(g, cp.gold_grid, cp.gold_avg),
        red: implied_target_grid(r, cp.red_grid, cp.red_avg),
    };
    if use_total_grid_target {
        if let Some(total_grid_target) = cp.total_grid_target {
            let slots = [
                ("gw", gw, grid_gw, targets.gw),
                ("blue", b, grid_b, targets.blue),
                ("purple", p, grid_p, targets.purple),
                ("gold", g, grid_g, targets.gold),
                ("red", r, grid_r, targets.red),
            ];
            let mut known = 0.0;
            let mut unknown_count = 0;
            let mut model_sum = 0.0;
            let mut fallback_sum = 0.0;
            for (_, count, model_grid, target) in slots {
                if let Some(target) = target {
                    known += target as f64;
                } else if count > 0 {
                    unknown_count += 1;
                    model_sum += model_grid.max(0.0001);
                    fallback_sum += count.max(1) as f64;
                }
            }
            let residual = total_grid_target - known;
            if unknown_count > 0 && residual >= 0.0 {
                if model_sum <= 0.0 {
                    model_sum = fallback_sum;
                }
                let alloc = |count: i32, model_grid: f64| -> Option<i32> {
                    if count <= 0 {
                        return None;
                    }
                    let ratio = if model_sum > 0.0 {
                        model_grid.max(0.0001) / model_sum
                    } else {
                        1.0 / unknown_count as f64
                    };
                    Some(clamp_grid_target(count, round_to_i32(residual * ratio)))
                };
                if targets.gw.is_none() {
                    targets.gw = alloc(gw, grid_gw);
                }
                if targets.blue.is_none() {
                    targets.blue = alloc(b, grid_b);
                }
                if targets.purple.is_none() {
                    targets.purple = alloc(p, grid_p);
                }
                if targets.gold.is_none() {
                    targets.gold = alloc(g, grid_g);
                }
                if targets.red.is_none() {
                    targets.red = alloc(r, grid_r);
                }
            }
        }
    }
    targets
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
            valid[fixed_count as usize] = is_valid_color_count(fixed_count, grid, avg);
        }
        return valid;
    }
    for count in min_count.max(0)..=max_count {
        valid[count as usize] = is_valid_color_count(count, grid, avg);
    }
    valid
}

fn is_valid_color_count(count: i32, grid: Option<f64>, avg: Option<f64>) -> bool {
    let rounded_grid = grid.map(round_to_i32);
    if let Some(rounded_grid) = rounded_grid {
        if count == 0 {
            if rounded_grid == 0 {
                return avg.map(|v| v.abs() < 1e-9).unwrap_or(true);
            }
            return false;
        }
        if rounded_grid < count || rounded_grid > 18 * count {
            return false;
        }
        if avg.is_some() && !avg_match(rounded_grid, count, avg) {
            return false;
        }
    } else if let Some(avg) = avg {
        if count == 0 {
            return avg == 0.0;
        }
        if !avg_count_match(count, avg) {
            return false;
        }
    }
    true
}

fn avg_match(grid: i32, count: i32, avg: Option<f64>) -> bool {
    let Some(avg) = avg else {
        return true;
    };
    if count == 0 {
        return true;
    }
    if !avg_count_match(count, avg) {
        return false;
    }
    let target = (avg * 100.0 + 1e-7).floor() as i32;
    let got = (grid as f64 * 100.0 / count as f64 + 1e-7).floor() as i32;
    got == target
}

fn avg_count_match(count: i32, avg: f64) -> bool {
    if count <= 0 || count > 40 || !avg.is_finite() || avg < 0.0 {
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

fn implied_target_grid(count: i32, grid_in: Option<f64>, avg_in: Option<f64>) -> Option<i32> {
    if count <= 0 {
        return None;
    }
    if let Some(grid) = grid_in {
        if grid.is_finite() {
            return Some(round_to_i32(grid));
        }
    }
    if let Some(avg) = avg_in {
        if avg.is_finite() && avg > 0.0 {
            return Some((round_to_i32(count as f64 * avg)).clamp(count, 18 * count));
        }
    }
    None
}

fn clamp_grid_target(count: i32, grid: i32) -> i32 {
    if count <= 0 {
        0
    } else {
        grid.clamp(count, 18 * count)
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

fn apply_revealed_value_adjustment(modeled_value: f64, cp: &CalcParams) -> f64 {
    let Some(revealed_count) = cp.revealed_count else {
        return modeled_value;
    };
    let Some(revealed_total_value) = cp.revealed_total_value else {
        return modeled_value;
    };
    if revealed_count <= 0 {
        return modeled_value;
    }
    let modeled_count = cp
        .high_quality_count
        .filter(|v| *v > 0)
        .unwrap_or(cp.total_count);
    if modeled_count <= 0 {
        return modeled_value;
    }
    let revealed = revealed_count.min(modeled_count);
    let ratio = (modeled_count - revealed).max(0) as f64 / modeled_count as f64;
    revealed_total_value + modeled_value * ratio
}

fn unrevealed_ratio(cp: &CalcParams) -> f64 {
    let Some(revealed_count) = cp.revealed_count else {
        return 1.0;
    };
    if revealed_count <= 0 || cp.revealed_total_value.is_none() {
        return 1.0;
    }
    let modeled_count = cp
        .high_quality_count
        .filter(|v| *v > 0)
        .unwrap_or(cp.total_count);
    if modeled_count <= 0 {
        return 1.0;
    }
    let revealed = revealed_count.min(modeled_count);
    (modeled_count - revealed).max(0) as f64 / modeled_count as f64
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
