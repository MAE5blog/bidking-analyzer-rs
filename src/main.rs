use anyhow::{Context, Result};

use bidking_rs::{CalcParams, StaticData, importer, load_core, ocr};
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "bidking")]
#[command(about = "Open reimplementation of the BidKing auction analyzer core")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    calc: CalcArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    Calc(CalcArgs),
    ImportExe(ImportExeArgs),
    ListMaps(DataArgs),
    OcrImage(OcrImageArgs),
    OcrScreen(OcrScreenArgs),
}

#[derive(Debug, Args, Clone)]
struct CalcArgs {
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(
        long,
        default_value = "../decompiled_4_12_3/MapBidCalculator.calculator_data_merged.csv"
    )]
    data: PathBuf,
    #[arg(long, default_value = "../core_algorithm/static_data.json")]
    static_data: PathBuf,
    #[arg(long, default_value = "101")]
    tier: String,
    #[arg(long)]
    map_id: Option<String>,
    #[arg(long)]
    nest_id: Option<String>,
    #[arg(long, default_value_t = 30)]
    total: i32,
    #[arg(long)]
    total_grid: Option<f64>,
    #[arg(long)]
    avg_grid_all: Option<f64>,
    #[arg(long, default_value_t = 0.85)]
    safety: f64,
    #[arg(long, default_value_t = 10)]
    max_show: usize,
    #[arg(long)]
    min_value_floor: Option<f64>,
    #[arg(long)]
    high_quality_count: Option<i32>,
    #[arg(long)]
    gw_count: Option<i32>,
    #[arg(long, default_value_t = 0)]
    gw_min: i32,
    #[arg(long)]
    gw_grid: Option<f64>,
    #[arg(long)]
    gw_avg: Option<f64>,
    #[arg(long)]
    blue_count: Option<i32>,
    #[arg(long, default_value_t = 0)]
    blue_min: i32,
    #[arg(long)]
    blue_grid: Option<f64>,
    #[arg(long)]
    blue_avg: Option<f64>,
    #[arg(long)]
    purple_count: Option<i32>,
    #[arg(long, default_value_t = 0)]
    purple_min: i32,
    #[arg(long)]
    purple_grid: Option<f64>,
    #[arg(long)]
    purple_avg: Option<f64>,
    #[arg(long)]
    gold_count: Option<i32>,
    #[arg(long, default_value_t = 0)]
    gold_min: i32,
    #[arg(long)]
    gold_grid: Option<f64>,
    #[arg(long)]
    gold_avg: Option<f64>,
    #[arg(long)]
    red_count: Option<i32>,
    #[arg(long, default_value_t = 0)]
    red_min: i32,
    #[arg(long)]
    red_grid: Option<f64>,
    #[arg(long)]
    red_avg: Option<f64>,
}

#[derive(Debug, Args, Clone)]
struct DataArgs {
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long, default_value = "../core_algorithm/static_data.json")]
    static_data: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct OcrImageArgs {
    #[arg(long)]
    image: PathBuf,
    #[arg(long)]
    fallback_total: Option<i32>,
}

#[derive(Debug, Args, Clone)]
struct OcrScreenArgs {
    #[arg(long)]
    fallback_total: Option<i32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Calc(args)) => run_calc(args),
        Some(Command::ImportExe(args)) => run_import_exe(args),
        Some(Command::ListMaps(args)) => run_list_maps(args),
        Some(Command::OcrImage(args)) => run_ocr_image(args),
        Some(Command::OcrScreen(args)) => run_ocr_screen(args),
        None => run_calc(cli.calc),
    }
}

#[derive(Debug, Args)]
struct ImportExeArgs {
    #[arg(long)]
    exe: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    static_template: Option<PathBuf>,
}

fn run_import_exe(args: ImportExeArgs) -> Result<()> {
    let report = importer::import_exe(&args.exe, &args.out, args.static_template.as_deref())?;
    println!("bundle_entries={}", report.bundle_entries);
    for path in &report.extracted_files {
        println!("extracted_file={}", path.display());
    }
    for path in &report.extracted_resources {
        println!("extracted_resource={}", path.display());
    }
    if let Some(path) = &report.static_data {
        println!("static_data={}", path.display());
    } else {
        println!("static_data=not_written");
    }
    println!(
        "report={}",
        report.out_dir.join("import_report.json").display()
    );
    Ok(())
}

fn run_list_maps(args: DataArgs) -> Result<()> {
    let (_, static_data_path) =
        resolve_data_paths(args.data_dir.as_deref(), Path::new(""), &args.static_data)?;
    let static_data = StaticData::load(&static_data_path)?;
    let mut maps: Vec<_> = static_data.map_to_nest.iter().collect();
    maps.sort_by(|a, b| a.0.cmp(b.0));
    println!("map_id,nest_id,name");
    for (map_id, nest_id) in maps {
        let name = static_data
            .map_names
            .get(map_id)
            .cloned()
            .unwrap_or_default();
        println!("{map_id},{nest_id},{name}");
    }
    Ok(())
}

fn run_ocr_image(args: OcrImageArgs) -> Result<()> {
    let scan = ocr::scan_screenshot_with_ppocrv4_onnx(&args.image, args.fallback_total)?;
    print_ocr_scan(scan);
    Ok(())
}

fn run_ocr_screen(args: OcrScreenArgs) -> Result<()> {
    let scan = ocr::scan_primary_screen_with_ppocrv4_onnx(args.fallback_total)?;
    print_ocr_scan(scan);
    Ok(())
}

fn print_ocr_scan(scan: ocr::OcrScan) {
    println!("engine={}", scan.engine);
    println!("crop={}", scan.crop_path.display());
    println!("lines={}", scan.lines.len());
    for line in &scan.lines {
        println!("line={line}");
    }
    println!("parsed:");
    print_field("map_name", scan.parsed.map_name.as_deref());
    print_field("total_all", scan.parsed.total_all.as_deref());
    print_field(
        "high_quality_total_count",
        scan.parsed.high_quality_total_count.as_deref(),
    );
    print_field(
        "global_grid_total",
        scan.parsed.global_grid_total.as_deref(),
    );
    print_field("global_avg_grid", scan.parsed.global_avg_grid.as_deref());
    print_field("wg_count", scan.parsed.wg_count.as_deref());
    print_field("wg_grid", scan.parsed.wg_grid.as_deref());
    print_field("wg_avg", scan.parsed.wg_avg.as_deref());
    print_field("blue_count", scan.parsed.blue_count.as_deref());
    print_field("blue_grid", scan.parsed.blue_grid.as_deref());
    print_field("blue_avg", scan.parsed.blue_avg.as_deref());
    print_field("purple_count", scan.parsed.purple_count.as_deref());
    print_field("purple_grid", scan.parsed.purple_grid.as_deref());
    print_field("purple_avg", scan.parsed.purple_avg.as_deref());
    print_field("gold_count", scan.parsed.gold_count.as_deref());
    print_field("gold_grid", scan.parsed.gold_grid.as_deref());
    print_field("gold_avg", scan.parsed.gold_avg.as_deref());
    print_field("red_count", scan.parsed.red_count.as_deref());
    print_field("red_grid", scan.parsed.red_grid.as_deref());
    print_field("red_avg", scan.parsed.red_avg.as_deref());
    print_field("purple_avg_value", scan.parsed.purple_avg_value.as_deref());
    print_field("gold_avg_value", scan.parsed.gold_avg_value.as_deref());
    print_field("red_avg_value", scan.parsed.red_avg_value.as_deref());
    print_field("min_value_floor", scan.parsed.min_value_floor.as_deref());
    for sample in &scan.parsed.value_samples {
        println!(
            "value_sample=count:{} avg:{}",
            sample.count, sample.avg_value
        );
    }
    for warning in &scan.parsed.warnings {
        println!("warning={warning}");
    }
}

fn print_field(name: &str, value: Option<&str>) {
    if let Some(value) = value {
        println!("{name}={value}");
    }
}

fn run_calc(args: CalcArgs) -> Result<()> {
    let (data_path, static_data_path) =
        resolve_data_paths(args.data_dir.as_deref(), &args.data, &args.static_data)?;
    let mut core = load_core(&data_path, &static_data_path)?;
    let nest_id = args
        .nest_id
        .clone()
        .or_else(|| {
            args.map_id
                .as_ref()
                .and_then(|id| core.static_data.map_to_nest.get(id).cloned())
        })
        .unwrap_or_else(|| "2001".to_string());
    let cp = CalcParams {
        tier: args.tier.clone(),
        map_nest_id: nest_id.clone(),
        total_count: args.total,
        total_grid_target: args.total_grid,
        avg_grid_all: args.avg_grid_all,
        high_quality_count: args.high_quality_count,
        gw_count: args.gw_count,
        min_gw: args.gw_min,
        gw_grid: args.gw_grid,
        gw_avg: args.gw_avg,
        blue_count: args.blue_count,
        min_blue: args.blue_min,
        blue_grid: args.blue_grid,
        blue_avg: args.blue_avg,
        purple_count: args.purple_count,
        min_purple: args.purple_min,
        purple_grid: args.purple_grid,
        purple_avg: args.purple_avg,
        gold_count: args.gold_count,
        min_gold: args.gold_min,
        gold_grid: args.gold_grid,
        gold_avg: args.gold_avg,
        red_count: args.red_count,
        min_red: args.red_min,
        red_grid: args.red_grid,
        red_avg: args.red_avg,
        safety_factor: args.safety,
        max_show: args.max_show,
        min_value_floor: args.min_value_floor,
        ..Default::default()
    };
    let results = core.run(cp.clone())?;
    let (p25, p50, p75) = core.price_range(&results, &cp);
    let tier_weights = core.tier_weights(&cp.tier)?;
    let probs = core
        .loader
        .get_map_quality_probs(Some(&cp.map_nest_id), &tier_weights);
    let map_name = args
        .map_id
        .as_ref()
        .and_then(|id| core.static_data.map_names.get(id))
        .cloned()
        .unwrap_or_default();
    println!(
        "tier={} map={} {} nest={} source={}",
        cp.tier,
        args.map_id.unwrap_or_default(),
        map_name,
        cp.map_nest_id,
        probs.source
    );
    println!("combos={} raw={}", results.len(), core.raw_results.len());
    println!(
        "bid_p25={:.0} bid_p50={:.0} bid_p75={:.0}",
        p25 * cp.safety_factor,
        p50 * cp.safety_factor,
        p75 * cp.safety_factor
    );
    println!("greenwhite,blue,purple,gold,red,probability,final_value,total_grid_est");
    for r in results.iter().take(cp.max_show) {
        println!(
            "{},{},{},{},{},{:.6},{:.0},{:.1}",
            r.greenwhite_count,
            r.blue_count,
            r.purple_count,
            r.gold_count,
            r.red_count,
            r.probability,
            r.final_value,
            r.total_grid_est
        );
    }
    Ok(())
}

fn resolve_data_paths(
    data_dir: Option<&Path>,
    data: &Path,
    static_data: &Path,
) -> Result<(PathBuf, PathBuf)> {
    if let Some(data_dir) = data_dir {
        let data_path = data_dir
            .join("resources")
            .join("MapBidCalculator.calculator_data_merged.csv");
        let static_data_path = data_dir.join("static_data.json");
        if !data_path.exists() {
            anyhow::bail!("data file not found in data dir: {}", data_path.display());
        }
        if !static_data_path.exists() {
            anyhow::bail!(
                "static_data.json not found in data dir: {}",
                static_data_path.display()
            );
        }
        return Ok((data_path, static_data_path));
    }
    Ok((
        data.to_path_buf(),
        static_data
            .canonicalize()
            .with_context(|| format!("resolve {}", static_data.display()))?,
    ))
}
