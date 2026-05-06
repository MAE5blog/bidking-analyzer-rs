use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, imageops::FilterType};
use paddle_ocr_rs::ocr_lite::OcrLite;
use regex::Regex;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MAP_NAMES: &[&str] = &[
    "未知快递",
    "医院快递",
    "超市快递",
    "学区快递",
    "小区快递",
    "玩具店快递",
    "古董街快递",
    "未知仓库",
    "潮牌仓库",
    "硬核资产仓库",
    "民生储备仓库",
    "书店仓库",
    "杂货集装箱",
    "家居集装箱",
    "数码科技集装箱",
    "冷链集装箱",
    "古董工艺集装箱",
    "文博集装箱",
    "奢华集装箱",
    "医疗用品集装箱",
    "军用物资集装箱",
    "潮牌集装箱",
    "未知别墅",
    "设计师居所",
    "科学家居所",
    "养生学家居所",
    "望族居所",
    "学者居所",
    "私人金库",
    "奢华养老院",
    "末日庇护所",
    "极客改造屋",
    "未知残骸",
    "远洋客轮舱房",
    "军用舰艇保险库",
    "冷链货船隔离舱",
    "殖民商船宝库",
    "探险家座舰资料库",
    "皇家御用货舱",
    "生物实验室样本库",
    "私掠船军火舱",
    "现代货轮娱乐库",
    "隐秘拍卖会",
];

const MAP_ALIASES: &[(&str, &str)] = &[
    ("隐秘拍卖会", "隐秘拍卖会"),
    ("隐秘拍卖", "隐秘拍卖会"),
    ("隐密拍卖会", "隐秘拍卖会"),
    ("隐密拍卖", "隐秘拍卖会"),
];

static PPOCRV4_ENGINE: OnceLock<std::result::Result<Mutex<PpOcrV4Engine>, String>> =
    OnceLock::new();
static ORT_INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OcrResult {
    pub round_number: Option<String>,
    pub total_all: Option<String>,
    pub global_grid_total: Option<String>,
    pub global_avg_grid: Option<String>,
    pub high_quality_total_count: Option<String>,
    pub wg_count: Option<String>,
    pub wg_grid: Option<String>,
    pub wg_avg: Option<String>,
    pub blue_count: Option<String>,
    pub blue_grid: Option<String>,
    pub blue_avg: Option<String>,
    pub purple_count: Option<String>,
    pub purple_grid: Option<String>,
    pub purple_avg: Option<String>,
    pub gold_count: Option<String>,
    pub gold_grid: Option<String>,
    pub gold_avg: Option<String>,
    pub red_count: Option<String>,
    pub red_grid: Option<String>,
    pub red_avg: Option<String>,
    pub purple_avg_value: Option<String>,
    pub gold_avg_value: Option<String>,
    pub red_avg_value: Option<String>,
    pub min_value_floor: Option<String>,
    pub value_samples: Vec<OcrValueSample>,
    pub map_name: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrValueSample {
    pub count: String,
    pub avg_value: String,
}

#[derive(Debug, Clone)]
pub struct OcrScan {
    pub engine: String,
    pub lines: Vec<String>,
    pub parsed: OcrResult,
    pub crop_path: PathBuf,
    pub timings: OcrTimings,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OcrTimings {
    pub source_ms: u128,
    pub preprocess_ms: u128,
    pub crop_save_ms: u128,
    pub ocr_ms: u128,
    pub parse_ms: u128,
    pub total_ms: u128,
}

pub fn default_capture_rect(screen_width: u32, screen_height: u32) -> CaptureRect {
    if use_legacy_ocr_regions() {
        return legacy_capture_rect(screen_width, screen_height);
    }
    CaptureRect {
        x: (screen_width as f64 * 0.255) as u32,
        y: (screen_height as f64 * 0.105) as u32,
        width: (screen_width as f64 * 0.345) as u32,
        height: (screen_height as f64 * 0.58) as u32,
    }
}

pub fn default_min_value_rect(screen_width: u32, screen_height: u32) -> CaptureRect {
    if use_legacy_ocr_regions() {
        return legacy_min_value_rect(screen_width, screen_height);
    }
    CaptureRect {
        x: (screen_width as f64 * 0.735) as u32,
        y: (screen_height as f64 * 0.835) as u32,
        width: (screen_width as f64 * 0.25) as u32,
        height: (screen_height as f64 * 0.075) as u32,
    }
}

pub fn default_round_rect(screen_width: u32, screen_height: u32) -> CaptureRect {
    if use_legacy_ocr_regions() {
        return legacy_round_rect(screen_width, screen_height);
    }
    CaptureRect {
        x: (screen_width as f64 * 0.335) as u32,
        y: (screen_height as f64 * 0.070) as u32,
        width: (screen_width as f64 * 0.190) as u32,
        height: (screen_height as f64 * 0.050) as u32,
    }
}

fn legacy_capture_rect(screen_width: u32, screen_height: u32) -> CaptureRect {
    CaptureRect {
        x: (screen_width as f64 * 0.225) as u32,
        y: (screen_height as f64 * 0.105) as u32,
        width: (screen_width as f64 * 0.37) as u32,
        height: (screen_height as f64 * 0.635) as u32,
    }
}

fn legacy_min_value_rect(screen_width: u32, screen_height: u32) -> CaptureRect {
    CaptureRect {
        x: (screen_width as f64 * 0.66) as u32,
        y: (screen_height as f64 * 0.80) as u32,
        width: (screen_width as f64 * 0.33) as u32,
        height: (screen_height as f64 * 0.12) as u32,
    }
}

fn legacy_round_rect(screen_width: u32, screen_height: u32) -> CaptureRect {
    CaptureRect {
        x: (screen_width as f64 * 0.31) as u32,
        y: (screen_height as f64 * 0.055) as u32,
        width: (screen_width as f64 * 0.38) as u32,
        height: (screen_height as f64 * 0.07) as u32,
    }
}

pub fn crop_info_region(image: &DynamicImage) -> DynamicImage {
    let (screen_width, screen_height) = image.dimensions();
    let rect = default_capture_rect(screen_width, screen_height);
    crop_region(image, rect)
}

pub fn crop_round_region(image: &DynamicImage) -> DynamicImage {
    let (screen_width, screen_height) = image.dimensions();
    let rect = default_round_rect(screen_width, screen_height);
    crop_region(image, rect)
}

pub fn crop_min_value_region(image: &DynamicImage) -> DynamicImage {
    let (screen_width, screen_height) = image.dimensions();
    let rect = default_min_value_rect(screen_width, screen_height);
    crop_region(image, rect)
}

pub fn crop_ocr_regions(image: &DynamicImage) -> RgbImage {
    let round = crop_round_region(image).to_rgb8();
    let info = crop_info_region(image).to_rgb8();
    let min_value = crop_min_value_region(image).to_rgb8();
    stitch_regions(&[round, info, min_value])
}

pub fn capture_primary_screen_info_region() -> Result<RgbImage> {
    let (screen_width, screen_height) = primary_screen_size()?;
    let round = capture_screen_rect(default_round_rect(screen_width, screen_height))?;
    let info = capture_screen_rect(default_capture_rect(screen_width, screen_height))?;
    let min_value = capture_screen_rect(default_min_value_rect(screen_width, screen_height))?;
    Ok(stitch_regions(&[round, info, min_value]))
}

fn crop_region(image: &DynamicImage, rect: CaptureRect) -> DynamicImage {
    let (screen_width, screen_height) = image.dimensions();
    let x = rect.x.min(screen_width.saturating_sub(1));
    let y = rect.y.min(screen_height.saturating_sub(1));
    let width = rect.width.min(screen_width.saturating_sub(x)).max(1);
    let height = rect.height.min(screen_height.saturating_sub(y)).max(1);
    image.crop_imm(x, y, width, height)
}

fn stitch_regions(regions: &[RgbImage]) -> RgbImage {
    let width = regions
        .iter()
        .map(|image| image.width())
        .max()
        .unwrap_or(1)
        .max(1);
    let height = regions
        .iter()
        .map(|image| image.height())
        .sum::<u32>()
        .max(1);
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
    let mut y = 0;
    for region in regions {
        image::imageops::replace(&mut canvas, region, 0, y as i64);
        y += region.height();
    }
    canvas
}

#[cfg(target_os = "windows")]
fn primary_screen_size() -> Result<(u32, u32)> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

    let width = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let height = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    if width <= 0 || height <= 0 {
        anyhow::bail!("无法获取主屏幕尺寸");
    }
    Ok((width as u32, height as u32))
}

#[cfg(not(target_os = "windows"))]
fn primary_screen_size() -> Result<(u32, u32)> {
    anyhow::bail!("当前只实现了 Windows 屏幕截图")
}

#[cfg(target_os = "windows")]
fn capture_screen_rect(rect: CaptureRect) -> Result<RgbImage> {
    use std::mem::size_of;
    use std::ptr::null_mut;
    use windows_sys::Win32::Graphics::Gdi::{
        BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleBitmap,
        CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits, RGBQUAD,
        ReleaseDC, SRCCOPY, SelectObject,
    };

    let width = rect.width.max(1) as i32;
    let height = rect.height.max(1) as i32;
    unsafe {
        let screen_dc = GetDC(null_mut());
        if screen_dc.is_null() {
            anyhow::bail!("GetDC failed");
        }
        let memory_dc = CreateCompatibleDC(screen_dc);
        if memory_dc.is_null() {
            ReleaseDC(null_mut(), screen_dc);
            anyhow::bail!("CreateCompatibleDC failed");
        }
        let bitmap = CreateCompatibleBitmap(screen_dc, width, height);
        if bitmap.is_null() {
            DeleteDC(memory_dc);
            ReleaseDC(null_mut(), screen_dc);
            anyhow::bail!("CreateCompatibleBitmap failed");
        }
        let old_object = SelectObject(memory_dc, bitmap);
        let blt_ok = BitBlt(
            memory_dc,
            0,
            0,
            width,
            height,
            screen_dc,
            rect.x as i32,
            rect.y as i32,
            SRCCOPY | CAPTUREBLT,
        ) != 0;
        if !blt_ok {
            SelectObject(memory_dc, old_object);
            DeleteObject(bitmap);
            DeleteDC(memory_dc);
            ReleaseDC(null_mut(), screen_dc);
            anyhow::bail!("BitBlt failed");
        }

        let mut info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB,
                biSizeImage: (width * height * 4) as u32,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [RGBQUAD {
                rgbBlue: 0,
                rgbGreen: 0,
                rgbRed: 0,
                rgbReserved: 0,
            }],
        };
        let mut bgra = vec![0_u8; (width * height * 4) as usize];
        let lines = GetDIBits(
            memory_dc,
            bitmap,
            0,
            height as u32,
            bgra.as_mut_ptr().cast(),
            &mut info,
            DIB_RGB_COLORS,
        );

        SelectObject(memory_dc, old_object);
        DeleteObject(bitmap);
        DeleteDC(memory_dc);
        ReleaseDC(null_mut(), screen_dc);

        if lines == 0 {
            anyhow::bail!("GetDIBits failed");
        }

        let mut image = RgbImage::new(width as u32, height as u32);
        for (idx, pixel) in bgra.chunks_exact(4).enumerate() {
            let x = (idx as u32) % width as u32;
            let y = (idx as u32) / width as u32;
            image.put_pixel(x, y, Rgb([pixel[2], pixel[1], pixel[0]]));
        }
        Ok(image)
    }
}

#[cfg(not(target_os = "windows"))]
fn capture_screen_rect(_rect: CaptureRect) -> Result<RgbImage> {
    anyhow::bail!("当前只实现了 Windows 屏幕截图")
}

pub fn scan_screenshot_with_ppocrv4_onnx(
    image_path: impl AsRef<Path>,
    fallback_ceiling: Option<i32>,
) -> Result<OcrScan> {
    let started = Instant::now();
    let image_path = image_path.as_ref();
    let image =
        image::open(image_path).with_context(|| format!("open {}", image_path.display()))?;
    let crop = crop_ocr_regions(&image);
    run_ppocrv4_onnx(crop, image_path, fallback_ceiling, started)
}

pub fn scan_primary_screen_with_ppocrv4_onnx(fallback_ceiling: Option<i32>) -> Result<OcrScan> {
    let started = Instant::now();
    let crop = capture_primary_screen_info_region()?;
    run_ppocrv4_onnx(crop, Path::new("primary-screen"), fallback_ceiling, started)
}

pub fn warm_up_ppocrv4_onnx() -> Result<()> {
    with_ppocrv4_engine(|_| Ok(()))
}

fn run_ppocrv4_onnx(
    crop: RgbImage,
    source: &Path,
    fallback_ceiling: Option<i32>,
    started: Instant,
) -> Result<OcrScan> {
    let source_ms = started.elapsed().as_millis();
    let step = Instant::now();
    let scale = ocr_resize_scale();
    let enlarged = image::imageops::resize(
        &crop,
        crop.width() * scale,
        crop.height() * scale,
        ocr_resize_filter(),
    );
    let preprocess_ms = step.elapsed().as_millis();
    let step = Instant::now();
    let crop_path = save_crop_if_requested(&enlarged, source)?;
    let crop_save_ms = step.elapsed().as_millis();
    let step = Instant::now();
    let result = with_ppocrv4_engine(|engine| engine.detect(&enlarged))?;
    let ocr_ms = step.elapsed().as_millis();
    let step = Instant::now();
    let mut blocks = result.text_blocks;
    blocks.sort_by_key(|block| {
        let min_y = block
            .box_points
            .iter()
            .map(|p| p.y)
            .min()
            .unwrap_or_default();
        let min_x = block
            .box_points
            .iter()
            .map(|p| p.x)
            .min()
            .unwrap_or_default();
        (min_y / 16, min_x)
    });
    let lines = blocks
        .into_iter()
        .map(|block| block.text.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let parsed = parse_ocr_lines(&lines, fallback_ceiling);
    let parse_ms = step.elapsed().as_millis();
    Ok(OcrScan {
        engine: "ppocrv4-onnx".to_string(),
        lines,
        parsed,
        crop_path,
        timings: OcrTimings {
            source_ms,
            preprocess_ms,
            crop_save_ms,
            ocr_ms,
            parse_ms,
            total_ms: started.elapsed().as_millis(),
        },
    })
}

struct PpOcrV4Engine {
    ocr: OcrLite,
}

impl PpOcrV4Engine {
    fn new() -> Result<Self> {
        let model_dir = find_ppocrv4_model_dir().context(
            "找不到 PP-OCRv4 ONNX 模型目录。请设置 BIDKING_PPOCRV4_DIR，或把模型放在 models\\ppocrv4",
        )?;
        let models = PpOcrV4Models::new(&model_dir)?;
        init_onnxruntime(&models.onnxruntime)?;
        let mut ocr = OcrLite::new();
        ocr.init_models(
            &models.det.to_string_lossy(),
            &models.cls.to_string_lossy(),
            &models.rec.to_string_lossy(),
            2,
        )
        .map_err(|err| anyhow::anyhow!("init PP-OCRv4 ONNX models: {err}"))?;
        Ok(Self { ocr })
    }

    fn detect(&mut self, image: &RgbImage) -> Result<paddle_ocr_rs::ocr_result::OcrResult> {
        self.ocr
            .detect(image, 50, 1600, 0.45, 0.30, 1.60, false, false)
            .map_err(|err| anyhow::anyhow!("run PP-OCRv4 ONNX OCR: {err}"))
    }
}

fn with_ppocrv4_engine<T>(f: impl FnOnce(&mut PpOcrV4Engine) -> Result<T>) -> Result<T> {
    let engine = PPOCRV4_ENGINE.get_or_init(|| {
        PpOcrV4Engine::new()
            .map(Mutex::new)
            .map_err(|err| format!("{err:#}"))
    });
    let engine = engine
        .as_ref()
        .map_err(|err| anyhow::anyhow!("init PP-OCRv4 ONNX models: {err}"))?;
    let mut engine = engine
        .lock()
        .map_err(|_| anyhow::anyhow!("PP-OCRv4 ONNX engine lock poisoned"))?;
    f(&mut engine)
}

fn init_onnxruntime(path: &Path) -> Result<()> {
    ORT_INIT_RESULT
        .get_or_init(|| {
            ort::init_from(path.to_string_lossy())
                .commit()
                .map(|_| ())
                .map_err(|err| err.to_string())
        })
        .as_ref()
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("load ONNX Runtime: {err}"))
}

#[derive(Debug, Clone)]
struct PpOcrV4Models {
    det: PathBuf,
    cls: PathBuf,
    rec: PathBuf,
    onnxruntime: PathBuf,
}

impl PpOcrV4Models {
    fn new(dir: &Path) -> Result<Self> {
        let models = Self {
            det: dir.join("ch_PP-OCRv4_det_infer.onnx"),
            cls: dir.join("ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            rec: dir.join("ch_PP-OCRv4_rec_infer.onnx"),
            onnxruntime: dir.join(onnxruntime_library_name()),
        };
        for path in [&models.det, &models.cls, &models.rec, &models.onnxruntime] {
            if !path.exists() {
                anyhow::bail!("missing model file {}", path.display());
            }
        }
        Ok(models)
    }
}

fn onnxruntime_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

pub fn find_ppocrv4_model_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("BIDKING_PPOCRV4_DIR") {
        let path = PathBuf::from(path);
        if PpOcrV4Models::new(&path).is_ok() {
            return Some(path);
        }
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("models").join("ppocrv4"));
        candidates.push(cwd.join("..").join("models").join("ppocrv4"));
    }
    if let Ok(exe) = env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        candidates.push(exe_dir.join("models").join("ppocrv4"));
        candidates.push(exe_dir.join("..").join("models").join("ppocrv4"));
        candidates.push(exe_dir.join("..").join("..").join("models").join("ppocrv4"));
    }

    candidates
        .into_iter()
        .find(|path| PpOcrV4Models::new(path).is_ok())
}

pub fn parse_ocr_lines(ocr_lines: &[String], fallback_ceiling: Option<i32>) -> OcrResult {
    let text = normalize_ocr_text(&ocr_lines.join(""));
    let mut result = OcrResult {
        round_number: round_number_match(&text),
        ..Default::default()
    };

    for (alias, map_name) in MAP_ALIASES {
        if text.contains(alias) {
            result.map_name = Some((*map_name).to_string());
            break;
        }
    }
    if result.map_name.is_none() {
        for map_name in MAP_NAMES {
            if text.contains(map_name) {
                result.map_name = Some((*map_name).to_string());
                break;
            }
        }
    }

    result.total_all = try_match(
        &text,
        r"(?:本场拍卖共有道具|本次竞拍的总藏品数量为).*?(\d+)",
    );
    result.global_avg_grid = try_match(&text, r"每件藏品平均占用的格子数量约为.*?([\d\.]+)");
    result.global_grid_total = try_match(
        &text,
        r"(?:本次竞拍的总占位数为|所有藏品总占用的格子数量为).*?(\d+)",
    );
    result.high_quality_total_count = try_match(
        &text,
        r"本次竞拍共有品质(?:紫色、金色、红色|红色、金色、紫色|金色、紫色、红色)藏品.*?(\d+)",
    );
    result.blue_grid = try_match(
        &text,
        r"(?:所有蓝色品质藏品总占位数为|蓝色品质总占用的格子数量为).*?(\d+)",
    );
    result.purple_grid = try_match(
        &text,
        r"(?:所有紫色品质藏品总占位数为|紫色品质总占用的格子数量为).*?(\d+)",
    );
    result.gold_grid = try_match(
        &text,
        r"(?:所有(?:金色|橙色)品质藏品总占位数为|(?:金色|橙色)品质总占用的格子数量为).*?(\d+)",
    );
    result.red_grid = try_match(
        &text,
        r"(?:所有红色品质藏品总占位数为|红色品质总占用的格子数量为).*?(\d+)",
    );
    result.wg_grid = try_match(&text, r"所有白色和绿色品质藏品总占位数为.*?(\d+)");
    result.gold_avg = try_match(
        &text,
        r"(?:所有(?:金色|橙色)品质藏品平均占位约|本次竞拍(?:金色|橙色)品质藏品平均格数约为|(?:金色|橙色)品质藏品平均占用的格子数量约为).*?([\d\.]+)",
    );
    result.purple_avg = try_match(
        &text,
        r"(?:所有紫色品质藏品平均占位约|本次竞拍紫色品质藏品平均格数约为|紫色品质藏品平均占用的格子数量约为).*?([\d\.]+)",
    );
    result.blue_avg = try_match(
        &text,
        r"(?:所有蓝色品质藏品平均占位约|本次竞拍蓝色品质藏品平均格数约为|蓝色品质藏品平均占用的格子数量约为).*?([\d\.]+)",
    );
    result.red_avg = try_match(
        &text,
        r"(?:所有红色品质藏品平均占位约|本次竞拍红色品质藏品平均格数约为|红色品质藏品平均占用的格子数量约为).*?([\d\.]+)",
    );
    result.wg_avg = try_match(&text, r"所有白色和绿色品质藏品平均占位约.*?([\d\.]+)");
    result.blue_count = try_match(
        &text,
        r"(?:蓝色品质藏品的总数量为|本场拍卖共有蓝色品质道具).*?(\d+)",
    );
    result.purple_count = try_match(
        &text,
        r"(?:紫色品质藏品的总数量为|本场拍卖共有紫色品质道具).*?(\d+)",
    );
    result.gold_count = try_match(
        &text,
        r"(?:(?:金色|橙色)品质藏品的总数量为|本场拍卖共有(?:金色|橙色)品质道具).*?(\d+)",
    );
    result.red_count = try_match(
        &text,
        r"(?:红色品质藏品的总数量为|本场拍卖共有红色品质道具).*?(\d+)",
    );
    result.wg_count = try_match(&text, r"本次竞拍白色和绿色品质藏品数量为.*?(\d+)");
    result.gold_avg_value = try_match(
        &text,
        r"所有(?:金色|橙色)品质藏品的平均价值约为.*?([\d\.]+)",
    );
    result.purple_avg_value = try_match(&text, r"所有紫色品质藏品的平均价值约为.*?([\d\.]+)");
    result.red_avg_value = try_match(&text, r"所有红色品质藏品的平均价值约为.*?([\d\.]+)");
    result.min_value_floor = min_value_floor_match(&text);
    result.value_samples = value_sample_matches(&text);

    result.total_all = fix_trailing_noise(result.total_all, 150);
    let fallback_ceiling = fallback_ceiling.filter(|count| *count > 0);
    let total_count = result
        .total_all
        .as_ref()
        .and_then(|text| text.parse::<i32>().ok())
        .or(fallback_ceiling);
    if let Some(total_count) = total_count {
        result.blue_count =
            fix_color_count_overflow(result.blue_count, total_count, "蓝色", &mut result.warnings);
        result.purple_count = fix_color_count_overflow(
            result.purple_count,
            total_count,
            "紫色",
            &mut result.warnings,
        );
        result.gold_count =
            fix_color_count_overflow(result.gold_count, total_count, "金色", &mut result.warnings);
        result.red_count =
            fix_color_count_overflow(result.red_count, total_count, "红色", &mut result.warnings);
        result.wg_count =
            fix_color_count_overflow(result.wg_count, total_count, "白绿", &mut result.warnings);
        result.value_samples.retain(|sample| {
            sample
                .count
                .parse::<i32>()
                .is_ok_and(|count| count <= total_count)
        });
    }
    let global_grid_max = total_count.map_or(3000, |count| 20 * count.max(1));
    result.global_grid_total = fix_trailing_noise(result.global_grid_total, global_grid_max);
    result.wg_grid = fix_trailing_noise(
        result.wg_grid,
        color_grid_noise_limit(&result.wg_count, total_count, 20),
    );
    result.blue_grid = fix_trailing_noise(
        result.blue_grid,
        color_grid_noise_limit(&result.blue_count, total_count, 20),
    );
    result.purple_grid = fix_trailing_noise(
        result.purple_grid,
        color_grid_noise_limit(&result.purple_count, total_count, 12),
    );
    result.gold_grid = fix_trailing_noise(
        result.gold_grid,
        color_grid_noise_limit(&result.gold_count, total_count, 18),
    );
    result.red_grid = fix_trailing_noise(
        result.red_grid,
        color_grid_noise_limit(&result.red_count, total_count, 16),
    );
    result
}

pub fn normalize_ocr_text(text: &str) -> String {
    let mut text = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .replace(['，', '．', ','], ".");
    for (from, to) in [
        ("站拍", "竞拍"),
        ("章拍", "竞拍"),
        ("姑拍", "竞拍"),
        ("本仁", "本次"),
        ("木次", "本次"),
        ("葛品", "藏品"),
        ("茴品", "藏品"),
        ("仑库", "仓库"),
        ("伧库", "仓库"),
        ("楣色", "橙色"),
        ("棉色", "橙色"),
        ("汇联", "汇聚"),
        ("汇精", "汇聚"),
        ("情报汇联", "情报汇聚"),
        ("限扔", "随机"),
        ("扔晏示", "显示"),
        ("晏示", "显示"),
    ] {
        text = text.replace(from, to);
    }
    text
}

fn try_match(text: &str, pattern: &str) -> Option<String> {
    let re = Regex::new(pattern).ok()?;
    re.captures(text)
        .and_then(|captures| captures.get(1))
        .map(|m| m.as_str().trim_matches('.').to_string())
        .filter(|value| !value.is_empty())
}

fn round_number_match(text: &str) -> Option<String> {
    let re = Regex::new(r"第([1-5一二三四五])轮").ok()?;
    let raw = re.captures(text)?.get(1)?.as_str();
    let value = match raw {
        "1" | "一" => 1,
        "2" | "二" => 2,
        "3" | "三" => 3,
        "4" | "四" => 4,
        "5" | "五" => 5,
        _ => return None,
    };
    Some(value.to_string())
}

fn value_sample_matches(text: &str) -> Vec<OcrValueSample> {
    let Ok(re) = Regex::new(
        r"随机(?:选择|显示|抽取|挑选)的?(\d+)件(?:藏品|道具)平均价值(?:约为|为).*?([\d\.]+)",
    ) else {
        return Vec::new();
    };
    let mut samples = Vec::new();
    for captures in re.captures_iter(text) {
        let Some(count) = captures.get(1) else {
            continue;
        };
        let Some(avg_value) = captures.get(2) else {
            continue;
        };
        let sample = OcrValueSample {
            count: count.as_str().trim_matches('.').to_string(),
            avg_value: normalize_decimal_number(avg_value.as_str())
                .unwrap_or_else(|| avg_value.as_str().trim_matches('.').to_string()),
        };
        if sample.count.is_empty() || sample.avg_value.is_empty() {
            continue;
        }
        if !samples.iter().any(|existing| existing == &sample) {
            samples.push(sample);
        }
    }
    samples
}

fn min_value_floor_match(text: &str) -> Option<String> {
    let re = Regex::new(
        r"(?:当前预估最低价格|当前估值最低价格|当前预估最低价|预估最低价格|估值最低价格)[^\d]{0,8}(\d[\d\.]*)",
    )
    .ok()?;
    let raw = re
        .captures(text)
        .and_then(|captures| captures.get(1))
        .map(|m| m.as_str())?;
    normalize_price_number(raw)
}

fn normalize_price_number(raw: &str) -> Option<String> {
    let raw = raw.trim_matches('.');
    if raw.is_empty() {
        return None;
    }
    let groups = raw
        .split('.')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if groups.len() > 1
        && (1..=3).contains(&groups[0].len())
        && groups[1..]
            .iter()
            .all(|part| part.len() == 3 && part.chars().all(|ch| ch.is_ascii_digit()))
    {
        let value = groups.join("");
        if !value.is_empty() {
            return Some(value);
        }
    }
    let value = raw
        .chars()
        .filter(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect::<String>()
        .trim_matches('.')
        .to_string();
    (!value.is_empty()).then_some(value)
}

fn normalize_decimal_number(raw: &str) -> Option<String> {
    let raw = raw.trim_matches('.');
    if raw.is_empty() {
        return None;
    }
    let groups = raw
        .split('.')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if groups.len() > 2
        && (1..=3).contains(&groups[0].len())
        && groups[0].chars().all(|ch| ch.is_ascii_digit())
        && groups[1..groups.len() - 1]
            .iter()
            .all(|part| part.len() == 3 && part.chars().all(|ch| ch.is_ascii_digit()))
        && groups.last().is_some_and(|part| {
            (1..=2).contains(&part.len()) && part.chars().all(|ch| ch.is_ascii_digit())
        })
    {
        let integer = groups[..groups.len() - 1].join("");
        let decimal = groups[groups.len() - 1];
        return Some(format!("{integer}.{decimal}"));
    }
    Some(raw.to_string())
}

fn fix_color_count_overflow(
    text: Option<String>,
    total_count: i32,
    label: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let text = text?;
    let Ok(value) = text.parse::<i32>() else {
        return Some(text);
    };
    if value <= total_count {
        return Some(text);
    }
    if text.len() <= 2 {
        if label == "蓝色" && text.len() > 1 && (value as f64) > (total_count as f64 * 3.0 / 5.0)
        {
            let truncated = text[..text.len() - 1].to_string();
            warnings.push(format!(
                "{label}件数识别值 {text} 大于总件数 {total_count}，且超过总件数的五分之三，已截断为 {truncated}。"
            ));
            return Some(truncated);
        }
        return Some(text);
    }
    let mut truncated = text[..2].to_string();
    warnings.push(format!(
        "{label}件数识别值 {text} 大于总件数 {total_count}，已截断为 {truncated}。"
    ));
    if label == "蓝色"
        && truncated.len() > 1
        && truncated
            .parse::<i32>()
            .is_ok_and(|value| (value as f64) > (total_count as f64 * 3.0 / 5.0))
    {
        truncated.truncate(truncated.len() - 1);
        warnings.push(format!(
            "{label}件数截断后仍大于总件数的五分之三，已继续截断为 {truncated}。"
        ));
    }
    Some(truncated)
}

fn fix_trailing_noise(text: Option<String>, absolute_max: i32) -> Option<String> {
    let mut text = text?;
    while text.len() > 1 && text.parse::<i32>().is_ok_and(|value| value > absolute_max) {
        text.truncate(text.len() - 1);
    }
    Some(text)
}

fn color_grid_noise_limit(
    count_text: &Option<String>,
    total_count: Option<i32>,
    max_item_grid: i32,
) -> i32 {
    count_text
        .as_deref()
        .and_then(|text| text.parse::<i32>().ok())
        .filter(|count| *count > 0)
        .or(total_count)
        .map(|count| max_item_grid * count.max(1))
        .unwrap_or(3000)
}

fn save_crop_if_requested(image: &RgbImage, source: &Path) -> Result<PathBuf> {
    let crop_path = temp_crop_path(source);
    if should_save_crop(source) {
        image
            .save(&crop_path)
            .with_context(|| format!("save OCR crop {}", crop_path.display()))?;
        Ok(crop_path)
    } else {
        Ok(PathBuf::new())
    }
}

fn ocr_resize_scale() -> u32 {
    env::var("BIDKING_OCR_SCALE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| (1..=3).contains(value))
        .unwrap_or(1)
}

fn ocr_resize_filter() -> FilterType {
    match env::var("BIDKING_OCR_FILTER")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "nearest" => FilterType::Nearest,
        "triangle" => FilterType::Triangle,
        "catmullrom" | "catmull" => FilterType::CatmullRom,
        "gaussian" => FilterType::Gaussian,
        _ => FilterType::Lanczos3,
    }
}

fn use_legacy_ocr_regions() -> bool {
    env::var("BIDKING_OCR_LEGACY_ROI").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn should_save_crop(source: &Path) -> bool {
    let _ = source;
    env::var("BIDKING_OCR_SAVE_CROP").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn temp_crop_path(source: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("screenshot");
    env::temp_dir().join(format!("bidking_ocr_{}_{}_crop.png", stem, stamp))
}
