#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use bidking_rs::{CalcParams, ValueSample, load_embedded_core, load_embedded_static_data, ocr};
use eframe::egui;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const MAX_VISIBLE_COMBOS: usize = 10;
const TEXT_STROKE: f32 = 1.12;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("竞拍之王估价器")
            .with_inner_size([1220.0, 800.0])
            .with_min_inner_size([1080.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "竞拍之王估价器",
        options,
        Box::new(|cc| {
            install_chinese_font(&cc.egui_ctx);
            install_style(&cc.egui_ctx);
            Ok(Box::new(BidKingGui::with_context(&cc.egui_ctx)))
        }),
    )
}

#[derive(Debug, Clone)]
struct MapRow {
    map_id: String,
    nest_id: String,
    name: String,
}

#[derive(Debug, Clone)]
struct UiCombo {
    greenwhite: i32,
    blue: i32,
    purple: i32,
    gold: i32,
    red: i32,
    probability: f64,
    final_value: f64,
    total_grid: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct AutoFillSummary {
    fields: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct CountRange {
    min: i32,
    max: i32,
}

#[derive(Debug, Default, Clone)]
struct CalculationOutput {
    combos: usize,
    raw: usize,
    p25: i64,
    p50: i64,
    p75: i64,
    rows: Vec<UiCombo>,
    elapsed_ms: u128,
    map_label: String,
    purple_range: Option<CountRange>,
    gold_range: Option<CountRange>,
    red_range: Option<CountRange>,
    composition_lines: Vec<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ValueSampleInput {
    count: String,
    avg_value: String,
}

struct BidKingGui {
    maps: Vec<MapRow>,
    selected_map_id: String,
    tier: String,
    total: String,
    high_quality_count: String,
    avg_grid_all: String,
    total_grid: String,
    safety: String,
    display_count: String,
    gw_count: String,
    gw_min: String,
    gw_grid: String,
    gw_avg: String,
    blue_count: String,
    blue_min: String,
    blue_grid: String,
    blue_avg: String,
    purple_count: String,
    purple_min: String,
    purple_grid: String,
    purple_avg: String,
    gold_count: String,
    gold_min: String,
    gold_grid: String,
    gold_avg: String,
    red_count: String,
    red_min: String,
    red_grid: String,
    red_avg: String,
    min_value_floor: String,
    manual_purple_item: String,
    manual_purple_grid: String,
    manual_gold_item: String,
    manual_gold_grid: String,
    value_samples: Vec<ValueSampleInput>,
    status: String,
    output: Option<CalculationOutput>,
    window_topmost: bool,
    global_hotkeys: Option<GlobalHotkeys>,
}

struct GlobalHotkeys {
    rx: mpsc::Receiver<GlobalHotkeyAction>,
    registered: RegisteredHotkeys,
    warning: Option<String>,
}

#[derive(Debug, Default, Clone, Copy)]
struct RegisteredHotkeys {
    calculate: bool,
    scan_screen: bool,
    reset_conditions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlobalHotkeyAction {
    Calculate,
    ScanScreen,
    ResetConditions,
}

impl GlobalHotkeys {
    fn has(&self, action: GlobalHotkeyAction) -> bool {
        self.registered.has(action)
    }
}

impl RegisteredHotkeys {
    fn has(&self, action: GlobalHotkeyAction) -> bool {
        match action {
            GlobalHotkeyAction::Calculate => self.calculate,
            GlobalHotkeyAction::ScanScreen => self.scan_screen,
            GlobalHotkeyAction::ResetConditions => self.reset_conditions,
        }
    }

    fn any(&self) -> bool {
        self.calculate || self.scan_screen || self.reset_conditions
    }
}

impl Default for BidKingGui {
    fn default() -> Self {
        let mut app = Self {
            maps: Vec::new(),
            selected_map_id: "2101".to_string(),
            tier: "101".to_string(),
            total: "0".to_string(),
            high_quality_count: String::new(),
            avg_grid_all: String::new(),
            total_grid: String::new(),
            safety: "0.85".to_string(),
            display_count: MAX_VISIBLE_COMBOS.to_string(),
            gw_count: String::new(),
            gw_min: String::new(),
            gw_grid: String::new(),
            gw_avg: String::new(),
            blue_count: String::new(),
            blue_min: String::new(),
            blue_grid: String::new(),
            blue_avg: String::new(),
            purple_count: String::new(),
            purple_min: String::new(),
            purple_grid: String::new(),
            purple_avg: String::new(),
            gold_count: String::new(),
            gold_min: String::new(),
            gold_grid: String::new(),
            gold_avg: String::new(),
            red_count: String::new(),
            red_min: String::new(),
            red_grid: String::new(),
            red_avg: String::new(),
            min_value_floor: String::new(),
            manual_purple_item: String::new(),
            manual_purple_grid: String::new(),
            manual_gold_item: String::new(),
            manual_gold_grid: String::new(),
            value_samples: Vec::new(),
            status: String::new(),
            output: None,
            window_topmost: false,
            global_hotkeys: None,
        };
        match app.reload_maps() {
            Ok(()) => {}
            Err(err) => app.status = format!("内置数据加载失败: {err}"),
        }
        app
    }
}

impl eframe::App for BidKingGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.normalize_display_count();

        let mut global_actions = Vec::new();
        if let Some(hotkeys) = &self.global_hotkeys {
            while let Ok(action) = hotkeys.rx.try_recv() {
                global_actions.push(action);
            }
        }
        for action in global_actions {
            match action {
                GlobalHotkeyAction::Calculate => self.calculate(),
                GlobalHotkeyAction::ScanScreen => self.scan_screen(),
                GlobalHotkeyAction::ResetConditions => self.reset_conditions(),
            }
        }

        if !self.has_global_hotkey(GlobalHotkeyAction::Calculate)
            && ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Q))
        {
            self.calculate();
        }
        if !self.has_global_hotkey(GlobalHotkeyAction::ScanScreen)
            && ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::W))
        {
            self.scan_screen();
        }
        if !self.has_global_hotkey(GlobalHotkeyAction::ResetConditions)
            && ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::E))
        {
            self.reset_conditions();
        }

        egui::SidePanel::left("left_panel")
            .resizable(false)
            .exact_width(432.0)
            .frame(
                egui::Frame::default()
                    .fill(color_bg())
                    .inner_margin(egui::Margin::symmetric(12, 0)),
            )
            .show(ctx, |ui| {
                let action_area_height = 56.0;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .max_height((ui.available_height() - action_area_height).max(420.0))
                    .show(ui, |ui| {
                        ui.add_space(10.0);
                        self.map_section(ui);
                        ui.add_space(8.0);
                        self.color_constraints_section(ui);
                        ui.add_space(8.0);
                        self.valuation_section(ui);
                    });
                ui.add_space(9.0);
                self.action_buttons(ui, ctx);
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(color_bg())
                    .inner_margin(egui::Margin::symmetric(16, 0)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add_space(12.0);
                        self.results_section(ui);
                        ui.add_space(12.0);
                    });
            });
    }
}

impl BidKingGui {
    fn with_context(ctx: &egui::Context) -> Self {
        let mut app = Self::default();
        app.install_global_hotkeys(ctx);
        app
    }

    fn install_global_hotkeys(&mut self, ctx: &egui::Context) {
        match start_global_hotkeys(ctx.clone()) {
            Ok(hotkeys) => {
                let warning = hotkeys.warning.clone();
                self.global_hotkeys = Some(hotkeys);
                if let Some(warning) = warning {
                    self.status = warning;
                }
            }
            Err(err) => {
                self.status = format!("全局热键不可用，仍可在窗口前台使用: {err}");
            }
        }
    }

    fn has_global_hotkey(&self, action: GlobalHotkeyAction) -> bool {
        self.global_hotkeys
            .as_ref()
            .is_some_and(|hotkeys| hotkeys.has(action))
    }

    fn map_section(&mut self, ui: &mut egui::Ui) {
        section(ui, "地图与全局", |ui| {
            ui.columns(2, |cols| {
                field_label(&mut cols[0], "区域档");
                dark_widget_scope(&mut cols[0], |ui| {
                    egui::ComboBox::from_id_salt("tier_select")
                        .width(174.0)
                        .selected_text(
                            egui::RichText::new(tier_label(&self.tier)).color(color_text()),
                        )
                        .show_ui(ui, |ui| {
                            force_dark_visuals(ui);
                            for tier in tier_options() {
                                if ui
                                    .selectable_value(
                                        &mut self.tier,
                                        tier.id.to_string(),
                                        egui::RichText::new(tier.label()).color(color_text()),
                                    )
                                    .changed()
                                {
                                    self.select_first_map_for_tier();
                                }
                            }
                        });
                });

                field_label(&mut cols[1], "详细地图 (BidMap)");
                let selected_name = self.selected_map_label();
                let maps = self.filtered_maps();
                dark_widget_scope(&mut cols[1], |ui| {
                    egui::ComboBox::from_id_salt("map_select")
                        .width(174.0)
                        .selected_text(egui::RichText::new(selected_name).color(color_text()))
                        .show_ui(ui, |ui| {
                            force_dark_visuals(ui);
                            for map in maps {
                                ui.selectable_value(
                                    &mut self.selected_map_id,
                                    map.map_id.clone(),
                                    egui::RichText::new(format!(
                                        "{} {} -> {}",
                                        map.map_id, map.name, map.nest_id
                                    ))
                                    .color(color_text()),
                                );
                            }
                        });
                });
            });

            ui.add_space(10.0);
            ui.columns(2, |cols| {
                labeled_text(&mut cols[0], "总件数 (选填)", &mut self.total, 174.0);
                labeled_text(
                    &mut cols[1],
                    "紫金币总数 (选填)",
                    &mut self.high_quality_count,
                    174.0,
                );
            });
            ui.add_space(8.0);
            ui.columns(2, |cols| {
                labeled_text(
                    &mut cols[0],
                    "全部品均格 (选填)",
                    &mut self.avg_grid_all,
                    174.0,
                );
                labeled_text(&mut cols[1], "总格数 (选填)", &mut self.total_grid, 174.0);
            });
            ui.add_space(8.0);
            ui.columns(2, |cols| {
                labeled_text(
                    &mut cols[0],
                    "安全系数 (默认 0.85)",
                    &mut self.safety,
                    174.0,
                );
                readout_text(&mut cols[1], "展示组合数", "固定前 10 个", 174.0);
            });
        });
    }

    fn color_constraints_section(&mut self, ui: &mut egui::Ui) {
        section(ui, "蓝 / 紫 / 金 / 红约束", |ui| {
            ui.horizontal(|ui| {
                table_header(ui, "品质", 54.0);
                table_header(ui, "件数", 66.0);
                table_header(ui, "至少", 66.0);
                table_header(ui, "总格", 66.0);
                table_header(ui, "均格", 66.0);
            });
            color_row(
                ui,
                "绿白:",
                color_gw(),
                &mut self.gw_count,
                &mut self.gw_min,
                &mut self.gw_grid,
                &mut self.gw_avg,
            );
            color_row(
                ui,
                "蓝件:",
                color_blue(),
                &mut self.blue_count,
                &mut self.blue_min,
                &mut self.blue_grid,
                &mut self.blue_avg,
            );
            color_row(
                ui,
                "紫件:",
                color_purple(),
                &mut self.purple_count,
                &mut self.purple_min,
                &mut self.purple_grid,
                &mut self.purple_avg,
            );
            color_row(
                ui,
                "金件:",
                color_gold(),
                &mut self.gold_count,
                &mut self.gold_min,
                &mut self.gold_grid,
                &mut self.gold_avg,
            );
            color_row(
                ui,
                "红件:",
                color_red(),
                &mut self.red_count,
                &mut self.red_min,
                &mut self.red_grid,
                &mut self.red_avg,
            );
        });
    }

    fn valuation_section(&mut self, ui: &mut egui::Ui) {
        section(ui, "估值线索与手动定价", |ui| {
            labeled_text(ui, "当前预估最低价格", &mut self.min_value_floor, 356.0);
            ui.add_space(8.0);
            ui.columns(2, |cols| {
                labeled_text(
                    &mut cols[0],
                    "紫色每件均价",
                    &mut self.manual_purple_item,
                    172.0,
                );
                labeled_text(
                    &mut cols[1],
                    "紫色每格均价",
                    &mut self.manual_purple_grid,
                    172.0,
                );
            });
            ui.add_space(8.0);
            ui.columns(2, |cols| {
                labeled_text(
                    &mut cols[0],
                    "金色每件均价",
                    &mut self.manual_gold_item,
                    172.0,
                );
                labeled_text(
                    &mut cols[1],
                    "金色每格均价",
                    &mut self.manual_gold_grid,
                    172.0,
                );
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("随机样本价值线索")
                        .strong()
                        .color(color_gold()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("+").clicked() {
                        self.value_samples.push(ValueSampleInput::default());
                    }
                });
            });
            let mut remove_index = None;
            for (index, sample) in self.value_samples.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [34.0, 24.0],
                        egui::Label::new(
                            egui::RichText::new(format!("{}", index + 1)).color(color_muted()),
                        ),
                    );
                    field_label(ui, "件数");
                    text_input(ui, &mut sample.count, 70.0);
                    field_label(ui, "均价");
                    text_input(ui, &mut sample.avg_value, 98.0);
                    if ui.button("×").clicked() {
                        remove_index = Some(index);
                    }
                });
            }
            if let Some(index) = remove_index {
                self.value_samples.remove(index);
            }
        });
    }

    fn action_buttons(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 7.0;
            let button_width = ((ui.available_width() - 21.0) / 4.0).clamp(82.0, 102.0);
            if action_button(
                ui,
                "开始计算",
                "Alt+Q",
                button_width,
                color_gold(),
                color_bg(),
            )
            .clicked()
            {
                self.calculate();
            }
            if action_button(
                ui,
                "视觉扫描",
                "Alt+W",
                button_width,
                color_purple(),
                color_text(),
            )
            .clicked()
            {
                self.scan_screen();
            }
            if action_button(
                ui,
                "重置条件",
                "Alt+E",
                button_width,
                color_panel_alt(),
                color_text(),
            )
            .clicked()
            {
                self.reset_conditions();
            }
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new(if self.window_topmost {
                            "窗口置顶\n已开启"
                        } else {
                            "窗口置顶\n未开启"
                        })
                        .strong()
                        .color(color_text()),
                    )
                    .min_size(egui::vec2(button_width, 42.0))
                    .fill(if self.window_topmost {
                        color_green_dark()
                    } else {
                        color_panel_alt()
                    })
                    .stroke(egui::Stroke::new(1.0, color_border()))
                    .corner_radius(6),
                )
                .clicked()
            {
                self.window_topmost = !self.window_topmost;
                let level = if self.window_topmost {
                    egui::WindowLevel::AlwaysOnTop
                } else {
                    egui::WindowLevel::Normal
                };
                ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
            }
        });
    }

    fn results_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("计算结果")
                    .size(24.0)
                    .color(color_gold())
                    .strong(),
            );
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("概率拟合 / 出价参考 / 组合反推")
                    .size(13.0)
                    .color(color_muted()),
            );
        });
        ui.add_space(8.0);
        if !self.status.trim().is_empty() {
            status_bar(ui, &self.status);
            ui.add_space(12.0);
        }

        let Some(output) = &self.output else {
            empty_results(ui);
            return;
        };

        summary_row(
            ui,
            "总件数:",
            &format_number(parse_i32_or_zero(&self.total) as i64),
            "（全品类）",
            color_green(),
        );
        summary_row(
            ui,
            "可行组合数:",
            &format!(
                "{} / {}",
                format_number(output.combos as i64),
                format_number(output.raw as i64)
            ),
            &format!("； 计算耗时 {} ms", output.elapsed_ms),
            color_blue(),
        );
        high_range_row(ui, output);
        ui.add_space(12.0);

        section(ui, "出价参考", |ui| {
            ui.columns(3, |cols| {
                price_card(&mut cols[0], "保守出价 (P25)", output.p25, color_green());
                price_card(&mut cols[1], "均衡出价 (P50)", output.p50, color_orange());
                price_card(&mut cols[2], "激进出价 (P75)", output.p75, color_red());
            });
        });
        ui.add_space(12.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("可行组合列表").color(color_text()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                pill_label(ui, "自动回填唯一件数/格数", color_gold());
            });
        });
        result_table(ui, &output.rows);
        ui.add_space(12.0);

        section(ui, "物品级可能组成（价格/占格拟合）", |ui| {
            ui.label(
                egui::RichText::new(format!(
                    "组合 1: {}，概率 {}",
                    output
                        .rows
                        .first()
                        .map(combo_label)
                        .unwrap_or_else(|| "--".to_string()),
                    output
                        .rows
                        .first()
                        .map(|r| format_percent(r.probability))
                        .unwrap_or_else(|| "--".to_string())
                ))
                .strong()
                .color(color_text()),
            );
            for line in &output.composition_lines {
                ui.label(egui::RichText::new(line).color(color_muted()));
            }
        });

        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(format!("当前地图: {}", output.map_label)).color(color_muted()),
        );
    }

    fn reload_maps(&mut self) -> Result<()> {
        let static_data = load_embedded_static_data()?;
        let mut maps = Vec::new();
        for (map_id, nest_id) in &static_data.map_to_nest {
            maps.push(MapRow {
                map_id: map_id.clone(),
                nest_id: nest_id.clone(),
                name: static_data
                    .map_names
                    .get(map_id)
                    .cloned()
                    .unwrap_or_default(),
            });
        }
        maps.sort_by(|a, b| a.map_id.cmp(&b.map_id));
        self.maps = maps;
        if !self
            .filtered_maps()
            .iter()
            .any(|m| m.map_id == self.selected_map_id)
        {
            self.select_first_map_for_tier();
        }
        Ok(())
    }

    fn calculate(&mut self) {
        match self.calculate_inner() {
            Ok((output, auto_fill)) => {
                self.status = if auto_fill.fields > 0 {
                    format!(
                        "当前地图: {}。地图实算完成，已自动回填 {} 个唯一件数/格数字段。",
                        output.map_label, auto_fill.fields
                    )
                } else {
                    format!(
                        "当前地图: {}。地图实算完成，未发现可回填的唯一件数/格数。",
                        output.map_label
                    )
                };
                self.output = Some(output);
            }
            Err(err) => {
                self.status = format!("计算失败: {err:#}");
                self.output = None;
            }
        }
    }

    fn scan_screen(&mut self) {
        let fallback_total = parse_optional_i32(&self.total).ok().flatten();
        self.status = "正在截取屏幕情报区域并识别...".to_string();
        match ocr::scan_primary_screen_with_ppocrv4_onnx(fallback_total) {
            Ok(scan) => {
                let updated = self.apply_ocr_result(&scan.parsed);
                let map = scan.parsed.map_name.as_deref().unwrap_or("未识别地图");
                let warnings = if scan.parsed.warnings.is_empty() {
                    String::new()
                } else {
                    format!("；{}", scan.parsed.warnings.join("；"))
                };
                let scan_status = format!(
                    "视觉扫描完成：{}，填入 {} 个字段，{} 行 {}{}",
                    map,
                    updated,
                    scan.engine,
                    scan.lines.len(),
                    warnings
                );
                match self.calculate_inner() {
                    Ok((output, auto_fill)) => {
                        let auto_fill_status = if auto_fill.fields > 0 {
                            format!("；计算后回填 {} 个唯一件数/格数字段", auto_fill.fields)
                        } else {
                            String::new()
                        };
                        self.status = format!(
                            "{}。已自动计算出价：当前地图: {}{}。",
                            scan_status, output.map_label, auto_fill_status
                        );
                        self.output = Some(output);
                    }
                    Err(err) => {
                        self.status = format!("{scan_status}。自动计算失败: {err:#}");
                        self.output = None;
                    }
                }
            }
            Err(err) => {
                self.status = format!("视觉扫描失败: {err:#}");
            }
        }
    }

    fn apply_ocr_result(&mut self, result: &ocr::OcrResult) -> usize {
        let mut updated = 0;
        if let Some(map_name) = &result.map_name {
            if let Some(map) = self.maps.iter().find(|m| m.name == *map_name).cloned() {
                if let Some(tier) = tier_from_map_id(&map.map_id) {
                    self.tier = tier.to_string();
                }
                self.selected_map_id = map.map_id;
                updated += 1;
            }
        }
        updated += set_if_some(&mut self.total, &result.total_all);
        updated += set_if_some(
            &mut self.high_quality_count,
            &result.high_quality_total_count,
        );
        updated += set_if_some(&mut self.total_grid, &result.global_grid_total);
        updated += set_if_some(&mut self.avg_grid_all, &result.global_avg_grid);
        updated += set_if_some(&mut self.gw_count, &result.wg_count);
        updated += set_if_some(&mut self.gw_grid, &result.wg_grid);
        updated += set_if_some(&mut self.gw_avg, &result.wg_avg);
        updated += set_if_some(&mut self.blue_count, &result.blue_count);
        updated += set_if_some(&mut self.blue_grid, &result.blue_grid);
        updated += set_if_some(&mut self.blue_avg, &result.blue_avg);
        updated += set_if_some(&mut self.purple_count, &result.purple_count);
        updated += set_if_some(&mut self.purple_grid, &result.purple_grid);
        updated += set_if_some(&mut self.purple_avg, &result.purple_avg);
        updated += set_if_some(&mut self.gold_count, &result.gold_count);
        updated += set_if_some(&mut self.gold_grid, &result.gold_grid);
        updated += set_if_some(&mut self.gold_avg, &result.gold_avg);
        updated += set_if_some(&mut self.red_count, &result.red_count);
        updated += set_if_some(&mut self.red_grid, &result.red_grid);
        updated += set_if_some(&mut self.red_avg, &result.red_avg);
        updated += set_if_some(&mut self.manual_purple_item, &result.purple_avg_value);
        updated += set_if_some(&mut self.manual_gold_item, &result.gold_avg_value);
        updated += set_if_some(&mut self.min_value_floor, &result.min_value_floor);
        updated += self.merge_ocr_value_samples(&result.value_samples);
        updated += self.reconcile_grids_from_averages(result);
        updated
    }

    fn reconcile_grids_from_averages(&mut self, result: &ocr::OcrResult) -> usize {
        let mut updated = 0;
        let count = self.gw_count.clone();
        let avg = self.gw_avg.clone();
        updated += set_grid_from_average(&mut self.gw_grid, &count, &avg, result.wg_grid.is_some());
        let count = self.blue_count.clone();
        let avg = self.blue_avg.clone();
        updated += set_grid_from_average(
            &mut self.blue_grid,
            &count,
            &avg,
            result.blue_grid.is_some(),
        );
        let count = self.purple_count.clone();
        let avg = self.purple_avg.clone();
        updated += set_grid_from_average(
            &mut self.purple_grid,
            &count,
            &avg,
            result.purple_grid.is_some(),
        );
        let count = self.gold_count.clone();
        let avg = self.gold_avg.clone();
        updated += set_grid_from_average(
            &mut self.gold_grid,
            &count,
            &avg,
            result.gold_grid.is_some(),
        );
        let count = self.red_count.clone();
        let avg = self.red_avg.clone();
        updated +=
            set_grid_from_average(&mut self.red_grid, &count, &avg, result.red_grid.is_some());
        updated
    }

    fn apply_unique_fields(&mut self, results: &[bidking_rs::ComboResult]) -> AutoFillSummary {
        let mut summary = AutoFillSummary::default();
        summary.fields += set_unique_i32(
            &mut self.gw_count,
            unique_i32(results, |r| r.greenwhite_count),
        );
        summary.fields +=
            set_grid_from_average(&mut self.gw_grid, &self.gw_count, &self.gw_avg, false);
        summary.fields +=
            set_unique_i32(&mut self.blue_count, unique_i32(results, |r| r.blue_count));
        summary.fields +=
            set_grid_from_average(&mut self.blue_grid, &self.blue_count, &self.blue_avg, false);
        summary.fields += set_unique_i32(
            &mut self.purple_count,
            unique_i32(results, |r| r.purple_count),
        );
        summary.fields += set_grid_from_average(
            &mut self.purple_grid,
            &self.purple_count,
            &self.purple_avg,
            false,
        );
        summary.fields +=
            set_unique_i32(&mut self.gold_count, unique_i32(results, |r| r.gold_count));
        summary.fields +=
            set_grid_from_average(&mut self.gold_grid, &self.gold_count, &self.gold_avg, false);
        summary.fields += set_unique_i32(&mut self.red_count, unique_i32(results, |r| r.red_count));
        summary.fields +=
            set_grid_from_average(&mut self.red_grid, &self.red_count, &self.red_avg, false);
        summary
    }

    fn calculate_inner(&mut self) -> Result<(CalculationOutput, AutoFillSummary)> {
        let start = Instant::now();
        let mut core = load_embedded_core()?;
        if let Some(tier) = tier_from_map_id(&self.selected_map_id) {
            self.tier = tier.to_string();
        }
        let nest_id = core
            .static_data
            .map_to_nest
            .get(&self.selected_map_id)
            .cloned()
            .with_context(|| format!("未知地图 {}", self.selected_map_id))?;
        let display_count = MAX_VISIBLE_COMBOS;
        let total_count = parse_i32(&self.total, "总件数")?;
        if total_count <= 0 {
            anyhow::bail!("请先填写大于 0 的总件数");
        }
        let min_value_floor = parse_optional_f64(&self.min_value_floor)?;
        if min_value_floor.is_some_and(|value| value < 0.0) {
            anyhow::bail!("当前预估最低价格不能为负数");
        }
        let cp = CalcParams {
            tier: self.tier.trim().to_string(),
            map_nest_id: nest_id,
            total_count,
            total_grid_target: parse_optional_f64(&self.total_grid)?,
            avg_grid_all: parse_optional_f64(&self.avg_grid_all)?,
            high_quality_count: parse_optional_i32(&self.high_quality_count)?,
            safety_factor: parse_f64(&self.safety, "安全系数")?,
            max_show: display_count,
            gw_count: parse_optional_i32(&self.gw_count)?,
            min_gw: parse_optional_i32(&self.gw_min)?.unwrap_or_default(),
            gw_grid: parse_optional_f64(&self.gw_grid)?,
            gw_avg: parse_optional_f64(&self.gw_avg)?,
            blue_count: parse_optional_i32(&self.blue_count)?,
            min_blue: parse_optional_i32(&self.blue_min)?.unwrap_or_default(),
            blue_grid: parse_optional_f64(&self.blue_grid)?,
            blue_avg: parse_optional_f64(&self.blue_avg)?,
            purple_count: parse_optional_i32(&self.purple_count)?,
            min_purple: parse_optional_i32(&self.purple_min)?.unwrap_or_default(),
            purple_grid: parse_optional_f64(&self.purple_grid)?,
            purple_avg: parse_optional_f64(&self.purple_avg)?,
            gold_count: parse_optional_i32(&self.gold_count)?,
            min_gold: parse_optional_i32(&self.gold_min)?.unwrap_or_default(),
            gold_grid: parse_optional_f64(&self.gold_grid)?,
            gold_avg: parse_optional_f64(&self.gold_avg)?,
            red_count: parse_optional_i32(&self.red_count)?,
            min_red: parse_optional_i32(&self.red_min)?.unwrap_or_default(),
            red_grid: parse_optional_f64(&self.red_grid)?,
            red_avg: parse_optional_f64(&self.red_avg)?,
            min_value_floor,
            manual_purple_per_item: parse_optional_f64(&self.manual_purple_item)?,
            manual_purple_per_grid: parse_optional_f64(&self.manual_purple_grid)?,
            manual_gold_per_item: parse_optional_f64(&self.manual_gold_item)?,
            manual_gold_per_grid: parse_optional_f64(&self.manual_gold_grid)?,
            value_samples: self.parse_value_samples()?,
        };
        let results = core.run(cp.clone())?;
        let (p25, p50, p75) = core.price_range(&results, &cp);
        let purple_range = range_for(&results, |r| r.purple_count);
        let gold_range = range_for(&results, |r| r.gold_count);
        let red_range = range_for(&results, |r| r.red_count);
        let composition_lines = results
            .first()
            .map(|top| core.combo_composition_lines(top, &cp))
            .unwrap_or_default();
        let auto_fill = self.apply_unique_fields(&results);
        let rows = results
            .iter()
            .take(display_count)
            .map(|r| UiCombo {
                greenwhite: r.greenwhite_count,
                blue: r.blue_count,
                purple: r.purple_count,
                gold: r.gold_count,
                red: r.red_count,
                probability: r.probability,
                final_value: r.final_value,
                total_grid: r.total_grid_est,
            })
            .collect::<Vec<_>>();
        let map_label = self.selected_map_label();
        Ok((
            CalculationOutput {
                combos: results.len(),
                raw: core.raw_results.len(),
                p25: (p25 * cp.safety_factor).round() as i64,
                p50: (p50 * cp.safety_factor).round() as i64,
                p75: (p75 * cp.safety_factor).round() as i64,
                purple_range,
                gold_range,
                red_range,
                rows,
                elapsed_ms: start.elapsed().as_millis(),
                map_label,
                composition_lines,
            },
            auto_fill,
        ))
    }

    fn parse_value_samples(&self) -> Result<Vec<ValueSample>> {
        let mut samples = Vec::new();
        for (index, sample) in self.value_samples.iter().enumerate() {
            let count_text = sample.count.trim();
            let avg_text = sample.avg_value.trim();
            if count_text.is_empty() && avg_text.is_empty() {
                continue;
            }
            if count_text.is_empty() || avg_text.is_empty() {
                anyhow::bail!("随机样本第 {} 行需要同时填写件数和均价", index + 1);
            }
            let count = count_text
                .parse::<i32>()
                .with_context(|| format!("随机样本第 {} 行件数不是有效整数", index + 1))?;
            let avg_value = avg_text
                .parse::<f64>()
                .with_context(|| format!("随机样本第 {} 行均价不是有效数字", index + 1))?;
            if count <= 0 {
                anyhow::bail!("随机样本第 {} 行件数必须大于 0", index + 1);
            }
            if !avg_value.is_finite() || avg_value < 0.0 {
                anyhow::bail!("随机样本第 {} 行均价不能为负数", index + 1);
            }
            samples.push(ValueSample { count, avg_value });
        }
        Ok(samples)
    }

    fn merge_ocr_value_samples(&mut self, samples: &[ocr::OcrValueSample]) -> usize {
        let mut updated = 0;
        for sample in samples {
            if self.value_samples.iter().any(|existing| {
                existing.count.trim() == sample.count.trim()
                    && existing.avg_value.trim() == sample.avg_value.trim()
            }) {
                continue;
            }
            self.value_samples.push(ValueSampleInput {
                count: sample.count.clone(),
                avg_value: sample.avg_value.clone(),
            });
            updated += 1;
        }
        updated
    }

    fn normalize_display_count(&mut self) {
        let expected = MAX_VISIBLE_COMBOS.to_string();
        if self.display_count.trim() != expected {
            self.display_count = expected;
        }
    }

    fn reset_conditions(&mut self) {
        let maps = self.maps.clone();
        let global_hotkeys = self.global_hotkeys.take();
        let window_topmost = self.window_topmost;
        let mut fresh = Self::default();
        fresh.maps = maps;
        fresh.window_topmost = window_topmost;
        fresh.global_hotkeys = global_hotkeys;
        fresh.status = "已重置条件。".to_string();
        *self = fresh;
    }

    fn filtered_maps(&self) -> Vec<MapRow> {
        self.maps
            .iter()
            .filter(|m| tier_from_map_id(&m.map_id).unwrap_or("") == self.tier)
            .cloned()
            .collect()
    }

    fn select_first_map_for_tier(&mut self) {
        if let Some(map) = self.filtered_maps().first() {
            self.selected_map_id = map.map_id.clone();
        }
    }

    fn selected_map_label(&self) -> String {
        self.maps
            .iter()
            .find(|m| m.map_id == self.selected_map_id)
            .map(|m| format!("{} {} -> {}", m.map_id, m.name, m.nest_id))
            .unwrap_or_else(|| self.selected_map_id.clone())
    }
}

fn section<R>(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let outer_width = ui.available_width();
    egui::Frame::default()
        .fill(color_panel())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(13, 9))
        .show(ui, |ui| {
            ui.set_min_width((outer_width - 28.0).max(120.0));
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 15.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, egui::CornerRadius::same(2), color_gold());
                ui.add_space(5.0);
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .size(16.0)
                        .color(color_gold()),
                );
            });
            ui.add_space(7.0);
            add(ui)
        })
        .inner
}

fn field_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(13.0).color(color_muted()));
}

fn labeled_text(ui: &mut egui::Ui, label: &str, value: &mut String, width: f32) {
    field_label(ui, label);
    text_input(ui, value, width);
}

fn readout_text(ui: &mut egui::Ui, label: &str, value: &str, width: f32) {
    field_label(ui, label);
    egui::Frame::default()
        .fill(color_card())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(4)
        .inner_margin(egui::Margin::symmetric(6, 4))
        .show(ui, |ui| {
            ui.add_sized(
                [(width - 12.0).max(20.0), 24.0],
                egui::Label::new(egui::RichText::new(value).strong().color(color_gold())),
            );
        });
}

fn table_header(ui: &mut egui::Ui, label: &str, width: f32) {
    ui.add_sized(
        [width, 21.0],
        egui::Label::new(egui::RichText::new(label).color(color_muted()).size(13.0)),
    );
}

fn color_row(
    ui: &mut egui::Ui,
    label: &str,
    color: egui::Color32,
    count: &mut String,
    min_count: &mut String,
    grid: &mut String,
    avg: &mut String,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [58.0, 30.0],
            egui::Label::new(egui::RichText::new(label).strong().color(color)),
        );
        small_input(ui, count);
        small_input(ui, min_count);
        small_input(ui, grid);
        small_input(ui, avg);
    });
}

fn small_input(ui: &mut egui::Ui, value: &mut String) {
    text_input(ui, value, 66.0);
}

fn text_input(ui: &mut egui::Ui, value: &mut String, width: f32) {
    dark_widget_scope(ui, |ui| {
        egui::Frame::default()
            .fill(color_input())
            .stroke(egui::Stroke::new(1.0, color_border()))
            .corner_radius(4)
            .inner_margin(egui::Margin::symmetric(7, 5))
            .show(ui, |ui| {
                ui.add_sized(
                    [(width - 14.0).max(20.0), 24.0],
                    egui::TextEdit::singleline(value).frame(false),
                );
            });
    });
}

fn action_button(
    ui: &mut egui::Ui,
    label: &str,
    shortcut: &str,
    width: f32,
    fill: egui::Color32,
    text_color: egui::Color32,
) -> egui::Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(format!("{label}\n({shortcut})"))
                .strong()
                .color(text_color),
        )
        .min_size(egui::vec2(width, 46.0))
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6),
    )
}

fn pill_label(ui: &mut egui::Ui, text: &str, accent: egui::Color32) {
    egui::Frame::default()
        .fill(color_card())
        .stroke(egui::Stroke::new(1.0, accent))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(10, 3))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).strong().size(13.0).color(accent));
        });
}

fn status_bar(ui: &mut egui::Ui, text: &str) {
    egui::Frame::default()
        .fill(color_panel_alt())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(14.0).color(color_muted()));
        });
}

fn dark_widget_scope<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.scope(|ui| {
        force_dark_visuals(ui);
        add(ui)
    })
    .inner
}

fn force_dark_visuals(ui: &mut egui::Ui) {
    let visuals = ui.visuals_mut();
    visuals.override_text_color = Some(color_text());
    visuals.panel_fill = color_bg();
    visuals.window_fill = color_panel();
    visuals.faint_bg_color = color_panel_alt();
    visuals.extreme_bg_color = color_input();
    visuals.selection.bg_fill = color_blue();
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    visuals.widgets.noninteractive.bg_fill = color_panel();
    visuals.widgets.noninteractive.weak_bg_fill = color_panel();
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    visuals.widgets.inactive.bg_fill = color_input();
    visuals.widgets.inactive.weak_bg_fill = color_input();
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, color_border());
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    visuals.widgets.hovered.bg_fill = color_panel_alt();
    visuals.widgets.hovered.weak_bg_fill = color_panel_alt();
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, color_muted());
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    visuals.widgets.active.bg_fill = color_panel_alt();
    visuals.widgets.active.weak_bg_fill = color_panel_alt();
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, color_blue());
    visuals.widgets.active.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    visuals.widgets.open.bg_fill = color_panel_alt();
    visuals.widgets.open.weak_bg_fill = color_panel_alt();
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, color_border());
    visuals.widgets.open.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
}

fn empty_results(ui: &mut egui::Ui) {
    section(ui, "出价参考", |ui| {
        ui.columns(3, |cols| {
            price_placeholder(&mut cols[0], "保守出价 (P25)", color_green());
            price_placeholder(&mut cols[1], "均衡出价 (P50)", color_orange());
            price_placeholder(&mut cols[2], "激进出价 (P75)", color_red());
        });
    });
    ui.add_space(12.0);

    section(ui, "可行组合列表", |ui| {
        empty_table(ui);
    });
    ui.add_space(12.0);

    section(ui, "物品级可能组成（价格/占格拟合）", |ui| {
        ui.label(egui::RichText::new("暂无结果").color(color_muted()));
    });
}

fn summary_row(ui: &mut egui::Ui, label: &str, value: &str, suffix: &str, accent: egui::Color32) {
    egui::Frame::default()
        .fill(color_panel_alt())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(5)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(label).strong().color(color_text()));
                ui.label(egui::RichText::new(value).strong().color(accent));
                ui.label(egui::RichText::new(suffix).color(color_muted()));
            });
        });
    ui.add_space(6.0);
}

fn high_range_row(ui: &mut egui::Ui, output: &CalculationOutput) {
    egui::Frame::default()
        .fill(color_panel_alt())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(5)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.columns(4, |cols| {
                cols[0].label(
                    egui::RichText::new("品质件数可能性")
                        .strong()
                        .color(color_gold()),
                );
                range_label(
                    &mut cols[1],
                    "紫件(Q4)",
                    output.purple_range,
                    color_purple(),
                );
                range_label(&mut cols[2], "金件(Q5)", output.gold_range, color_gold());
                range_label(&mut cols[3], "红件(Q6)", output.red_range, color_red());
            });
        });
}

fn range_label(ui: &mut egui::Ui, label: &str, range: Option<CountRange>, color: egui::Color32) {
    ui.vertical(|ui| {
        ui.label(egui::RichText::new(label).size(13.0).color(color_muted()));
        let text = range
            .map(|r| format!("{} ~ {}", r.min, r.max))
            .unwrap_or_else(|| "--".to_string());
        ui.label(egui::RichText::new(text).strong().color(color));
    });
}

fn price_card(ui: &mut egui::Ui, label: &str, value: i64, accent: egui::Color32) {
    egui::Frame::default()
        .fill(color_card())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(16, 12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new(label).color(color_muted()).size(13.0));
                ui.label(
                    egui::RichText::new(format_number(value))
                        .size(30.0)
                        .strong()
                        .color(accent),
                );
                accent_bar(ui, accent);
            });
        });
}

fn price_placeholder(ui: &mut egui::Ui, label: &str, accent: egui::Color32) {
    egui::Frame::default()
        .fill(color_card())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(16, 12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new(label).color(color_muted()).size(13.0));
                ui.label(
                    egui::RichText::new("--")
                        .size(30.0)
                        .strong()
                        .color(color_muted()),
                );
                accent_bar(ui, accent);
            });
        });
}

fn accent_bar(ui: &mut egui::Ui, accent: egui::Color32) {
    ui.add_space(6.0);
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 3.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(2), accent);
}

fn result_table(ui: &mut egui::Ui, rows: &[UiCombo]) {
    let outer_width = ui.available_width();
    egui::Frame::default()
        .fill(color_panel())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_min_width((outer_width - 22.0).max(560.0));
            let col_width = (ui.available_width() / 8.0).max(64.0);
            egui::Grid::new("result_grid")
                .striped(true)
                .min_col_width(col_width)
                .min_row_height(25.0)
                .num_columns(8)
                .show(ui, |ui| {
                    table_head(ui, "绿白", color_gold());
                    table_head(ui, "蓝", color_gold());
                    table_head(ui, "紫", color_gold());
                    table_head(ui, "金", color_gold());
                    table_head(ui, "红", color_gold());
                    table_head(ui, "概率", color_gold());
                    table_head(ui, "组合估值", color_gold());
                    table_head(ui, "预计总格", color_gold());
                    ui.end_row();
                    for row in rows {
                        color_count(ui, row.greenwhite, color_gw());
                        color_count(ui, row.blue, color_blue());
                        color_count(ui, row.purple, color_purple());
                        color_count(ui, row.gold, color_gold());
                        color_count(ui, row.red, color_red());
                        ui.label(
                            egui::RichText::new(format_percent(row.probability))
                                .color(color_text()),
                        );
                        ui.label(
                            egui::RichText::new(format_number(row.final_value.round() as i64))
                                .color(color_text()),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.1}", row.total_grid))
                                .color(color_text()),
                        );
                        ui.end_row();
                    }
                });
        });
}

fn empty_table(ui: &mut egui::Ui) {
    let outer_width = ui.available_width();
    egui::Frame::default()
        .fill(color_panel())
        .stroke(egui::Stroke::new(1.0, color_border()))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_min_width((outer_width - 22.0).max(560.0));
            let col_width = (ui.available_width() / 8.0).max(64.0);
            egui::Grid::new("empty_result_grid")
                .striped(true)
                .min_col_width(col_width)
                .min_row_height(25.0)
                .num_columns(8)
                .show(ui, |ui| {
                    table_head(ui, "绿白", color_gold());
                    table_head(ui, "蓝", color_gold());
                    table_head(ui, "紫", color_gold());
                    table_head(ui, "金", color_gold());
                    table_head(ui, "红", color_gold());
                    table_head(ui, "概率", color_gold());
                    table_head(ui, "组合估值", color_gold());
                    table_head(ui, "预计总格", color_gold());
                    ui.end_row();
                    for _ in 0..4 {
                        for _ in 0..8 {
                            ui.label(egui::RichText::new("--").color(color_muted()));
                        }
                        ui.end_row();
                    }
                });
        });
}

fn table_head(ui: &mut egui::Ui, label: &str, color: egui::Color32) {
    ui.label(egui::RichText::new(label).strong().color(color));
}

fn color_count(ui: &mut egui::Ui, value: i32, color: egui::Color32) {
    ui.label(egui::RichText::new(value.to_string()).strong().color(color));
}

fn combo_label(row: &UiCombo) -> String {
    format!(
        "{}/{}/{}/{}/{} (绿白/蓝/紫/金/红)",
        row.greenwhite, row.blue, row.purple, row.gold, row.red
    )
}

fn range_for(
    rows: &[bidking_rs::ComboResult],
    f: impl Fn(&bidking_rs::ComboResult) -> i32,
) -> Option<CountRange> {
    let mut iter = rows.iter().map(f);
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    for value in iter {
        min = min.min(value);
        max = max.max(value);
    }
    Some(CountRange { min, max })
}

#[derive(Debug, Clone, Copy)]
struct TierOption {
    id: &'static str,
    name: &'static str,
}

impl TierOption {
    fn label(self) -> String {
        format!("{} {}", self.id, self.name)
    }
}

fn tier_options() -> &'static [TierOption] {
    &[
        TierOption {
            id: "101",
            name: "快递",
        },
        TierOption {
            id: "102",
            name: "仓库",
        },
        TierOption {
            id: "103",
            name: "集装箱",
        },
        TierOption {
            id: "104",
            name: "别墅",
        },
        TierOption {
            id: "105",
            name: "沉船",
        },
        TierOption {
            id: "106",
            name: "隐秘拍卖会",
        },
    ]
}

fn tier_label(tier: &str) -> String {
    let name = match tier {
        "101" => "快递",
        "102" => "仓库",
        "103" => "集装箱",
        "104" => "别墅",
        "105" => "沉船",
        "106" => "隐秘拍卖会",
        _ => "未知",
    };
    format!("{tier} {name}")
}

fn parse_optional_i32(text: &str) -> Result<Option<i32>> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    Ok(Some(text.parse()?))
}

fn parse_optional_f64(text: &str) -> Result<Option<f64>> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    Ok(Some(text.parse()?))
}

fn parse_i32(text: &str, name: &str) -> Result<i32> {
    text.trim()
        .parse()
        .with_context(|| format!("{name} 不是有效整数"))
}

fn parse_f64(text: &str, name: &str) -> Result<f64> {
    text.trim()
        .parse()
        .with_context(|| format!("{name} 不是有效数字"))
}

fn set_if_some(target: &mut String, value: &Option<String>) -> usize {
    if let Some(value) = value {
        *target = value.clone();
        1
    } else {
        0
    }
}

fn set_unique_i32(target: &mut String, value: Option<i32>) -> usize {
    let Some(value) = value else {
        return 0;
    };
    let value = value.to_string();
    if target.trim() == value {
        return 0;
    }
    *target = value;
    1
}

fn set_grid_from_average(
    target: &mut String,
    count_text: &str,
    avg_text: &str,
    direct_grid_seen: bool,
) -> usize {
    if direct_grid_seen {
        return 0;
    }
    let count = parse_optional_i32(count_text).ok().flatten();
    let avg = parse_optional_f64(avg_text).ok().flatten();
    set_unique_i32(target, infer_grid_from_average(count, avg))
}

fn infer_grid_from_average(count: Option<i32>, avg: Option<f64>) -> Option<i32> {
    let count = count?;
    let avg = avg?;
    if count <= 0 || !avg.is_finite() || avg <= 0.0 {
        return None;
    }
    let target = (avg * 100.0 + 1e-7).floor() as i32;
    let mut matches = (count..=18 * count)
        .filter(|grid| ((*grid as f64 * 100.0 / count as f64) + 1e-7).floor() as i32 == target);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn unique_i32(
    results: &[bidking_rs::ComboResult],
    f: impl Fn(&bidking_rs::ComboResult) -> i32,
) -> Option<i32> {
    let mut iter = results.iter();
    let first = f(iter.next()?);
    if iter.all(|result| f(result) == first) {
        Some(first)
    } else {
        None
    }
}

fn parse_i32_or_zero(text: &str) -> i32 {
    text.trim().parse().unwrap_or_default()
}

fn format_number(value: i64) -> String {
    let mut s = value.abs().to_string();
    let mut out = String::new();
    while s.len() > 3 {
        let tail = s.split_off(s.len() - 3);
        if out.is_empty() {
            out = tail;
        } else {
            out = format!("{tail},{out}");
        }
    }
    if out.is_empty() {
        out = s;
    } else {
        out = format!("{s},{out}");
    }
    if value < 0 { format!("-{out}") } else { out }
}

fn format_percent(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

fn tier_from_map_id(map_id: &str) -> Option<&'static str> {
    match map_id.chars().next()? {
        '2' => match map_id.chars().nth(1)? {
            '1' => Some("101"),
            '2' => Some("102"),
            '3' => Some("103"),
            '4' => Some("104"),
            '5' => Some("105"),
            '6' => Some("106"),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(target_os = "windows")]
fn start_global_hotkeys(ctx: egui::Context) -> Result<GlobalHotkeys> {
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        MOD_ALT, MOD_NOREPEAT, RegisterHotKey, UnregisterHotKey, VK_E, VK_Q, VK_W,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG, WM_HOTKEY};

    const HOTKEY_CALCULATE: i32 = 0x4251;
    const HOTKEY_SCAN_SCREEN: i32 = 0x4257;
    const HOTKEY_RESET_CONDITIONS: i32 = 0x4245;

    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();

    std::thread::Builder::new()
        .name("bidking-global-hotkey".to_string())
        .spawn(move || {
            let hwnd: HWND = null_mut();
            let mut registered = RegisteredHotkeys::default();
            let mut failures = Vec::new();

            if unsafe {
                RegisterHotKey(hwnd, HOTKEY_CALCULATE, MOD_ALT | MOD_NOREPEAT, VK_Q as u32)
            } == 0
            {
                failures.push(format!("Alt+Q: {}", std::io::Error::last_os_error()));
            } else {
                registered.calculate = true;
            }

            if unsafe {
                RegisterHotKey(
                    hwnd,
                    HOTKEY_SCAN_SCREEN,
                    MOD_ALT | MOD_NOREPEAT,
                    VK_W as u32,
                )
            } == 0
            {
                failures.push(format!("Alt+W: {}", std::io::Error::last_os_error()));
            } else {
                registered.scan_screen = true;
            }

            if unsafe {
                RegisterHotKey(
                    hwnd,
                    HOTKEY_RESET_CONDITIONS,
                    MOD_ALT | MOD_NOREPEAT,
                    VK_E as u32,
                )
            } == 0
            {
                failures.push(format!("Alt+E: {}", std::io::Error::last_os_error()));
            } else {
                registered.reset_conditions = true;
            }

            if !registered.any() {
                let _ = ready_tx.send(Err(failures.join("；")));
                return;
            }

            let warning = if failures.is_empty() {
                None
            } else {
                Some(format!("部分热键注册失败: {}", failures.join("；")))
            };
            let _ = ready_tx.send(Ok((registered, warning)));

            let mut msg: MSG = unsafe { std::mem::zeroed() };
            loop {
                let result = unsafe { GetMessageW(&mut msg, null_mut(), 0, 0) };
                if result <= 0 {
                    break;
                }
                if msg.message == WM_HOTKEY {
                    let action = match msg.wParam as i32 {
                        HOTKEY_CALCULATE => Some(GlobalHotkeyAction::Calculate),
                        HOTKEY_SCAN_SCREEN => Some(GlobalHotkeyAction::ScanScreen),
                        HOTKEY_RESET_CONDITIONS => Some(GlobalHotkeyAction::ResetConditions),
                        _ => None,
                    };
                    if let Some(action) = action {
                        if tx.send(action).is_err() {
                            break;
                        }
                        ctx.request_repaint();
                    }
                }
            }

            if registered.calculate {
                unsafe {
                    UnregisterHotKey(hwnd, HOTKEY_CALCULATE);
                }
            }
            if registered.scan_screen {
                unsafe {
                    UnregisterHotKey(hwnd, HOTKEY_SCAN_SCREEN);
                }
            }
            if registered.reset_conditions {
                unsafe {
                    UnregisterHotKey(hwnd, HOTKEY_RESET_CONDITIONS);
                }
            }
        })
        .context("启动全局热键线程失败")?;

    match ready_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok((registered, warning))) => Ok(GlobalHotkeys {
            rx,
            registered,
            warning,
        }),
        Ok(Err(err)) => Err(anyhow::anyhow!("{err}")),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow::anyhow!("等待热键注册超时")),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(anyhow::anyhow!("热键线程提前退出")),
    }
}

#[cfg(not(target_os = "windows"))]
fn start_global_hotkeys(_ctx: egui::Context) -> Result<GlobalHotkeys> {
    anyhow::bail!("当前只支持 Windows 全局热键")
}

fn install_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(24.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(13.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(14.0, egui::FontFamily::Monospace),
    );
    style.spacing.item_spacing = egui::vec2(8.0, 9.0);
    style.spacing.button_padding = egui::vec2(13.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(10);
    style.spacing.indent = 16.0;
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = color_bg();
    style.visuals.window_fill = color_panel();
    style.visuals.extreme_bg_color = color_input();
    style.visuals.override_text_color = Some(color_text());
    style.visuals.widgets.inactive.bg_fill = color_input();
    style.visuals.widgets.inactive.weak_bg_fill = color_input();
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, color_border());
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    style.visuals.widgets.hovered.bg_fill = color_panel_alt();
    style.visuals.widgets.hovered.weak_bg_fill = color_panel_alt();
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, color_muted());
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    style.visuals.widgets.active.bg_fill = color_panel_alt();
    style.visuals.widgets.active.weak_bg_fill = color_panel_alt();
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, color_blue());
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    style.visuals.widgets.open.bg_fill = color_panel_alt();
    style.visuals.widgets.open.weak_bg_fill = color_panel_alt();
    style.visuals.widgets.open.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(TEXT_STROKE, color_text());
    ctx.set_style(style);
}

fn install_chinese_font(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ];
    let Some((path, bytes)) = candidates
        .iter()
        .find_map(|path| std::fs::read(path).ok().map(|bytes| (*path, bytes)))
    else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    let font_name = format!(
        "cjk-{}",
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("font")
    );
    fonts
        .font_data
        .insert(font_name.clone(), egui::FontData::from_owned(bytes).into());
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, font_name.clone());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, font_name);
    ctx.set_fonts(fonts);
}

fn color_bg() -> egui::Color32 {
    egui::Color32::from_rgb(11, 17, 23)
}

fn color_panel() -> egui::Color32 {
    egui::Color32::from_rgb(25, 36, 52)
}

fn color_panel_alt() -> egui::Color32 {
    egui::Color32::from_rgb(18, 29, 42)
}

fn color_card() -> egui::Color32 {
    egui::Color32::from_rgb(14, 25, 35)
}

fn color_input() -> egui::Color32 {
    egui::Color32::from_rgb(15, 27, 40)
}

fn color_border() -> egui::Color32 {
    egui::Color32::from_rgb(47, 65, 88)
}

fn color_text() -> egui::Color32 {
    egui::Color32::from_rgb(232, 240, 250)
}

fn color_muted() -> egui::Color32 {
    egui::Color32::from_rgb(168, 193, 220)
}

fn color_gold() -> egui::Color32 {
    egui::Color32::from_rgb(255, 213, 74)
}

fn color_orange() -> egui::Color32 {
    egui::Color32::from_rgb(244, 172, 82)
}

fn color_green() -> egui::Color32 {
    egui::Color32::from_rgb(116, 211, 154)
}

fn color_green_dark() -> egui::Color32 {
    egui::Color32::from_rgb(27, 94, 68)
}

fn color_gw() -> egui::Color32 {
    egui::Color32::from_rgb(88, 225, 151)
}

fn color_blue() -> egui::Color32 {
    egui::Color32::from_rgb(45, 170, 255)
}

fn color_purple() -> egui::Color32 {
    egui::Color32::from_rgb(194, 93, 255)
}

fn color_red() -> egui::Color32 {
    egui::Color32::from_rgb(255, 113, 124)
}
