use bidking_rs::ocr::{
    default_capture_rect, default_min_value_rect, default_round_rect, normalize_ocr_text,
    parse_ocr_lines,
};

#[test]
fn capture_rect_matches_original_ratio_for_1080p() {
    assert_eq!(
        default_capture_rect(1920, 1080),
        bidking_rs::ocr::CaptureRect {
            x: 489,
            y: 113,
            width: 662,
            height: 626,
        }
    );
}

#[test]
fn min_value_capture_rect_matches_right_loot_panel_for_1080p() {
    assert_eq!(
        default_min_value_rect(1920, 1080),
        bidking_rs::ocr::CaptureRect {
            x: 1411,
            y: 901,
            width: 480,
            height: 81,
        }
    );
}

#[test]
fn round_capture_rect_matches_top_center_round_badge_for_1080p() {
    assert_eq!(
        default_round_rect(1920, 1080),
        bidking_rs::ocr::CaptureRect {
            x: 643,
            y: 75,
            width: 364,
            height: 54,
        }
    );
}

#[test]
fn parser_handles_original_style_lines() {
    let lines = vec![
        "第2轮".to_string(),
        "小区快递:竞拍信息".to_string(),
        "本次竞拍的总藏品数量为23件".to_string(),
        "随机显示4件藏品".to_string(),
        "本次竞拍橙色品质藏品平均格数约为1格".to_string(),
    ];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.round_number.as_deref(), Some("2"));
    assert_eq!(result.map_name.as_deref(), Some("小区快递"));
    assert_eq!(result.total_all.as_deref(), Some("23"));
    assert_eq!(result.gold_avg.as_deref(), Some("1"));
}

#[test]
fn parser_keeps_multiple_random_value_samples() {
    let lines = vec![
        "硬核资产仓库:竞拍信息".to_string(),
        "本次竞拍的总藏品数量为63件".to_string(),
        "随机选择的3件藏品平均价值约为611.33".to_string(),
        "随机选择的6件藏品平均价值约为1412.21".to_string(),
    ];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.value_samples.len(), 2);
    assert_eq!(result.value_samples[0].count, "3");
    assert_eq!(result.value_samples[0].avg_value, "611.33");
    assert_eq!(result.value_samples[1].count, "6");
    assert_eq!(result.value_samples[1].avg_value, "1412.21");
}

#[test]
fn parser_normalizes_grouped_random_value_sample_decimal() {
    let lines = vec!["随机选择的6件藏品平均价值约为1,412.21".to_string()];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.value_samples.len(), 1);
    assert_eq!(result.value_samples[0].count, "6");
    assert_eq!(result.value_samples[0].avg_value, "1412.21");
}

#[test]
fn parser_ignores_nonpositive_fallback_total_ceiling() {
    let lines = vec!["随机选择的3件藏品平均价值约为611.33".to_string()];

    let result = parse_ocr_lines(&lines, Some(0));

    assert_eq!(result.value_samples.len(), 1);
    assert_eq!(result.value_samples[0].count, "3");
    assert_eq!(result.value_samples[0].avg_value, "611.33");
}

#[test]
fn parser_extracts_current_min_value_floor() {
    let lines = vec![
        "战利品".to_string(),
        "当前预估最低价格： 10,114".to_string(),
    ];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.min_value_floor.as_deref(), Some("10114"));
}

#[test]
fn parser_treats_dot_grouped_min_value_as_thousands() {
    let lines = vec!["当前预估最低价格:10.114".to_string()];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.min_value_floor.as_deref(), Some("10114"));
}

#[test]
fn parser_keeps_large_color_grid_when_count_allows_it() {
    let lines = vec![
        "本次竞拍的总藏品数量为110件".to_string(),
        "本场拍卖共有蓝色品质道具52件".to_string(),
        "所有蓝色品质藏品总占位数为130格".to_string(),
    ];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.blue_count.as_deref(), Some("52"));
    assert_eq!(result.blue_grid.as_deref(), Some("130"));
}

#[test]
fn parser_trims_large_color_grid_trailing_noise_against_color_count() {
    let lines = vec![
        "本次竞拍的总藏品数量为110件".to_string(),
        "本场拍卖共有蓝色品质道具52件".to_string(),
        "所有蓝色品质藏品总占位数为1301格".to_string(),
    ];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.blue_grid.as_deref(), Some("130"));
}

#[test]
fn parser_handles_noisy_ocr_lines_from_sample() {
    let lines = vec![
        "万 7 - 许 硬 核 资 产 仑 库 竞 拍 信 息".to_string(),
        "Ma 本 次 站 拍 楣 色 品 质 藏 品 平 均 格 数 约 为 0 格".to_string(),
        "Ma 本 次 站 拍 紫 色 品 质 藏 品 平 均 格 数 约 为 3.25 格".to_string(),
        "虞 0 本 次 站 拍 蓝 色 品 质 藏 品 平 均 格 数 约 为 3.3 格".to_string(),
        "Ma 本 仁 站 拍 白 色 和 绿 色 品 质 葛 品 数 量 为 16 件".to_string(),
    ];

    let result = parse_ocr_lines(&lines, Some(63));

    assert_eq!(result.map_name.as_deref(), Some("硬核资产仓库"));
    assert_eq!(result.gold_avg.as_deref(), Some("0"));
    assert_eq!(result.purple_avg.as_deref(), Some("3.25"));
    assert_eq!(result.blue_avg.as_deref(), Some("3.3"));
    assert_eq!(result.wg_count.as_deref(), Some("16"));
}

#[test]
fn normalizer_removes_spaces_and_common_ocr_confusions() {
    assert_eq!(
        normalize_ocr_text("本 次 站 拍 白 色 和 绿 色 品 质 葛 品 数 量 为 16 件"),
        "本次竞拍白色和绿色品质藏品数量为16件"
    );
}
