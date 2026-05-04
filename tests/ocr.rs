use bidking_rs::ocr::{default_capture_rect, normalize_ocr_text, parse_ocr_lines};

#[test]
fn capture_rect_matches_original_ratio_for_1080p() {
    assert_eq!(
        default_capture_rect(1920, 1080),
        bidking_rs::ocr::CaptureRect {
            x: 432,
            y: 113,
            width: 710,
            height: 685,
        }
    );
}

#[test]
fn parser_handles_original_style_lines() {
    let lines = vec![
        "小区快递:竞拍信息".to_string(),
        "本次竞拍的总藏品数量为23件".to_string(),
        "随机显示4件藏品".to_string(),
        "本次竞拍橙色品质藏品平均格数约为1格".to_string(),
    ];

    let result = parse_ocr_lines(&lines, None);

    assert_eq!(result.map_name.as_deref(), Some("小区快递"));
    assert_eq!(result.total_all.as_deref(), Some("23"));
    assert_eq!(result.gold_avg.as_deref(), Some("1"));
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
