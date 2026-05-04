use anyhow::{Context, Result, bail};
use flate2::read::DeflateDecoder;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub exe: PathBuf,
    pub out_dir: PathBuf,
    pub bundle_entries: usize,
    pub extracted_files: Vec<PathBuf>,
    pub extracted_resources: Vec<PathBuf>,
    pub static_data: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct BundleEntry {
    offset: usize,
    size: usize,
    compressed_size: usize,
    kind: u8,
    name: String,
}

#[derive(Debug, Clone)]
struct Section {
    virtual_address: u32,
    virtual_size: u32,
    raw_ptr: u32,
    raw_size: u32,
}

#[derive(Debug, Clone)]
struct MetadataStream {
    offset: usize,
    size: usize,
}

pub fn import_exe(
    exe: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    static_template: Option<&Path>,
) -> Result<ImportReport> {
    let exe = exe.as_ref();
    let out_dir = out_dir.as_ref();
    let bundle_dir = out_dir.join("bundle");
    let resource_dir = out_dir.join("resources");
    fs::create_dir_all(&bundle_dir).with_context(|| format!("create {}", bundle_dir.display()))?;
    fs::create_dir_all(&resource_dir)
        .with_context(|| format!("create {}", resource_dir.display()))?;

    let bytes = fs::read(exe).with_context(|| format!("read {}", exe.display()))?;
    let entries = parse_bundle_manifest(&bytes)?;
    let manifest_path = out_dir.join("bundle_manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&entries_as_json(&entries))?,
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;

    let wanted_bundle_names = [
        "MapBidCalculator.dll",
        "MapBidCalculator.deps.json",
        "MapBidCalculator.runtimeconfig.json",
    ];
    let mut extracted_files = Vec::new();
    for name in wanted_bundle_names {
        if let Some(entry) = entries.iter().find(|entry| entry.name == name) {
            let data = read_bundle_entry(&bytes, entry)
                .with_context(|| format!("extract bundle entry {}", entry.name))?;
            let dst = bundle_dir.join(&entry.name);
            fs::write(&dst, data).with_context(|| format!("write {}", dst.display()))?;
            extracted_files.push(dst);
        }
    }

    let dll_path = bundle_dir.join("MapBidCalculator.dll");
    if !dll_path.exists() {
        bail!("MapBidCalculator.dll was not found in the single-file bundle");
    }

    let resources = extract_managed_resources(&dll_path)
        .with_context(|| format!("extract managed resources from {}", dll_path.display()))?;
    let wanted_resource_suffixes = [
        "calculator_data_merged.csv",
        "drop_table_weights.csv",
        "item_prices.csv",
    ];
    let mut extracted_resources = Vec::new();
    for suffix in wanted_resource_suffixes {
        let Some((name, data)) = resources.iter().find(|(name, _)| name.ends_with(suffix)) else {
            bail!("resource ending with {suffix} was not found in MapBidCalculator.dll");
        };
        let dst = resource_dir.join(name);
        fs::write(&dst, data).with_context(|| format!("write {}", dst.display()))?;
        extracted_resources.push(dst);
    }

    let static_data = if let Some(template) = static_template {
        let dst = out_dir.join("static_data.json");
        fs::copy(template, &dst)
            .with_context(|| format!("copy {} to {}", template.display(), dst.display()))?;
        Some(dst)
    } else {
        None
    };

    let report = ImportReport {
        exe: exe.to_path_buf(),
        out_dir: out_dir.to_path_buf(),
        bundle_entries: entries.len(),
        extracted_files,
        extracted_resources,
        static_data,
    };
    let report_path = out_dir.join("import_report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    Ok(report)
}

fn parse_bundle_manifest(bytes: &[u8]) -> Result<Vec<BundleEntry>> {
    let name = b"MapBidCalculator.dll";
    let name_pos = bytes
        .windows(name.len())
        .rposition(|window| window == name)
        .context("MapBidCalculator.dll was not found in bundle manifest")?;
    if name_pos < 26 {
        bail!("invalid bundle manifest entry position");
    }
    let mut p = name_pos - 26;
    let mut entries = Vec::new();
    while p + 26 <= bytes.len() {
        let offset = read_u64(bytes, p)? as usize;
        let size = read_u64(bytes, p + 8)? as usize;
        let compressed_size = read_u64(bytes, p + 16)? as usize;
        let kind = bytes[p + 24];
        let name_len = bytes[p + 25] as usize;
        p += 26;
        if name_len == 0 || p + name_len > bytes.len() {
            break;
        }
        let name = std::str::from_utf8(&bytes[p..p + name_len])
            .context("bundle entry name is not utf-8")?
            .to_string();
        p += name_len;
        let stored_size = if compressed_size > 0 {
            compressed_size
        } else {
            size
        };
        if offset >= bytes.len() || offset.saturating_add(stored_size) > bytes.len() {
            break;
        }
        entries.push(BundleEntry {
            offset,
            size,
            compressed_size,
            kind,
            name,
        });
    }
    if entries.is_empty() {
        bail!("no bundle entries parsed");
    }
    Ok(entries)
}

fn read_bundle_entry(bytes: &[u8], entry: &BundleEntry) -> Result<Vec<u8>> {
    let stored_size = if entry.compressed_size > 0 {
        entry.compressed_size
    } else {
        entry.size
    };
    let data = &bytes[entry.offset..entry.offset + stored_size];
    let out = if entry.compressed_size > 0 {
        let mut decoder = DeflateDecoder::new(data);
        let mut decoded = Vec::with_capacity(entry.size);
        decoder.read_to_end(&mut decoded)?;
        decoded
    } else {
        data.to_vec()
    };
    if out.len() != entry.size {
        bail!(
            "{} decoded size mismatch: got {}, expected {}",
            entry.name,
            out.len(),
            entry.size
        );
    }
    Ok(out)
}

fn extract_managed_resources(dll_path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let bytes = fs::read(dll_path).with_context(|| format!("read {}", dll_path.display()))?;
    let sections = parse_pe_sections(&bytes)?;
    let cli_rva = pe_data_directory(&bytes, 14)?.0;
    if cli_rva == 0 {
        bail!("assembly has no CLR header");
    }
    let cli = rva_to_offset(cli_rva, &sections).context("map CLR header RVA")?;
    let metadata_rva = read_u32(&bytes, cli + 8)?;
    let resources_rva = read_u32(&bytes, cli + 0x18)?;
    let metadata = rva_to_offset(metadata_rva, &sections).context("map metadata RVA")?;
    let resources = rva_to_offset(resources_rva, &sections).context("map resources RVA")?;
    let streams = parse_metadata_streams(&bytes, metadata)?;
    let tables = streams
        .get("#~")
        .or_else(|| streams.get("#-"))
        .context("metadata tables stream not found")?;
    let strings = streams
        .get("#Strings")
        .context("#Strings stream not found")?;
    let manifest = parse_manifest_resource_rows(&bytes, tables.offset, strings)?;

    let mut out = Vec::new();
    for row in manifest {
        if row.implementation != 0 {
            continue;
        }
        let start = resources + row.offset as usize;
        let len = read_u32(&bytes, start)? as usize;
        let data_start = start + 4;
        let data_end = data_start + len;
        if data_end > bytes.len() {
            bail!("resource {} points outside file", row.name);
        }
        out.push((row.name, bytes[data_start..data_end].to_vec()));
    }
    Ok(out)
}

fn parse_pe_sections(bytes: &[u8]) -> Result<Vec<Section>> {
    if bytes.get(0..2) != Some(b"MZ") {
        bail!("not a PE file");
    }
    let pe = read_u32(bytes, 0x3c)? as usize;
    if bytes.get(pe..pe + 4) != Some(b"PE\0\0") {
        bail!("invalid PE signature");
    }
    let section_count = read_u16(bytes, pe + 6)? as usize;
    let optional_size = read_u16(bytes, pe + 20)? as usize;
    let section_table = pe + 24 + optional_size;
    let mut sections = Vec::with_capacity(section_count);
    for idx in 0..section_count {
        let p = section_table + idx * 40;
        sections.push(Section {
            virtual_size: read_u32(bytes, p + 8)?,
            virtual_address: read_u32(bytes, p + 12)?,
            raw_size: read_u32(bytes, p + 16)?,
            raw_ptr: read_u32(bytes, p + 20)?,
        });
    }
    Ok(sections)
}

fn pe_data_directory(bytes: &[u8], index: usize) -> Result<(u32, u32)> {
    let pe = read_u32(bytes, 0x3c)? as usize;
    let optional = pe + 24;
    let magic = read_u16(bytes, optional)?;
    let data_dir_start = match magic {
        0x10b => optional + 96,
        0x20b => optional + 112,
        _ => bail!("unsupported PE optional header magic {magic:#x}"),
    };
    let p = data_dir_start + index * 8;
    Ok((read_u32(bytes, p)?, read_u32(bytes, p + 4)?))
}

fn rva_to_offset(rva: u32, sections: &[Section]) -> Option<usize> {
    for section in sections {
        let start = section.virtual_address;
        let span = section.virtual_size.max(section.raw_size);
        let end = start.saturating_add(span);
        if rva >= start && rva < end {
            return Some((section.raw_ptr + (rva - start)) as usize);
        }
    }
    None
}

fn parse_metadata_streams(
    bytes: &[u8],
    metadata: usize,
) -> Result<HashMap<String, MetadataStream>> {
    if read_u32(bytes, metadata)? != 0x424a5342 {
        bail!("invalid metadata root signature");
    }
    let version_len = read_u32(bytes, metadata + 12)? as usize;
    let after_version = align4(metadata + 16 + version_len);
    let stream_count = read_u16(bytes, after_version + 2)? as usize;
    let mut p = after_version + 4;
    let mut streams = HashMap::new();
    for _ in 0..stream_count {
        let offset = read_u32(bytes, p)? as usize;
        let size = read_u32(bytes, p + 4)? as usize;
        p += 8;
        let name_start = p;
        while p < bytes.len() && bytes[p] != 0 {
            p += 1;
        }
        let name = std::str::from_utf8(&bytes[name_start..p])
            .context("metadata stream name is not utf-8")?
            .to_string();
        p = align4(p + 1);
        streams.insert(
            name,
            MetadataStream {
                offset: metadata + offset,
                size,
            },
        );
    }
    Ok(streams)
}

#[derive(Debug, Clone)]
struct ManifestResourceRow {
    offset: u32,
    name: String,
    implementation: u32,
}

fn parse_manifest_resource_rows(
    bytes: &[u8],
    tables: usize,
    strings: &MetadataStream,
) -> Result<Vec<ManifestResourceRow>> {
    let heap_sizes = bytes[tables + 6];
    let string_index_size = if heap_sizes & 0x01 != 0 { 4 } else { 2 };
    let guid_index_size = if heap_sizes & 0x02 != 0 { 4 } else { 2 };
    let blob_index_size = if heap_sizes & 0x04 != 0 { 4 } else { 2 };
    let valid = read_u64(bytes, tables + 8)?;
    let mut row_counts = [0u32; 64];
    let mut p = tables + 24;
    for (table, count) in row_counts.iter_mut().enumerate() {
        if (valid >> table) & 1 == 1 {
            *count = read_u32(bytes, p)?;
            p += 4;
        }
    }

    let mut table_offsets = [0usize; 64];
    let mut current = p;
    for table in 0..64 {
        table_offsets[table] = current;
        current += row_counts[table] as usize
            * table_row_size(
                table,
                &row_counts,
                string_index_size,
                guid_index_size,
                blob_index_size,
            );
    }

    let manifest_table = 40usize;
    let row_size = table_row_size(
        manifest_table,
        &row_counts,
        string_index_size,
        guid_index_size,
        blob_index_size,
    );
    let coded_impl_size = coded_index_size(&row_counts, &[38, 35, 39], 2);
    let mut rows = Vec::new();
    for row in 0..row_counts[manifest_table] as usize {
        let r = table_offsets[manifest_table] + row * row_size;
        let offset = read_u32(bytes, r)?;
        let name_index = read_index(bytes, r + 8, string_index_size)?;
        let implementation = read_index(bytes, r + 8 + string_index_size, coded_impl_size)?;
        let name = read_string_heap(bytes, strings, name_index)?;
        rows.push(ManifestResourceRow {
            offset,
            name,
            implementation,
        });
    }
    Ok(rows)
}

fn table_row_size(
    table: usize,
    rows: &[u32; 64],
    str_size: usize,
    guid_size: usize,
    blob_size: usize,
) -> usize {
    let idx = |table: usize| table_index_size(rows[table]);
    let coded = |tables: &[usize], bits: usize| coded_index_size(rows, tables, bits);
    match table {
        0 => 2 + str_size + guid_size * 3,
        1 => coded(&[0, 26, 35, 1], 2) + str_size + str_size,
        2 => 4 + str_size + str_size + coded(&[2, 1, 27], 2) + idx(4) + idx(6),
        4 => 2 + str_size + blob_size,
        6 => 4 + 2 + 2 + str_size + blob_size + idx(8),
        8 => 2 + 2 + str_size,
        9 => idx(2) + coded(&[2, 1, 27], 2),
        10 => coded(&[2, 1, 26, 6, 27], 3) + str_size + blob_size,
        11 => 2 + coded(&[4, 8, 23], 2) + blob_size,
        12 => {
            coded(
                &[
                    6, 4, 1, 2, 8, 9, 10, 0, 14, 23, 20, 17, 26, 27, 32, 35, 38, 39, 40, 42, 44,
                    43, 45,
                ],
                5,
            ) + coded(&[0, 0, 6, 10, 0], 3)
                + blob_size
        }
        13 => coded(&[4, 8], 1) + blob_size,
        14 => 2 + coded(&[2, 6, 32], 2) + blob_size,
        15 => 2 + 4 + idx(2),
        16 => 4 + idx(4),
        17 => blob_size,
        18 => idx(2) + idx(20),
        20 => 2 + str_size + coded(&[2, 1, 27], 2),
        21 => idx(2) + idx(23),
        23 => 2 + str_size + blob_size,
        24 => 2 + idx(6) + coded(&[20, 23], 1),
        25 => idx(2) + coded(&[6, 10], 1) + coded(&[6, 10], 1),
        26 => str_size,
        27 => blob_size,
        28 => 2 + coded(&[4, 6], 1) + str_size + idx(26),
        29 => 4 + idx(4),
        32 => 4 + 2 + 2 + 2 + 2 + 4 + blob_size + str_size + str_size,
        35 => 2 + 2 + 2 + 2 + 4 + blob_size + str_size + str_size + blob_size,
        38 => 4 + str_size + blob_size,
        39 => 4 + 4 + str_size + str_size + coded(&[38, 35, 39], 2),
        40 => 4 + 4 + str_size + coded(&[38, 35, 39], 2),
        41 => idx(2) + idx(2),
        42 => 2 + 2 + coded(&[2, 6], 1) + str_size,
        43 => idx(42) + coded(&[2, 1, 27], 2),
        44 => coded(&[6, 10], 1) + blob_size,
        _ => 0,
    }
}

fn table_index_size(rows: u32) -> usize {
    if rows < 0x10000 { 2 } else { 4 }
}

fn coded_index_size(rows: &[u32; 64], tables: &[usize], tag_bits: usize) -> usize {
    let max_rows = tables.iter().map(|table| rows[*table]).max().unwrap_or(0);
    if max_rows < (1u32 << (16 - tag_bits)) {
        2
    } else {
        4
    }
}

fn read_string_heap(bytes: &[u8], strings: &MetadataStream, index: u32) -> Result<String> {
    if index == 0 {
        return Ok(String::new());
    }
    let start = strings.offset + index as usize;
    let end_limit = strings.offset + strings.size;
    if start >= end_limit || end_limit > bytes.len() {
        bail!("string heap index out of range");
    }
    let mut end = start;
    while end < end_limit && bytes[end] != 0 {
        end += 1;
    }
    Ok(std::str::from_utf8(&bytes[start..end])
        .context("string heap value is not utf-8")?
        .to_string())
}

fn entries_as_json(entries: &[BundleEntry]) -> Vec<serde_json::Value> {
    entries
        .iter()
        .map(|entry| {
            serde_json::json!({
                "offset": entry.offset,
                "size": entry.size,
                "compressed_size": entry.compressed_size,
                "type": entry.kind,
                "name": entry.name,
            })
        })
        .collect()
}

fn read_index(bytes: &[u8], offset: usize, size: usize) -> Result<u32> {
    match size {
        2 => Ok(read_u16(bytes, offset)? as u32),
        4 => read_u32(bytes, offset),
        _ => bail!("invalid metadata index size {size}"),
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .with_context(|| format!("read u16 at {offset}"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .with_context(|| format!("read u32 at {offset}"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .with_context(|| format!("read u64 at {offset}"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn align4(value: usize) -> usize {
    (value + 3) & !3
}
