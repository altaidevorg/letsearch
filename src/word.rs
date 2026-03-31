//! Extract plain text from Word `.docx` (OOXML) and legacy `.doc` files.
//!
//! `.docx` is read in-process (ZIP + `word/document.xml`).
//! `.doc` uses an external tool if available: `antiword`, `catdoc`, or LibreOffice (`soffice`).

use anyhow::{anyhow, Context};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

/// Read `.docx` or `.doc` and return UTF-8 plain text (lossy where needed).
pub fn word_document_to_plain_text(path: &Path) -> anyhow::Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "docx" => docx_to_plain_text(path),
        "doc" => legacy_doc_to_plain_text(path),
        _ => Err(anyhow!(
            "expected .doc or .docx, got extension {:?}",
            path.extension()
        )),
    }
}

fn docx_to_plain_text(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("not a valid .docx zip: {}", path.display()))?;
    let mut xml = String::new();
    {
        let mut entry = archive
            .by_name("word/document.xml")
            .map_err(|_| anyhow!("word/document.xml missing (not a valid .docx)"))?;
        entry
            .read_to_string(&mut xml)
            .with_context(|| format!("read word/document.xml in {}", path.display()))?;
    }

    let mut reader = Reader::from_str(&xml);
    reader.config_mut().trim_text(false);

    let mut out = String::new();
    let mut in_w_t = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) | Event::Empty(ref e) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"t" => in_w_t = true,
                    b"tab" => out.push('\t'),
                    b"br" | b"cr" => out.push('\n'),
                    _ => {}
                }
            }
            Event::Text(e) => {
                if in_w_t {
                    out.push_str(&e.unescape()?);
                }
            }
            Event::End(ref e) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"t" => in_w_t = false,
                    b"p" => out.push('\n'),
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    Ok(out)
}

fn legacy_doc_to_plain_text(path: &Path) -> anyhow::Result<String> {
    if let Ok(s) = try_antiword(path) {
        return Ok(s);
    }
    if let Ok(s) = try_catdoc(path) {
        return Ok(s);
    }
    if let Ok(s) = try_soffice_txt(path) {
        return Ok(s);
    }
    Err(anyhow!(
        "Could not read .doc '{}'. Install one of: antiword, catdoc, or LibreOffice (soffice in PATH), \
         or convert the file to .docx.",
        path.display()
    ))
}

fn try_antiword(path: &Path) -> anyhow::Result<String> {
    let out = Command::new("antiword")
        .arg(path.as_os_str())
        .output()
        .map_err(|_| anyhow!("antiword not available"))?;
    if !out.status.success() {
        return Err(anyhow!("antiword failed"));
    }
    String::from_utf8(out.stdout).map_err(|e| anyhow!("antiword output not UTF-8: {}", e))
}

fn try_catdoc(path: &Path) -> anyhow::Result<String> {
    let out = Command::new("catdoc")
        .arg(path.as_os_str())
        .output()
        .map_err(|_| anyhow!("catdoc not available"))?;
    if !out.status.success() {
        return Err(anyhow!("catdoc failed"));
    }
    String::from_utf8(out.stdout).map_err(|e| anyhow!("catdoc output not UTF-8: {}", e))
}

fn try_soffice_txt(path: &Path) -> anyhow::Result<String> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad .doc path"))?;

    let mut last_err = anyhow!("LibreOffice (soffice / libreoffice) not available");
    for bin in ["soffice", "libreoffice"] {
        let out_dir = std::env::temp_dir().join(format!(
            "letsearch_doc_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            bin
        ));
        if std::fs::create_dir_all(&out_dir).is_err() {
            continue;
        }
        let txt_path = out_dir.join(format!("{}.txt", stem));

        let status = match Command::new(bin)
            .args([
                "--headless",
                "--nologo",
                "--nofirststartwizard",
                "--convert-to",
                "txt:Text",
            ])
            .arg(path.as_os_str())
            .arg("--outdir")
            .arg(&out_dir)
            .status()
        {
            Ok(s) => s,
            Err(e) => {
                last_err = anyhow!("{}: {}", bin, e);
                let _ = std::fs::remove_dir_all(&out_dir);
                continue;
            }
        };

        let read = if status.success() && txt_path.is_file() {
            std::fs::read_to_string(&txt_path).map_err(|e| anyhow!("read converted txt: {}", e))
        } else {
            Err(anyhow!("{} conversion failed or output missing", bin))
        };
        let _ = std::fs::remove_dir_all(&out_dir);
        match read {
            Ok(s) => return Ok(s),
            Err(e) => last_err = e,
        }
    }

    Err(last_err)
}
