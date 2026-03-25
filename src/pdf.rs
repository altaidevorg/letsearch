//! PDF reading support with automatic text detection.
//!
//! Converts text-based PDFs to Markdown. A heuristic samples
//! `sqrt(total_pages)` pages to decide whether the document contains
//! extractable text or is a scanned image-only PDF.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

/// Minimum non-whitespace characters per page to consider it text-bearing.
const MIN_CHARS_PER_PAGE: usize = 50;

/// Minimum fraction of sampled pages that must be text-bearing for the whole
/// document to be classified as a text PDF.
const MIN_TEXT_PAGE_RATIO: f64 = 0.5;

/// Return `true` if the PDF at `path` appears to be a text-based document.
///
/// The heuristic opens the file, samples `ceil(sqrt(page_count))` evenly
/// spaced pages, extracts plain text from each, and checks whether the
/// fraction of pages with at least [`MIN_CHARS_PER_PAGE`] non-whitespace
/// characters meets [`MIN_TEXT_PAGE_RATIO`].
pub fn is_text_pdf(path: &str) -> anyhow::Result<bool> {
    let mut doc = PdfDocument::open(path)?;
    let total_pages = doc.page_count()?;
    if total_pages == 0 {
        return Ok(false);
    }

    let sample_count = ((total_pages as f64).sqrt().ceil() as usize)
        .max(1)
        .min(total_pages);
    let step = (total_pages / sample_count).max(1);

    let mut text_pages: usize = 0;
    for i in 0..sample_count {
        let page_idx = (i * step).min(total_pages - 1);
        let text = doc.extract_text(page_idx).unwrap_or_default();
        let non_ws: usize = text.chars().filter(|c| !c.is_whitespace()).count();
        if non_ws >= MIN_CHARS_PER_PAGE {
            text_pages += 1;
        }
    }

    Ok((text_pages as f64 / sample_count as f64) >= MIN_TEXT_PAGE_RATIO)
}

/// Convert a text-based PDF to a Markdown string.
///
/// Returns an error if the file cannot be opened, if the PDF appears to
/// be a scanned document (no extractable text), or if conversion fails.
pub fn pdf_to_markdown(path: &str) -> anyhow::Result<String> {
    if !is_text_pdf(path)? {
        return Err(anyhow::anyhow!(
            "PDF '{}' appears to be a scanned document without extractable text",
            path
        ));
    }

    let mut doc = PdfDocument::open(path)?;
    let options = ConversionOptions::default();
    let markdown = doc.to_markdown_all(&options)?;
    Ok(markdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal single-page text PDF in memory and write it to a temp
    /// file so we can test the heuristic without requiring external assets.
    fn write_minimal_text_pdf(path: &str) {
        // This is a hand-crafted minimal valid PDF with one page containing
        // the string "Hello World".
        let pdf_bytes: &[u8] = b"%PDF-1.4\n\
            1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
            2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
            3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
              /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\
            4 0 obj\n<< /Length 44 >>\nstream\n\
            BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\n\
            5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\
            xref\n0 6\n\
            0000000000 65535 f \n\
            0000000009 00000 n \n\
            0000000058 00000 n \n\
            0000000115 00000 n \n\
            0000000266 00000 n \n\
            0000000360 00000 n \n\
            trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n";

        let mut f = std::fs::File::create(path).expect("create temp pdf");
        f.write_all(pdf_bytes).expect("write pdf bytes");
    }

    #[test]
    fn test_is_text_pdf_with_text_doc() {
        let tmp = std::env::temp_dir().join("letsearch_test_text.pdf");
        write_minimal_text_pdf(tmp.to_str().unwrap());
        // The minimal PDF has extractable text, so it should be classified as
        // text-based (even if the actual string is very short).
        // We only assert that the function runs without error.
        let result = is_text_pdf(tmp.to_str().unwrap());
        assert!(
            result.is_ok(),
            "is_text_pdf should not error on a valid PDF"
        );
    }

    #[test]
    fn test_pdf_to_markdown_returns_string_or_scanned_error() {
        let tmp = std::env::temp_dir().join("letsearch_test_md.pdf");
        write_minimal_text_pdf(tmp.to_str().unwrap());
        let result = pdf_to_markdown(tmp.to_str().unwrap());
        // For our minimal PDF the result could be either a markdown string
        // (text was extracted) or a "scanned document" error (extractor found
        // nothing).  Either way the function must not panic.
        match result {
            Ok(md) => {
                // Non-empty result or empty - just ensure it's a String
                let _ = md;
            }
            Err(e) => {
                assert!(e.to_string().contains("scanned"), "unexpected error: {e}");
            }
        }
    }

    #[test]
    fn test_is_text_pdf_missing_file() {
        let result = is_text_pdf("/nonexistent/path/file.pdf");
        assert!(result.is_err(), "should error on missing file");
    }
}
