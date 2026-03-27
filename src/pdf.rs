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

/// Inner check that works on an already-open document, avoiding a second
/// file-open in [`pdf_to_markdown`].
fn check_text_pdf(doc: &mut PdfDocument) -> anyhow::Result<bool> {
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

/// Return `true` if the PDF at `path` appears to be a text-based document.
///
/// The heuristic opens the file, samples `ceil(sqrt(page_count))` evenly
/// spaced pages, extracts plain text from each, and checks whether the
/// fraction of pages with at least [`MIN_CHARS_PER_PAGE`] non-whitespace
/// characters meets [`MIN_TEXT_PAGE_RATIO`].
pub fn is_text_pdf(path: &str) -> anyhow::Result<bool> {
    let mut doc = PdfDocument::open(path)?;
    check_text_pdf(&mut doc)
}

/// Convert a text-based PDF to a Markdown string.
///
/// The PDF is opened **once**: the text-heuristic check and the full
/// Markdown conversion both operate on the same in-memory document,
/// avoiding redundant file I/O.
///
/// Returns an error if the file cannot be opened, if the PDF appears to
/// be a scanned document (no extractable text), or if conversion fails.
pub fn pdf_to_markdown(path: &str) -> anyhow::Result<String> {
    let mut doc = PdfDocument::open(path)?;
    if !check_text_pdf(&mut doc)? {
        return Err(anyhow::anyhow!(
            "PDF '{}' appears to be a scanned document without extractable text",
            path
        ));
    }
    let options = ConversionOptions::default();
    let markdown = doc.to_markdown_all(&options)?;
    Ok(markdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a single-page text PDF with ≥50 non-whitespace characters so that
    /// [`is_text_pdf`] reliably classifies it as text-based (passes
    /// `MIN_CHARS_PER_PAGE`).
    fn write_minimal_text_pdf(path: &str) {
        // Hand-crafted minimal valid PDF (PDF 1.4).
        // Content stream contains 50 non-whitespace ASCII characters using the
        // Helvetica Type1 font.  The /Length value and all xref byte-offsets
        // have been computed exactly.
        let pdf_bytes: &[u8] = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 82 >>\nstream\nBT /F1 12 Tf 50 700 Td (AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEE) Tj ET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000243 00000 n \n0000000374 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n444\n%%EOF\n";

        let mut f = std::fs::File::create(path).expect("create temp pdf");
        f.write_all(pdf_bytes).expect("write pdf bytes");
    }

    #[test]
    fn test_is_text_pdf_with_text_doc() {
        let tmp = std::env::temp_dir().join("letsearch_test_text.pdf");
        write_minimal_text_pdf(tmp.to_str().unwrap());
        // The PDF contains 50 non-whitespace characters on its single page,
        // which meets MIN_CHARS_PER_PAGE (50). is_text_pdf must return Ok(true).
        let result = is_text_pdf(tmp.to_str().unwrap());
        assert!(
            result.is_ok(),
            "is_text_pdf should not error on a valid PDF"
        );
        assert_eq!(
            result.unwrap(),
            true,
            "PDF with >=50 non-whitespace chars should be classified as text-based"
        );
    }

    #[test]
    fn test_pdf_to_markdown_succeeds_on_text_pdf() {
        let tmp = std::env::temp_dir().join("letsearch_test_md.pdf");
        write_minimal_text_pdf(tmp.to_str().unwrap());
        // The PDF has enough text to pass the heuristic, so conversion must succeed.
        let result = pdf_to_markdown(tmp.to_str().unwrap());
        assert!(
            result.is_ok(),
            "pdf_to_markdown should succeed on a text-based PDF, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_is_text_pdf_missing_file() {
        let result = is_text_pdf("/nonexistent/path/file.pdf");
        assert!(result.is_err(), "should error on missing file");
    }
}
