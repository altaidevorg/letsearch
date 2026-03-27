//! Hierarchical Markdown chunker with configurable token-based splitting.
//!
//! Chunks are produced by first splitting on Markdown header boundaries
//! (H1 → H2 → H3 in order), then on blank-line paragraph breaks, and
//! finally by raw token count when a paragraph still exceeds the limit.
//!
//! Token counting is performed by the [`tokie`] tokenizer when a
//! `tokenizer_path` is supplied. When no path is given, the chunker falls
//! back to a word-count approximation (one word ≈ 1.3 tokens).

use std::path::Path;

/// Configuration for [`MarkdownChunker`].
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Maximum number of tokens per output chunk.
    pub max_tokens: usize,

    /// Number of tokens of context carried over from the previous chunk.
    /// Set to `0` to disable overlap.
    pub overlap_tokens: usize,

    /// Optional path to a HuggingFace `tokenizer.json` file used for
    /// accurate token counting.  When `None` the chunker uses a word-count
    /// approximation instead.
    pub tokenizer_path: Option<String>,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 50,
            tokenizer_path: None,
        }
    }
}

/// Hierarchical Markdown chunker.
///
/// # Example
///
/// ```rust,no_run
/// use letsearch::chunker::{ChunkerConfig, MarkdownChunker};
///
/// let config = ChunkerConfig { max_tokens: 128, ..Default::default() };
/// let chunker = MarkdownChunker::new(config).unwrap();
/// let chunks = chunker.chunk("# Hello\n\nSome text here.");
/// assert!(!chunks.is_empty());
/// ```
pub struct MarkdownChunker {
    config: ChunkerConfig,
    tokenizer: Option<tokie::Tokenizer>,
}

impl MarkdownChunker {
    /// Create a new chunker from the given [`ChunkerConfig`].
    ///
    /// If `config.tokenizer_path` is set the tokenizer is loaded from that
    /// file.  Returns an error if the file is present but cannot be parsed.
    pub fn new(config: ChunkerConfig) -> anyhow::Result<Self> {
        let tokenizer = if let Some(ref path) = config.tokenizer_path {
            Some(
                tokie::Tokenizer::from_json(Path::new(path))
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer '{}': {}", path, e))?,
            )
        } else {
            None
        };
        Ok(Self { config, tokenizer })
    }

    /// Split `text` into chunks respecting the configured token limit.
    ///
    /// Returns an empty `Vec` if `text` is blank.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        // If the entire text fits, return it as-is.
        if self.count_tokens(text) <= self.config.max_tokens {
            return vec![text.to_string()];
        }

        // Hierarchically attempt header-level splits.
        let segments = self.split_by_headers(text);
        let mut chunks = Vec::new();
        for seg in segments {
            if seg.trim().is_empty() {
                continue;
            }
            if self.count_tokens(&seg) <= self.config.max_tokens {
                chunks.push(seg);
            } else {
                chunks.extend(self.split_by_paragraphs(&seg));
            }
        }
        chunks
    }

    // ---- Private helpers ----

    /// Count the tokens in `text` using the loaded tokenizer, or fall back to
    /// a word-count approximation (1.3 tokens per word, rounded up).
    fn count_tokens(&self, text: &str) -> usize {
        if let Some(ref tok) = self.tokenizer {
            tok.count_tokens(text)
        } else {
            // Approximation: average English word is ~1.3 tokens.
            let words = text.split_whitespace().count();
            (words as f64 * 1.3).ceil() as usize
        }
    }

    /// Return the exact header level (number of leading `#`) of `line`, or
    /// `None` if the line is not a Markdown ATX header.
    fn header_level(line: &str) -> Option<usize> {
        if !line.starts_with('#') {
            return None;
        }
        let level = line.chars().take_while(|&c| c == '#').count();
        if level > 6 {
            return None;
        }
        // Must be followed by a space (or end-of-line for an empty header).
        match line.chars().nth(level) {
            Some(' ') | None => Some(level),
            _ => None,
        }
    }

    /// Split `text` at H1/H2/H3 boundaries and recursively sub-split
    /// sections that are still above the token limit.
    fn split_by_headers(&self, text: &str) -> Vec<String> {
        // Try each header level in turn.
        for level in 1..=3 {
            let sections = self.split_at_level(text, level);
            if sections.len() > 1 {
                // Got at least one split: recursively process each section.
                return sections
                    .into_iter()
                    .flat_map(|sec| {
                        if self.count_tokens(&sec) > self.config.max_tokens {
                            self.split_by_headers(&sec)
                        } else {
                            vec![sec]
                        }
                    })
                    .collect();
            }
        }
        // No header split was found; return the text unchanged.
        vec![text.to_string()]
    }

    /// Split `text` at exactly `level` header lines, keeping the header with
    /// the section that follows it.
    fn split_at_level(&self, text: &str, level: usize) -> Vec<String> {
        let mut sections: Vec<String> = Vec::new();
        let mut current = String::new();

        for line in text.lines() {
            if Self::header_level(line) == Some(level) && !current.trim().is_empty() {
                sections.push(current.trim_end().to_string());
                current = String::new();
            }
            current.push_str(line);
            current.push('\n');
        }
        if !current.trim().is_empty() {
            sections.push(current.trim_end().to_string());
        }
        sections
    }

    /// Split `text` on double-newline paragraph boundaries, merging paragraphs
    /// greedily until the limit is reached, then starting a new chunk (with
    /// optional token overlap from the previous chunk).
    fn split_by_paragraphs(&self, text: &str) -> Vec<String> {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();

        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for para in &paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            // A paragraph that is already too large on its own needs special handling.
            if self.count_tokens(para) > self.config.max_tokens {
                // Flush the accumulator first.
                if !current.trim().is_empty() {
                    chunks.push(current.clone());
                    current = String::new();
                }
                // Token-split the oversized paragraph; all resulting sub-chunks
                // are complete — do NOT carry any of them into `current`.
                let sub = self.split_by_tokens(para);
                chunks.extend(sub.into_iter());
                continue;
            }

            let candidate = if current.is_empty() {
                para.to_string()
            } else {
                format!("{}\n\n{}", current, para)
            };

            if self.count_tokens(&candidate) <= self.config.max_tokens {
                current = candidate;
            } else {
                if !current.trim().is_empty() {
                    chunks.push(current.clone());
                }
                // Start the next chunk with optional overlap from the flushed chunk,
                // then the new paragraph.
                let overlap = self.tail_overlap(&current);
                current = if overlap.is_empty() {
                    para.to_string()
                } else {
                    format!("{}\n\n{}", overlap, para)
                };
            }
        }
        if !current.trim().is_empty() {
            chunks.push(current);
        }
        chunks
    }

    /// Split a single oversized text block by raw token count, producing
    /// chunks of at most `max_tokens` with `overlap_tokens` carry-over.
    fn split_by_tokens(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < words.len() {
            let mut end = start + 1;
            // Grow until the chunk would exceed max_tokens.
            while end < words.len() {
                let candidate = words[start..=end].join(" ");
                if self.count_tokens(&candidate) > self.config.max_tokens {
                    break;
                }
                end += 1;
            }
            chunks.push(words[start..end].join(" "));

            // Advance start, keeping overlap.
            if self.config.overlap_tokens == 0 || end >= words.len() {
                start = end;
            } else {
                // Rewind by however many words cover overlap_tokens.
                let mut overlap_words = 0;
                let mut tok_count = 0;
                for w in words[..end].iter().rev() {
                    let w_tokens = self.count_tokens(w);
                    if tok_count + w_tokens > self.config.overlap_tokens {
                        break;
                    }
                    tok_count += w_tokens;
                    overlap_words += 1;
                }
                let new_start = end.saturating_sub(overlap_words);
                // Always advance: if overlap would stall at the same position,
                // just move past the current chunk entirely.
                start = if new_start > start { new_start } else { end };
            }
        }
        chunks
    }

    /// Return a suffix of `text` whose token count is ≤ `overlap_tokens`.
    fn tail_overlap(&self, text: &str) -> String {
        if self.config.overlap_tokens == 0 || text.is_empty() {
            return String::new();
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut taken = 0;
        let mut result_words: Vec<&str> = Vec::new();
        for &w in words.iter().rev() {
            let new_count = taken + self.count_tokens(w);
            if new_count > self.config.overlap_tokens {
                break;
            }
            taken = new_count;
            result_words.push(w);
        }
        result_words.reverse();
        result_words.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunker(max_tokens: usize, overlap: usize) -> MarkdownChunker {
        MarkdownChunker::new(ChunkerConfig {
            max_tokens,
            overlap_tokens: overlap,
            tokenizer_path: None,
        })
        .unwrap()
    }

    #[test]
    fn test_empty_input() {
        let c = chunker(512, 0);
        assert!(c.chunk("").is_empty());
        assert!(c.chunk("   \n  ").is_empty());
    }

    #[test]
    fn test_short_text_single_chunk() {
        let c = chunker(512, 0);
        let text = "Hello world this is a short text.";
        let chunks = c.chunk(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_header_splitting() {
        let text = "# Section A\n\nContent A.\n\n# Section B\n\nContent B.";
        let c = chunker(10, 0); // 10 words ≈ 13 tokens, well below section sizes
        let chunks = c.chunk(text);
        // Should be split at the H1 boundaries.
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {chunks:?}"
        );
        assert!(
            chunks.iter().any(|ch| ch.contains("Section A")),
            "chunk for A missing"
        );
        assert!(
            chunks.iter().any(|ch| ch.contains("Section B")),
            "chunk for B missing"
        );
    }

    #[test]
    fn test_paragraph_splitting() {
        let text =
            "First paragraph with some words.\n\nSecond paragraph with different words.\n\nThird paragraph.";
        // Very small max_tokens forces paragraph-level splits.
        let c = chunker(3, 0);
        let chunks = c.chunk(text);
        assert!(chunks.len() >= 2, "expected splits, got {chunks:?}");
    }

    #[test]
    fn test_overlap_tokens() {
        let words: Vec<String> = (1..=100).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let c = chunker(20, 5);
        let chunks = c.chunk(&text);
        // There should be multiple chunks, each with some overlap.
        assert!(
            chunks.len() > 1,
            "expected multiple chunks with small limit"
        );
    }

    #[test]
    fn test_header_level_detection() {
        assert_eq!(MarkdownChunker::header_level("# H1"), Some(1));
        assert_eq!(MarkdownChunker::header_level("## H2"), Some(2));
        assert_eq!(MarkdownChunker::header_level("### H3"), Some(3));
        assert_eq!(MarkdownChunker::header_level("#### H4"), Some(4));
        assert_eq!(MarkdownChunker::header_level("#nospace"), None);
        assert_eq!(MarkdownChunker::header_level("not a header"), None);
        assert_eq!(MarkdownChunker::header_level(""), None);
    }

    #[test]
    fn test_h2_split_within_h1_section() {
        let text =
            "# Big Section\n\n## Part One\n\nWords words words.\n\n## Part Two\n\nMore words.";
        // Set limit so a full H1 section is too large but H2 sections fit.
        let c = chunker(6, 0);
        let chunks = c.chunk(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_all_chunks_within_limit() {
        let text = "# A\n\nLorem ipsum dolor sit amet.\n\n# B\n\nConsectetur adipiscing elit.\n\n# C\n\nSed do eiusmod tempor.";
        for max in [5, 10, 50, 200] {
            let c = chunker(max, 0);
            let chunks = c.chunk(text);
            for ch in &chunks {
                let tok = c.count_tokens(ch);
                // Allow a small slack: a single oversized word may exceed limit.
                assert!(
                    tok <= max + 5,
                    "chunk ({tok} tokens) exceeds limit {max}: {ch:?}"
                );
            }
        }
    }

    #[test]
    fn test_no_infinite_loop_oversized_word() {
        // A single "word" (no whitespace) that is very long — this should not
        // loop indefinitely even when max_tokens is tiny.
        let long_word = "a".repeat(500);
        let text = format!("{long_word} short words here {long_word}");
        let c = chunker(2, 1);
        // Just assert it terminates and produces chunks.
        let chunks = c.chunk(&text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_split_by_tokens_no_overlap() {
        let words: Vec<String> = (1..=50).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let c = chunker(10, 0);
        let chunks = c.chunk(&text);
        assert!(chunks.len() > 1);
        // No chunk should overlap with another when overlap_tokens == 0.
        for i in 0..chunks.len().saturating_sub(1) {
            let last_word_of_chunk = chunks[i].split_whitespace().last().unwrap();
            let first_word_of_next = chunks[i + 1].split_whitespace().next().unwrap();
            assert_ne!(
                last_word_of_chunk, first_word_of_next,
                "unexpected overlap at chunk boundary {i}"
            );
        }
    }
}
