#!/usr/bin/env python3
"""
Fix text preprocessing issues in the processed chunks.

This script identifies and fixes common text preprocessing problems:
- Excessive spacing between letters
- Hyphenated words with spaces (song lyrics, hymne)
- OCR artifacts from PDF processing
- Table garbage and number sequences
- Meaningless short chunks
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import os

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


def is_garbage_content(text: str) -> Tuple[bool, str]:
    """
    Detect if chunk contains garbage/non-informative content.
    
    Returns:
        Tuple of (is_garbage, reason)
    """
    if not text or len(text.strip()) < 50:
        return True, "too_short"
    
    # Count meaningful words vs garbage
    words = text.split()
    if len(words) < 10:
        return True, "too_few_words"
    
    # Check for song lyrics pattern (excessive hyphens between letters)
    hyphen_letter_ratio = text.count(' - ') / max(len(words), 1)
    if hyphen_letter_ratio > 0.15:
        return True, "song_lyrics"
    
    # Check for hymne/lagu pattern
    hymne_patterns = [
        r'hymne\s+umb',
        r'lagu\s*[&:]\s*syair',
        r'g\s*=\s*1\s*;\s*4\s*/\s*4',  # Musical notation
        r'berlandaskan\s+pan\s*-\s*casila',
        r'tri\s*-?\s*dharma',
    ]
    for pattern in hymne_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, "hymne_lyrics"
    
    # Check for table garbage (excessive numbers and dashes)
    numbers = len(re.findall(r'\d+', text))
    dashes = text.count('-') + text.count('–')
    if numbers > 20 and dashes > 10:
        # Check if it's meaningful (has enough words)
        alpha_words = len([w for w in words if w.isalpha() and len(w) > 2])
        if alpha_words < 15:
            return True, "table_garbage"
    
    # Check for excessive dots/dashes sequences
    garbage_sequences = len(re.findall(r'[-\.]{5,}', text))
    if garbage_sequences > 3:
        return True, "formatting_garbage"
    
    # Check for [UNK] tokens ratio
    unk_count = text.count('[UNK]')
    if unk_count > 5:
        return True, "too_many_unknowns"
    
    # Check for mostly numbers
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.4:
        return True, "mostly_numbers"
    
    return False, "ok"


def fix_spacing_issues(text: str) -> str:
    """
    Fix common spacing issues in OCR/text processing.
    """
    if not text:
        return text
    
    # Remove [UNK] tokens
    text = re.sub(r'\[UNK\]', '', text)
    
    # Fix single letter spacing: "u n i v e r s i t a s" -> "universitas"
    text = re.sub(r'\b([a-zA-Z])\s+(?=[a-zA-Z]\s)', r'\1', text)
    
    # Fix multiple single spaces: "t e k n i k" -> "teknik"
    text = re.sub(r'\b([a-zA-Z](?:\s+[a-zA-Z]){2,})\b',
                  lambda m: re.sub(r'\s+', '', m.group(1)), text)
    
    # Fix hyphen spaced words: "ti a pa - da - mu" -> "tiapada mu"
    text = re.sub(r'([a-zA-Z])\s*-\s*([a-zA-Z])', r'\1\2', text)
    
    # Remove musical notation patterns
    text = re.sub(r'g\s*=\s*\d+\s*;\s*\d+\s*/\s*\d+', '', text, flags=re.IGNORECASE)
    
    # Fix specific problematic patterns with simple replacements
    patterns = [
        (r'u\s+n\s+i\s+v\s+e\s+r\s+s\s+i\s+t\s+a\s+', 'universitas '),
        (r'm\s+e\s+r\s+c\s+u\s+b\s+u\s+a\s+n\s+a', 'mercubuana'),
        (r'f\s+a\s+k\s+u\s+l\s+t\s+a\s+', 'fakultas '),
        (r'p\s+r\s+o\s+g\s+r\s+a\s+m\s+', 'program '),
        (r's\s+t\s+u\s+d\s+i\s+', 'studi '),
        (r'j\s+u\s+r\s+u\s+s\s+a\s+n\s+', 'jurusan '),
        (r't\s+e\s+k\s+n\s+i\s+k\s+', 'teknik '),
        (r'm\s+a\s+h\s+a\s+s\s+i\s+s\s+w\s+a', 'mahasiswa'),
        (r'p\s+a\s+n\s+c\s+a\s+s\s+i\s+l\s+a', 'pancasila'),
        (r't\s+r\s+i\s+d\s+h\s+a\s+r\s+m\s+a', 'tridharma'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove garbage number sequences (like from tables)
    # Pattern: sequences of numbers with spaces/dashes
    text = re.sub(r'(\d+\s*[-–]\s*){3,}', '', text)
    
    # Remove standalone number sequences with dots
    text = re.sub(r'([\d\.]+\s+){5,}', '', text)
    
    # Remove course code sequences without context
    text = re.sub(r'\b[a-z]\d{6,}\b', '', text, flags=re.IGNORECASE)
    
    # Fix multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Fix weird spacing around punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    
    # Fix spaced numbers: "2 0 2 4" -> "2024"
    text = re.sub(r'\b(\d)\s+(?=\d)\b', r'\1', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_chunk_text(text: str) -> str:
    """
    Aggressively clean chunk text.
    """
    if not text:
        return ""
    
    # First pass: fix spacing issues
    text = fix_spacing_issues(text)
    
    # Remove empty parentheses
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # Remove standalone punctuation sequences
    text = re.sub(r'^[^\w\s]+$', '', text, flags=re.MULTILINE)
    
    # Remove lines with mostly numbers/dashes
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        alpha_count = sum(1 for c in line if c.isalpha())
        total_count = len(line.strip())
        if total_count > 0 and alpha_count / total_count > 0.3:
            cleaned_lines.append(line)
        elif alpha_count > 20:  # Keep if has enough letters
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def detect_text_issues(text: str) -> Dict[str, Any]:
    """
    Detect if text has preprocessing issues.
    """
    if not text:
        return {"has_issues": False, "issues": [], "is_garbage": False}
    
    issues = []
    
    # Check if garbage
    is_garbage, reason = is_garbage_content(text)
    
    # Check for excessive spacing
    space_count = text.count(' ')
    letter_count = sum(1 for c in text if c.isalpha())
    if letter_count > 0 and space_count / letter_count > 0.3:
        issues.append("excessive_spacing")
    
    # Check for single-letter patterns
    single_letter_spaces = len(re.findall(r'\b([a-zA-Z])\s+(?=[a-zA-Z])', text))
    if single_letter_spaces > 5:
        issues.append("single_letter_spacing")
    
    # Check for hyphen-spaced patterns
    hyphen_spaces = len(re.findall(r'[a-zA-Z]\s*-\s*[a-zA-Z]', text))
    if hyphen_spaces > 2:
        issues.append("hyphen_spacing")
    
    # Check for spaced numbers
    spaced_numbers = len(re.findall(r'\b\d\s+\d\b', text))
    if spaced_numbers > 0:
        issues.append("spaced_numbers")
    
    # Check for [UNK] tokens
    if '[UNK]' in text:
        issues.append("unknown_tokens")
    
    return {
        "has_issues": len(issues) > 0 or is_garbage,
        "issues": issues,
        "is_garbage": is_garbage,
        "garbage_reason": reason,
        "space_ratio": space_count / max(letter_count, 1),
        "single_letter_spaces": single_letter_spaces,
        "hyphen_spaces": hyphen_spaces,
        "spaced_numbers": spaced_numbers
    }


def analyze_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze chunks for preprocessing issues.
    """
    total_chunks = len(chunks)
    issues_summary = {
        "excessive_spacing": 0,
        "single_letter_spacing": 0,
        "hyphen_spacing": 0,
        "spaced_numbers": 0,
        "unknown_tokens": 0,
        "garbage_chunks": 0,
        "no_issues": 0
    }
    
    garbage_reasons = {}
    examples = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        analysis = detect_text_issues(text)
        
        if analysis["is_garbage"]:
            issues_summary["garbage_chunks"] += 1
            reason = analysis["garbage_reason"]
            garbage_reasons[reason] = garbage_reasons.get(reason, 0) + 1
        
        if analysis["has_issues"]:
            for issue in analysis["issues"]:
                if issue in issues_summary:
                    issues_summary[issue] += 1
            
            if len(examples) < 10:
                examples.append({
                    "chunk_index": i,
                    "issues": analysis["issues"],
                    "is_garbage": analysis["is_garbage"],
                    "garbage_reason": analysis.get("garbage_reason", ""),
                    "text_preview": text[:300] + "..." if len(text) > 300 else text,
                    "space_ratio": analysis["space_ratio"]
                })
        else:
            issues_summary["no_issues"] += 1
    
    return {
        "total_chunks": total_chunks,
        "issues_summary": issues_summary,
        "garbage_reasons": garbage_reasons,
        "problematic_chunks": sum(issues_summary[k] for k in issues_summary if k != "no_issues"),
        "examples": examples
    }


def fix_chunks(chunks: List[Dict[str, Any]], dry_run: bool = False, remove_garbage: bool = True) -> List[Dict[str, Any]]:
    """
    Fix text preprocessing issues in chunks.
    
    Args:
        chunks: List of document chunks
        dry_run: If True, only show what would be fixed
        remove_garbage: If True, remove garbage chunks entirely
    
    Returns:
        Fixed chunks
    """
    fixed_chunks = []
    fixes_count = 0
    removed_count = 0
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        original_text = text
        
        if not text:
            continue
        
        # Check if garbage
        is_garbage, reason = is_garbage_content(text)
        
        if is_garbage and remove_garbage:
            removed_count += 1
            if not dry_run:
                print(f"Removed chunk {i+1}: {reason}")
            continue
        
        # Clean the text
        fixed_text = clean_chunk_text(text)
        
        # Check if cleaned text is now too short
        if len(fixed_text.strip()) < 50:
            removed_count += 1
            if not dry_run:
                print(f"Removed chunk {i+1}: cleaned_too_short")
            continue
        
        if fixed_text != original_text:
            fixes_count += 1
            if not dry_run:
                chunk['text'] = fixed_text
                chunk['fixed'] = True
        else:
            if not dry_run:
                chunk['fixed'] = False
        
        fixed_chunks.append(chunk)
    
    print(f"\nFixed {fixes_count} chunks")
    print(f"Removed {removed_count} garbage chunks")
    print(f"Remaining: {len(fixed_chunks)} chunks")
    
    return fixed_chunks


def main():
    parser = argparse.ArgumentParser(description="Fix text preprocessing issues in document chunks")
    
    # Get defaults from env
    default_input = os.getenv("PROCESSED_DIR", "../data/processed") + "/chunks.json"
    default_output = os.getenv("PROCESSED_DIR", "../data/processed") + "/chunks_cleaned.json"
    
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input chunks file (default: {default_input})")
    parser.add_argument("--output", "-o", default=default_output,
                       help=f"Output file for fixed chunks (default: {default_output})")
    parser.add_argument("--analyze", "--analyze-only", action="store_true",
                       help="Only analyze issues, don't fix them")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be fixed without making changes")
    parser.add_argument("--keep-garbage", action="store_true",
                       help="Keep garbage chunks instead of removing them")
    parser.add_argument("--replace", action="store_true",
                       help="Replace original chunks.json with cleaned version")

    args = parser.parse_args()

    print("=" * 60)
    print("Academic RAG - Text Preprocessing Fixer")
    print("=" * 60)
    
    # Load chunks
    print(f"\nLoading chunks from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.input}: {e}")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks")

    # Analyze issues
    print("\nAnalyzing text preprocessing issues...")
    analysis = analyze_chunks(chunks)

    print(f"\n{'=' * 40}")
    print("ANALYSIS RESULTS")
    print(f"{'=' * 40}")
    print(f"Total chunks: {analysis['total_chunks']}")
    print(f"Chunks with issues: {analysis['problematic_chunks']}")
    print(f"Garbage chunks: {analysis['issues_summary']['garbage_chunks']}")
    print(f"Clean chunks: {analysis['issues_summary']['no_issues']}")

    print(f"\nIssue breakdown:")
    for issue, count in analysis['issues_summary'].items():
        if issue not in ['no_issues', 'garbage_chunks'] and count > 0:
            percentage = (count / analysis['total_chunks']) * 100
            print(f"  {issue}: {count} ({percentage:.1f}%)")
    
    if analysis['garbage_reasons']:
        print(f"\nGarbage reasons:")
        for reason, count in sorted(analysis['garbage_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Show examples
    if analysis['examples']:
        print(f"\n{'=' * 40}")
        print("EXAMPLE PROBLEMATIC CHUNKS")
        print(f"{'=' * 40}")
        for i, example in enumerate(analysis['examples'][:5]):
            print(f"\nExample {i+1} (chunk {example['chunk_index']}):")
            print(f"  Issues: {', '.join(example['issues']) if example['issues'] else 'none'}")
            print(f"  Is garbage: {example['is_garbage']} ({example['garbage_reason']})")
            print(f"  Preview: {example['text_preview'][:150]}...")

    if args.analyze:
        print("\nAnalysis complete. Use --dry-run or run without --analyze to fix issues.")
        return

    # Fix issues
    if args.dry_run:
        print(f"\n{'=' * 40}")
        print("DRY RUN - Would fix the following:")
        print(f"{'=' * 40}")
        fixed_chunks = fix_chunks(chunks, dry_run=True, remove_garbage=not args.keep_garbage)
        print("\nNo files were modified.")
    else:
        print(f"\n{'=' * 40}")
        print("FIXING ISSUES")
        print(f"{'=' * 40}")
        fixed_chunks = fix_chunks(chunks, dry_run=False, remove_garbage=not args.keep_garbage)

        # Determine output path
        if args.replace:
            output_path = Path(args.input)
        else:
            output_path = Path(args.output)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving cleaned chunks to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Cleaned chunks saved to {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Rebuild indexes:")
        print(f"     python scripts/build_indexes.py --documents {output_path} --force --verify")
        print(f"  2. Test the RAG system")


if __name__ == "__main__":
    main()
