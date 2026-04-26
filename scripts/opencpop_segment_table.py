#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SegmentRecord:
    utt_id: str
    text: str
    phonemes: List[str]
    notes: List[str]
    note_durations: List[float]
    phoneme_durations: List[float]
    slurs: List[int]


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


def parse_segments_line(line: str) -> SegmentRecord:
    parts = line.rstrip("\n").split("|")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 fields, got {len(parts)}: {line[:120]}")
    utt_id, text, phoneme_s, note_s, note_dur_s, ph_dur_s, slur_s = parts
    phonemes = phoneme_s.split()
    notes = note_s.split()
    note_durations = [float(x) for x in note_dur_s.split()]
    phoneme_durations = [float(x) for x in ph_dur_s.split()]
    slurs = [int(x) for x in slur_s.split()]
    n = len(phonemes)
    if not (len(notes) == len(note_durations) == len(phoneme_durations) == len(slurs) == n):
        raise ValueError(
            f"Length mismatch for {utt_id}: "
            f"phonemes={len(phonemes)} notes={len(notes)} note_durations={len(note_durations)} "
            f"phoneme_durations={len(phoneme_durations)} slurs={len(slurs)}"
        )
    return SegmentRecord(
        utt_id=utt_id,
        text=text,
        phonemes=phonemes,
        notes=notes,
        note_durations=note_durations,
        phoneme_durations=phoneme_durations,
        slurs=slurs,
    )


def load_segment_record(segments_file: Path, utt_id: str) -> SegmentRecord:
    with segments_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{utt_id}|"):
                return parse_segments_line(line)
    raise FileNotFoundError(f"utt_id {utt_id} not found in {segments_file}")


def parse_textgrid(path: Path) -> Dict[str, List[Interval]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tiers: Dict[str, List[Interval]] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
            intervals: List[Interval] = []
            i += 1
            while i < len(lines):
                cur = lines[i].strip()
                if cur.startswith("item [") or cur.startswith("name = "):
                    i -= 1
                    break
                if cur.startswith("intervals ["):
                    xmin = float(lines[i + 1].split("=", 1)[1].strip())
                    xmax = float(lines[i + 2].split("=", 1)[1].strip())
                    text = lines[i + 3].split("=", 1)[1].strip().strip('"')
                    intervals.append(Interval(xmin=xmin, xmax=xmax, text=text))
                    i += 4
                    continue
                i += 1
            tiers[name] = intervals
        i += 1
    return tiers


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def find_sentence_interval(
    text_intervals: List[Interval],
    text: str,
    duration: float,
    tol: float = 0.08,
) -> Interval:
    candidates = [
        it for it in text_intervals
        if it.text == text and abs((it.xmax - it.xmin) - duration) <= tol
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return min(candidates, key=lambda x: abs((x.xmax - x.xmin) - duration))
    # Fallback: match by text only
    by_text = [it for it in text_intervals if it.text == text]
    if len(by_text) == 1:
        return by_text[0]
    if by_text:
        return min(by_text, key=lambda x: abs((x.xmax - x.xmin) - duration))
    raise ValueError(f"Could not locate sentence interval for text={text!r}")


def locate_chars_for_phonemes(record: SegmentRecord, textgrid_root: Path) -> List[Optional[str]]:
    song_id = record.utt_id[:4]
    tg_path = textgrid_root / f"{song_id}.TextGrid"
    if not tg_path.exists():
        return [None] * len(record.phonemes)
    tiers = parse_textgrid(tg_path)
    if "句子" not in tiers or "汉字" not in tiers:
        return [None] * len(record.phonemes)

    total_duration = sum(record.phoneme_durations)
    sent = find_sentence_interval(tiers["句子"], record.text, total_duration)
    char_intervals = [
        it for it in tiers["汉字"]
        if overlap(it.xmin, it.xmax, sent.xmin, sent.xmax) > 0 and it.text not in ("silence", "SP", "AP", "", "_")
    ]

    out: List[Optional[str]] = []
    cur = sent.xmin
    for dur in record.phoneme_durations:
        start = cur
        end = cur + dur
        best_char: Optional[str] = None
        best_ov = -1.0
        for ch in char_intervals:
            ov = overlap(start, end, ch.xmin, ch.xmax)
            if ov > best_ov:
                best_ov = ov
                best_char = ch.text
        out.append(best_char if best_ov > 0 else None)
        cur = end
    return out


def meaning_for_position(
    idx: int,
    record: SegmentRecord,
    chars: List[Optional[str]],
) -> str:
    ph = record.phonemes[idx]
    note = record.notes[idx]
    slur = record.slurs[idx]
    ch = chars[idx]

    if ph == "SP":
        return "停顿 SP"
    if ph == "AP":
        return "气口 AP"
    if ch is None:
        base = f"{ph} 对应当前位置发音"
    else:
        same_char_positions = [i for i, c in enumerate(chars) if c == ch]
        local_rank = same_char_positions.index(idx) if idx in same_char_positions else 0
        group_size = len(same_char_positions)
        if group_size == 1:
            base = f"“{ch}”的发音 {ph}"
        elif local_rank == 0:
            base = f"“{ch}”的声母 {ph}"
        elif slur == 1:
            base = f"“{ch}”的连音延续 {ph}"
        else:
            base = f"“{ch}”的韵母 {ph}"

    prev_same_ph = idx > 0 and record.phonemes[idx - 1] == ph
    prev_same_note = idx > 0 and record.notes[idx - 1] == note
    if slur == 1:
        if prev_same_ph and not prev_same_note:
            return f"{base}，切到新 note 的连音"
        return f"{base}，slur=1"
    if idx > 0 and prev_same_note:
        return f"{base}，同一个 note"
    return base


def format_float(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def markdown_table(record: SegmentRecord, chars: List[Optional[str]], only_slur: bool) -> str:
    lines = [
        f"### {record.utt_id} | {record.text}",
        "",
        "| 位置 | phoneme | note | note_duration | phoneme_duration | slur | 含义 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for i, (ph, note, nd, pd, sl) in enumerate(
        zip(record.phonemes, record.notes, record.note_durations, record.phoneme_durations, record.slurs),
        start=1,
    ):
        if only_slur and sl != 1:
            continue
        meaning = meaning_for_position(i - 1, record, chars)
        lines.append(
            f"| {i} | {ph} | {note} | {format_float(nd)} | {format_float(pd, digits=5)} | {sl} | {meaning} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OpenCpop segment rows as markdown tables.")
    parser.add_argument("--segments-file", default="datasets/opencpop/segments/train.txt")
    parser.add_argument("--textgrid-root", default="datasets/opencpop/textgrids")
    parser.add_argument("--utt-id", action="append", required=True, help="Utterance id such as 2001000005. Can be repeated.")
    parser.add_argument("--only-slur", action="store_true", help="Only show rows where slur == 1.")
    parser.add_argument("--out", help="Write markdown output to a .md file instead of stdout.")
    args = parser.parse_args()

    segments_file = Path(args.segments_file)
    textgrid_root = Path(args.textgrid_root)
    blocks: List[str] = []
    for n, utt_id in enumerate(args.utt_id):
        record = load_segment_record(segments_file, utt_id)
        chars = locate_chars_for_phonemes(record, textgrid_root)
        blocks.append(markdown_table(record, chars, only_slur=args.only_slur))

    output = "\n\n".join(blocks) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote markdown to {out_path}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
