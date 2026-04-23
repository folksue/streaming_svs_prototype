from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence

from utils import ensure_dir, load_json, save_json


STANDARD_MANIFEST = "manifest"
M4SINGER_ADAPTER = "m4singer"
OPENCPOP_ADAPTER = "opencpop"
SOULX_SINGER_EVAL_ADAPTER = "soulx-singer-eval"
ACE_OPENCPOP_SEGMENTS_ADAPTER = "ace-opencpop-segments"


def normalize_adapter_name(name: str | None) -> str:
    if not name:
        return STANDARD_MANIFEST
    return str(name).strip().lower()


def adapter_needs_phoneme_vocab(adapter_name: str) -> bool:
    return normalize_adapter_name(adapter_name) in {
        M4SINGER_ADAPTER,
        OPENCPOP_ADAPTER,
        SOULX_SINGER_EVAL_ADAPTER,
        ACE_OPENCPOP_SEGMENTS_ADAPTER,
    }


def load_manifest_entries(
    manifest_path: str,
    adapter_name: str | None = None,
    audio_root: str | None = None,
) -> List[Dict[str, Any]]:
    adapter = normalize_adapter_name(adapter_name)
    if adapter == STANDARD_MANIFEST:
        entries = load_json(manifest_path)
        if not isinstance(entries, list):
            raise ValueError("Manifest must be a list of utterance dicts")
        return [normalize_standard_entry(x, base_dir=os.path.dirname(manifest_path)) for x in entries]
    if adapter == M4SINGER_ADAPTER:
        return load_m4singer_entries(manifest_path, audio_root=audio_root)
    if adapter == OPENCPOP_ADAPTER:
        return load_opencpop_entries(manifest_path, audio_root=audio_root)
    if adapter == SOULX_SINGER_EVAL_ADAPTER:
        return load_soulx_singer_eval_entries(manifest_path, audio_root=audio_root)
    if adapter == ACE_OPENCPOP_SEGMENTS_ADAPTER:
        return load_ace_opencpop_segments_entries(manifest_path, audio_root=audio_root)
    raise ValueError(f"Unsupported manifest adapter: {adapter_name}")


def build_or_load_phoneme_vocab(
    vocab_path: str,
    all_entries: Sequence[Sequence[Dict[str, Any]]],
    allow_create: bool,
) -> Dict[str, int]:
    if os.path.exists(vocab_path):
        raw_vocab = load_json(vocab_path)
        if not isinstance(raw_vocab, dict):
            raise ValueError(f"Phoneme vocab must be a JSON object: {vocab_path}")
        vocab = {str(k): int(v) for k, v in raw_vocab.items()}
        if "<pad>" not in vocab:
            vocab["<pad>"] = 0
        return vocab

    if not allow_create:
        raise FileNotFoundError(
            f"Phoneme vocab not found: {vocab_path}. "
            "Run preprocessing with the training split first or provide a prebuilt vocab."
        )

    symbols = sorted(
        {
            symbol
            for entries in all_entries
            for entry in entries
            for symbol in entry.get("phoneme_symbols", [])
            if symbol
        }
    )
    vocab = {"<pad>": 0}
    for idx, symbol in enumerate(symbols, start=1):
        vocab[symbol] = idx

    ensure_dir(os.path.dirname(vocab_path) or ".")
    save_json(vocab_path, vocab)
    return vocab


def attach_phoneme_ids(entries: Sequence[Dict[str, Any]], phoneme_vocab: Dict[str, int]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if "phoneme_ids" in entry:
            normalized.append(entry)
            continue

        symbols = entry.get("phoneme_symbols")
        if not isinstance(symbols, list):
            raise ValueError(f"Entry {entry.get('utt_id', '<unknown>')} is missing phoneme_ids/phoneme_symbols")

        try:
            phoneme_ids = [int(phoneme_vocab[symbol]) for symbol in symbols]
        except KeyError as exc:
            raise KeyError(
                f"Unknown phoneme symbol {exc.args[0]!r} in entry {entry.get('utt_id', '<unknown>')}. "
                f"Update {len(phoneme_vocab)}-item phoneme vocab."
            ) from exc

        item = dict(entry)
        item["phoneme_ids"] = phoneme_ids
        normalized.append(item)
    return normalized


def normalize_standard_entry(entry: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    normalized = dict(entry)
    normalized["utt_id"] = str(entry["utt_id"])
    normalized["audio_path"] = resolve_audio_path(entry["audio_path"], base_dir)
    normalized["phoneme_intervals"] = to_intervals(entry["phoneme_intervals"])
    normalized["note_intervals"] = to_intervals(entry["note_intervals"])
    normalized["note_pitch_midi"] = [float(x) for x in entry["note_pitch_midi"]]
    if "note_slur" in entry:
        normalized["note_slur"] = [int(x) for x in entry["note_slur"]]
    elif "slur" in entry:
        normalized["note_slur"] = parse_int_sequence(entry["slur"])
    else:
        normalized["note_slur"] = [0 for _ in normalized["note_intervals"]]
    if "phoneme_ids" in entry:
        normalized["phoneme_ids"] = [int(x) for x in entry["phoneme_ids"]]
    if "phoneme_symbols" in entry:
        normalized["phoneme_symbols"] = parse_symbol_sequence(entry["phoneme_symbols"])
    if "duration" in entry:
        normalized["duration"] = float(entry["duration"])
    return normalized


def load_m4singer_entries(manifest_path: str, audio_root: str | None = None) -> List[Dict[str, Any]]:
    base_dir = os.path.dirname(manifest_path)
    dataset_root = audio_root or base_dir
    raw_entries = load_json(manifest_path)
    if not isinstance(raw_entries, list):
        raise ValueError("M4Singer metadata must be a JSON list")

    entries: List[Dict[str, Any]] = []
    for raw in raw_entries:
        item_name = first_present(raw, "utt_id", "item_name")
        if item_name is None:
            raise ValueError("M4Singer entry is missing item_name/utt_id")

        phoneme_symbols = parse_symbol_sequence(first_present(raw, "phs", "ph", "phones", "phonemes"))
        phoneme_intervals = intervals_from_entry(
            raw,
            start_key="ph_start",
            end_key="ph_end",
            dur_key="ph_dur",
            dur_keys=("ph_duration", "phone_dur", "phone_duration"),
        )
        note_values = first_present(raw, "notes", "note", "note_seq")
        if isinstance(note_values, list):
            note_pitch_midi = [note_to_midi(x) for x in note_values]
        else:
            note_symbols = parse_symbol_sequence(note_values)
            note_pitch_midi = [note_to_midi(x) for x in note_symbols]
        note_intervals = intervals_from_entry(
            raw,
            start_key="note_start",
            end_key="note_end",
            dur_key="notes_dur",
            dur_keys=("note_dur", "note_duration"),
        )

        ensure_aligned_lengths(item_name, "phoneme", phoneme_symbols, phoneme_intervals)
        ensure_aligned_lengths(item_name, "note", note_pitch_midi, note_intervals)

        default_rel_audio_path = m4singer_item_to_audio_path(str(item_name))
        audio_path = resolve_audio_path(
            first_present(raw, "audio_path", "wav_path", "wav_fn", "wav"),
            dataset_root,
            fallback_relative=default_rel_audio_path,
        )
        slur_values = first_present(raw, "is_slur", "note_slur", "slur")
        if slur_values is None:
            note_slur = [0 for _ in note_intervals]
        else:
            note_slur = parse_int_sequence(slur_values)
            if len(note_slur) != len(note_intervals):
                raise ValueError(
                    f"M4Singer item '{item_name}' slur length mismatch: "
                    f"notes={len(note_intervals)} slur={len(note_slur)}"
                )

        duration = max_duration(phoneme_intervals, note_intervals, raw.get("duration"))
        entries.append(
            {
                "utt_id": str(item_name),
                "audio_path": audio_path,
                "phoneme_symbols": phoneme_symbols,
                "phoneme_intervals": phoneme_intervals,
                "note_pitch_midi": note_pitch_midi,
                "note_intervals": note_intervals,
                "note_slur": note_slur,
                "duration": duration,
            }
        )
    return entries


def load_opencpop_entries(manifest_path: str, audio_root: str | None = None) -> List[Dict[str, Any]]:
    base_dir = os.path.dirname(manifest_path)
    dataset_root = audio_root or os.path.join(base_dir, "wavs")
    entries: List[Dict[str, Any]] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 6:
                raise ValueError(f"Unexpected OpenCpop metadata format at line {line_id}: {line}")

            utt_id = parts[0].strip()
            phoneme_symbols = parse_symbol_sequence(parts[2])
            note_symbols = parse_symbol_sequence(parts[3])
            note_durations = parse_float_sequence(parts[4])
            phoneme_durations = parse_float_sequence(parts[5])

            if len(parts) >= 7:
                slur_flags = parse_int_sequence(parts[6])
                if len(slur_flags) != len(note_symbols):
                    raise ValueError(
                        f"OpenCpop line {line_id} slur length mismatch: notes={len(note_symbols)} slur={len(slur_flags)}"
                    )
            else:
                slur_flags = [0 for _ in note_symbols]

            phoneme_intervals = durations_to_intervals(phoneme_durations)
            note_intervals = durations_to_intervals(note_durations)
            note_pitch_midi = [note_to_midi(x) for x in note_symbols]

            ensure_aligned_lengths(utt_id, "phoneme", phoneme_symbols, phoneme_intervals)
            ensure_aligned_lengths(utt_id, "note", note_pitch_midi, note_intervals)

            audio_path = resolve_audio_path(
                None,
                dataset_root,
                fallback_relative=f"{utt_id}.wav",
            )

            entries.append(
                {
                    "utt_id": utt_id,
                    "audio_path": audio_path,
                    "phoneme_symbols": phoneme_symbols,
                    "phoneme_intervals": phoneme_intervals,
                    "note_pitch_midi": note_pitch_midi,
                    "note_intervals": note_intervals,
                    "note_slur": slur_flags,
                    "duration": max_duration(phoneme_intervals, note_intervals, None),
                }
            )
    return entries


def load_soulx_singer_eval_entries(manifest_path: str, audio_root: str | None = None) -> List[Dict[str, Any]]:
    base_dir = os.path.dirname(manifest_path)
    dataset_root = audio_root or base_dir
    raw_entries = load_jsonl_or_json(manifest_path)

    entries: List[Dict[str, Any]] = []
    for raw in raw_entries:
        utt_id = str(first_present(raw, "utt_id", "item_name", "uid", "id"))
        if not utt_id or utt_id == "None":
            raise ValueError("SoulX-Singer-Eval entry is missing utt_id/item_name")

        phoneme_symbols = parse_symbol_sequence(
            first_present(raw, "phs", "phonemes", "phones", "ph_seq", "ph")
        )
        phoneme_intervals = intervals_from_entry(
            raw,
            start_key="ph_start",
            end_key="ph_end",
            dur_key="ph_dur",
            dur_keys=("ph_durs", "phoneme_dur", "durations", "phone_dur"),
        )

        note_values = first_present(raw, "ep_pitches", "notes", "note_seq", "note", "midi", "note_pitch_midi")
        if isinstance(note_values, list) and note_values and isinstance(note_values[0], (int, float)):
            note_pitch_midi = [float(x) for x in note_values]
        else:
            note_pitch_midi = [note_to_midi(x) for x in parse_symbol_sequence(note_values)]
        note_intervals = intervals_from_entry(
            raw,
            start_key="note_start",
            end_key="note_end",
            dur_key="notes_dur",
            dur_keys=("ep_notedurs", "note_dur", "note_duration", "midi_dur"),
        )

        ensure_aligned_lengths(utt_id, "phoneme", phoneme_symbols, phoneme_intervals)
        ensure_aligned_lengths(utt_id, "note", note_pitch_midi, note_intervals)

        audio_path = resolve_audio_path(
            first_present(raw, "audio_path", "wav_path", "wav_fn", "wav", "wav_name"),
            dataset_root,
            fallback_relative=guess_soulx_audio_path(raw, utt_id),
        )

        entries.append(
            {
                "utt_id": utt_id,
                "audio_path": audio_path,
                "phoneme_symbols": phoneme_symbols,
                "phoneme_intervals": phoneme_intervals,
                "note_pitch_midi": note_pitch_midi,
                "note_intervals": note_intervals,
                "note_slur": [0 for _ in note_intervals],
                "duration": max_duration(phoneme_intervals, note_intervals, raw.get("duration")),
            }
        )
    return entries


def load_ace_opencpop_segments_entries(manifest_path: str, audio_root: str | None = None) -> List[Dict[str, Any]]:
    """
    Load ACE-OpenCpop segment metadata exported from HF (or converted sample JSON).

    Supported manifest shapes:
    1) list[dict], where each dict is a segment entry.
    2) HF datasets-server payload with top-level {"rows": [{"row": {...}}, ...]}.
    """
    base_dir = os.path.dirname(manifest_path)
    dataset_root = audio_root or base_dir
    raw_payload = load_json(manifest_path)

    raw_entries: List[Dict[str, Any]]
    if isinstance(raw_payload, list):
        raw_entries = raw_payload
    elif isinstance(raw_payload, dict) and isinstance(raw_payload.get("rows"), list):
        raw_entries = []
        for item in raw_payload["rows"]:
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                raw_entries.append(item["row"])
            elif isinstance(item, dict):
                raw_entries.append(item)
            else:
                raise ValueError("Invalid ACE-OpenCpop rows payload item; expected dict/row-dict.")
    else:
        raise ValueError("ACE-OpenCpop manifest must be a list or a {'rows': [...]} dict payload.")

    entries: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"ACE-OpenCpop entry at index {idx} is not a dict.")

        utt_id = str(first_present(raw, "segment_id", "utt_id", "id"))
        if not utt_id or utt_id == "None":
            raise ValueError(f"ACE-OpenCpop entry at index {idx} is missing segment_id/utt_id.")

        phoneme_symbols = parse_symbol_sequence(first_present(raw, "phn", "phoneme_symbols", "phones", "phonemes"))
        ph_starts = parse_float_sequence(first_present(raw, "phn_start_time", "ph_start"))
        ph_ends = parse_float_sequence(first_present(raw, "phn_end_time", "ph_end"))
        if len(ph_starts) != len(ph_ends):
            raise ValueError(f"{utt_id}: phn_start_time/phn_end_time length mismatch.")
        phoneme_intervals = [[float(s), float(e)] for s, e in zip(ph_starts, ph_ends)]

        note_values = first_present(raw, "note_midi", "note_pitch_midi", "notes", "note")
        if isinstance(note_values, list) and (not note_values or isinstance(note_values[0], (int, float))):
            note_pitch_midi = [float(x) for x in note_values]
        else:
            note_pitch_midi = [note_to_midi(x) for x in parse_symbol_sequence(note_values)]

        note_starts = parse_float_sequence(first_present(raw, "note_start_times", "note_start"))
        note_ends = parse_float_sequence(first_present(raw, "note_end_times", "note_end"))
        if len(note_starts) != len(note_ends):
            raise ValueError(f"{utt_id}: note_start_times/note_end_times length mismatch.")
        note_intervals = [[float(s), float(e)] for s, e in zip(note_starts, note_ends)]

        ensure_aligned_lengths(utt_id, "phoneme", phoneme_symbols, phoneme_intervals)
        ensure_aligned_lengths(utt_id, "note", note_pitch_midi, note_intervals)

        # ACE metadata does not provide explicit slur labels in the same way as OpenCpop;
        # we keep a neutral zero vector so downstream protocol remains compatible.
        note_slur = [0 for _ in note_intervals]

        audio_candidate = first_present(raw, "audio_path", "wav_path", "wav_fn", "wav")
        if not audio_candidate:
            audio_meta = raw.get("audio")
            if isinstance(audio_meta, str):
                audio_candidate = audio_meta
            elif isinstance(audio_meta, dict):
                audio_candidate = first_present(audio_meta, "path", "src", "url", "filename")
            elif isinstance(audio_meta, list) and audio_meta:
                first = audio_meta[0]
                if isinstance(first, dict):
                    audio_candidate = first_present(first, "path", "src", "url", "filename")
                elif isinstance(first, str):
                    audio_candidate = first

        if isinstance(audio_candidate, str) and audio_candidate.startswith(("http://", "https://")):
            # Keep local loading semantics: require local sample wav path when remote URL is present.
            audio_candidate = None

        audio_path = resolve_audio_path(
            audio_candidate,
            dataset_root,
            fallback_relative=f"{utt_id}.wav",
        )

        entries.append(
            {
                "utt_id": utt_id,
                "audio_path": audio_path,
                "phoneme_symbols": phoneme_symbols,
                "phoneme_intervals": phoneme_intervals,
                "note_pitch_midi": note_pitch_midi,
                "note_intervals": note_intervals,
                "note_slur": note_slur,
                "duration": max_duration(phoneme_intervals, note_intervals, raw.get("duration")),
            }
        )
    return entries


def intervals_from_entry(
    entry: Dict[str, Any],
    start_key: str,
    end_key: str,
    dur_key: str,
    dur_keys: Sequence[str],
) -> List[List[float]]:
    if start_key in entry and end_key in entry:
        starts = parse_float_sequence(entry[start_key])
        ends = parse_float_sequence(entry[end_key])
        if len(starts) != len(ends):
            raise ValueError(f"Interval start/end length mismatch for {start_key}/{end_key}")
        return [[float(s), float(e)] for s, e in zip(starts, ends)]

    for key in (dur_key, *dur_keys):
        if key in entry:
            return durations_to_intervals(parse_float_sequence(entry[key]))

    raise ValueError(f"Missing interval metadata: expected {start_key}/{end_key} or {dur_key}")


def to_intervals(values: Any) -> List[List[float]]:
    if not isinstance(values, list):
        raise ValueError("Intervals must be a list")
    intervals: List[List[float]] = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid interval item: {item}")
        intervals.append([float(item[0]), float(item[1])])
    return intervals


def parse_symbol_sequence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x for x in value.strip().split() if x]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    raise TypeError(f"Unsupported symbol sequence type: {type(value)!r}")


def parse_float_sequence(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [x for x in value.replace(",", " ").split() if x]
        return [float(x) for x in parts]
    if isinstance(value, list):
        return [float(x) for x in value]
    raise TypeError(f"Unsupported float sequence type: {type(value)!r}")


def parse_int_sequence(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [x for x in value.replace(",", " ").split() if x]
        return [int(x) for x in parts]
    if isinstance(value, list):
        return [int(x) for x in value]
    raise TypeError(f"Unsupported int sequence type: {type(value)!r}")


def load_jsonl_or_json(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".json"):
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return payload

    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = load_json_from_line(line, path=path, line_id=line_id)
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict JSON object in {path}:{line_id}")
            entries.append(item)
    return entries


def load_json_from_line(line: str, path: str, line_id: int) -> Any:
    import json

    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL line in {path}:{line_id}") from exc


def durations_to_intervals(durations: Sequence[float]) -> List[List[float]]:
    intervals: List[List[float]] = []
    t = 0.0
    for dur in durations:
        dur_f = max(float(dur), 0.0)
        intervals.append([t, t + dur_f])
        t += dur_f
    return intervals


def resolve_audio_path(audio_path: Any, base_dir: str, fallback_relative: str | None = None) -> str:
    candidate = str(audio_path).strip() if audio_path is not None and str(audio_path).strip() else fallback_relative
    if not candidate:
        raise ValueError("Could not resolve audio path")
    if os.path.isabs(candidate):
        return candidate
    return os.path.normpath(os.path.join(base_dir, candidate))


def m4singer_item_to_audio_path(item_name: str) -> str:
    parts = item_name.split("#")
    if len(parts) >= 3:
        song_dir = "#".join(parts[:-1])
        utt_name = parts[-1]
        return os.path.join(song_dir, f"{utt_name}.wav")
    return f"{item_name}.wav"


def guess_soulx_audio_path(entry: Dict[str, Any], utt_id: str) -> str:
    candidates = [
        first_present(entry, "wav_name", "wav"),
        f"{utt_id}.wav",
        os.path.join("wavs", f"{utt_id}.wav"),
        os.path.join("segments", "wavs", f"{utt_id}.wav"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return f"{utt_id}.wav"


def first_present(entry: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def note_to_midi(note: str) -> float:
    token = str(note).strip()
    if not token or token.lower() in {"rest", "sil", "sp", "pau"}:
        return 0.0
    if "/" in token:
        token = token.split("/", 1)[0].strip()
    try:
        return float(token)
    except ValueError:
        pass

    if len(token) < 2:
        raise ValueError(f"Unsupported note token: {note!r}")

    name = token[:-1]
    octave = token[-1]
    if len(token) >= 3 and token[-2].isdigit():
        name = token[:-2]
        octave = token[-2:]

    name = name.upper()
    note_map = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    if name not in note_map:
        raise ValueError(f"Unsupported note name: {note!r}")
    return float((int(octave) + 1) * 12 + note_map[name])


def ensure_aligned_lengths(utt_id: str, label: str, seq_a: Sequence[Any], seq_b: Sequence[Any]) -> None:
    if len(seq_a) != len(seq_b):
        raise ValueError(f"{utt_id}: {label} length mismatch ({len(seq_a)} vs {len(seq_b)})")


def max_duration(
    phoneme_intervals: Sequence[Sequence[float]],
    note_intervals: Sequence[Sequence[float]],
    explicit_duration: Any,
) -> float:
    if explicit_duration is not None:
        return float(explicit_duration)
    max_ph = max((float(x[1]) for x in phoneme_intervals), default=0.0)
    max_note = max((float(x[1]) for x in note_intervals), default=0.0)
    return max(max_ph, max_note)
