#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <repo_id> <file_list> <output_dir> <log_file>" >&2
  exit 2
fi

repo_id="$1"
file_list="$2"
output_dir="$3"
log_file="$4"

mkdir -p "$output_dir" "$(dirname "$log_file")"

download_one() {
  rel="$1"
  url="https://huggingface.co/datasets/${repo_id}/resolve/main/${rel}"
  dest="${output_dir}/${rel}"
  tmp="${dest}.part"
  mkdir -p "$(dirname "$dest")"

  if [ -s "$dest" ]; then
    printf '%s\tSKIP\t%s\t%s\n' "$(date -Is)" "$rel" "$dest" >> "$log_file"
    return 0
  fi

  printf '%s\tSTART_DIRECT\t%s\n' "$(date -Is)" "$rel" >> "$log_file"
  result="$(
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl -sS -L --fail --retry 0 --connect-timeout 8 --max-time 12 \
      -C - -o "$tmp" \
      -w 'http=%{http_code} speed=%{speed_download} size=%{size_download} time=%{time_total}' \
      "$url" 2>&1
  )"
  rc=$?
  printf '%s\tDIRECT_RESULT\trc=%s\t%s\t%s\n' "$(date -Is)" "$rc" "$rel" "$result" >> "$log_file"

  if [ "$rc" -ne 0 ]; then
    printf '%s\tSTART_PROXY\t%s\n' "$(date -Is)" "$rel" >> "$log_file"
    result="$(
      curl -sS -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 7200 \
        -C - -o "$tmp" \
        -w 'http=%{http_code} speed=%{speed_download} size=%{size_download} time=%{time_total}' \
        "$url" 2>&1
    )"
    rc=$?
    printf '%s\tPROXY_RESULT\trc=%s\t%s\t%s\n' "$(date -Is)" "$rc" "$rel" "$result" >> "$log_file"
  fi

  if [ "$rc" -eq 0 ]; then
    mv "$tmp" "$dest"
    printf '%s\tDONE\t%s\t%s\n' "$(date -Is)" "$rel" "$dest" >> "$log_file"
    return 0
  fi

  printf '%s\tFAILED\t%s\n' "$(date -Is)" "$rel" >> "$log_file"
  return "$rc"
}

while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  attempt=1
  while [ "$attempt" -le 5 ]; do
    if download_one "$rel"; then
      break
    fi
    if [ "$attempt" -eq 5 ]; then
      exit 1
    fi
    sleep_seconds=$((attempt * 15))
    printf '%s\tRETRY_AFTER_FAILURE\tattempt=%s\tsleep=%s\t%s\n' "$(date -Is)" "$attempt" "$sleep_seconds" "$rel" >> "$log_file"
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
done < "$file_list"
