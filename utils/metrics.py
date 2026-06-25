"""Read an Ultralytics results.csv, rank epochs by fitness, and render reports.

fitness = 0.1*mAP50 + 0.9*mAP50-95 — the exact score Ultralytics uses to pick
best.pt. Rank 1 by fitness is the epoch best.pt was saved from (NOT always the
highest mAP50, since mAP50-95 is weighted 9x).

save_ranking_png() renders a dark-theme report (best.pt 평가 + fitness/recall
top-N) via headless Chrome, so Korean text and color medals render correctly;
it falls back to a plain matplotlib table when Chrome is unavailable.
"""
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path


def fitness(map50: float, map5095: float) -> float:
    """Ultralytics best.pt selection score (mAP50-95 weighted 9x)."""
    return 0.1 * map50 + 0.9 * map5095


def read_epochs(results_csv) -> list:
    """Parse results.csv -> one dict per epoch (with fitness), in file order."""
    rows = []
    with open(results_csv, newline="") as f:
        for raw in csv.DictReader(f):
            r = {k.strip(): v for k, v in raw.items()}  # tolerate spaced headers
            m50 = float(r["metrics/mAP50(B)"])
            m5095 = float(r["metrics/mAP50-95(B)"])
            rows.append({
                "epoch": int(float(r["epoch"])),
                "mAP50": m50,
                "mAP50-95": m5095,
                "precision": float(r["metrics/precision(B)"]),
                "recall": float(r["metrics/recall(B)"]),
                "fitness": fitness(m50, m5095),
            })
    return rows


def top_epochs(results_csv, top: int = 10) -> list:
    """Epochs sorted by fitness (desc), best first."""
    return sorted(read_epochs(results_csv), key=lambda d: d["fitness"], reverse=True)[:top]


def format_top_table(rows: list) -> str:
    """ASCII ranking table for the terminal; rank 1 (= best.pt) is flagged."""
    header = f"{'rank':>4} {'epoch':>6} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7} {'fitness':>9}"
    lines = [header, "-" * len(header)]
    for i, d in enumerate(rows, 1):
        star = "  <- best.pt" if i == 1 else ""
        lines.append(
            f"{i:>4} {d['epoch']:>6} {d['mAP50']:>8.3f} {d['mAP50-95']:>9.3f} "
            f"{d['precision']:>7.3f} {d['recall']:>7.3f} {d['fitness']:>9.4f}{star}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rich report PNG (headless Chrome, matplotlib fallback)
# ---------------------------------------------------------------------------

_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
_ORDER = {"bad": 0, "mid": 1, "good": 2, "great": 3}

# best.pt 등급 기준: 값 < 상한 -> 그 등급. 마지막 튜플은 그 이상 전부.
_ASSESS = {
    "mAP50":    {"bands": [(0.5, "거의 못 씀", "bad"), (0.7, "보통", "mid"),
                           (0.9, "실용 가능", "good"), (99, "잘 학습됨", "great")],
                 "ref": "&lt;0.5 못씀 · 0.7~0.9 실용 · 0.9+ 잘됨"},
    "mAP50-95": {"bands": [(0.4, "약함", "bad"), (0.5, "보통", "mid"),
                           (0.6, "괜찮음", "good"), (99, "강함", "great")],
                 "ref": "&lt;0.4 약함 · 0.5+ 괜찮음 · 0.6+ 강함 (mAP50보다 낮은 게 정상)"},
    "Recall":   {"bands": [(0.7, "놓침 많음", "bad"), (0.8, "보통", "mid"),
                           (0.95, "실용", "good"), (99, "안전 용도 가능", "great")],
                 "ref": "&lt;0.7 놓침많음 · 0.8~0.95 실용 · 0.95+ 안전용도"},
    "Precision": {"bands": [(0.8, "낮음", "bad"), (0.95, "실용", "good"),
                            (99, "오탐 거의 없음", "great")],
                  "ref": "0.8~0.95 실용 · 0.95+ 오탐 치명적 용도"},
}

_CSS = """
body{background:#0d0d0d;margin:0;padding:36px 40px;font-family:'Noto Sans CJK KR','Noto Sans CJK SC',sans-serif;}
h2{color:#e4e4e4;font-size:22px;font-weight:600;margin:22px 0 14px 2px;}
h2:first-child{margin-top:0;}
h2 .meta{color:#7a7a7a;font-size:15px;font-weight:400;}
table{border-collapse:collapse;margin-bottom:6px;}
th,td{border:1px solid #333;padding:11px 22px;text-align:center;font-size:21px;color:#b4b4b4;}
th{color:#dcdcdc;font-weight:500;background:#161616;}
th.hl{color:#fff;font-weight:700;}
td.hl{color:#ededed;}
tr.best td{background:#11240f;}
td b{color:#fff;font-weight:700;}
td.rank{font-size:18px;color:#9a9a9a;}
.medal{font-size:26px;line-height:1.1;}
.sub{color:#fff;font-size:13px;margin-top:3px;font-weight:700;}
.note{color:#5fbf5f;font-size:12px;margin-top:3px;}
td.ref{color:#808080;font-size:14px;text-align:left;}
.bad{color:#e05a5a !important;font-weight:700;}
.mid{color:#d9b54a !important;font-weight:700;}
.good{color:#7bc96f !important;font-weight:700;}
.great{color:#43c043 !important;font-weight:700;}
.verdict{color:#d2d2d2;font-size:18px;margin:10px 0 6px 2px;}
.verdict b{color:#fff;}
"""


def _find_chrome():
    for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(b)
        if p:
            return p
    return None


def _assess_one(metric, value):
    for thr, label, level in _ASSESS[metric]["bands"]:
        if value < thr:
            return label, level
    last = _ASSESS[metric]["bands"][-1]
    return last[1], last[2]


def _assessment_html(best):
    spec = [("mAP50", best["mAP50"]), ("mAP50-95", best["mAP50-95"]),
            ("Recall", best["recall"]), ("Precision", best["precision"])]
    th = "<tr><th>지표</th><th>best.pt 값</th><th>평가</th><th>기준</th></tr>"
    body = ""
    worst = None
    for metric, val in spec:
        label, level = _assess_one(metric, val)
        if worst is None or _ORDER[level] < _ORDER[worst[2]]:
            worst = (metric, label, level)
        body += (f"<tr><td>{metric}</td><td><b>{val:.3f}</b></td>"
                 f"<td class='{level}'>● {label}</td>"
                 f"<td class='ref'>{_ASSESS[metric]['ref']}</td></tr>")
    if worst[2] in ("bad", "mid"):
        verdict = f"⚠ 약점: <b>{worst[0]} = {worst[1]}</b> — 이 지표가 용도에 충분한지 확인하세요"
    else:
        verdict = "전반적으로 <b>잘 학습됨</b> ✅"
    return f"<table>{th}{body}</table><div class='verdict'>{verdict}</div>"


def _rank_cell(i, sub=None):
    medal = _MEDALS.get(i)
    if medal:
        return f'<div class="medal">{medal}</div>' + (f'<div class="sub">{sub}</div>' if sub else "")
    return f"{i}위"


def _fitness_html(rows, map50_best_ep):
    th = "<tr><th>순위</th><th>epoch</th><th>mAP50</th><th>mAP50-95</th><th class='hl'>fitness</th></tr>"
    body = ""
    for i, d in enumerate(rows, 1):
        best = i == 1
        m50 = f"{d['mAP50']:.3f}"
        if d["epoch"] == map50_best_ep:
            m50 = f"<b>{m50}</b><div class='note'>← mAP50 최고</div>"
        ep = f"<b>{d['epoch']}</b>" if best else d["epoch"]
        m5095 = f"<b>{d['mAP50-95']:.3f}</b>" if best else f"{d['mAP50-95']:.3f}"
        fit = f"<b>{d['fitness']:.3f}</b>" if best else f"{d['fitness']:.3f}"
        body += (f"<tr class='{'best' if best else ''}'>"
                 f"<td class='rank'>{_rank_cell(i, 'best.pt' if best else None)}</td>"
                 f"<td>{ep}</td><td>{m50}</td><td>{m5095}</td><td class='hl'>{fit}</td></tr>")
    return f"<table>{th}{body}</table>"


def _recall_html(rows):
    th = ("<tr><th>순위</th><th>epoch</th><th class='hl'>recall</th>"
          "<th>precision</th><th>mAP50</th><th>fitness</th></tr>")
    body = ""
    for i, d in enumerate(rows, 1):
        best = i == 1
        rec = (f"<b>{d['recall']:.3f}</b><div class='note'>← recall 최고</div>"
               if best else f"{d['recall']:.3f}")
        body += (f"<tr class='{'best' if best else ''}'><td class='rank'>{_rank_cell(i)}</td>"
                 f"<td>{d['epoch']}</td><td class='hl'>{rec}</td><td>{d['precision']:.3f}</td>"
                 f"<td>{d['mAP50']:.3f}</td><td>{d['fitness']:.3f}</td></tr>")
    return f"<table>{th}{body}</table>"


def save_ranking_png(results_csv, out_path, name="model", top=10):
    """best.pt 평가 + fitness/recall 상위 N 을 한 장의 다크 PNG 로 저장 (헤드리스 Chrome).

    Chrome 이 없으면 matplotlib 의 단순 fitness 표로 폴백한다.
    epoch 데이터가 없으면 None 을 반환한다.
    """
    epochs = read_epochs(results_csv)
    if not epochs:
        return None
    by_fit = sorted(epochs, key=lambda d: d["fitness"], reverse=True)[:top]
    by_rec = sorted(epochs, key=lambda d: d["recall"], reverse=True)[:top]
    map50_best_ep = max(epochs, key=lambda d: d["mAP50"])["epoch"]
    best = by_fit[0]

    chrome = _find_chrome()
    if chrome is None:
        return _save_fitness_png_matplotlib(by_fit, out_path, name)

    html = (
        f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>"
        f"<h2>{name} · best.pt 평가 <span class='meta'>(epoch {best['epoch']})</span></h2>"
        f"{_assessment_html(best)}"
        f"<h2>Fitness 기준 상위 {top} <span class='meta'>(best.pt = 1위)</span></h2>"
        f"{_fitness_html(by_fit, map50_best_ep)}"
        f"<h2>Recall 기준 상위 {top} <span class='meta'>(가장 안 놓치는 epoch)</span></h2>"
        f"{_recall_html(by_rec)}"
        f"</body></html>"
    )
    tmp = Path(tempfile.mkdtemp())
    try:
        html_path = tmp / "report.html"
        raw = tmp / "raw.png"
        html_path.write_text(html, encoding="utf-8")
        win_h = 1000 + top * 210  # 넉넉하게: 콘텐츠보다 크면 아래는 트림됨
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
             "--force-device-scale-factor=2", f"--screenshot={raw}",
             f"--window-size=980,{win_h}", html_path.as_uri()],
            capture_output=True, text=True, timeout=120,
        )
        if not raw.exists():
            return _save_fitness_png_matplotlib(by_fit, out_path, name)
        from PIL import Image, ImageChops
        im = Image.open(raw).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            pad = 28
            l, t, r, b = bbox
            im = im.crop((max(0, l - pad), max(0, t - pad),
                          min(im.width, r + pad), min(im.height, b + pad)))
        im.save(out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path


def _save_fitness_png_matplotlib(rows, out_path, name):
    """Chrome 가 없을 때 폴백: 단순 fitness 표 (이모지·한글 평가 없음)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["rank", "epoch", "mAP50", "mAP50-95", "P", "R", "fitness"]
    cell_text = [
        [str(i), str(d["epoch"]), f"{d['mAP50']:.3f}", f"{d['mAP50-95']:.3f}",
         f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['fitness']:.4f}"]
        for i, d in enumerate(rows, 1)
    ]
    fig, ax = plt.subplots(figsize=(8, 0.45 * (len(rows) + 2)))
    ax.axis("off")
    ax.set_title(f"{name}: fitness ranking (rank 1 = best.pt)", fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)
    for c in range(len(cols)):
        tbl[(0, c)].set_facecolor("#4C72B0")
        tbl[(0, c)].set_text_props(color="white", fontweight="bold")
        tbl[(1, c)].set_facecolor("#d8f5d8")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
