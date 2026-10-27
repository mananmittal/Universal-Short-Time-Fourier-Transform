"""Index existing plots without recomputing or changing any figure.

python -m benchmarks.plot_gallery --from-edition results/paper_wavs/<edition>

Each PNG/PDF also gets a readable filename in one flat ``all_plots`` folder.
Hard links avoid duplicating the figures; copied files remain ordinary PNGs/PDFs.
"""

import argparse
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import quote


WAV_NAMES = {
    "humpback_whale_song.wav": (1, "Humpback whale song", "Humpback_whale_song"),
    "humpback8kmono.wav": (2, "Humpback (8 kHz)", "Humpback_8kHz"),
    "humpback_whale_and_dolphins.wav": (3, "Humpback whale and dolphins", "Humpback_whale_and_dolphins"),
    "snapping_shrimp.wav": (4, "Snapping shrimp", "Snapping_shrimp"),
    "fish_chorus.wav": (5, "Fish chorus", "Fish_chorus"),
    "cool_soliloquy_A.wav": (6, "Cool soliloquy A", "Cool_soliloquy_A"),
    "english_speech_1.wav": (7, "English speech 1", "English_speech_1"),
    "english_speech_2.wav": (8, "English speech 2", "English_speech_2"),
    "hyperbolic_chirp.wav": (9, "Hyperbolic chirp", "Hyperbolic_chirp"),
    "mixed_test_signal.wav": (10, "Mixed test signal", "Mixed_test_signal"),
}
METRICS = {
    "02_regret_time_frequency_total": (2, "1-D entropy gaps · time, frequency, and total", "1D_time_frequency_total", "regret"),
    "03_regret_2d_renyi3": (3, "2-D Rényi entropy gap", "2D_Renyi_entropy", "regret"),
    "04_blend_weights_pmf": (4, "Blend weights · PMF over time", "Blend_weights_PMF", "weights"),
    "regret_2d_renyi3": (1, "2-D Rényi entropy regret", "2D_Renyi_regret", "regret"),
    "regret_1d_renyi3_time": (2, "1-D time regret · cumulative average", "1D_time_regret_average", "regret"),
    "regret_1d_renyi3_frequency": (3, "1-D frequency regret · cumulative average", "1D_frequency_regret_average", "regret"),
    "regret_1d_renyi3_total": (4, "1-D total regret · cumulative average", "1D_total_regret_average", "regret"),
    "blend_weights_map": (5, "Blend weights · 2-D heatmap", "Blend_weights_heatmap", "weights"),
}


def _describe(relative, record):
    if relative.stem == "03_regret_2d_renyi3" and record.get("joint_entropy_oracle"):
        return (3, "2-D Rényi regret · separate prefix optima", "2D_Renyi_regret", "regret")
    if relative.stem == "01_spectrograms":
        bounds = record.get("frequency_view", {}).get("focus_hz")
        title = "Spectrograms · U-STFT, best, middle, and worst"
        if bounds:
            title += f" · {bounds[0] / 1000:g}–{bounds[1] / 1000:g} kHz"
        return 1, title, "Spectrograms_U_STFT_best_middle_worst", "spectrogram"
    if relative.stem in METRICS:
        return METRICS[relative.stem]
    if relative.stem.startswith("spectrograms_u_"):
        full_band = relative.stem.endswith("_full_band")
        view = record.get("frequency_view", {})
        bounds = view.get("full_band_hz" if full_band else "focus_hz")
        frequency = f"{bounds[0] / 1000:g}–{bounds[1] / 1000:g} kHz" if bounds else ""
        suffix = "full band" if full_band else "main view"
        method = relative.parts[-2] if "methods" in relative.parts else None
        if method == "posterior_blend":
            offset, title, filename = 1, "U-STFT spectrogram", "Spectrogram_U_STFT"
        elif method and re.fullmatch(r"fixed_\d+", method):
            index = int(method.split("_")[-1])
            lengths = record.get("candidate_lengths_samples", [])
            length = lengths[index] if index < len(lengths) else f"{2 ** index}H"
            offset, title, filename = 2 + index, f"Fixed {length} · spectrogram", f"Spectrogram_fixed_{length}"
        else:
            offset, title, filename = 0, "Spectrograms · U-STFT and all fixed windows", "Spectrogram_all_windows"
        title += f" · {frequency} ({suffix})" if frequency else f" · {suffix}"
        return 6 + offset + (7 if full_band else 0), title, filename + ("_full_band" if full_band else "_main"), "spectrogram"
    title = relative.stem.replace("_", " ").replace("-", " ")
    return 100, title, re.sub(r"[^A-Za-z0-9_]+", "_", str(relative.with_suffix(""))), "other"


def _file_links(files):
    return "".join(
        f'<div class="file-row"><span>{label}</span>'
        + "".join(f'<a href="{escape(files[style][ext], quote=True)}" target="_blank" rel="noopener">{ext.upper()}</a>' for ext in ("pdf", "png"))
        + "</div>"
        for style, label in (("dissertation", "Dissertation"), ("slides", "Slides"))
    )


def _html(groups, figures):
    navigation = ['<button class="wav-button active" data-wav="all" aria-pressed="true">All WAV files<span>' + str(len(figures)) + ' plots</span></button>']
    sections = []
    for group in groups:
        dataset, name = group["dataset"], group["name"]
        members = [figure for figure in figures if figure["dataset"] == dataset]
        navigation.append(f'<button class="wav-button" data-wav="{escape(dataset, quote=True)}" aria-pressed="false">{escape(name)}<span>{len(members)} plots</span></button>')
        cards = []
        for figure in members:
            files = figure["files"]
            cards.append(
                f'<article class="card" data-id="{figure["id"]}" data-type="{figure["type"]}">'
                f'<a class="preview" href="{escape(files["dissertation"]["png"], quote=True)}" target="_blank" rel="noopener" aria-label="Preview {escape(figure["title"], quote=True)}">'
                f'<img src="{escape(files["dissertation"]["png"], quote=True)}" alt="{escape(figure["title"], quote=True)}" loading="lazy" decoding="async"></a>'
                f'<div class="card-body"><h3>{escape(figure["title"])}</h3>{_file_links(files)}</div></article>'
            )
        excerpt = f' · {group["duration"]:g} s excerpt' if group["duration"] is not None else ""
        sections.append(
            f'<section class="wav-section" data-wav="{escape(dataset, quote=True)}"><div class="section-heading"><h2>{escape(name)}</h2><span>{len(members)} plots</span></div>'
            f'<p class="source-name">{escape(group["filename"])}{excerpt}</p><div class="cards">{"".join(cards)}</div></section>'
        )
    payload = json.dumps(figures, ensure_ascii=False).replace("<", "\\u003c")
    return _PAGE.replace("__NAV__", "".join(navigation)).replace("__SECTIONS__", "".join(sections)).replace("__FIGURES__", payload).replace("__WAV_COUNT__", str(len(groups))).replace("__PLOT_COUNT__", str(len(figures))).replace("__FILE_COUNT__", str(4 * len(figures)))


def build_gallery(edition, publish=True):
    """Add a complete gallery and flat files to an existing paired WAV edition."""
    edition = Path(edition).expanduser().resolve()
    inventory = json.loads((edition / "data/wav_inventory.json").read_text())
    records = {record["dataset"]: record for record in inventory}
    dissertation = {path.relative_to(edition / "dissertation") for path in (edition / "dissertation").rglob("*.png")}
    slides = {path.relative_to(edition / "slides") for path in (edition / "slides").rglob("*.png")}
    if not dissertation or dissertation != slides:
        raise ValueError("Cannot index an incomplete dissertation/slide collection")
    datasets = {path.parts[0] for path in dissertation}
    if datasets != records.keys():
        raise ValueError("WAV inventory and plotted datasets disagree")
    groups = []
    for dataset, record in records.items():
        order, name, filename = WAV_NAMES.get(record["filename"], (100, Path(record["filename"]).stem, dataset.replace("-", "_")))
        duration = record["excerpt_end_s"] - record["excerpt_start_s"] if "excerpt_end_s" in record and "excerpt_start_s" in record else None
        groups.append({"dataset": dataset, "name": name, "filename": record["filename"], "order": order,
                       "file_prefix": filename, "duration": duration})
    groups.sort(key=lambda group: (group["order"], group["name"]))
    flat = edition / "all_plots"
    flat.mkdir(exist_ok=True)
    figures, filenames = [], set()
    for wav_number, group in enumerate(groups, 1):
        dataset = group["dataset"]
        described = [(relative, _describe(relative, records[dataset])) for relative in dissertation if relative.parts[0] == dataset]
        described.sort(key=lambda item: (item[1][0], str(item[0])))
        for plot_number, (relative, (_, title, filename, kind)) in enumerate(described, 1):
            files = {}
            prefix = f"{wav_number:02d}_{group['file_prefix']}__{plot_number:02d}_{filename}"
            for style_number, style in enumerate(("dissertation", "slides"), 1):
                files[style] = {}
                for extension in ("png", "pdf"):
                    source = edition / style / relative.with_suffix("." + extension)
                    if not source.is_file():
                        raise ValueError(f"Missing figure file: {source}")
                    target = flat / f"{prefix}__{style_number}_{style}.{extension}"
                    if target.name in filenames:
                        raise ValueError(f"Duplicate gallery filename: {target.name}")
                    filenames.add(target.name)
                    if target.exists():
                        if not target.samefile(source):
                            raise ValueError(f"Existing flat file does not match its figure: {target}")
                    else:
                        target.hardlink_to(source)
                    files[style][extension] = "all_plots/" + quote(target.name)
            figures.append({"id": len(figures), "dataset": dataset, "wav": group["name"], "filename": group["filename"], "title": title,
                            "type": kind, "source": str(relative), "files": files})
    extras = {path.name for path in flat.iterdir()} - filenames
    if extras:
        raise ValueError(f"Unexpected files in the flat plot folder: {sorted(extras)}")
    page = _html(groups, figures)
    (edition / "index.html").write_text(page, encoding="utf-8")
    (edition / "plot_index.json").write_text(json.dumps({"wavs": groups, "plots": figures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = edition / "README.md"
    access = "[Open all plots](index.html) · [All PNGs and PDFs in one folder](all_plots/)"
    if readme.is_file() and access not in readme.read_text():
        heading, separator, body = readme.read_text().partition("\n")
        readme.write_text(heading + separator + "\n" + access + "\n" + body)
    if publish:
        # Keep the top-level entry point stable while preserving every edition.
        # Replace only a gallery symlink created by this helper.
        shortcut = edition.parent / "all_plots"
        if shortcut.exists() and not shortcut.is_symlink():
            raise ValueError(f"Refusing to replace a real folder: {shortcut}")
        temporary = edition.parent / ".all_plots_next"
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f"Unexpected temporary gallery link: {temporary}")
        temporary.symlink_to(Path(edition.name) / "all_plots", target_is_directory=True)
        temporary.replace(shortcut)
        (edition.parent / "ALL_PLOTS.html").write_text(page, encoding="utf-8")
        guide = edition.parent / "README.md"
        if not guide.exists() or guide.read_text().startswith("# All WAV plots\n"):
            guide.write_text(
                "# All WAV plots\n\n[Open the gallery](ALL_PLOTS.html) · [Open the flat plot folder](all_plots/)\n\n"
                f"{len(groups)} WAV files, {len(figures)} plots, {4 * len(figures)} PNGs and PDFs. "
                "Every plot has dissertation and slide versions.\n\n"
                "Use the gallery to select a WAV, search plots, enlarge thumbnails, or open either format. "
                "The flat folder contains all files, named in WAV / plot / style order. "
                "Earlier dated editions remain available below this folder.\n"
            )
    return edition / "index.html"


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>All WAV plots</title><style>
:root{font-family:Arial,Helvetica,sans-serif;color:#202735;background:#f5f6f8;color-scheme:light}
*{box-sizing:border-box}body{margin:0}button,input,select{font:inherit}button,a,input,select{-webkit-tap-highlight-color:transparent}
a{color:#244a9b;text-decoration:none}a:hover{text-decoration:underline}button{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #8cacf0;outline-offset:3px}
[hidden]{display:none!important}.masthead{padding:26px 32px 22px;background:#fff;border-bottom:1px solid #e0e4ea}
.masthead h1{font-size:30px;letter-spacing:-.6px;margin:0 0 8px}.masthead p{margin:0;color:#586275;line-height:1.6;font-size:14px}.folder-link{display:inline-block;margin-top:12px;font-size:14px;font-weight:600}
.layout{display:grid;grid-template-columns:228px minmax(0,1fr);max-width:1800px;margin:auto}.sidebar{padding:22px 14px;position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid #e0e4ea}
.sidebar h2{font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#687385;margin:0 12px 12px}.wav-button{display:block;text-align:left;width:100%;border:0;background:transparent;border-radius:7px;padding:10px 12px;margin:3px 0;color:#344154;font-size:14px;line-height:1.3}.wav-button span{display:block;font-size:12px;color:#687385;margin-top:4px}.wav-button:hover{background:#e9edf4}.wav-button.active{color:#163e91;background:#e5edfc}.wav-button.active span{color:#486698}
main{min-width:0;padding:0 28px 50px}.toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:14px;padding:20px 0;background:#f5f6f8;position:sticky;top:0;z-index:2;border-bottom:1px solid #e0e4ea}.control{display:flex;flex-direction:column;gap:6px}.control label,.preview-label{font-size:12px;color:#586275}.search{flex:1;min-width:180px}input,select{border:1px solid #c9d1dc;border-radius:6px;padding:9px 11px;background:#fff;color:#202735;width:100%}.preview-toggle{display:flex;border:1px solid #c9d1dc;border-radius:6px;background:#fff;overflow:hidden}.preview-toggle button{border:0;background:transparent;padding:10px 12px;color:#4a5669;font-size:13px}.preview-toggle button.active{background:#244a9b;color:#fff}
.results-count{font-size:13px;color:#687385;margin:16px 0 4px}.section-heading{display:flex;align-items:center;gap:12px;margin:26px 0 7px}.section-heading h2{font-size:21px;letter-spacing:-.3px;margin:0}.section-heading span{font-size:12px;padding:4px 8px;border-radius:20px;background:#e7ebf1;color:#526074}.source-name{font-size:12px;color:#6a7587;margin:0 0 16px;overflow-wrap:anywhere;line-height:1.5}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:#fff;border:1px solid #e0e4ea;border-radius:9px;overflow:hidden}.preview{display:flex;align-items:center;justify-content:center;aspect-ratio:16/9;background:#fff;border-bottom:1px solid #edf0f4;padding:7px}.preview img{display:block;width:100%;height:100%;object-fit:contain}.preview:hover{background:#fafbfe}.card-body{padding:13px 15px 15px}h3{font-size:14px;line-height:1.5;margin:0 0 12px;font-weight:600;min-height:42px}.file-row{display:flex;align-items:center;gap:8px;margin-top:6px;font-size:12px}.file-row span{width:90px;color:#687385}.file-row a{border:1px solid #d4ddea;border-radius:4px;padding:4px 10px;line-height:1.2}.file-row a:hover{background:#edf3ff;text-decoration:none}.empty{padding:50px 0;color:#687385}
dialog{border:0;border-radius:10px;padding:0;width:min(1280px,96vw);max-height:96vh;box-shadow:0 12px 60px #0005;background:#fff;color:#202735}dialog::backdrop{background:#17202be0}.viewer-head{display:flex;justify-content:space-between;align-items:start;gap:18px;padding:14px 18px;border-bottom:1px solid #e0e4ea}.viewer-head h2{font-size:17px;margin:0;line-height:1.45}.viewer-head p{font-size:12px;color:#687385;margin:5px 0 0}.close{border:0;background:#edf0f5;border-radius:6px;font-size:23px;line-height:1;padding:6px 10px}.viewer-image{display:block;width:100%;height:min(67vh,720px);object-fit:contain;padding:8px}.viewer-bottom{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:10px 18px 16px;border-top:1px solid #e0e4ea}.viewer-nav{display:flex;gap:8px}.viewer-nav button{padding:9px 13px;border:1px solid #c9d1dc;border-radius:6px;background:#fff;color:#244a9b}.viewer-nav button:disabled{opacity:.4;cursor:default}.viewer-files{display:flex;gap:20px}.viewer-files .file-row span{width:auto}
@media(min-width:1450px){.cards{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){.layout{grid-template-columns:190px minmax(0,1fr)}main{padding:0 18px 40px}.cards{grid-template-columns:1fr}.toolbar{gap:10px}.search{flex-basis:100%}.viewer-bottom,.viewer-files{flex-wrap:wrap}}
@media(max-width:600px){.masthead{padding:20px}.masthead h1{font-size:26px}.layout{display:block}.sidebar{height:auto;position:static;border-right:0;border-bottom:1px solid #e0e4ea;padding:12px;display:flex;overflow-x:auto;gap:6px}.sidebar h2{display:none}.wav-button{width:auto;min-width:150px;margin:0}.wav-button span{font-size:11px}.toolbar{position:static}.section-heading h2{font-size:19px}.viewer-image{height:56vh}.viewer-head h2{font-size:14px}.viewer-bottom{gap:12px}.viewer-files{gap:10px}.viewer-files .file-row{font-size:11px}}
</style></head><body>
<header class="masthead"><h1>All WAV plots</h1><p>__WAV_COUNT__ WAV files · __PLOT_COUNT__ plots · __FILE_COUNT__ PNGs and PDFs.<br>Click a thumbnail to enlarge it. Every plot has dissertation and slide versions.</p><a class="folder-link" href="all_plots/">Open all files in one folder ↗</a></header>
<div class="layout"><nav class="sidebar" aria-label="Choose a WAV file"><h2>WAV files</h2>__NAV__</nav><main>
<div class="toolbar"><div class="control search"><label for="search">Find a plot</label><input id="search" type="search" placeholder="Search titles, WAV names, or window lengths…"></div>
<div class="control"><label for="plot-type">Plot type</label><select id="plot-type"><option value="all">All plot types</option><option value="regret">Entropy comparisons</option><option value="weights">Blend weights</option><option value="spectrogram">Spectrograms</option></select></div>
<div class="control"><span class="preview-label">Thumbnail style</span><div class="preview-toggle" aria-label="Thumbnail style"><button class="active" data-style="dissertation" aria-pressed="true">Dissertation</button><button data-style="slides" aria-pressed="false">Slides</button></div></div></div>
<p id="results-count" class="results-count" aria-live="polite">Showing __PLOT_COUNT__ plots across __WAV_COUNT__ WAV files</p>__SECTIONS__<p id="empty" class="empty" hidden>No plots match this search. Clear the search or choose another WAV file.</p>
</main></div>
<dialog id="viewer" aria-labelledby="viewer-title"><div class="viewer-head"><div><h2 id="viewer-title"></h2><p id="viewer-context"></p></div><button id="close" class="close" aria-label="Close preview">×</button></div><a id="viewer-full" target="_blank" rel="noopener" title="Open full-size PNG"><img id="viewer-image" class="viewer-image" alt=""></a><div class="viewer-bottom"><div class="viewer-nav"><button id="previous">← Previous</button><button id="next">Next →</button></div><div id="viewer-files" class="viewer-files"></div></div></dialog>
<script>
const figures=__FIGURES__;
const byId=new Map(figures.map(f=>[String(f.id),f]));
const cards=Array.from(document.querySelectorAll('.card'));
const sections=Array.from(document.querySelectorAll('.wav-section'));
const viewer=document.getElementById('viewer');
let selectedWav='all',style='dissertation',visible=figures,selectedId=null;
function applyFilters(){
  const words=document.getElementById('search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const type=document.getElementById('plot-type').value;
  visible=figures.filter(f=>(selectedWav==='all'||f.dataset===selectedWav)&&(type==='all'||f.type===type)&&words.every(word=>(f.title+' '+f.wav+' '+f.filename+' '+f.dataset+' '+f.source).toLowerCase().includes(word)));
  const ids=new Set(visible.map(f=>String(f.id)));
  cards.forEach(card=>card.hidden=!ids.has(card.dataset.id));
  sections.forEach(section=>{
    const total=section.querySelectorAll('.card').length,count=section.querySelectorAll('.card:not([hidden])').length;
    section.hidden=count===0;
    section.querySelector('.section-heading span').textContent=count===total?`${total} plots`:`${count} of ${total} plots`;
  });
  const wavs=new Set(visible.map(f=>f.dataset)).size;
  document.getElementById('results-count').textContent=`Showing ${visible.length} plots across ${wavs} WAV ${wavs===1?'file':'files'}`;
  document.getElementById('empty').hidden=visible.length!==0;
}
document.querySelectorAll('.wav-button').forEach(button=>button.addEventListener('click',()=>{
  selectedWav=button.dataset.wav;
  document.querySelectorAll('.wav-button').forEach(peer=>{const active=peer===button;peer.classList.toggle('active',active);peer.setAttribute('aria-pressed',String(active));});
  applyFilters();window.scrollTo({top:0,behavior:'smooth'});
}));
document.getElementById('search').addEventListener('input',applyFilters);
document.getElementById('plot-type').addEventListener('change',applyFilters);
document.querySelectorAll('[data-style]').forEach(button=>button.addEventListener('click',()=>{
  style=button.dataset.style;
  document.querySelectorAll('[data-style]').forEach(peer=>{const active=peer===button;peer.classList.toggle('active',active);peer.setAttribute('aria-pressed',String(active));});
  cards.forEach(card=>{const figure=byId.get(card.dataset.id),url=figure.files[style].png;card.querySelector('img').src=url;card.querySelector('.preview').href=url;});
}));
function showFigure(id){
  const figure=byId.get(String(id));selectedId=figure.id;
  document.getElementById('viewer-title').textContent=figure.title;
  document.getElementById('viewer-context').textContent=figure.wav+' · '+(style==='slides'?'Slides':'Dissertation');
  const image=document.getElementById('viewer-image');image.src=figure.files[style].png;image.alt=figure.wav+' — '+figure.title;
  document.getElementById('viewer-full').href=image.getAttribute('src');
  const links=document.getElementById('viewer-files');links.replaceChildren();
  for(const [key,label] of [['dissertation','Dissertation'],['slides','Slides']]){
    const row=document.createElement('div');row.className='file-row';const span=document.createElement('span');span.textContent=label;row.append(span);
    for(const ext of ['pdf','png']){const link=document.createElement('a');link.href=figure.files[key][ext];link.textContent=ext.toUpperCase();link.target='_blank';link.rel='noopener';row.append(link);}links.append(row);
  }
  const index=visible.findIndex(f=>f.id===selectedId);
  document.getElementById('previous').disabled=index<=0;
  document.getElementById('next').disabled=index>=visible.length-1;
  if(!viewer.open)viewer.showModal();
}
cards.forEach(card=>card.querySelector('.preview').addEventListener('click',event=>{
  if(event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
  event.preventDefault();showFigure(card.dataset.id);
}));
function move(direction){const index=visible.findIndex(f=>f.id===selectedId);if(visible[index+direction])showFigure(visible[index+direction].id);}
document.getElementById('previous').addEventListener('click',()=>move(-1));
document.getElementById('next').addEventListener('click',()=>move(1));
document.getElementById('close').addEventListener('click',()=>viewer.close());
viewer.addEventListener('click',event=>{if(event.target===viewer){const box=viewer.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)viewer.close();}});
document.addEventListener('keydown',event=>{if(viewer.open&&event.key==='ArrowLeft')move(-1);if(viewer.open&&event.key==='ArrowRight')move(1);});
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-edition", type=Path, required=True)
    parser.add_argument("--no-publish", action="store_true", help="Do not update the parent folder's All plots shortcut")
    arguments = parser.parse_args()
    gallery = build_gallery(arguments.from_edition, publish=not arguments.no_publish)
    print(f"Open all plots: {gallery.parent.parent / 'ALL_PLOTS.html' if not arguments.no_publish else gallery}")


if __name__ == "__main__":
    main()
