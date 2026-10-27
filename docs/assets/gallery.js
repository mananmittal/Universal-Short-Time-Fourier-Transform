/* Generated from results/paper_wavs/2026-10-04_105324_935286/plot_index.json */
var RECORDINGS = [{"name": "Humpback whale song", "file": "humpback_whale_song.wav", "duration": 5.0, "windows": ["H", "4H", "16H"], "plots": [{"src": "assets/plots/humpback-whale-song/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–4 kHz"}, {"src": "assets/plots/humpback-whale-song/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/humpback-whale-song/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/humpback-whale-song/blend-weights.png", "title": "Blend weights · PMF over time"}]}, {"name": "Humpback whale and dolphins", "file": "humpback_whale_and_dolphins.wav", "duration": 4.94, "windows": ["H", "4H", "16H"], "plots": [{"src": "assets/plots/humpback-whale-and-dolphins/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–4 kHz"}, {"src": "assets/plots/humpback-whale-and-dolphins/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/humpback-whale-and-dolphins/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/humpback-whale-and-dolphins/blend-weights.png", "title": "Blend weights · PMF over time"}]}, {"name": "Snapping shrimp", "file": "snapping_shrimp.wav", "duration": 5.0, "windows": ["H", "2H", "8H"], "plots": [{"src": "assets/plots/snapping-shrimp/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–24 kHz"}, {"src": "assets/plots/snapping-shrimp/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/snapping-shrimp/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/snapping-shrimp/blend-weights.png", "title": "Blend weights · PMF over time"}]}, {"name": "Fish chorus", "file": "fish_chorus.wav", "duration": 5.0, "windows": ["H", "4H", "16H"], "plots": [{"src": "assets/plots/fish-chorus/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–4 kHz"}, {"src": "assets/plots/fish-chorus/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/fish-chorus/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/fish-chorus/blend-weights.png", "title": "Blend weights · PMF over time"}]}, {"name": "English speech 1", "file": "english_speech_1.wav", "duration": 5.0, "windows": ["H", "4H", "16H"], "plots": [{"src": "assets/plots/english-speech-1/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–8 kHz"}, {"src": "assets/plots/english-speech-1/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/english-speech-1/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/english-speech-1/blend-weights.png", "title": "Blend weights · PMF over time"}]}, {"name": "English speech 2", "file": "english_speech_2.wav", "duration": 5.0, "windows": ["H", "4H", "16H"], "plots": [{"src": "assets/plots/english-speech-2/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–8 kHz"}, {"src": "assets/plots/english-speech-2/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/english-speech-2/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/english-speech-2/blend-weights.png", "title": "Blend weights · PMF over time"}]}, {"name": "Hyperbolic chirp", "file": "hyperbolic_chirp.wav", "duration": 15.0, "windows": ["H", "4H", "16H"], "plots": [{"src": "assets/plots/hyperbolic-chirp/spectrograms.jpg", "title": "Spectrograms · U-STFT, best, middle, and worst · 0–4 kHz"}, {"src": "assets/plots/hyperbolic-chirp/entropy-gaps.png", "title": "1-D entropy gaps · time, frequency, and total"}, {"src": "assets/plots/hyperbolic-chirp/renyi-regret.png", "title": "2-D Rényi regret · separate prefix optima"}, {"src": "assets/plots/hyperbolic-chirp/blend-weights.png", "title": "Blend weights · PMF over time"}]}];


/* Where each recording comes from; [n] points to the Data sources list. */
var SANCTSOUND = "NOAA Office of National Marine Sanctuaries and U.S. Navy, SanctSound passive acoustic monitoring, ";
var SOURCES = {
  "humpback_whale_song.wav": {text: SANCTSOUND + "Channel Islands site CI04, 2020-02-25.", ref: 1},
  "humpback_whale_and_dolphins.wav": {text: SANCTSOUND + "Hawaiian Islands site HI04, 2020-01-29.", ref: 1},
  "snapping_shrimp.wav": {text: SANCTSOUND + "Florida Keys site FK02, 2019-11-03.", ref: 1},
  "fish_chorus.wav": {text: SANCTSOUND + "Gray's Reef site GR01, 2019-09-02.", ref: 1},
  "english_speech_1.wav": {text: "CSTR VCTK Corpus, speaker p227.", ref: 2},
  "english_speech_2.wav": {text: "CSTR VCTK Corpus, speaker p228.", ref: 2},
  "hyperbolic_chirp.wav": {text: "Synthetic hyperbolic chirp generated for this work.", ref: null}
};

(function () {
  "use strict";

  function caption(rec, i) {
    switch (i) {
      case 0:
        return "U-STFT compared with fixed-window STFTs at supports " +
          rec.windows.slice(0, -1).join(", ") + " and " + rec.windows[rec.windows.length - 1] +
          "; panel titles mark which fixed window is best, middle and worst.";
      case 1:
        return "Running entropy gap of the U-STFT against hindsight comparators, shown separately for time, frequency and total concentration.";
      case 2:
        return "Joint 2-D Rényi-3 regret against the finite-trellis hindsight reference, with each prefix optimum computed independently.";
      default:
        return "Posterior weight over the dyadic window supports H, 2H, 4H, … at every frame.";
    }
  }

  var picker = document.getElementById("picker");
  var select = document.getElementById("picker-select");
  var meta = document.getElementById("rec-meta");
  var grid = document.getElementById("plots");
  var current = 0;

  RECORDINGS.forEach(function (rec, idx) {
    var b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "tab");
    b.textContent = rec.name;
    b.addEventListener("click", function () { show(idx); });
    picker.insertBefore(b, select);
    var o = document.createElement("option");
    o.value = idx;
    o.textContent = rec.name;
    select.appendChild(o);
  });
  select.addEventListener("change", function () { show(+select.value); });

  function show(idx, initial) {
    current = idx;
    var rec = RECORDINGS[idx];
    Array.prototype.forEach.call(picker.querySelectorAll("button"), function (b, j) {
      b.setAttribute("aria-selected", j === idx ? "true" : "false");
    });
    select.value = idx;
    meta.innerHTML = "";
    meta.appendChild(document.createTextNode(rec.duration + " s excerpt · "));
    var c = document.createElement("code");
    c.textContent = rec.file;
    meta.appendChild(c);

    var source = SOURCES[rec.file];
    var line = document.createElement("div");
    line.className = "rec-source";
    line.appendChild(document.createTextNode("Source: " + source.text + " "));
    if (source.ref) {
      var a = document.createElement("a");
      a.href = "#data-sources";
      a.textContent = "[" + source.ref + "]";
      line.appendChild(a);
    }
    meta.appendChild(line);

    grid.innerHTML = "";
    rec.plots.forEach(function (p, i) {
      var fig = document.createElement("figure");
      fig.className = "plot";
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "thumb";
      btn.setAttribute("aria-label", "Enlarge: " + p.title);
      var img = document.createElement("img");
      img.src = p.src;
      img.loading = "lazy";
      img.decoding = "async";
      img.alt = rec.name + " — " + p.title;
      btn.appendChild(img);
      btn.addEventListener("click", function () { openLb(i); });
      var cap = document.createElement("figcaption");
      var s = document.createElement("strong");
      s.textContent = p.title;
      var t = document.createElement("span");
      t.textContent = caption(rec, i);
      cap.appendChild(s);
      cap.appendChild(t);
      fig.appendChild(btn);
      fig.appendChild(cap);
      grid.appendChild(fig);
    });
    if (!initial) try { history.replaceState(null, "", "#rec-" + (idx + 1)); } catch (e) {}
  }

  // Lightbox
  var lb = document.getElementById("lightbox");
  var lbImg = document.getElementById("lb-img");
  var lbCap = document.getElementById("lb-cap");
  var lbIndex = 0;
  var lastFocus = null;

  function renderLb() {
    var rec = RECORDINGS[current];
    var p = rec.plots[lbIndex];
    lbImg.src = p.src;
    lbImg.alt = rec.name + " — " + p.title;
    lbCap.textContent = rec.name + " · " + p.title + " (" + (lbIndex + 1) + "/" + rec.plots.length + ")";
  }
  function openLb(i) {
    lastFocus = document.activeElement;
    lbIndex = i;
    renderLb();
    lb.classList.add("open");
    document.body.style.overflow = "hidden";
    document.getElementById("lb-close").focus();
  }
  function closeLb() {
    lb.classList.remove("open");
    document.body.style.overflow = "";
    if (lastFocus) lastFocus.focus();
  }
  function step(d) {
    var n = RECORDINGS[current].plots.length;
    lbIndex = (lbIndex + d + n) % n;
    renderLb();
  }
  document.getElementById("lb-close").addEventListener("click", closeLb);
  document.getElementById("lb-prev").addEventListener("click", function (e) { e.stopPropagation(); step(-1); });
  document.getElementById("lb-next").addEventListener("click", function (e) { e.stopPropagation(); step(1); });
  lb.addEventListener("click", function (e) { if (e.target === lb) closeLb(); });
  document.addEventListener("keydown", function (e) {
    if (!lb.classList.contains("open")) return;
    if (e.key === "Escape") closeLb();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
  });

  // Copy buttons
  Array.prototype.forEach.call(document.querySelectorAll("pre"), function (pre) {
    var b = document.createElement("button");
    var isBib = pre.classList.contains("bibtex");
    var label = isBib ? "Copy BibTeX" : "Copy";
    b.type = "button";
    b.className = isBib ? "copy copy-bib" : "copy";
    b.textContent = label;
    b.addEventListener("click", function () {
      var text = pre.querySelector("code").innerText;
      var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = label; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
    });
    pre.appendChild(b);
  });

  var m = /^#rec-(\d+)$/.exec(location.hash);
  var start = m ? Math.min(Math.max(+m[1] - 1, 0), RECORDINGS.length - 1) : 0;
  show(start, true);
})();
