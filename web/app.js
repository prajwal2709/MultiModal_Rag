(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const app = $("#appShell");
  const canvas = $("#canvasWorkspace");
  const cardLayer = $("#evidenceCardLayer");
  const connectorSvg = $("#connectorLayer");
  const fileInput = $("#fileInput");
  const figureDialog = $("#figureDialog");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

  let docs = [];
  let currentDoc = null;
  let currentPage = 1;
  let zoom = 100;
  let activeView = "workspace";
  let activeTab = "overview";
  let cards = [];
  let links = [];
  let connectMode = false;
  let pendingConnection = null;
  let lastQuestion = "";
  let latestAnswer = "";
  let isAsking = false;
  let toastTimer;
  let recentQuestions = JSON.parse(localStorage.getItem("doculens.recent") || "[]");

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function request(url, options = {}) {
    const response = await fetch(url, options);
    const type = response.headers.get("content-type") || "";
    const data = type.includes("application/json") ? await response.json() : {};
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  function toast(message, kind = "") {
    const node = $("#toast");
    node.textContent = message;
    node.className = `toast ${kind ? `is-${kind}` : ""}`;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 4500);
  }

  function docForId(id) { return docs.find((doc) => doc.document_id === id); }

  function setView(name) {
    activeView = name;
    $$('[data-screen]').forEach((screen) => { screen.hidden = screen.dataset.screen !== name; });
    $$('[data-view-link]').forEach((link) => link.classList.toggle("is-active", link.dataset.viewLink === name));
    app.classList.remove("mobile-nav-open");
    $("#mobileBackdrop").hidden = true;
    if (name === "map") renderMap();
    if (name === "collections") renderCollections();
    if (name === "research") renderResearchPrompts();
    if (name === "notes") loadNote();
  }

  function setStatus(doc) {
    const badge = $("#indexStatus");
    badge.className = "status-pill";
    if (!doc) { badge.innerHTML = "<i></i> Waiting"; return; }
    if (doc.indexing) { badge.classList.add("is-indexing"); badge.innerHTML = "<i></i> Indexing"; }
    else if (doc.indexed) { badge.classList.add("is-indexed"); badge.innerHTML = "<i></i> Indexed"; }
    else { badge.classList.add("is-failed"); badge.innerHTML = "<i></i> Not indexed"; }
  }

  function renderRecent() {
    const root = $("#recentDocuments");
    root.replaceChildren();
    const recent = docs.filter((doc) => JSON.parse(localStorage.getItem("doculens.opened") || "[]").includes(doc.name)).slice(0, 6);
    if (!recent.length) {
      root.innerHTML = '<p class="rail-muted">Your documents will appear here.</p>';
      return;
    }
    recent.forEach((doc) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = `recent-doc${doc.name === currentDoc?.name ? " is-current" : ""}`;
      button.innerHTML = `<span class="mini-pdf" aria-hidden="true">▤</span><span>${escapeHtml(doc.name)}</span>`;
      button.title = doc.name; button.addEventListener("click", () => selectDocument(doc)); root.append(button);
    });
  }

  function renderDocumentCards() {
    const root = $("#libraryCards");
    const filter = ($("#librarySearch").value || "").toLocaleLowerCase();
    root.replaceChildren();
    const visible = docs.filter((doc) => doc.name.toLocaleLowerCase().includes(filter));
    $("#librarySummary").textContent = `${visible.length} document${visible.length === 1 ? "" : "s"}`;
    if (!visible.length) {
      root.innerHTML = `<div class="secondary-paper"><h3>${docs.length ? "No matching documents" : "Your archive is waiting"}</h3><p>${docs.length ? "Try another title." : "Add a PDF to extract its text and figures, then explore it here."}</p></div>`;
      return;
    }
    visible.forEach((doc) => {
      const article = document.createElement("article"); article.className = "library-card";
      const status = doc.indexing ? "Indexing" : doc.indexed ? "Indexed and searchable" : "Not indexed";
      article.innerHTML = `<span class="pdf-emblem">PDF</span><h3 title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</h3><p>${status} · ${doc.pages} pages · ${doc.figures} figures · ${escapeHtml(doc.size)}</p><div class="library-card-actions"><button class="open-document" type="button" data-open-document="${escapeHtml(doc.name)}">Open workspace</button><button class="remove-document" type="button" data-remove-document="${escapeHtml(doc.name)}" ${doc.indexing ? "disabled" : ""}>Remove</button></div>`;
      root.append(article);
    });
  }

  async function loadDocuments() {
    const data = await request("/api/documents");
    docs = Array.isArray(data.documents) ? data.documents : [];
    $("#navDocCount").textContent = docs.length;
    renderDocumentCards();
    renderRecent();
    if (currentDoc) {
      const refreshed = docs.find((doc) => doc.name === currentDoc.name);
      if (!refreshed) {
        currentDoc = null;
        renderCurrentDocument();
        if (activeView === "workspace") setView("documents");
      } else {
        const changed = refreshed.document_id !== currentDoc.document_id;
        currentDoc = refreshed;
        if (changed) await selectDocument(refreshed);
        else { setStatus(refreshed); updateCounts(refreshed); }
      }
    } else if (docs.length) {
      const opened = JSON.parse(localStorage.getItem("doculens.opened") || "[]");
      await selectDocument(docs.find((doc) => opened.includes(doc.name)) || docs[0]);
    } else {
      renderCurrentDocument();
    }
  }

  function updateCounts(doc) {
    $("#pageCount").textContent = doc?.pages ?? 0;
    $("#figureCount").textContent = doc?.figures ?? 0;
    $("#totalPages").textContent = `/ ${doc?.pages || "—"}`;
    $("#noteDocumentName").textContent = doc?.name || "No document selected";
  }

  function renderCurrentDocument() {
    $("#activeDocumentName").textContent = currentDoc?.name || "Choose a document";
    $("#activeDocumentSub").textContent = currentDoc ? `${currentDoc.size} · ${currentDoc.indexed ? "Ready to search" : "Available to read"}` : "Your research library";
    updateCounts(currentDoc);
    setStatus(currentDoc);
    $("#askScope").textContent = "Searching all indexed PDFs";
    $("#pageRail").innerHTML = '<p class="empty-hint">Select a PDF to browse its pages.</p>';
    $("#pdfFrameWrap").innerHTML = '<div class="reader-empty"><span>▤</span><strong>Your source page will appear here</strong><small>Add or select an indexed PDF to begin.</small></div>';
    $("#readerPageLabel").textContent = "— / —";
    $("#pageNumber").value = 1;
    $("#figureStripCount").textContent = "Select a PDF";
    $("#figureItems").innerHTML = '<p class="empty-hint">Figures from the selected PDF appear here after indexing.</p>';
    renderRecent();
  }

  async function selectDocument(doc) {
    if (!doc) return;
    currentDoc = doc;
    currentPage = 1;
    zoom = 100;
    const opened = JSON.parse(localStorage.getItem("doculens.opened") || "[]").filter((name) => name !== doc.name);
    localStorage.setItem("doculens.opened", JSON.stringify([doc.name, ...opened].slice(0, 8)));
    renderCurrentDocument();
    await Promise.allSettled([renderPageRail(), renderPage(), loadFigures()]);
    renderRecent();
    if (activeView !== "workspace") setView("workspace");
  }

  function pdfUrl(page = currentPage) {
    return `/api/pdf?name=${encodeURIComponent(currentDoc.name)}#page=${page}&zoom=${zoom}`;
  }

  async function renderPage() {
    if (!currentDoc) return;
    const page = Math.max(1, Math.min(currentDoc.pages || 1, Number(currentPage) || 1));
    currentPage = page;
    $("#pageNumber").value = page;
    $("#readerPageLabel").textContent = `${page} / ${currentDoc.pages || "—"}`;
    $("#zoomReadout").textContent = `${zoom}%`;
    let frame = $("#pdfFrame");
    if (!frame) {
      $("#pdfFrameWrap").innerHTML = '<iframe class="pdf-native" id="pdfFrame" title="Original PDF page"></iframe>';
      frame = $("#pdfFrame");
    }
    frame.src = pdfUrl(page);
    $$(".page-thumb").forEach((thumb) => thumb.classList.toggle("is-current", Number(thumb.dataset.page) === page));
  }

  async function renderPageRail() {
    const rail = $("#pageRail");
    rail.replaceChildren();
    if (!currentDoc?.pages) { rail.innerHTML = '<p class="empty-hint">No pages available.</p>'; return; }
    const frag = document.createDocumentFragment();
    for (let page = 1; page <= currentDoc.pages; page += 1) {
      const button = document.createElement("button"); button.type = "button"; button.className = `page-thumb${page === currentPage ? " is-current" : ""}`; button.dataset.page = page; button.title = `Open page ${page}`;
      const img = document.createElement("img"); img.loading = "lazy"; img.alt = `Thumbnail of page ${page}`; img.src = `/api/pdf-page?name=${encodeURIComponent(currentDoc.name)}&page=${page}&scale=0.28`;
      const label = document.createElement("span"); label.textContent = page;
      button.append(img, label); frag.append(button);
    }
    rail.append(frag);
  }

  async function loadFigures() {
    const root = $("#figureItems");
    if (!currentDoc) { root.innerHTML = '<p class="empty-hint">Select a PDF to see its extracted figures.</p>'; return; }
    try {
      const data = await request(`/api/figures?name=${encodeURIComponent(currentDoc.name)}`);
      const figures = data.figures || [];
      $("#figureStripCount").textContent = `${figures.length} extracted`;
      $("#figureCount").textContent = currentDoc.figures ?? figures.length;
      root.replaceChildren();
      if (!figures.length) {
        root.innerHTML = `<p class="empty-hint">${currentDoc.indexed ? "No figures were extracted from this PDF." : "Index this PDF to make extracted figures available."}</p>`;
        return;
      }
      figures.forEach((figure, index) => {
        const tile = document.createElement("div"); tile.className = "figure-tile"; tile.dataset.figureIndex = index;
        const open = document.createElement("button"); open.type = "button"; open.className = "figure-open"; open.dataset.openFigure = index; open.setAttribute("aria-label", `Preview figure from page ${figure.page ?? "unknown"}`);
        const img = document.createElement("img"); img.loading = "lazy"; img.src = `/api/image?path=${encodeURIComponent(figure.path)}`; img.alt = `Extracted figure from page ${figure.page ?? "unknown"}`; open.append(img);
        const page = document.createElement("span"); page.textContent = figure.page ? `Page ${figure.page}` : "Figure";
        const add = document.createElement("button"); add.type = "button"; add.className = "figure-add"; add.dataset.addFigure = index; add.textContent = "+"; add.title = "Pin figure to the evidence canvas"; add.setAttribute("aria-label", "Pin figure to the evidence canvas");
        tile.append(open, page, add); root.append(tile);
      });
      root._figures = figures;
    } catch (error) {
      root.innerHTML = `<p class="empty-hint">${escapeHtml(error.message)}</p>`;
    }
  }

  function showFigure(imageUrl, caption) {
    $("#figurePreview").src = imageUrl;
    $("#figureCaption").textContent = caption;
    figureDialog.showModal();
  }

  function addCard(source, type = "source", content = "") {
    const key = `card-${crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)}`;
    const card = { key, source, type, content, x: 13 + (cards.length % 2) * 245, y: 16 + Math.floor(cards.length / 2) * 155 };
    cards.push(card);
    renderCards();
    return card;
  }

  function addFigureCard(figure) {
    const doc = currentDoc;
    addCard({ type: "image", document_id: doc.document_id, document_name: doc.name, page: figure.page, image_path: figure.path, source_id: "Figure" }, "figure", "Extracted figure");
    toast("Figure pinned to the evidence canvas.", "success");
  }

  function renderCards() {
    cardLayer.replaceChildren();
    $("#canvasEmpty").hidden = cards.length > 0;
    cards.forEach((card) => {
      const source = card.source || {};
      const page = source.page ? `Page ${source.page}` : "Research note";
      const kind = card.type === "answer" ? "DocuLens insight" : card.type === "figure" || source.type === "image" ? "Figure" : "Key passage";
      const article = document.createElement("article");
      article.className = `evidence-card${card.selected ? " is-selected" : ""}`;
      article.dataset.cardKey = card.key;
      article.style.left = `${card.x}px`; article.style.top = `${card.y}px`;
      const title = document.createElement("div"); title.className = "card-topline";
      title.innerHTML = `<span class="card-type"><i></i>${escapeHtml(kind)}</span><span class="card-page">${escapeHtml(page)}</span>`;
      const metadata = document.createElement("p"); metadata.className = "card-document";
      metadata.textContent = [source.document_name, source.source_id ? `[${source.source_id}]` : ""].filter(Boolean).join(" · ");
      const copy = document.createElement("p"); copy.className = `card-copy${card.type === "answer" ? " is-insight" : ""}`;
      copy.textContent = card.content || source.text || "Visual source extracted from this PDF page.";
      article.append(title);
      if (metadata.textContent) article.append(metadata);
      article.append(copy);
      if (source.type === "image" && source.image_path) {
        const img = document.createElement("img"); img.className = "card-image"; img.loading = "lazy"; img.src = `/api/image?path=${encodeURIComponent(source.image_path)}`; img.alt = `Figure from ${source.document_name || "PDF"}, page ${source.page || "unknown"}`; article.append(img);
      }
      const actions = document.createElement("div"); actions.className = "card-actions";
      const open = document.createElement("button"); open.type = "button"; open.className = "card-action"; open.dataset.jumpCard = card.key; open.textContent = source.page ? `Open ${source.document_name ? "source" : "page"} →` : "Insight";
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "card-action"; remove.dataset.removeCard = card.key; remove.textContent = "Remove";
      actions.append(open, remove); article.append(actions); cardLayer.append(article);
    });
    drawLinks();
    updateConnectionsPanel();
    const selectedCount = cards.filter((item) => item.selected).length;
    $("#askScope").textContent = selectedCount ? `${selectedCount} selected evidence card${selectedCount === 1 ? "" : "s"} guide retrieval across the indexed PDFs` : "Searching all indexed PDFs";
  }

  function drawLinks() {
    const width = Math.max(canvas.clientWidth, cardLayer.scrollWidth || 0);
    const height = Math.max(canvas.clientHeight, cardLayer.scrollHeight || 0);
    cardLayer.style.width = `${width}px`; cardLayer.style.height = `${height}px`;
    connectorSvg.setAttribute("width", width); connectorSvg.setAttribute("height", height); connectorSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    connectorSvg.replaceChildren();
    links.forEach(([from, to]) => {
      const a = cards.find((item) => item.key === from), b = cards.find((item) => item.key === to);
      if (!a || !b) return;
      const nodeA = $(`[data-card-key="${CSS.escape(from)}"]`), nodeB = $(`[data-card-key="${CSS.escape(to)}"]`);
      if (!nodeA || !nodeB) return;
      const x1 = a.x + nodeA.offsetWidth / 2, y1 = a.y + nodeA.offsetHeight / 2;
      const x2 = b.x + nodeB.offsetWidth / 2, y2 = b.y + nodeB.offsetHeight / 2;
      const bend = Math.max(30, Math.abs(x2 - x1) * .35);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`); path.setAttribute("class", "connector-line"); connectorSvg.append(path);
    });
  }

  function updateConnectionsPanel() {
    const root = $("#insightContent");
    const mode = $(".insight-tabs .is-active")?.dataset.insight || "summary";
    if (mode === "connections") {
      if (!links.length) {
        root.innerHTML = '<div class="insight-placeholder"><span>↗</span><h3>No connections yet</h3><p>Choose Connect, then select two evidence cards to draw a relationship you want to remember.</p></div>';
      } else {
        root.innerHTML = `<h3 class="insight-subhead">Your evidence links</h3>${links.map(([a, b], i) => {
          const left = cards.find((c) => c.key === a), right = cards.find((c) => c.key === b);
          return `<button class="insight-source-link" data-focus-card="${escapeHtml(a)}">${escapeHtml(left?.source?.source_id || left?.type || "Evidence")} ↔ ${escapeHtml(right?.source?.source_id || right?.type || "Evidence")} <span>Link ${i + 1}</span></button>`;
        }).join("")}`;
      }
    }
  }

  function renderInsightSummary(answer, result) {
    const root = $("#insightContent");
    const sources = result.retrieved_sources || result.sources || [];
    root.innerHTML = `<h3 class="insight-subhead">Latest finding</h3><p class="insight-summary">${escapeHtml(answer)}</p><h3 class="insight-subhead">Evidence used</h3>${sources.slice(0, 6).map((source) => `<button class="insight-source-link" data-jump-source="${escapeHtml(source.source_id)}">[${escapeHtml(source.source_id)}] ${escapeHtml(source.document_name || "Document")} · Page ${escapeHtml(source.page ?? "—")}</button>`).join("") || '<p class="insight-summary">No source cards were returned.</p>'}<p class="verification-note">${escapeHtml(verificationLabel(result.verification_status))}</p>`;
    $(".insight-tabs [data-insight='summary']").classList.add("is-active");
    $(".insight-tabs [data-insight='connections']").classList.remove("is-active");
    $(".insight-tabs [data-insight='questions']").classList.remove("is-active");
  }

  function verificationLabel(status) {
    if (status === "verification_failed") return "Citation checks could not be completed.";
    if (status === "completed_with_contradictions") return "At least one claim conflicts with its cited evidence.";
    if (status === "completed_with_insufficient_evidence") return "At least one claim needs more evidence.";
    if (status === "skipped_no_cited_sources") return "No recognized cited sources were available to check.";
    return "Review the cited passages before relying on an answer.";
  }

  function setConnectMode(value) {
    connectMode = value;
    pendingConnection = null;
    $("#connectMode").classList.toggle("is-active", value);
    $("#canvasStatus").textContent = value ? "Select two evidence cards to connect." : "Drag cards to arrange them; select Connect to link evidence.";
  }

  function selectCard(card) {
    if (connectMode) {
      if (!pendingConnection) {
        pendingConnection = card.key; card.selected = true; $("#canvasStatus").textContent = "Select one more card to complete this connection."; renderCards();
      } else if (pendingConnection === card.key) {
        card.selected = false; pendingConnection = null; $("#canvasStatus").textContent = "Choose two different cards to connect.";
      } else {
        links.push([pendingConnection, card.key]);
        cards.forEach((item) => { item.selected = false; });
        setConnectMode(false); renderCards();
        toast("Evidence connection saved in this workspace.", "success");
      }
    } else {
      cards.forEach((item) => { item.selected = item.key === card.key ? !item.selected : false; });
      renderCards();
    }
  }

  async function jumpToSource(source) {
    const doc = docForId(source.document_id) || docs.find((item) => item.name === source.document_name);
    if (doc && doc.name !== currentDoc?.name) await selectDocument(doc);
    if (source.page && currentDoc) {
      currentPage = Number(source.page); await renderPage();
      const thumb = $(`.page-thumb[data-page="${currentPage}"]`); thumb?.scrollIntoView({ block: "nearest", behavior: reducedMotion.matches ? "auto" : "smooth" });
    }
  }

  async function jumpToCard(key) {
    const card = cards.find((item) => item.key === key);
    if (card?.source?.page) await jumpToSource(card.source);
  }

  function appendAnswer(result, question) {
    const answer = String(result.answer || "DocuLens returned no answer.");
    addCard({ type: "insight", source_id: "Insight" }, "answer", answer);
    const retrieved = result.retrieved_sources || result.sources || [];
    const sourceCards = new Map();
    retrieved.forEach((source) => {
      const card = addCard(source);
      sourceCards.set(source.source_id, card);
    });
    renderFinalAnswer(answer, result, sourceCards);
    renderInsightSummary(answer, result);
    recentQuestions = [question, ...recentQuestions.filter((item) => item !== question)].slice(0, 8);
    localStorage.setItem("doculens.recent", JSON.stringify(recentQuestions));
    toast("Answer and retrieved sources pinned to the research canvas.", "success");
    $("#finalAnswer").scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "center" });
  }

  function renderFinalAnswer(answer, result, sourceCards) {
    latestAnswer = answer;
    $("#finalAnswer").hidden = false;
    const citedSources = result.sources || [];
    const cardById = new Map(citedSources.map((source) => [source.source_id, sourceCards.get(source.source_id)]).filter(([, card]) => card));
    const safeAnswer = escapeHtml(answer).replace(/\[((?:[TI]\d+)(?:\s*,\s*[TI]\d+)*)\]/g, (citation, ids) => {
      const card = cardById.get(ids.split(",")[0].trim());
      return card ? `<button class="answer-citation" type="button" data-jump-card="${escapeHtml(card.key)}">${citation}</button>` : citation;
    }).replace(/\n/g, "<br>");
    $("#finalAnswerBody").innerHTML = safeAnswer;
    const status = result.verification_status || "unknown";
    $("#answerCheck").textContent = verificationLabel(status);
    const links = $("#answerSourceLinks"); links.replaceChildren();
    citedSources.forEach((source) => {
      const card = sourceCards.get(source.source_id); if (!card) return;
      const button = document.createElement("button"); button.type = "button"; button.dataset.jumpCard = card.key;
      button.textContent = `[${source.source_id}] ${source.document_name || "Document"} · Page ${source.page ?? "—"}`;
      links.append(button);
    });
  }

  async function pollAnswer(jobId) {
    while (true) {
      const job = await request(`/api/answers/${encodeURIComponent(jobId)}`);
      if (job.status === "complete") return job.result;
      if (job.status === "error") throw new Error(job.error || "Answer generation failed.");
      $("#sendButton").setAttribute("aria-label", `Working: ${job.phase || "retrieving"}`);
      await wait(850);
    }
  }

  async function ask(question) {
    const clean = question.trim();
    if (!clean || isAsking) return;
    if (!docs.some((doc) => doc.indexed)) { toast("Index a PDF before asking a question.", "error"); return; }
    isAsking = true; lastQuestion = clean; $("#sendButton").disabled = true; $("#sendButton").classList.add("is-loading"); $("#canvasStatus").textContent = "Searching indexed PDFs and retrieving evidence…";
    try {
      const selectedHint = cards.filter((card) => card.selected).map((card) => {
        const source = card.source || {};
        return [source.document_name, source.page ? `page ${source.page}` : "", card.content || source.text || ""].filter(Boolean).join(": ").slice(0, 800);
      }).join("\n").slice(0, 3000);
      const { job_id: jobId } = await request("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: clean, retrieval_hint: selectedHint }) });
      const result = await pollAnswer(jobId);
      appendAnswer(result, clean);
      $("#question").value = "";
    } catch (error) {
      $("#canvasStatus").textContent = "The request did not complete. You can retry it.";
      toast(error.message, "error");
    } finally {
      isAsking = false; $("#sendButton").disabled = false; $("#sendButton").classList.remove("is-loading"); $("#sendButton").setAttribute("aria-label", "Ask DocuLens");
    }
  }

  async function pollUpload(jobId) {
    const labels = { extracting: "Extracting PDF text and figures…", text_indexing: "Embedding and indexing text…", image_indexing: "Embedding and indexing figures…" };
    while (true) {
      const job = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (job.status === "complete") return job;
      if (job.status === "error") throw new Error(job.error || "PDF indexing failed.");
      $("#uploadProgressText").textContent = labels[job.phase] || "Uploading PDF…";
      await wait(700);
    }
  }

  async function addPdf(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf") || (file.type && file.type !== "application/pdf")) { toast("Choose a PDF file.", "error"); return; }
    if (file.size > 25 * 1024 * 1024) { toast("PDF uploads are limited to 25 MB.", "error"); return; }
    setView("documents");
    $("#uploadProgress").hidden = false; $("#uploadProgressText").textContent = "Reading PDF upload…"; fileInput.disabled = true;
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = ""; const block = 0x8000;
      for (let start = 0; start < bytes.length; start += block) binary += String.fromCharCode(...bytes.subarray(start, start + block));
      const { job_id: jobId } = await request("/api/upload", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: file.name, data: btoa(binary) }) });
      await loadDocuments(); await pollUpload(jobId); await loadDocuments();
      toast("PDF indexed and ready to explore.", "success");
    } catch (error) { toast(error.message, "error"); }
    finally { $("#uploadProgress").hidden = true; fileInput.disabled = false; fileInput.value = ""; await loadDocuments().catch(() => {}); }
  }

  async function removeDocument(name, button) {
    if (!confirm(`Remove “${name}” and its indexed text and figures?`)) return;
    const old = button.textContent; button.disabled = true; button.textContent = "Removing…";
    try {
      await request("/api/remove", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
      if (currentDoc?.name === name) { currentDoc = null; cards = []; links = []; renderCards(); renderCurrentDocument(); }
      await loadDocuments(); toast("PDF and its indexed evidence were removed.", "success");
    } catch (error) { button.disabled = false; button.textContent = old; toast(error.message, "error"); }
  }

  function renderMap() {
    const root = $("#mapContent"); root.className = "secondary-paper map-canvas";
    if (!links.length) { root.innerHTML = '<div class="map-empty"><span>⌘</span><h3>Your map begins with a connection</h3><p>Create a connection between two evidence cards in the workspace. This map only shows links you explicitly make.</p></div>'; return; }
    root.innerHTML = links.map(([a, b], index) => {
      const left = cards.find((card) => card.key === a), right = cards.find((card) => card.key === b);
      return `<button class="map-edge" data-focus-card="${escapeHtml(a)}" type="button"><strong>${escapeHtml(left?.source?.document_name || left?.content?.slice(0, 40) || "Evidence")}</strong><span>↔</span><strong>${escapeHtml(right?.source?.document_name || right?.content?.slice(0, 40) || "Evidence")}</strong><small>Connection ${index + 1}</small></button>`;
    }).join("");
  }

  function getCollections() { try { return JSON.parse(localStorage.getItem("doculens.collections") || "[]"); } catch { return []; } }
  function saveCollections(data) { localStorage.setItem("doculens.collections", JSON.stringify(data)); renderCollections(); }
  function renderCollections() {
    const root = $("#collectionCards"), collections = getCollections(); root.replaceChildren();
    if (!collections.length) { root.innerHTML = '<div class="secondary-paper"><h3>No collections yet</h3><p>Create a collection, then use “Add to collection” on an open document.</p></div>'; return; }
    collections.forEach((collection, index) => {
      const card = document.createElement("article"); card.className = "collection-card";
      card.innerHTML = `<h3>${escapeHtml(collection.name)}</h3><p>${collection.documents.length} document${collection.documents.length === 1 ? "" : "s"}</p>${collection.documents.length ? `<ul>${collection.documents.map((name) => `<li>${escapeHtml(name)}</li>`).join("")}</ul>` : "<p>Add a document from the workspace header.</p>"}<button type="button" data-delete-collection="${index}">Delete collection</button>`;
      root.append(card);
    });
  }

  function loadNote() {
    const key = currentDoc ? `doculens.note.${currentDoc.document_id}` : "doculens.note.empty";
    $("#documentNote").value = localStorage.getItem(key) || "";
    $("#noteDocumentName").textContent = currentDoc?.name || "No document selected";
  }

  function renderResearchPrompts() {
    const prompts = [["Summarize", "Summarize the main argument and conclusions in the indexed PDFs."], ["Compare", "Compare the approaches and findings in the indexed PDFs."], ["Find a figure", "Find figures that illustrate the main concepts in the indexed PDFs."], ["Trace a claim", "Identify evidence for the main claim and cite the relevant pages."], ["Prepare a report", "Create a concise research report with source citations."]];
    $("#researchPrompts").innerHTML = prompts.map(([title, text]) => `<button class="research-prompt" data-research-prompt="${escapeHtml(text)}" type="button"><strong>${escapeHtml(title)}</strong><small>${escapeHtml(text)}</small></button>`).join("");
  }

  function applyTab(tab) {
    activeTab = tab;
    $$(".workspace-tab").forEach((button) => button.classList.toggle("is-active", button.dataset.tab === tab));
    const reader = $("#readerPanel"), canvasPanel = $("#evidenceCanvas"), grid = $("#researchGrid");
    grid.classList.remove("insights-open", "insights-hidden");
    if (tab === "pages") { reader.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "nearest" }); }
    if (tab === "figures") { $("#figureGallery").scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "center" }); }
    if (tab === "insights") grid.classList.add("insights-open");
    if (tab === "tables") toast("Table extraction is not available in this backend yet.");
    if (tab === "overview") canvasPanel.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "nearest" });
  }

  document.addEventListener("click", async (event) => {
    const target = event.target;
    const view = target.closest("[data-view-link]");
    if (view) { event.preventDefault(); setView(view.dataset.viewLink); return; }
    const openDoc = target.closest("[data-open-document]");
    if (openDoc) { await selectDocument(docs.find((doc) => doc.name === openDoc.dataset.openDocument)); return; }
    const removeDoc = target.closest("[data-remove-document]");
    if (removeDoc) { await removeDocument(removeDoc.dataset.removeDocument, removeDoc); return; }
    const thumb = target.closest(".page-thumb");
    if (thumb) { currentPage = Number(thumb.dataset.page); await renderPage(); return; }
    const tab = target.closest("[data-tab]");
    if (tab) { applyTab(tab.dataset.tab); return; }
    const prompt = target.closest("[data-prompt]");
    if (prompt) { $("#question").value = prompt.dataset.prompt; ask(prompt.dataset.prompt); return; }
    const researchPrompt = target.closest("[data-research-prompt]");
    if (researchPrompt) { setView("workspace"); $("#question").value = researchPrompt.dataset.researchPrompt; ask(researchPrompt.dataset.researchPrompt); return; }
    const addFigure = target.closest("[data-add-figure]");
    if (addFigure) { const figure = $("#figureItems")._figures?.[Number(addFigure.dataset.addFigure)]; if (figure) addFigureCard(figure); return; }
    const openFigure = target.closest("[data-open-figure]");
    if (openFigure) { const figure = $("#figureItems")._figures?.[Number(openFigure.dataset.openFigure)]; if (figure) showFigure(`/api/image?path=${encodeURIComponent(figure.path)}`, `${currentDoc?.name || "PDF"} · Page ${figure.page ?? "—"}`); return; }
    const cardRemove = target.closest("[data-remove-card]");
    if (cardRemove) { cards = cards.filter((card) => card.key !== cardRemove.dataset.removeCard); links = links.filter((link) => !link.includes(cardRemove.dataset.removeCard)); renderCards(); return; }
    const cardJump = target.closest("[data-jump-card]");
    if (cardJump) { await jumpToCard(cardJump.dataset.jumpCard); return; }
    const sourceJump = target.closest("[data-jump-source]");
    if (sourceJump) { const card = cards.find((item) => item.source?.source_id === sourceJump.dataset.jumpSource); if (card) await jumpToCard(card.key); return; }
    const focusCard = target.closest("[data-focus-card]");
    if (focusCard) { setView("workspace"); const card = $(`[data-card-key="${CSS.escape(focusCard.dataset.focusCard)}"]`); card?.scrollIntoView({ block: "center", behavior: reducedMotion.matches ? "auto" : "smooth" }); card?.classList.add("is-selected"); return; }
    const card = target.closest(".evidence-card");
    if (card && !target.closest("button")) { const data = cards.find((item) => item.key === card.dataset.cardKey); if (data) selectCard(data); return; }
    const insightTab = target.closest("[data-insight]");
    if (insightTab) { $$("[data-insight]").forEach((button) => button.classList.toggle("is-active", button === insightTab)); if (insightTab.dataset.insight === "connections") updateConnectionsPanel(); else if (insightTab.dataset.insight === "questions") { $("#insightContent").innerHTML = '<h3 class="insight-subhead">Questions to explore</h3>' + ["What evidence supports the central claim?", "Which figures illustrate the main method?", "How do the documents differ?"] .map((text) => `<button class="insight-source-link" data-prompt="${escapeHtml(text)}">${escapeHtml(text)}</button>`).join(""); } else if (!$("#insightContent .insight-summary")) $("#insightContent").innerHTML = '<div class="insight-placeholder"><span>✧</span><h3>Research notes</h3><p>Ask DocuLens a question to bring source-backed insights into this panel.</p></div>'; return; }
    const deleteCollection = target.closest("[data-delete-collection]");
    if (deleteCollection) { const collections = getCollections(); collections.splice(Number(deleteCollection.dataset.deleteCollection), 1); saveCollections(collections); }
  });

  cardLayer.addEventListener("pointerdown", (event) => {
    const node = event.target.closest(".evidence-card");
    if (!node || event.target.closest("button")) return;
    const card = cards.find((item) => item.key === node.dataset.cardKey); if (!card) return;
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault(); node.setPointerCapture(event.pointerId); node.classList.add("is-dragging");
    node._drag = { x: event.clientX, y: event.clientY, left: card.x, top: card.y };
  });
  cardLayer.addEventListener("pointermove", (event) => {
    const node = event.target.closest(".evidence-card"); if (!node?._drag) return;
    const card = cards.find((item) => item.key === node.dataset.cardKey); if (!card) return;
    card.x = Math.max(4, node._drag.left + event.clientX - node._drag.x); card.y = Math.max(5, node._drag.top + event.clientY - node._drag.y);
    node.style.left = `${card.x}px`; node.style.top = `${card.y}px`; drawLinks();
  });
  cardLayer.addEventListener("pointerup", (event) => { const node = event.target.closest(".evidence-card"); if (node) { node._drag = null; node.classList.remove("is-dragging"); } });
  canvas.addEventListener("scroll", drawLinks);
  window.addEventListener("resize", drawLinks);

  $("#askForm").addEventListener("submit", (event) => { event.preventDefault(); ask($("#question").value); });
  $("#copyAnswer").addEventListener("click", async () => {
    if (!latestAnswer) return;
    try { await navigator.clipboard.writeText(latestAnswer); toast("Final answer copied.", "success"); }
    catch { toast("Clipboard access is unavailable in this browser.", "error"); }
  });
  $("#globalSearchForm").addEventListener("submit", (event) => { event.preventDefault(); const q = $("#globalSearch").value.trim(); if (q) { setView("workspace"); $("#question").value = q; ask(q); } });
  $("#librarySearch").addEventListener("input", renderDocumentCards);
  fileInput.addEventListener("change", () => addPdf(fileInput.files[0]));
  $("#previousPage").addEventListener("click", () => { currentPage = Math.max(1, currentPage - 1); renderPage(); });
  $("#nextPage").addEventListener("click", () => { currentPage = Math.min(currentDoc?.pages || 1, currentPage + 1); renderPage(); });
  $("#pageNumber").addEventListener("change", () => { currentPage = Number($("#pageNumber").value); renderPage(); });
  $("#zoomIn").addEventListener("click", () => { zoom = Math.min(200, zoom + 10); renderPage(); });
  $("#zoomOut").addEventListener("click", () => { zoom = Math.max(50, zoom - 10); renderPage(); });
  $("#readerFullscreen").addEventListener("click", () => { const frame = $("#pdfFrameWrap"); if (frame.requestFullscreen) frame.requestFullscreen(); });
  $("#connectMode").addEventListener("click", () => setConnectMode(!connectMode));
  $("#fitCanvas").addEventListener("click", () => { cards.forEach((card, index) => { card.x = 13 + (index % 2) * 245; card.y = 16 + Math.floor(index / 2) * 155; card.selected = false; }); setConnectMode(false); renderCards(); canvas.scrollTo({ top: 0, left: 0, behavior: reducedMotion.matches ? "auto" : "smooth" }); });
  $("#closeInsights").addEventListener("click", () => $("#researchGrid").classList.toggle("insights-hidden"));
  $("#addToCollection").addEventListener("click", () => {
    if (!currentDoc) { toast("Open a PDF before adding it to a collection.", "error"); return; }
    const collections = getCollections();
    if (!collections.length) { toast("Create a collection first in the Collections section."); setView("collections"); return; }
    const choice = prompt(`Add ${currentDoc.name} to which collection?\n${collections.map((item, index) => `${index + 1}. ${item.name}`).join("\n")}`, "1");
    const selected = collections[Number(choice) - 1];
    if (selected && !selected.documents.includes(currentDoc.name)) { selected.documents.push(currentDoc.name); saveCollections(collections); toast(`Added to ${selected.name}.`, "success"); }
  });
  $("#newCollection").addEventListener("click", () => { const name = prompt("Name this collection:"); if (!name?.trim()) return; const collections = getCollections(); collections.push({ name: name.trim(), documents: [] }); saveCollections(collections); });
  $("#documentNote").addEventListener("input", () => { const key = currentDoc ? `doculens.note.${currentDoc.document_id}` : "doculens.note.empty"; localStorage.setItem(key, $("#documentNote").value); $("#noteSaveStatus").textContent = "Saved in this browser"; });
  $("#themeToggle").addEventListener("click", () => { document.body.classList.toggle("night-theme"); localStorage.setItem("doculens.theme", document.body.classList.contains("night-theme") ? "night" : "day"); });
  $("#headerThemeToggle").addEventListener("click", () => $("#themeToggle").click());
  $("#railToggle").addEventListener("click", () => app.classList.toggle("rail-collapsed"));
  $("#mobileMenu").addEventListener("click", () => { app.classList.add("mobile-nav-open"); $("#mobileBackdrop").hidden = false; });
  $("#mobileBackdrop").addEventListener("click", () => { app.classList.remove("mobile-nav-open"); $("#mobileBackdrop").hidden = true; });
  $("#figureDialog").addEventListener("click", (event) => { if (event.target === figureDialog) figureDialog.close(); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") { app.classList.remove("mobile-nav-open"); $("#mobileBackdrop").hidden = true; } });

  if (localStorage.getItem("doculens.theme") === "night") document.body.classList.add("night-theme");
  renderResearchPrompts();
  renderCards();
  request("/api/status").then((status) => { $("#keyStatus").textContent = status.key_configured ? "Research key ready" : "Add API key to enable answers"; }).catch(() => { $("#keyStatus").textContent = "Local server unavailable"; });
  loadDocuments().catch((error) => { toast(`Could not load your document library: ${error.message}`, "error"); $("#libraryCards").innerHTML = '<div class="secondary-paper"><h3>Library unavailable</h3><p>Check that the local DocuLens server is running.</p></div>'; });
})();
