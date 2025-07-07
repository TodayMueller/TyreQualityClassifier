async function fetchJSON(url, opts = {}) {
  const resp = await fetch(url, opts);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}

async function fetchLatestBatch(limit = 60) {
  document.getElementById('main-spinner').classList.remove('hidden');

  const resultsList     = document.getElementById("results-list");
  const resultContainer = document.getElementById("result-container");
  resultsList.innerHTML = "";
  modalItems = [];

  let data;
  try {
    data = await fetchJSON(`/api/history?meta=true&limit=${limit}`);
  } catch (e) {
    console.error("fetchLatestBatch:", e);
    document.getElementById('main-spinner').classList.add('hidden');
    return;
  }
  const { history } = data;
  if (!history.length) {
    document.getElementById('main-spinner').classList.add('hidden');
    return;
  }

  resultContainer.classList.remove("hidden");
  for (const r of history) {
    createResult({
      id:        r.id,
      filename:  r.filename,
      verdict:   r.verdict,
      timestamp: r.timestamp,
      imageUrl:  `/api/history/${r.id}/image`
    });
  }

  document.getElementById('main-spinner').classList.add('hidden');
}

async function fetchHistoryMeta(page = 1, perPage = 100, status = 'all') {
  try {
    return await fetchJSON(
      `/api/history?meta=true&page=${page}&per_page=${perPage}&status=${status}`
    );
  } catch (e) {
    console.error("fetchHistoryMeta:", e);
    return { history: [], total: 0 };
  }
}

async function classifyFiles(files) {
  if (!files.length) return;
  const dropZone = document.getElementById("drop-zone");
  dropZone.classList.add("spinner-active");
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    try {
      await fetch("/api/classify", { method: "POST", body: fd });
    } catch (err) {
      console.error("classifyFiles:", err);
    }
  }
  dropZone.classList.remove("spinner-active");
  await fetchLatestBatch(60);
}

async function getFilesFromDataTransfer(items) {
  const out = [];
  for (const it of items) {
    const ent = it.webkitGetAsEntry?.();
    if (ent?.isDirectory) {
      out.push(...await readDirectory(ent));
    } else if (it.getAsFile) {
      out.push(it.getAsFile());
    }
  }
  return out;
}

async function readDirectory(dir) {
  const rdr = dir.createReader(), arr = [];
  let ents;
  while ((ents = await new Promise(r => rdr.readEntries(r)))?.length) {
    for (const ent of ents) {
      if (ent.isDirectory) {
        arr.push(...await readDirectory(ent));
      } else {
        arr.push(await new Promise(r => ent.file(r)));
      }
    }
  }
  return arr;
}

window.api = {
  fetchJSON,
  fetchLatestBatch,
  fetchHistoryMeta,        
  classifyFiles,
  getFilesFromDataTransfer,
  readDirectory
};
