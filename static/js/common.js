const modal = document.getElementById("image-modal")
const modalImg = document.getElementById("image-modal-img")
const modalFilename = document.getElementById("image-modal-filename")
const modalVerdict = document.getElementById("image-modal-verdict")
const modalTime = document.getElementById("image-modal-time")
const modalClose = document.getElementById("image-modal-close")
const modalNext = document.getElementById("image-modal-next")
const modalPrev = document.getElementById("image-modal-prev")
let modalItems = []
let currentIndex = -1

function createResult(data) {
  const div = document.createElement("div")
  div.className = "result-item"
  div.dataset.status = data.verdict
  div.dataset.modalIndex = modalItems.length

  const img = document.createElement("img")
  img.className = "result-thumbnail"
  img.src = data.imageUrl
  img.loading = "lazy"

  const content = document.createElement("div")
  content.className = "result-content"

  const nameDiv = document.createElement("div")
  nameDiv.textContent = data.filename

  const timeDiv = document.createElement("div")
  timeDiv.textContent = new Date(data.timestamp).toLocaleString()

  const catDiv = document.createElement("div")
  catDiv.textContent = getVerdictText(data.verdict)
  catDiv.classList.add("category-label", `status-${data.verdict}`)

  content.append(nameDiv, timeDiv, catDiv)
  div.append(img, content)

  div.addEventListener("click", () => showModalItem(+div.dataset.modalIndex))
  document.getElementById("results-list").appendChild(div)

  modalItems.push({
    imgSrc: data.imageUrl,
    filename: data.filename,
    verdict: data.verdict,
    time: data.timestamp
  })
}

function getVerdictText(verdict) {
  const texts = {
    good: 'Пригодное',
    defective: 'Непригодное',
    external: 'Сторонний объект',
    error: 'Отклонено'
  }
  return texts[verdict] || verdict
}

function getVisibleIndices() {
  return Array.from(document.querySelectorAll(".result-item, .history-item"))
    .filter(el => el.style.display !== "none")
    .map(el => +el.dataset.modalIndex)
}

function showModalItem(idx) {
  const item = modalItems[idx]
  modalImg.src = item.imgSrc
  modalFilename.textContent = item.filename
  modalVerdict.textContent = getVerdictText(item.verdict)
  modalVerdict.className = ""
  modalVerdict.classList.add("category-label", `status-${item.verdict}`)
  modalTime.textContent = new Date(item.time).toLocaleString()
  modal.classList.add("active")
  currentIndex = idx

  const vis = getVisibleIndices()
  const pos = vis.indexOf(idx)
  modalPrev.classList.toggle("disabled", pos <= 0)
  modalNext.classList.toggle("disabled", pos >= vis.length - 1)
}

function navigate(delta) {
  const vis = getVisibleIndices()
  const pos = vis.indexOf(currentIndex)
  const next = vis[pos + delta]
  if (typeof next === "number") showModalItem(next)
}

modalClose.addEventListener("click", () => modal.classList.remove("active"))
modal.addEventListener("click", e => { if (e.target === modal) modal.classList.remove("active") })
modalPrev.addEventListener("click", () => navigate(-1))
modalNext.addEventListener("click", () => navigate(1))
document.addEventListener("keydown", e => {
  if (!modal.classList.contains("active")) return
  if (e.key === "Escape")      modal.classList.remove("active")
  if (e.key === "ArrowLeft")   navigate(-1)
  if (e.key === "ArrowRight")  navigate(1)
})
