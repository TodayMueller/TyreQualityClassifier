let currentPage = 1, totalPages = 1;
const perPage = 100;
let currentStatus = 'all';

function setActiveFilter(btn) {
  document.querySelectorAll('.filter-button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentStatus = btn.dataset.status;
  currentPage = 1;
  loadPage(1);
}

async function loadPage(page) {
  document.getElementById('history-spinner').classList.remove('hidden');
  const { history, total } = await api.fetchHistoryMeta(page, perPage, currentStatus);
  totalPages = Math.ceil(total / perPage) || 1;
  renderPage(history);
  updatePagination(page);
  document.getElementById('history-spinner').classList.add('hidden');
  window.scrollTo(0, 0);
}

function renderPage(items) {
  const container = document.getElementById('history-body');
  container.innerHTML = '';
  modalItems = [];

  items.forEach(r => {
    const card = document.createElement('div');
    card.className = 'history-item';
    card.dataset.modalIndex = modalItems.length;

    const img = document.createElement('img');
    img.className = 'history-thumb';
    img.src = `/api/history/${r.id}/image`;
    img.loading = 'lazy';

    const content = document.createElement('div');
    content.className = 'history-content';

    const filename = document.createElement('div');
    filename.className = 'history-filename';
    filename.textContent = r.filename;

    const time = document.createElement('div');
    time.className = 'history-time';
    time.textContent = new Date(r.timestamp).toLocaleString();

    const verdict = document.createElement('div');
    verdict.className = 'history-verdict';
    verdict.dataset.verdict = r.verdict;
    verdict.textContent = getVerdictText(r.verdict);

    content.append(filename, time, verdict);
    card.append(img, content);
    card.addEventListener('click', () => showModalItem(+card.dataset.modalIndex));
    container.appendChild(card);

    modalItems.push({
      imgSrc: `/api/history/${r.id}/image`,
      filename: r.filename,
      verdict: r.verdict,
      time: r.timestamp
    });
  });
}

function getVerdictText(verdict) {
  const texts = {
    good: 'Пригодное',
    defective: 'Непригодное',
    external: 'Сторонний объект',
    error: 'Отклонено'
  };
  return texts[verdict] || verdict;
}

function updatePagination(page) {
  currentPage = page;
  const prev = document.getElementById('prev-page');
  const next = document.getElementById('next-page');
  const pagination = document.querySelector('.pagination');

  prev.disabled = page <= 1;
  next.disabled = page >= totalPages;
  document.getElementById('page-info').textContent = `Страница ${page} из ${totalPages}`;

  if (totalPages > 1) {
    pagination.style.display = 'flex';
  } else {
    pagination.style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.filter-button').forEach(btn =>
    btn.addEventListener('click', () => setActiveFilter(btn))
  );
  document.getElementById('prev-page').addEventListener('click', () => {
    if (currentPage > 1) loadPage(currentPage - 1);
  });
  document.getElementById('next-page').addEventListener('click', () => {
    if (currentPage < totalPages) loadPage(currentPage + 1);
  });
  document.getElementById('btn-clear-history').addEventListener('click', async () => {
    if (confirm('Вы уверены, что хотите удалить всю историю?')) {
      await fetch('/api/history', { method: 'DELETE' });
      loadPage(1);
    }
  });
  loadPage(1);
});
