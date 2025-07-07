const style = document.createElement("style");
style.textContent = `
  .hidden { display: none !important; }
  .flatpickr-calendar {
    position: fixed !important;
    bottom: 10px !important;
    right: 10px !important;
    margin: 0 !important;
  }
`;
document.head.appendChild(style);

let currentRange = "all";
const btns = document.querySelectorAll(".filter-button");
const filtersEl = document.querySelector(".stats-filters");
const pickerEl = document.getElementById("date-range-picker");
const spinner = document.getElementById("stats-spinner");
const summary = document.querySelector(".stats-summary");
const periodEl = document.getElementById("stats-period");
const totalEl = document.getElementById("stats-total");
const dataEls = {
  good: { cnt: "count-good", pct: "percent-good" },
  defective: { cnt: "count-defective", pct: "percent-defective" },
  external: { cnt: "count-external", pct: "percent-external" },
  error: { cnt: "count-error", pct: "percent-error" },
};
const pieCard = document.querySelector("#stats-chart").closest(".chart-card");
const histCard = document
  .querySelector("#histogram-chart")
  .closest(".chart-card");
const expActions = document.querySelector(".export-actions-bottom");
const msgEl = document.getElementById("no-data-message");
const metricsEl = document.getElementById("metrics-container");
let pieChart, histChart;

function hide(el) { el && el.classList.add("hidden"); }
function show(el) { el && el.classList.remove("hidden"); }
function destroyCharts() {
  if (pieChart) { pieChart.destroy(); pieChart = null; }
  if (histChart) { histChart.destroy(); histChart = null; }
}

const picker = flatpickr(pickerEl, {
  locale: flatpickr.l10ns.ru, 
  mode: "range",
  dateFormat: "Y-m-d",
  clickOpens: false,
  appendTo: document.body,
  onChange(selectedDates, _, instance) {
    if (selectedDates.length === 2) {
      instance.close();
      load("custom");
    }
  },
});


const ranges = {
  all: () => ({}),
  today: () => {
    const d = new Date().toISOString().slice(0, 10);
    return { from: d, to: d };
  },
  week: () => {
    let e = new Date(), s = new Date(e);
    s.setDate(e.getDate() - 6);
    return { from: s.toISOString().slice(0, 10), to: e.toISOString().slice(0, 10) };
  },
  month: () => {
    let e = new Date(), s = new Date(e);
    s.setDate(e.getDate() - 29);
    return { from: s.toISOString().slice(0, 10), to: e.toISOString().slice(0, 10) };
  },
  custom: () => {
    if (picker.selectedDates.length === 2) {
      const [s, e] = picker.selectedDates;
      const pad = n => String(n).padStart(2, "0");
      return {
        from: `${s.getFullYear()}-${pad(s.getMonth() + 1)}-${pad(s.getDate())}`,
        to: `${e.getFullYear()}-${pad(e.getMonth() + 1)}-${pad(e.getDate())}`,
      };
    }
    return {};
  },
};

function updateSummary(data) {
  periodEl.textContent = data.period;
  totalEl.textContent = data.total;
  for (let k in dataEls) {
    document.getElementById(dataEls[k].cnt).textContent = data.counts[k];
    document.getElementById(dataEls[k].pct).textContent =
      data.percents[k].toFixed(1) + "%";
  }
}

function renderPie(data) {
  const ctx = document.getElementById("stats-chart").getContext("2d");
  pieChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Пригодные", "Непригодные", "Сторонние", "Отклонённые"],
      datasets: [{
        data: ["good", "defective", "external", "error"].map(k => data.counts[k]),
        backgroundColor: ["#2e7d32", "#2852c6", "#b910bc", "#ef6c00"]
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", align: "start" } },
      animation: {
        duration: 900,
        easing: "easeOutQuart",
        animateRotate: true,
        animateScale: true
      }
    }
  });
}

function renderHist(data, singleDay) {
  const ctx = document.getElementById("histogram-chart").getContext("2d");
  let labels, datasets;
  if (singleDay) {
    const buckets = Array.from({ length: 24 }, (_, h) => ({ h, good:0, defective:0, external:0, error:0 }));
    data.records.forEach(r => { buckets[new Date(r.ts).getHours()][r.verdict]++; });
    labels = buckets.map(b => String(b.h).padStart(2, "0") + ":00");
    datasets = ["good","defective","external","error"].map((k,i) => ({
      label: ["Пригодные","Непригодные","Сторонние","Отклонённые"][i],
      data: buckets.map(b => b[k]),
      backgroundColor: ["#2e7d32","#2852c6","#b910bc","#ef6c00"][i]
    }));
  } else {
    const buckets = {};
    data.records.forEach(r => {
      const d = r.ts.slice(0, 10);
      buckets[d] = buckets[d] || { good:0, defective:0, external:0, error:0 };
      buckets[d][r.verdict]++;
    });
    if (currentRange === "all") {
      labels = Object.keys(buckets);
    } else {
      const [start, end] = data.period.split(" – ");
      labels = []; let cur = new Date(start), last = new Date(end);
      while (cur <= last) {
        labels.push(cur.toISOString().slice(0, 10));
        cur.setDate(cur.getDate() + 1);
      }
    }
    datasets = ["good","defective","external","error"].map((k,i) => ({
      label: ["Пригодные","Непригодные","Сторонние","Отклонённые"][i],
      data: labels.map(d => buckets[d]?.[k] || 0),
      backgroundColor: ["#2e7d32","#2852c6","#b910bc","#ef6c00"][i]
    }));
  }
  histChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 300,
        easing: "easeOutQuad",
        delay: ctx => ctx.dataIndex * 20
      },
      scales: {
        x: { stacked: true, ticks: { autoSkip: !singleDay, maxTicksLimit: singleDay ? 24 : 12 } },
        y: { stacked: true, beginAtZero: true, grace: "5%" }
      },
      plugins: { legend: { position: "bottom", align: "start" } }
    }
  });
}

function load(range) {
  currentRange = range;
  hide(msgEl);
  destroyCharts();
  show(filtersEl);
  hide(summary);
  hide(pieCard);
  hide(histCard);
  hide(metricsEl);
  hide(expActions);
  hide(pickerEl);
  show(spinner);

  const opts = ranges[range]();
  if (range === "custom" && !opts.from) {
    show(pickerEl);
    picker.open();
    hide(spinner);
    return;
  }

  fetch(`/api/stats?${new URLSearchParams(opts)}`)
    .then((r) => r.json())
    .then((d) => {
      hide(spinner);

      if (d.total === 0) {
        btns.forEach((b) =>
          b.classList.toggle("active", b.dataset.range === range)
        );
        show(filtersEl);
        hide(summary);
        hide(pieCard);
        hide(histCard);
        hide(metricsEl);
        hide(expActions);
        show(msgEl);
        msgEl.textContent =
          "Невозможно собрать статистику, так как за выбранный период времени запросов не было";
        return;
      }

      if (range === "all") {
        const dates = d.records.map((r) => r.ts.slice(0, 10));
        const first = dates[0];
        const last = dates[dates.length - 1];
        d.period = first === last ? first : `${first} – ${last}`;
      } else {
        if (opts.from && opts.to && opts.from === opts.to) {
          d.period = opts.from;
        } else {
          d.period = `${opts.from} – ${opts.to}`;
        }
      }

      btns.forEach((b) =>
        b.classList.toggle("active", b.dataset.range === range)
      );
      updateSummary(d);
      show(summary);

      show(pieCard);
      renderPie(d);

      let singleDay = false;
      if (range === "all") {
        const dates = d.records.map((r) => r.ts.slice(0, 10));
        singleDay = dates.length > 0 && dates[0] === dates[dates.length - 1];
      } else {
        singleDay = !!(opts.from && opts.to && opts.from === opts.to);
      }

      show(histCard);
      renderHist(d, singleDay);

      metricsEl.innerHTML = `
        <ul class="metrics-list">
          <li>Среднее количество запросов в день: <strong>${d.metrics.avg_per_day.toFixed(
            2
          )}</strong></li>
          <li>Мин. время обработки запроса: <strong>${(
            d.metrics.min_time * 1000
          ).toFixed(0)} мс</strong></li>
          <li>Ср. время обработки запроса: <strong>${(
            d.metrics.avg_time * 1000
          ).toFixed(0)} мс</strong></li>
          <li>Макс. время обработки запроса: <strong>${(
            d.metrics.max_time * 1000
          ).toFixed(0)} мс</strong></li>
        </ul>`;
      show(metricsEl);
      show(expActions);
    })
    .catch((e) => {
      hide(spinner);
      btns.forEach((b) => b.classList.remove("active"));
      show(filtersEl);
      hide(summary);
      hide(pieCard);
      hide(histCard);
      hide(metricsEl);
      hide(expActions);
      show(msgEl);
      msgEl.textContent = "Ошибка при загрузке статистики";
    });
}


btns.forEach(b => {
  if (b.dataset.range === "custom") {
    b.addEventListener("click", () => {
      currentRange = "custom";
      btns.forEach(x => x.classList.toggle("active", x === b));
      show(pickerEl);
      picker.open();
    });
  } else {
    b.addEventListener("click", () => load(b.dataset.range));
  }
});

load("all");

const exportBtn = document.getElementById('export-excel');
if (exportBtn) {
  exportBtn.addEventListener('click', () => {
    const opts = ranges[currentRange]();
    const params = new URLSearchParams(opts);
    window.location.href = `/api/stats/export.xlsx?${params.toString()}`;
  });
}
