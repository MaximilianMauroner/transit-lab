import { api } from "./api.js";
import { linkHandler, routeFromLocation } from "./routes/router.js";
import { renderOverview } from "./features/overview/index.js";
import { renderData } from "./features/data/index.js";
import { renderRuns } from "./features/runs/index.js";
import { renderNetwork } from "./features/network/index.js";
import { renderCriticality } from "./features/criticality/index.js";
import { renderSimilarity } from "./features/similarity/index.js";
import { renderEmbeddings } from "./features/embeddings/index.js";
import { renderEvaluation } from "./features/evaluation/index.js";

const view = document.querySelector("#view");
const routeTitle = document.querySelector("#route-title");
const routeKicker = document.querySelector("#route-kicker");
const sidebarStatus = document.querySelector("#sidebar-status");
const toast = document.querySelector("#toast");

const ROUTE_META = {
  overview: ["Control plane", "Overview"],
  data: ["Repository inventory", "Data & lineage"],
  runs: ["Operational control", "Runs"],
  network: ["Spatial inspection", "Network"],
  criticality: ["Model output", "Criticality"],
  similarity: ["Representation search", "Similarity"],
  embeddings: ["Representation space", "Embeddings"],
  evaluation: ["Quality signals", "Evaluation"]
};

const RENDERERS = { overview: renderOverview, data: renderData, runs: renderRuns, network: renderNetwork, criticality: renderCriticality, similarity: renderSimilarity, embeddings: renderEmbeddings, evaluation: renderEvaluation };

document.addEventListener("click", linkHandler);
window.addEventListener("popstate", renderRoute);
document.querySelector("#refresh-button")?.addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  try {
    await api("/api/inventory/refresh", { method: "POST" });
    await renderRoute();
    showToast("Inventory refreshed from the configured data directory.");
  } catch (error) { showToast(error.message, true); }
  finally { event.currentTarget.disabled = false; }
});

async function renderRoute() {
  const route = routeFromLocation();
  const [kicker, title] = ROUTE_META[route];
  routeKicker.textContent = kicker;
  routeTitle.textContent = title;
  document.querySelectorAll("a[data-route]").forEach((link) => link.classList.toggle("active", link.dataset.route === route));
  await RENDERERS[route](view);
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("danger", isError);
  toast.hidden = false;
  setTimeout(() => { toast.hidden = true; }, 3500);
}

async function checkHealth() {
  try {
    await api("/api/health");
    sidebarStatus.textContent = "API connected";
  } catch {
    sidebarStatus.textContent = "API unavailable";
  }
}

await checkHealth();
await renderRoute();
