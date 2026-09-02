import { api } from "./api.ts";
import { linkHandler, routeFromLocation } from "./routes/router.ts";
import { renderOverview } from "./features/overview/index.ts";
import { renderData } from "./features/data/index.ts";
import { renderRuns } from "./features/runs/index.ts";
import { renderNetwork } from "./features/network/index.ts";
import { renderCriticality } from "./features/criticality/index.ts";
import { renderSimilarity } from "./features/similarity/index.ts";
import { renderEmbeddings } from "./features/embeddings/index.ts";
import { renderEvaluation } from "./features/evaluation/index.ts";

const view = document.querySelector<HTMLElement>("#view");
const routeTitle = document.querySelector<HTMLElement>("#route-title");
const routeKicker = document.querySelector<HTMLElement>("#route-kicker");
const sidebarStatus = document.querySelector<HTMLElement>("#sidebar-status");
const toast = document.querySelector<HTMLElement>("#toast");

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
document.querySelector<HTMLButtonElement>("#refresh-button")?.addEventListener("click", async (event) => {
  const button = event.currentTarget as HTMLButtonElement;
  button.disabled = true;
  try {
    await api("/api/inventory/refresh", { method: "POST" });
    await renderRoute();
    showToast("Inventory refreshed from the configured data directory.");
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; }
});

async function renderRoute() {
  const route = routeFromLocation();
  const [kicker, title] = ROUTE_META[route];
  routeKicker.textContent = kicker;
  routeTitle.textContent = title;
  document.querySelectorAll<HTMLAnchorElement>("a[data-route]").forEach((link) => link.classList.toggle("active", link.dataset.route === route));
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
