import { api } from "../../../packages/api-client/src/index.ts";
import { errorCard, loading } from "../../../packages/ui/src/index.ts";
import { routeFromLocation, linkHandler } from "./router.ts";
import { renderOverview } from "./features/overview.ts";
import { renderNetwork } from "./features/network.ts";
import { renderCriticality } from "./features/criticality.ts";
import { renderSimilarity } from "./features/similarity.ts";
import { renderEmbeddings, renderEvaluation } from "./features/analysis.ts";

const view = document.querySelector<HTMLElement>("#view");
const routeTitle = document.querySelector<HTMLElement>("#route-title");
const routeKicker = document.querySelector<HTMLElement>("#route-kicker");
const status = document.querySelector<HTMLElement>("#status");
const RENDERERS = { overview: renderOverview, network: renderNetwork, criticality: renderCriticality, similarity: renderSimilarity, embeddings: renderEmbeddings, evaluation: renderEvaluation };
const META = {
  overview: ["Published workspace", "Overview"],
  network: ["Published network", "Network"],
  criticality: ["Published model output", "Criticality"],
  similarity: ["Published representation", "Similarity"],
  embeddings: ["Published representation", "Embeddings"],
  evaluation: ["Published quality", "Evaluation"]
};

document.addEventListener("click", linkHandler);
window.addEventListener("popstate", renderRoute);

async function renderRoute() {
  const route = routeFromLocation();
  const [kicker, title] = META[route];
  routeKicker.textContent = kicker;
  routeTitle.textContent = title;
  document.querySelectorAll<HTMLAnchorElement>("a[data-route]").forEach((link) => link.classList.toggle("active", link.dataset.route === route));
  view.innerHTML = loading("Loading published results…");
  try {
    await RENDERERS[route](view);
    status.textContent = "Published data only";
  } catch (error) {
    view.innerHTML = errorCard(error);
    status.textContent = "Publication unavailable";
  }
}

try {
  await api("/api/public/catalog");
  await renderRoute();
} catch {
  status.textContent = "Control API unavailable";
  view.innerHTML = errorCard(new Error("The public publication service is unavailable."));
}
