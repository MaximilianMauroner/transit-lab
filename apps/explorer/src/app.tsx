import { useEffect, useState, type ComponentType, type MouseEvent } from "react";
import { createRoot } from "react-dom/client";
import { api } from "../../../packages/api-client/src/index.ts";
import { navigate, routeFromLocation } from "./router.ts";
import { OverviewView } from "./features/overview.tsx";
import { NetworkView } from "./features/network.tsx";
import { CriticalityView } from "./features/criticality.tsx";
import { SimilarityView } from "./features/similarity.tsx";
import { EmbeddingsView, EvaluationView } from "./features/analysis.tsx";

type Route = "overview" | "network" | "criticality" | "similarity" | "embeddings" | "evaluation";

const routeMeta: Record<Route, [string, string]> = {
  overview: ["Published workspace", "Overview"],
  network: ["Published network", "Network"],
  criticality: ["Published model output", "Criticality"],
  similarity: ["Published representation", "Similarity"],
  embeddings: ["Published representation", "Embeddings"],
  evaluation: ["Published quality", "Evaluation"]
};

const routeViews: Record<Route, ComponentType> = {
  overview: OverviewView,
  network: NetworkView,
  criticality: CriticalityView,
  similarity: SimilarityView,
  embeddings: EmbeddingsView,
  evaluation: EvaluationView
};

const routes: Array<[Route, string]> = [["overview", "Overview"], ["network", "Network"], ["criticality", "Criticality"], ["similarity", "Similarity"], ["embeddings", "Embeddings"], ["evaluation", "Evaluation"]];

function App() {
  const [route, setRoute] = useState<Route>(routeFromLocation() as Route);
  const [status, setStatus] = useState("Connecting…");
  const [kicker, title] = routeMeta[route];
  const View = routeViews[route];

  useEffect(() => {
    const handlePopState = () => setRoute(routeFromLocation() as Route);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    let active = true;
    api("/api/public/catalog")
      .then(() => active && setStatus("Published data only"))
      .catch(() => active && setStatus("Control API unavailable"));
    return () => { active = false; };
  }, []);

  return (
    <div className="mx-auto min-h-screen max-w-[1280px] px-4 pb-12 pt-6 sm:px-8 lg:px-14">
      <header className="flex items-center justify-between gap-4 border-b border-line pb-5">
        <a className="text-xl font-extrabold tracking-[0.02em] text-copy" href="/" onClick={linkClick("overview")}>Transit Lab <span className="text-mint">Explorer</span></a>
        <span className="inline-flex items-center gap-2 rounded-full border border-[#2e7667] px-2.5 py-1 text-xs text-mint"><span className="size-1.5 rounded-full bg-mint-strong" aria-hidden="true" />{status}</span>
      </header>
      <nav className="flex flex-wrap gap-2 py-4 pb-10" aria-label="Published views">
        {routes.map(([routeName, label]) => <a key={routeName} className={`rounded-lg px-2.5 py-1.5 text-sm transition-colors hover:bg-panel-raised hover:text-copy ${route === routeName ? "bg-panel-raised text-copy" : "text-muted"}`} href={routeName === "overview" ? "/" : `/${routeName}`} onClick={linkClick(routeName)}>{label}</a>)}
      </nav>
      <main>
        <p className="mb-2 text-[11px] font-extrabold uppercase tracking-[0.14em] text-mint">{kicker}</p>
        <h1 className="mb-6 mt-0 text-[clamp(30px,5vw,50px)] font-semibold leading-none tracking-[-0.045em] text-copy">{title}</h1>
        <section aria-live="polite"><View key={route} /></section>
      </main>
      <footer className="pt-8 text-xs text-muted">Published artifacts only · private runs remain in Studio</footer>
    </div>
  );

  function linkClick(target: Route) {
    return (event: MouseEvent<HTMLAnchorElement>) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(target);
    };
  }
}

const root = document.querySelector("#root");
if (!root) throw new Error("Explorer root element is missing");
createRoot(root).render(<App />);
