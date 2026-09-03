import { useEffect, useState, type ComponentType, type MouseEvent } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api.ts";
import { navigate, routeFromLocation } from "./routes/router.ts";
import { OverviewView } from "./features/overview/index.tsx";
import { DataView } from "./features/data/index.tsx";
import { RunsView } from "./features/runs/index.tsx";
import { NetworkView } from "./features/network/index.tsx";
import { CriticalityView } from "./features/criticality/index.tsx";
import { SimilarityView } from "./features/similarity/index.tsx";
import { EmbeddingsView } from "./features/embeddings/index.tsx";
import { EvaluationView } from "./features/evaluation/index.tsx";
import { Button, cn, LiveBadge } from "./components/ui.tsx";

type Route = "overview" | "data" | "runs" | "network" | "criticality" | "similarity" | "embeddings" | "evaluation";

const routeMeta: Record<Route, [string, string]> = {
  overview: ["Control plane", "Overview"],
  data: ["Repository inventory", "Data & lineage"],
  runs: ["Operational control", "Runs"],
  network: ["Spatial inspection", "Network"],
  criticality: ["Model output", "Criticality"],
  similarity: ["Representation search", "Similarity"],
  embeddings: ["Representation space", "Embeddings"],
  evaluation: ["Quality signals", "Evaluation"]
};

const routeViews: Record<Route, ComponentType> = {
  overview: OverviewView,
  data: DataView,
  runs: RunsView,
  network: NetworkView,
  criticality: CriticalityView,
  similarity: SimilarityView,
  embeddings: EmbeddingsView,
  evaluation: EvaluationView
};

const workspaceRoutes: Array<[Route, string, string]> = [
  ["overview", "⌂", "Overview"],
  ["data", "▦", "Data & lineage"],
  ["runs", "↗", "Runs"],
  ["network", "◎", "Network"],
  ["criticality", "⌁", "Criticality"],
  ["similarity", "≋", "Similarity"]
];

const analysisRoutes: Array<[Route, string, string]> = [
  ["embeddings", "◌", "Embeddings"],
  ["evaluation", "▤", "Evaluation"]
];

function App() {
  const [route, setRoute] = useState<Route>(routeFromLocation() as Route);
  const [sidebarStatus, setSidebarStatus] = useState("Connecting…");
  const [refreshing, setRefreshing] = useState(false);
  const [toast, setToast] = useState<{ message: string; error: boolean } | null>(null);
  const [kicker, title] = routeMeta[route];
  const View = routeViews[route];

  useEffect(() => {
    const handlePopState = () => setRoute(routeFromLocation() as Route);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    let active = true;
    api("/api/health")
      .then(() => active && setSidebarStatus("API connected"))
      .catch(() => active && setSidebarStatus("API unavailable"));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timeout = window.setTimeout(() => setToast(null), 3500);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  async function refreshIndex() {
    setRefreshing(true);
    try {
      await api("/api/inventory/refresh", { method: "POST" });
      setToast({ message: "Inventory refreshed from the configured data directory.", error: false });
      window.dispatchEvent(new PopStateEvent("popstate"));
    } catch (error) {
      setToast({ message: error instanceof Error ? error.message : String(error), error: true });
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[244px_minmax(0,1fr)]">
      <aside className="flex flex-col border-b border-line bg-ink-raised px-4 py-5 lg:min-h-screen lg:border-b-0 lg:border-r lg:py-7">
        <a className="mb-7 inline-flex items-center gap-2.5 px-3 font-bold tracking-[-0.02em]" href="/" onClick={linkClick("overview")} aria-label="Transit Lab Studio overview">
          <span className="inline-flex h-[25px] w-[26px] items-end gap-[3px] rounded-lg border border-[#3e7061] bg-[#18342e] p-0.5" aria-hidden="true">
            <i className="h-[9px] w-[5px] rounded-full bg-mint opacity-65" />
            <i className="h-[15px] w-[5px] rounded-full bg-mint" />
            <i className="h-5 w-[5px] rounded-full bg-mint opacity-80" />
          </span>
          <span>Transit Lab <em className="font-medium not-italic text-mint">Studio</em></span>
        </a>
        <p className="mb-2 px-3 text-[10px] font-extrabold uppercase tracking-[0.14em] text-faint">Workspace</p>
        <nav className="grid gap-0.5" aria-label="Primary navigation">
          {workspaceRoutes.map(([routeName, icon, label]) => <NavLink key={routeName} route={routeName} icon={icon} label={label} active={route === routeName} />)}
        </nav>
        <p className="mb-2 mt-7 px-3 text-[10px] font-extrabold uppercase tracking-[0.14em] text-faint">Analysis</p>
        <nav className="grid gap-0.5" aria-label="Analysis navigation">
          {analysisRoutes.map(([routeName, icon, label]) => <NavLink key={routeName} route={routeName} icon={icon} label={label} active={route === routeName} />)}
        </nav>
        <div className="mt-7 flex items-center gap-2 border-t border-line px-2 pt-[18px] text-[11px] text-muted lg:mt-auto">
          <span className="size-1.5 rounded-full bg-mint-strong shadow-[0_0_0_4px_rgba(66,198,149,0.1)]" aria-hidden="true" />
          <span><strong className="block text-[11px] font-semibold text-copy">Local workspace</strong><small className="mt-0.5 block text-[10px] text-faint">{sidebarStatus}</small></span>
        </div>
      </aside>
      <main className="min-w-0 px-[22px] pb-[70px] pt-7 sm:px-[4vw] lg:px-[clamp(22px,4vw,64px)]">
        <header className="mx-auto mb-[34px] flex max-w-[1440px] items-end justify-between gap-5 border-b border-line pb-6 max-[720px]:items-start max-[720px]:flex-col">
          <div>
            <p className="mb-1.5 text-[10px] font-extrabold uppercase tracking-[0.14em] text-mint">{kicker}</p>
            <h1 className="m-0 text-[clamp(25px,3vw,36px)] font-semibold tracking-[-0.045em] text-copy">{title}</h1>
          </div>
          <div className="flex items-center gap-3 max-[720px]:flex-wrap">
            <LiveBadge>Rust artifacts indexed</LiveBadge>
            <Button variant="quiet" onClick={refreshIndex} disabled={refreshing}>{refreshing ? "Refreshing…" : "Refresh index"}</Button>
          </div>
        </header>
        <section className="mx-auto max-w-[1440px]" aria-live="polite">
          <View key={route} />
        </section>
      </main>
      {toast ? <div className={cn("fixed bottom-[22px] right-[22px] max-w-[360px] rounded-lg border px-3.5 py-2.5 text-xs shadow-[0_18px_50px_rgba(0,0,0,0.2)]", toast.error ? "border-[#70433f] bg-[#2b1d1c] text-coral" : "border-[#3c675a] bg-[#17312b] text-mint")} role="status" aria-live="polite">{toast.message}</div> : null}
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

function NavLink({ route, icon, label, active }: { route: Route; icon: string; label: string; active: boolean }) {
  return (
    <a
      className={cn("flex items-center gap-2.5 rounded-lg border border-transparent px-3 py-2.5 text-[13px] text-muted transition-colors hover:bg-[#152020] hover:text-copy", active && "border-[#2f5a4f] bg-[#17312b] text-mint")}
      href={route === "overview" ? "/" : `/${route}`}
      onClick={(event) => {
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        navigate(route);
      }}
    >
      <span className={cn("w-[17px] text-center text-[17px] text-faint", active && "text-mint")} aria-hidden="true">{icon}</span>
      {label}
    </a>
  );
}

const root = document.querySelector("#root");
if (!root) throw new Error("Studio root element is missing");
createRoot(root).render(<App />);
