const ROUTES = new Set(["overview", "network", "criticality", "similarity", "embeddings", "evaluation"]);

export function routeFromLocation(location = window.location) {
  const name = location.pathname.replace(/^\/+|\/+$/g, "") || "overview";
  return ROUTES.has(name) ? name : "overview";
}

export function navigate(route) {
  const target = route === "overview" ? "/" : `/${route}`;
  window.history.pushState({}, "", target);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function linkHandler(event) {
  const link = event.target.closest("a[data-route]");
  if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  navigate(link.dataset.route);
}
