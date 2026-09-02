const root = import.meta.dir;
const port = Number(process.env.PORT || 3000);

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};

function responseHeaders(contentType) {
  return {
    "Cache-Control": "no-store",
    "Content-Type": contentType
  };
}

const server = Bun.serve({
  port,
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "transit-lab-web-wrapper" });
    }

    let pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    pathname = decodeURIComponent(pathname);

    if (pathname.includes("..")) {
      return new Response("Not found", { status: 404 });
    }

    const file = Bun.file(`${root}${pathname}`);
    if (!(await file.exists())) {
      return new Response("Not found", { status: 404 });
    }

    const extension = pathname.slice(pathname.lastIndexOf("."));
    return new Response(file, {
      headers: responseHeaders(contentTypes[extension] || "application/octet-stream")
    });
  }
});

console.log(`Transit Lab web wrapper listening on http://localhost:${server.port}`);
