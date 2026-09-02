import { modeRgb } from "./modes.ts";

type Point = { x: number; y: number; z: number };
type ScreenPoint = { x: number; y: number };
type NetworkStation = Record<string, any>;
type NetworkLine = Record<string, any>;
type Network = {
  stations: NetworkStation[];
  lines: NetworkLine[];
  patterns?: Array<Record<string, any>>;
  transit_edges?: Array<Record<string, any>>;
  transfers?: Array<Record<string, any>>;
  interchanges?: Array<Record<string, any>>;
};
type RendererLocations = {
  position: number;
  color: number;
  size: number;
  round: number;
  projection: WebGLUniformLocation | null;
  view: WebGLUniformLocation | null;
};
type RendererBuffers = Record<string, WebGLBuffer | null>;
type RendererCounts = Record<string, number>;
type StationStats = { maxDepartures: number; maxLines: number; maxService: number };
type HoverPosition = ScreenPoint | null;
type RendererOptions = {
  onHover: (station: number | null, position: HoverPosition) => void;
  onSelect: (station: number | null) => void;
};

const vertexSource = `
  attribute vec3 a_position;
  attribute vec4 a_color;
  attribute float a_size;
  attribute float a_round;
  uniform mat4 u_projection;
  uniform mat4 u_view;
  varying vec4 v_color;
  varying float v_round;
  void main() {
    gl_Position = u_projection * u_view * vec4(a_position, 1.0);
    gl_PointSize = a_size;
    v_color = a_color;
    v_round = a_round;
  }
`;

const fragmentSource = `
  precision mediump float;
  varying vec4 v_color;
  varying float v_round;
  void main() {
    if (v_round > 0.5) {
      vec2 point = gl_PointCoord - vec2(0.5);
      if (dot(point, point) > 0.25) discard;
    }
    gl_FragColor = v_color;
  }
`;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function multiply(left: ArrayLike<number>, right: ArrayLike<number>) {
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      result[column * 4 + row] =
        left[row] * right[column * 4] +
        left[4 + row] * right[column * 4 + 1] +
        left[8 + row] * right[column * 4 + 2] +
        left[12 + row] * right[column * 4 + 3];
    }
  }
  return result;
}

function perspective(fieldOfView: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fieldOfView / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * range, -1,
    0, 0, 2 * far * near * range, 0
  ]);
}

function normalize3(vector: number[]) {
  const length = Math.hypot(...vector) || 1;
  return vector.map((value) => value / length);
}

function cross(left: number[], right: number[]) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0]
  ];
}

function lookAt(eye: number[], target: number[], up: number[]) {
  const forward = normalize3(target.map((value, index) => value - eye[index]));
  const side = normalize3(cross(forward, up));
  const correctedUp = cross(side, forward);
  return new Float32Array([
    side[0], correctedUp[0], -forward[0], 0,
    side[1], correctedUp[1], -forward[1], 0,
    side[2], correctedUp[2], -forward[2], 0,
    -side[0] * eye[0] - side[1] * eye[1] - side[2] * eye[2],
    -correctedUp[0] * eye[0] - correctedUp[1] * eye[1] - correctedUp[2] * eye[2],
    forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2],
    1
  ]);
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "Unknown shader error";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext) {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "Unable to link the graph renderer");
  }
  return program;
}

export function stationsForVisibleLines(network: Network, visibleLines: Set<number>) {
  const visibleStations = new Set<number>();
  for (const edge of network.transit_edges || []) {
    if (!visibleLines.has(Number(edge.line))) continue;
    visibleStations.add(Number(edge.from));
    visibleStations.add(Number(edge.to));
  }
  for (const pattern of network.patterns || []) {
    if (!visibleLines.has(Number(pattern.signature?.line))) continue;
    for (const station of pattern.signature?.stops || []) visibleStations.add(Number(station));
  }
  return visibleStations;
}

export class GraphRenderer {
  canvas: HTMLCanvasElement;
  onHover: RendererOptions["onHover"];
  onSelect: RendererOptions["onSelect"];
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: RendererLocations;
  buffers: RendererBuffers;
  network: Network | null;
  positions: Point[];
  stationPoints: Point[];
  visibleStations: Set<number>;
  visibleLines: Set<number>;
  selectedLine: number | null;
  hoveredStation: number | null;
  heightMode: string;
  showTransfers: boolean;
  showStations: boolean;
  yaw: number;
  pitch: number;
  distance: number;
  dragging: boolean;
  moved: boolean;
  lastPointer: { x: number; y: number };
  resizeObserver: ResizeObserver;
  counts: RendererCounts;

  constructor(canvas: HTMLCanvasElement, { onHover, onSelect }: RendererOptions) {
    this.canvas = canvas;
    this.onHover = onHover;
    this.onSelect = onSelect;
    const gl = canvas.getContext("webgl", { antialias: true, alpha: true });
    if (!gl) throw new Error("This browser does not support WebGL.");
    this.gl = gl;
    this.program = createProgram(gl);
    this.locations = {
      position: this.gl.getAttribLocation(this.program, "a_position"),
      color: this.gl.getAttribLocation(this.program, "a_color"),
      size: this.gl.getAttribLocation(this.program, "a_size"),
      round: this.gl.getAttribLocation(this.program, "a_round"),
      projection: this.gl.getUniformLocation(this.program, "u_projection"),
      view: this.gl.getUniformLocation(this.program, "u_view")
    };
    this.buffers = {};
    this.network = null;
    this.positions = [];
    this.stationPoints = [];
    this.visibleStations = new Set();
    this.visibleLines = new Set();
    this.selectedLine = null;
    this.hoveredStation = null;
    this.yaw = 0.68;
    this.pitch = 0.76;
    this.distance = 10;
    this.dragging = false;
    this.moved = false;
    this.lastPointer = { x: 0, y: 0 };
    this.attachEvents();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }

  attachEvents() {
    this.canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.moved = false;
      this.lastPointer = { x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (this.dragging) {
        const deltaX = event.clientX - this.lastPointer.x;
        const deltaY = event.clientY - this.lastPointer.y;
        if (Math.abs(deltaX) + Math.abs(deltaY) > 1) this.moved = true;
        this.yaw += deltaX * 0.008;
        this.pitch = clamp(this.pitch + deltaY * 0.008, 0.12, 1.45);
        this.lastPointer = { x: event.clientX, y: event.clientY };
        this.render();
      }
      if (this.network && !this.dragging) this.updateHover(event);
    });
    this.canvas.addEventListener("pointerup", (event) => {
      this.dragging = false;
      this.canvas.releasePointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("click", (event) => {
      if (!this.moved && this.network) {
        const station = this.pickStation(event.clientX, event.clientY);
        this.onSelect(station);
      }
    });
    this.canvas.addEventListener("pointerleave", () => {
      if (!this.dragging) {
        this.hoveredStation = null;
        this.onHover(null, null);
      }
    });
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = clamp(this.distance * Math.exp(event.deltaY * 0.001), 3.4, 28);
      this.render();
    }, { passive: false });
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.render();
    }
  }

  setNetwork(network: Network) {
    this.network = network;
    this.selectedLine = null;
    this.hoveredStation = null;
    this.visibleLines = new Set(network.lines.map((line) => line.index));
    this.positions = this.makePositions(network);
    this.buildBuffers();
    this.fit();
  }

  makePositions(network: Network) {
    if (!network.stations.length) return [];
    const latitudes = network.stations.map((station) => Number(station.latitude));
    const longitudes = network.stations.map((station) => Number(station.longitude));
    const minLat = Math.min(...latitudes);
    const maxLat = Math.max(...latitudes);
    const minLon = Math.min(...longitudes);
    const maxLon = Math.max(...longitudes);
    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;
    const scale = Math.max(maxLat - minLat, maxLon - minLon, 0.0001);
    return network.stations.map((station) => ({
      x: ((Number(station.longitude) - centerLon) / scale) * 8,
      z: -((Number(station.latitude) - centerLat) / scale) * 8,
      y: 0
    }));
  }

  heightFor(station: NetworkStation, heightMode: string, stats: StationStats) {
    if (!this.network) return 0.1;
    if (heightMode === "departures") {
      return 0.1 + Math.sqrt((Number(station.daily_departures) || 0) / stats.maxDepartures) * 1.5;
    }
    if (heightMode === "service") {
      const span = Math.max(0, (Number(station.last_departure) || 0) - (Number(station.first_departure) || 0));
      return 0.1 + Math.sqrt(span / stats.maxService) * 1.5;
    }
    return 0.12 + Math.sqrt((Number(station.line_count) || 0) / stats.maxLines) * 1.55;
  }

  setHeightMode(heightMode: string) {
    this.heightMode = heightMode;
    if (this.network) {
      this.buildBuffers();
      this.render();
    }
  }

  setVisibleLines(lines: Iterable<number>) {
    this.visibleLines = new Set(lines);
    this.buildBuffers();
    if (this.hoveredStation !== null && !this.visibleStations.has(this.hoveredStation)) {
      this.hoveredStation = null;
      this.onHover(null, null);
    }
    this.render();
  }

  setShowTransfers(showTransfers: boolean) {
    this.showTransfers = showTransfers;
    this.buildBuffers();
    this.render();
  }

  setShowStations(showStations: boolean) {
    this.showStations = showStations;
    this.render();
  }

  setSelectedLine(lineIndex: number | null) {
    this.selectedLine = lineIndex;
    this.buildBuffers();
    this.render();
  }

  fit() {
    this.yaw = 0.68;
    this.pitch = 0.76;
    this.distance = 10;
    this.render();
  }

  topView() {
    this.yaw = 0;
    this.pitch = 1.46;
    this.render();
  }

  createBuffer(key: string, values: number[]) {
    const gl = this.gl;
    if (!this.buffers[key]) this.buffers[key] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[key]);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.STATIC_DRAW);
  }

  buildBuffers() {
    if (!this.network) return;
    const network = this.network;
    const lines = [];
    const transfers = [];
    const stations = [];
    const stems = [];
    const grid = [];
    this.visibleStations = stationsForVisibleLines(network, this.visibleLines);
    const stationStats = {
      maxDepartures: Math.max(...network.stations.map((station) => Number(station.daily_departures) || 0), 1),
      maxLines: Math.max(...network.stations.map((station) => Number(station.line_count) || 0), 1),
      maxService: Math.max(...network.stations.map((station) => Math.max(0, (Number(station.last_departure) || 0) - (Number(station.first_departure) || 0))), 1)
    };
    const vertex = (position, color, size = 1, round = 0) => [
      position.x, position.y, position.z, color[0], color[1], color[2], color[3] ?? 1, size, round
    ];
    const edgeColor = (lineIndex) => {
      const color = modeRgb(network.lines[lineIndex]?.mode);
      if (this.selectedLine !== null && lineIndex !== this.selectedLine) {
        return [color[0] * 0.34, color[1] * 0.34, color[2] * 0.34, 0.22];
      }
      return [color[0], color[1], color[2], 0.9];
    };

    for (let index = -10; index <= 10; index += 1) {
      const value = index * 0.8;
      grid.push(...vertex({ x: value, y: -0.01, z: -8 }, [0.17, 0.22, 0.28, 0.18]));
      grid.push(...vertex({ x: value, y: -0.01, z: 8 }, [0.17, 0.22, 0.28, 0.18]));
      grid.push(...vertex({ x: -8, y: -0.01, z: value }, [0.17, 0.22, 0.28, 0.18]));
      grid.push(...vertex({ x: 8, y: -0.01, z: value }, [0.17, 0.22, 0.28, 0.18]));
    }

    this.stationPoints = network.stations.map((station, index) => {
      const point = this.positions[index];
      point.y = this.heightFor(station, this.heightMode || "connectivity", stationStats);
      const isVisible = this.visibleStations.has(index);
      const isInterchange = Number(station.line_count) > 1;
      const color = isInterchange ? [1.0, 0.78, 0.37, 1] : [0.88, 0.95, 0.96, 0.96];
      const radius = isInterchange ? 11 + Math.min(7, Number(station.line_count) * 1.5) : 8;
      if (isVisible) {
        if (this.showStations !== false) stations.push(...vertex(point, color, radius, 1));
        stems.push(...vertex({ ...point, y: 0 }, [color[0], color[1], color[2], 0.12]));
        stems.push(...vertex(point, [color[0], color[1], color[2], isInterchange ? 0.6 : 0.28]));
      }
      return point;
    });

    for (const edge of network.transit_edges || []) {
      const lineIndex = Number(edge.line);
      if (!this.visibleLines.has(lineIndex)) continue;
      const from = this.stationPoints[Number(edge.from)];
      const to = this.stationPoints[Number(edge.to)];
      if (!from || !to) continue;
      const color = edgeColor(lineIndex);
      lines.push(...vertex(from, color), ...vertex(to, color));
    }

    if (this.showTransfers !== false) {
      for (const transfer of network.transfers || []) {
        const from = this.stationPoints[Number(transfer.from)];
        const to = this.stationPoints[Number(transfer.to)];
        if (!from || !to || !this.visibleStations.has(Number(transfer.from)) || !this.visibleStations.has(Number(transfer.to))) continue;
        transfers.push(...vertex(from, [0.55, 0.64, 0.72, 0.5]), ...vertex(to, [0.55, 0.64, 0.72, 0.5]));
      }
    }

    this.createBuffer("grid", grid);
    this.createBuffer("stems", stems);
    this.createBuffer("lines", lines);
    this.createBuffer("transfers", transfers);
    this.createBuffer("stations", stations);
    this.counts = {
      grid: grid.length / 9,
      stems: stems.length / 9,
      lines: lines.length / 9,
      transfers: transfers.length / 9,
      stations: stations.length / 9
    };
  }

  bindBuffer(key: string) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[key]);
    const stride = 9 * Float32Array.BYTES_PER_ELEMENT;
    gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, stride, 0);
    gl.vertexAttribPointer(this.locations.color, 4, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribPointer(this.locations.size, 1, gl.FLOAT, false, stride, 7 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribPointer(this.locations.round, 1, gl.FLOAT, false, stride, 8 * Float32Array.BYTES_PER_ELEMENT);
  }

  draw(key: string, mode: number) {
    if (!this.counts?.[key]) return;
    this.bindBuffer(key);
    this.gl.drawArrays(mode, 0, this.counts[key]);
  }

  cameraMatrices() {
    const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    const eye = [
      Math.sin(this.yaw) * Math.cos(this.pitch) * this.distance,
      Math.sin(this.pitch) * this.distance,
      Math.cos(this.yaw) * Math.cos(this.pitch) * this.distance
    ];
    return {
      projection: perspective(Math.PI / 4, aspect, 0.1, 60),
      view: lookAt(eye, [0, 0.5, 0], [0, 1, 0])
    };
  }

  render() {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.03, 0.045, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.network) return;
    gl.useProgram(this.program);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const matrices = this.cameraMatrices();
    gl.uniformMatrix4fv(this.locations.projection, false, matrices.projection);
    gl.uniformMatrix4fv(this.locations.view, false, matrices.view);
    gl.enableVertexAttribArray(this.locations.position);
    gl.enableVertexAttribArray(this.locations.color);
    gl.enableVertexAttribArray(this.locations.size);
    gl.enableVertexAttribArray(this.locations.round);
    gl.lineWidth(1);
    this.draw("grid", gl.LINES);
    this.draw("transfers", gl.LINES);
    this.draw("stems", gl.LINES);
    this.draw("lines", gl.LINES);
    this.draw("stations", gl.POINTS);
  }

  project(point: Point): ScreenPoint | null {
    const matrices = this.cameraMatrices();
    const combined = multiply(matrices.projection, matrices.view);
    const x = combined[0] * point.x + combined[4] * point.y + combined[8] * point.z + combined[12];
    const y = combined[1] * point.x + combined[5] * point.y + combined[9] * point.z + combined[13];
    const w = combined[3] * point.x + combined[7] * point.y + combined[11] * point.z + combined[15];
    if (w <= 0) return null;
    return {
      x: (x / w + 1) * 0.5 * this.canvas.clientWidth,
      y: (1 - y / w) * 0.5 * this.canvas.clientHeight
    };
  }

  pickStation(clientX: number, clientY: number): number | null {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let closest = null;
    let distance = 20;
    for (let index = 0; index < this.stationPoints.length; index += 1) {
      if (!this.visibleStations.has(index) || this.showStations === false) continue;
      const screen = this.project(this.stationPoints[index]);
      if (!screen) continue;
      const current = Math.hypot(screen.x - x, screen.y - y);
      if (current < distance) {
        closest = index;
        distance = current;
      }
    }
    return closest;
  }

  updateHover(event: PointerEvent) {
    const station = this.pickStation(event.clientX, event.clientY);
    if (station === this.hoveredStation) return;
    this.hoveredStation = station;
    this.canvas.style.cursor = station === null ? "grab" : "pointer";
    this.onHover(station, station === null ? null : this.project(this.stationPoints[station]));
  }
}
