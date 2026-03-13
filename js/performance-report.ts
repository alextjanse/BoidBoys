export type BenchmarkSettings = {
  boidCount: number;
  separationWeight: number;
  alignmentWeight: number;
  cohesionWeight: number;
  maxSpeed: number;
  updateFrequency?: number;
  projectName?: string;
  groupName?: string;
  version?: string;
};

export type HardwareInfo = {
  cpu: string;
  gpu: string;
  os: string;
};

export type RuntimeMetrics = {
  avgRenderTime: number;
  avgSimTime: number;
};

type ExportInput = {
  frameTimes: number[];
  settings: BenchmarkSettings;
  hardware: HardwareInfo;
  metrics: RuntimeMetrics;
};

type SummaryStats = {
  avgFrameTime: number;
  avgFps: number;
  onePercentLowFrameTime: number;
  onePercentLowFps: number;
};

type BenchmarkExportPayload = {
  schemaVersion: string;
  generatedAtIso: string;
  filenameBase: string;
  settings: BenchmarkSettings;
  hardware: HardwareInfo;
  metrics: RuntimeMetrics;
  stats: SummaryStats;
  frameTimes: number[];
  rollingAverage: {
    windowSize: number;
    frameTimes: number[];
  };
};

export class PerformanceReportExporter {
  public async exportPerformanceReport(input: ExportInput): Promise<Blob | null> {
    const cleaned = input.frameTimes.filter((v) => Number.isFinite(v) && v > 0);
    if (cleaned.length === 0) {
      window.alert("No frame-time data available to export.");
      return null;
    }

    const date = new Date();
    const dateStamp = this.formatDateStamp(date);
    const boidLabel = this.formatBoidLabel(input.settings.boidCount);
    const benchmarkName = await this.promptBenchmarkNameWithPreview(dateStamp, boidLabel);
    if (!benchmarkName) {
      return null;
    }

    const filename = `${dateStamp}_${boidLabel}_Boids_${benchmarkName}.png`;
    const filenameBase = `${dateStamp}_${boidLabel}_Boids_${benchmarkName}`;
    const stats = this.computeStats(cleaned);
    const canvas = this.renderReportCanvas(cleaned, input, stats, date);

    const blob = await this.canvasToPngBlob(canvas);
    this.downloadBlob(blob, filename);

    const payload = this.buildBenchmarkExportPayload(cleaned, input, stats, date, filenameBase);
    this.downloadJson(payload, `${filenameBase}.json`);

    return blob;
  }

  public async importBenchmarkJsonAndExport(file: File): Promise<Blob | null> {
    const raw = await file.text();
    let parsed: unknown;

    try {
      parsed = JSON.parse(raw);
    } catch {
      window.alert("Invalid JSON file.");
      return null;
    }

    const payload = this.parseBenchmarkPayload(parsed);
    if (!payload) {
      window.alert("JSON does not match expected benchmark export format.");
      return null;
    }

    const cleaned = payload.frameTimes.filter((v) => Number.isFinite(v) && v > 0);
    if (cleaned.length === 0) {
      window.alert("Imported JSON has no valid frame-time data.");
      return null;
    }

    const reportDate = payload.generatedAtIso ? new Date(payload.generatedAtIso) : new Date();
    const safeDate = Number.isFinite(reportDate.getTime()) ? reportDate : new Date();

    const stats = this.computeStats(cleaned);
    const input: ExportInput = {
      frameTimes: cleaned,
      settings: payload.settings,
      hardware: payload.hardware,
      metrics: payload.metrics,
    };

    const dateStamp = this.formatDateStamp(new Date());
    const boidLabel = this.formatBoidLabel(payload.settings.boidCount);
    const benchmarkName = await this.promptBenchmarkNameWithPreview(dateStamp, boidLabel);
    if (!benchmarkName) {
      return null;
    }

    const filename = `${dateStamp}_${boidLabel}_Boids_${benchmarkName}.png`;
    const canvas = this.renderReportCanvas(cleaned, input, stats, safeDate);
    const blob = await this.canvasToPngBlob(canvas);
    this.downloadBlob(blob, filename);
    return blob;
  }

  public async openBenchmarkPreviewFromJsonFiles(files: File[]): Promise<void> {
    const parsed: Array<{ fileName: string; payload: BenchmarkExportPayload; }> = [];
    const failed: string[] = [];

    for (const file of files) {
      try {
        const raw = await file.text();
        const json = JSON.parse(raw) as unknown;
        const payload = this.parseBenchmarkPayload(json);
        if (!payload) {
          failed.push(file.name);
          continue;
        }
        parsed.push({ fileName: file.name, payload });
      } catch {
        failed.push(file.name);
      }
    }

    if (parsed.length === 0) {
      window.alert("No valid benchmark JSON files were selected.");
      return;
    }

    const previewCanvas = this.renderImportedComparisonCanvas(parsed);
    const defaultBaseName = parsed.length === 1
      ? (parsed[0].payload.filenameBase || parsed[0].fileName.replace(/\.json$/i, ""))
      : `${this.formatDateStamp(new Date())}_Benchmark_Comparison`;

    this.openGraphPreviewModal(previewCanvas, `${this.sanitizeFilePart(defaultBaseName)}.png`, parsed.length);

    if (failed.length > 0) {
      window.alert(`Skipped ${failed.length} invalid file(s): ${failed.join(", ")}`);
    }
  }

  private renderImportedComparisonCanvas(
    entries: Array<{ fileName: string; payload: BenchmarkExportPayload; }>
  ): HTMLCanvasElement {
    const variablePanels = [
      { label: "Boid Count", unit: "#", values: entries.map((e) => e.payload.settings.boidCount) },
      { label: "Separation Weight", unit: "x", values: entries.map((e) => e.payload.settings.separationWeight) },
      { label: "Alignment Weight", unit: "x", values: entries.map((e) => e.payload.settings.alignmentWeight) },
      { label: "Cohesion Weight", unit: "x", values: entries.map((e) => e.payload.settings.cohesionWeight) },
      { label: "Max Speed", unit: "u/s", values: entries.map((e) => e.payload.settings.maxSpeed) },
      { label: "Update Frequency", unit: "wg", values: entries.map((e) => e.payload.settings.updateFrequency ?? 0) },
      { label: "Avg Sim Time", unit: "ms", values: entries.map((e) => e.payload.metrics.avgSimTime) },
      { label: "Avg Render Time", unit: "ms", values: entries.map((e) => e.payload.metrics.avgRenderTime) },
      { label: "Avg Frame Time", unit: "ms", values: entries.map((e) => e.payload.stats.avgFrameTime) },
      { label: "Avg FPS", unit: "FPS", values: entries.map((e) => e.payload.stats.avgFps) },
      { label: "1% Low Frame Time", unit: "ms", values: entries.map((e) => e.payload.stats.onePercentLowFrameTime) },
      { label: "1% Low FPS", unit: "FPS", values: entries.map((e) => e.payload.stats.onePercentLowFps) },
    ];

    const margin = 80;
    const titleH = 120;
    const trendH = 760;
    const legendH = 130;
    const panelGap = 24;
    const panelH = 210;
    const panelCols = 2;
    const panelRows = Math.ceil(variablePanels.length / panelCols);
    const panelsAreaH = panelRows * panelH + (panelRows - 1) * panelGap;

    const canvas = document.createElement("canvas");
    canvas.width = 2400;
    canvas.height = margin * 2 + titleH + trendH + legendH + 30 + panelsAreaH;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to create 2D canvas context.");
    }

    const bg = "#FFFFFF";
    const text = "#111111";
    const subText = "#555555";
    const border = "#D9D9D9";
    const targetColor = "#D23B3B";
    const palette = ["#2F80ED", "#0AA174", "#E07A12", "#8A5CF6", "#D23B3B", "#3FA9F5", "#1F7A8C"];

    const chartX = margin;
    const chartY = margin + titleH;
    const chartW = canvas.width - margin * 2;
    const chartH = trendH;

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = border;
    ctx.lineWidth = 3;
    ctx.strokeRect(6, 6, canvas.width - 12, canvas.height - 12);

    ctx.fillStyle = text;
    ctx.font = "700 52px Arial";
    ctx.fillText(entries.length > 1 ? "BENCHMARK COMPARISON" : "BENCHMARK GRAPH PREVIEW", margin, margin + 52);
    ctx.fillStyle = subText;
    ctx.font = "500 24px Arial";
    ctx.fillText(
      entries.length > 1
        ? "Comparison across all numeric benchmark variables + 60-frame rolling frame-time trend"
        : "Detailed variable view + frame-time trend",
      margin,
      margin + 92
    );

    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(chartX, chartY, chartW, chartH);

    const padL = 95;
    const padR = 30;
    const padT = 25;
    const padB = 80;
    const plotX = chartX + padL;
    const plotY = chartY + padT;
    const plotW = chartW - padL - padR;
    const plotH = chartH - padT - padB;

    const allSeries = entries.map((entry) => {
      const source = entry.payload.frameTimes.filter((v) => Number.isFinite(v) && v > 0);
      return source.length > 0
        ? this.computeRollingAverage(source, 60)
        : [];
    });
    const flattened = allSeries.flat();
    const maxFrames = Math.max(1, ...allSeries.map((series) => series.length));
    const maxMs = Math.max(22, ...flattened, 16.67) * 1.08;

    const yTicks = [0, 8, 16.67, 24, 33, 40, 50, 66].filter((v) => v <= maxMs + 2);
    ctx.strokeStyle = "#E9E9E9";
    ctx.lineWidth = 1;
    for (const tick of yTicks) {
      const yy = plotY + plotH - (tick / maxMs) * plotH;
      ctx.beginPath();
      ctx.moveTo(plotX, yy);
      ctx.lineTo(plotX + plotW, yy);
      ctx.stroke();

      ctx.fillStyle = subText;
      ctx.font = "500 19px Arial";
      ctx.fillText(`${tick.toFixed(tick === 16.67 ? 2 : 0)}ms`, chartX + 10, yy + 6);
    }

    const targetY = plotY + plotH - (16.67 / maxMs) * plotH;
    ctx.strokeStyle = targetColor;
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 8]);
    ctx.beginPath();
    ctx.moveTo(plotX, targetY);
    ctx.lineTo(plotX + plotW, targetY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = targetColor;
    ctx.font = "600 20px Arial";
    ctx.fillText("16.67ms target", plotX + 10, targetY - 8);

    for (let i = 0; i < allSeries.length; i++) {
      const series = allSeries[i];
      if (series.length === 0) continue;

      const color = palette[i % palette.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();

      for (let j = 0; j < series.length; j++) {
        const nx = maxFrames <= 1 ? 0 : j / (maxFrames - 1);
        const px = plotX + nx * plotW;
        const py = plotY + plotH - (series[j] / maxMs) * plotH;
        if (j === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }

      ctx.stroke();
    }

    ctx.fillStyle = text;
    ctx.font = "600 24px Arial";
    ctx.fillText("Frame Time (ms)", chartX + 14, chartY + 30);
    ctx.fillText("Frame Index", plotX + plotW - 160, chartY + chartH - 20);

    ctx.save();
    ctx.translate(chartX + 26, chartY + chartH / 2 + 80);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Frame Time (ms)", 0, 0);
    ctx.restore();

    const legendY = chartY + chartH + 32;
    ctx.font = "600 20px Arial";
    for (let i = 0; i < entries.length; i++) {
      const color = palette[i % palette.length];
      const col = i % 2;
      const row = Math.floor(i / 2);
      const lx = margin + col * ((canvas.width - margin * 2) / 2);
      const ly = legendY + row * 36;
      const name = entries[i].fileName.replace(/\.json$/i, "");
      const shortName = name.length > 58 ? `${name.slice(0, 55)}...` : name;

      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(lx + 34, ly);
      ctx.stroke();

      ctx.fillStyle = text;
      ctx.fillText(`${i + 1}. ${shortName}`, lx + 44, ly + 7);
    }

    const panelsTop = legendY + Math.ceil(entries.length / 2) * 36 + 34;
    const panelW = Math.floor((canvas.width - margin * 2 - panelGap) / 2);

    for (let i = 0; i < variablePanels.length; i++) {
      const col = i % panelCols;
      const row = Math.floor(i / panelCols);
      const px = margin + col * (panelW + panelGap);
      const py = panelsTop + row * (panelH + panelGap);
      this.drawScalarComparisonPanel(
        ctx,
        px,
        py,
        panelW,
        panelH,
        variablePanels[i].label,
        variablePanels[i].unit,
        variablePanels[i].values,
        palette,
        text,
        subText,
        border
      );
    }

    return canvas;
  }

  private drawScalarComparisonPanel(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    title: string,
    unit: string,
    values: number[],
    palette: string[],
    text: string,
    subText: string,
    border: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = text;
    ctx.font = "700 24px Arial";
    ctx.fillText(`${title} (${unit})`, x + 14, y + 32);

    const padL = 44;
    const padR = 18;
    const padT = 48;
    const padB = 36;
    const plotX = x + padL;
    const plotY = y + padT;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const finiteValues = values.filter((v) => Number.isFinite(v));
    const maxVal = Math.max(1, ...finiteValues) * 1.1;

    ctx.strokeStyle = "#EFEFEF";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const yy = plotY + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(plotX, yy);
      ctx.lineTo(plotX + plotW, yy);
      ctx.stroke();
    }

    const barCount = Math.max(1, values.length);
    const gap = Math.max(8, Math.floor(plotW * 0.015));
    const barW = Math.max(8, Math.floor((plotW - gap * (barCount - 1)) / barCount));

    for (let i = 0; i < values.length; i++) {
      const value = Number.isFinite(values[i]) ? values[i] : 0;
      const bh = maxVal <= 0 ? 0 : (value / maxVal) * plotH;
      const bx = plotX + i * (barW + gap);
      const by = plotY + plotH - bh;
      const color = palette[i % palette.length];

      ctx.fillStyle = color;
      ctx.fillRect(bx, by, barW, bh);

      ctx.fillStyle = subText;
      ctx.font = "600 14px Arial";
      ctx.fillText(String(i + 1), bx + Math.max(0, barW / 2 - 4), plotY + plotH + 16);

      ctx.fillStyle = text;
      ctx.font = "500 13px Arial";
      const label = value >= 1000 ? value.toFixed(0) : value.toFixed(2);
      ctx.fillText(label, bx, Math.max(y + 46, by - 4));
    }
  }

  private openGraphPreviewModal(canvas: HTMLCanvasElement, defaultFilename: string, seriesCount: number): void {
    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(0,0,0,0.45)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.zIndex = "10010";

    const modal = document.createElement("div");
    modal.style.width = "min(94vw, 1320px)";
    modal.style.maxHeight = "90vh";
    modal.style.background = "#FFFFFF";
    modal.style.border = "1px solid #DADADA";
    modal.style.borderRadius = "12px";
    modal.style.boxShadow = "0 20px 60px rgba(0,0,0,0.25)";
    modal.style.padding = "16px";
    modal.style.display = "flex";
    modal.style.flexDirection = "column";
    modal.style.gap = "12px";
    modal.style.fontFamily = "Arial, sans-serif";

    const title = document.createElement("div");
    title.style.fontSize = "22px";
    title.style.fontWeight = "700";
    title.textContent = seriesCount > 1 ? "Benchmark Comparison Preview" : "Benchmark Preview";

    const subtitle = document.createElement("div");
    subtitle.style.fontSize = "14px";
    subtitle.style.color = "#555";
    subtitle.textContent = "Showing 60-frame rolling average; use Export PNG to save this preview.";

    const viewport = document.createElement("div");
    viewport.style.overflow = "auto";
    viewport.style.border = "1px solid #E5E5E5";
    viewport.style.borderRadius = "8px";
    viewport.style.background = "#FAFAFA";

    canvas.style.width = "100%";
    canvas.style.height = "auto";
    canvas.style.display = "block";
    viewport.appendChild(canvas);

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.justifyContent = "flex-end";
    actions.style.gap = "10px";

    const close = document.createElement("button");
    close.textContent = "Close";
    close.style.padding = "8px 12px";
    close.style.border = "1px solid #CCC";
    close.style.background = "#FFF";
    close.style.borderRadius = "8px";
    close.style.cursor = "pointer";

    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export PNG";
    exportBtn.style.padding = "8px 12px";
    exportBtn.style.border = "none";
    exportBtn.style.background = "#2F80ED";
    exportBtn.style.color = "#FFF";
    exportBtn.style.borderRadius = "8px";
    exportBtn.style.cursor = "pointer";

    const finish = () => {
      window.removeEventListener("keydown", onEsc);
      overlay.remove();
    };

    const onEsc = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        finish();
      }
    };

    close.addEventListener("click", finish);
    exportBtn.addEventListener("click", async () => {
      const blob = await this.canvasToPngBlob(canvas);
      const safeName = this.sanitizeFilePart(defaultFilename.replace(/\.png$/i, "")) || "Benchmark_Preview";
      this.downloadBlob(blob, `${safeName}.png`);
    });
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) finish();
    });
    window.addEventListener("keydown", onEsc);

    actions.appendChild(close);
    actions.appendChild(exportBtn);
    modal.appendChild(title);
    modal.appendChild(subtitle);
    modal.appendChild(viewport);
    modal.appendChild(actions);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
  }

  private parseBenchmarkPayload(data: unknown): BenchmarkExportPayload | null {
    if (!data || typeof data !== "object") return null;
    const d = data as Record<string, unknown>;

    if (!Array.isArray(d.frameTimes)) return null;
    const frameTimes = d.frameTimes.filter((v): v is number => typeof v === "number");
    if (frameTimes.length === 0) return null;

    const settingsRaw = d.settings as Record<string, unknown> | undefined;
    const hardwareRaw = d.hardware as Record<string, unknown> | undefined;
    const metricsRaw = d.metrics as Record<string, unknown> | undefined;

    if (!settingsRaw || !hardwareRaw || !metricsRaw) return null;

    if (
      typeof settingsRaw.boidCount !== "number" ||
      typeof settingsRaw.separationWeight !== "number" ||
      typeof settingsRaw.alignmentWeight !== "number" ||
      typeof settingsRaw.cohesionWeight !== "number" ||
      typeof settingsRaw.maxSpeed !== "number"
    ) {
      return null;
    }

    if (
      typeof hardwareRaw.cpu !== "string" ||
      typeof hardwareRaw.gpu !== "string" ||
      typeof hardwareRaw.os !== "string"
    ) {
      return null;
    }

    if (
      typeof metricsRaw.avgRenderTime !== "number" ||
      typeof metricsRaw.avgSimTime !== "number"
    ) {
      return null;
    }

    const computedStats = this.computeStats(frameTimes);

    return {
      schemaVersion: typeof d.schemaVersion === "string" ? d.schemaVersion : "boid-benchmark-export-v1",
      generatedAtIso: typeof d.generatedAtIso === "string" ? d.generatedAtIso : new Date().toISOString(),
      filenameBase: typeof d.filenameBase === "string" ? d.filenameBase : "imported_benchmark",
      settings: {
        boidCount: settingsRaw.boidCount,
        separationWeight: settingsRaw.separationWeight,
        alignmentWeight: settingsRaw.alignmentWeight,
        cohesionWeight: settingsRaw.cohesionWeight,
        maxSpeed: settingsRaw.maxSpeed,
        updateFrequency: typeof settingsRaw.updateFrequency === "number" ? settingsRaw.updateFrequency : undefined,
        projectName: typeof settingsRaw.projectName === "string" ? settingsRaw.projectName : undefined,
        groupName: typeof settingsRaw.groupName === "string" ? settingsRaw.groupName : undefined,
        version: typeof settingsRaw.version === "string" ? settingsRaw.version : undefined,
      },
      hardware: {
        cpu: hardwareRaw.cpu,
        gpu: hardwareRaw.gpu,
        os: hardwareRaw.os,
      },
      metrics: {
        avgRenderTime: metricsRaw.avgRenderTime,
        avgSimTime: metricsRaw.avgSimTime,
      },
      stats: computedStats,
      frameTimes,
      rollingAverage: {
        windowSize: 60,
        frameTimes: this.computeRollingAverage(frameTimes, 60),
      },
    };
  }

  private buildBenchmarkExportPayload(
    frameTimes: number[],
    input: ExportInput,
    stats: SummaryStats,
    date: Date,
    filenameBase: string
  ): BenchmarkExportPayload {
    return {
      schemaVersion: "boid-benchmark-export-v1",
      generatedAtIso: date.toISOString(),
      filenameBase,
      settings: {
        ...input.settings,
      },
      hardware: {
        ...input.hardware,
      },
      metrics: {
        ...input.metrics,
      },
      stats: {
        ...stats,
      },
      frameTimes: [...frameTimes],
      rollingAverage: {
        windowSize: 60,
        frameTimes: this.computeRollingAverage(frameTimes, 60),
      },
    };
  }

  private computeRollingAverage(frameTimes: number[], windowSize: number): number[] {
    const out: number[] = [];
    if (windowSize <= 0) return out;

    let rollingSum = 0;
    for (let i = 0; i < frameTimes.length; i++) {
      rollingSum += frameTimes[i];
      if (i >= windowSize) {
        rollingSum -= frameTimes[i - windowSize];
      }
      const samples = Math.min(i + 1, windowSize);
      out.push(rollingSum / samples);
    }

    return out;
  }

  private computeStats(frameTimes: number[]): SummaryStats {
    const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    const avgFps = 1000 / avgFrameTime;

    const sortedDescending = [...frameTimes].sort((a, b) => b - a);
    const count = Math.max(1, Math.floor(sortedDescending.length * 0.01));
    const worstOnePercent = sortedDescending.slice(0, count);
    const onePercentLowFrameTime = worstOnePercent.reduce((a, b) => a + b, 0) / worstOnePercent.length;
    const onePercentLowFps = 1000 / onePercentLowFrameTime;

    return {
      avgFrameTime,
      avgFps,
      onePercentLowFrameTime,
      onePercentLowFps,
    };
  }

  private renderReportCanvas(
    frameTimes: number[],
    input: ExportInput,
    stats: SummaryStats,
    date: Date
  ): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.width = 2800;
    canvas.height = 1800;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to create 2D canvas context.");
    }

    const bg = "#FFFFFF";
    const border = "#D9D9D9";
    const text = "#111111";
    const subText = "#555555";
    const accent = "#2F80ED";
    const good = "#1A9E55";
    const bad = "#D23B3B";

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = border;
    ctx.lineWidth = 4;
    ctx.strokeRect(8, 8, canvas.width - 16, canvas.height - 16);

    const margin = 80;
    const headerH = 190;
    const footerH = 340;
    const contentTop = margin + headerH;
    const contentBottom = canvas.height - margin - footerH;
    const contentH = contentBottom - contentTop;

    const guideW = 280;
    const graphW = 1500;
    const statsW = 360;
    const narrativeW = 460;
    const gap = 30;

    const guideX = margin;
    const graphX = guideX + guideW + gap;
    const statsX = graphX + graphW + gap;
    const narrativeX = statsX + statsW + gap;

    this.drawHeader(ctx, margin, margin, canvas.width - margin * 2, headerH, date, input.settings.version ?? "v1.0.0", text, subText);
    this.drawYAxisGuideBox(ctx, guideX, contentTop, guideW, contentH, text, subText, border);
    this.drawGraph(ctx, frameTimes, graphX, contentTop, graphW, contentH, text, subText, border, accent, bad);
    this.drawStatsSidebar(ctx, statsX, contentTop, statsW, contentH, stats, text, subText, border, good, bad);
    this.drawNarrativeBox(ctx, narrativeX, contentTop, narrativeW, contentH, text, subText, border);
    this.drawFooter(ctx, margin, canvas.height - margin - footerH, canvas.width - margin * 2, footerH, input, text, subText, border);

    return canvas;
  }

  private drawHeader(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    date: Date,
    version: string,
    text: string,
    subText: string
  ): void {
    ctx.fillStyle = text;
    ctx.font = "700 58px Arial";
    ctx.fillText("PERFORMANCE BENCHMARK REPORT", x, y + 68);

    ctx.fillStyle = subText;
    ctx.font = "500 28px Arial";
    ctx.fillText(`Date: ${date.toISOString().slice(0, 10)}`, x, y + 120);
    ctx.fillText(`Version: ${version}`, x + 460, y + 120);
    ctx.fillText("Target: 60Hz (16.67ms)", x + 880, y + 120);

    ctx.strokeStyle = "#D9D9D9";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y + h - 5);
    ctx.lineTo(x + w, y + h - 5);
    ctx.stroke();
  }

  private drawGraph(
    ctx: CanvasRenderingContext2D,
    data: number[],
    x: number,
    y: number,
    w: number,
    h: number,
    text: string,
    subText: string,
    border: string,
    lineColor: string,
    targetColor: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const padL = 110;
    const padR = 20;
    const padT = 30;
    const padB = 95;
    const plotX = x + padL;
    const plotY = y + padT;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const maxMs = Math.max(22, ...data) * 1.05;
    const yTicks = [0, 8, 16.67, 24, 33, 40].filter((v) => v <= maxMs + 3);

    ctx.strokeStyle = "#E9E9E9";
    ctx.lineWidth = 1;
    for (const tick of yTicks) {
      const yy = plotY + plotH - (tick / maxMs) * plotH;
      ctx.beginPath();
      ctx.moveTo(plotX, yy);
      ctx.lineTo(plotX + plotW, yy);
      ctx.stroke();

      ctx.fillStyle = subText;
      ctx.font = "500 20px Arial";
      ctx.fillText(`${tick.toFixed(tick === 16.67 ? 2 : 0)}ms`, x + 12, yy + 6);
    }

    const targetY = plotY + plotH - (16.67 / maxMs) * plotH;
    ctx.strokeStyle = targetColor;
    ctx.lineWidth = 3;
    ctx.setLineDash([12, 8]);
    ctx.beginPath();
    ctx.moveTo(plotX, targetY);
    ctx.lineTo(plotX + plotW, targetY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = targetColor;
    ctx.font = "600 22px Arial";
    ctx.fillText("16.67ms (60 FPS)", plotX + 10, targetY - 10);

    // Raw frame-time line
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const nx = data.length <= 1 ? 0 : i / (data.length - 1);
      const px = plotX + nx * plotW;
      const py = plotY + plotH - (data[i] / maxMs) * plotH;
      if (i === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Rolling 60-frame average line
    const WINDOW = 60;
    const rollingAvgColor = "#F0A500";
    ctx.strokeStyle = rollingAvgColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - WINDOW + 1);
      let sum = 0;
      for (let j = start; j <= i; j++) sum += data[j];
      const avgMs = sum / (i - start + 1);
      const nx = data.length <= 1 ? 0 : i / (data.length - 1);
      const px = plotX + nx * plotW;
      const py = plotY + plotH - (avgMs / maxMs) * plotH;
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();

    // Legend
    const legendX = plotX + 10;
    const legendY = plotY + plotH - 14;
    ctx.font = "600 20px Arial";

    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 36, legendY);
    ctx.stroke();
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = lineColor;
    ctx.fillText("Frame Time", legendX + 44, legendY + 6);

    ctx.strokeStyle = rollingAvgColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX + 220, legendY);
    ctx.lineTo(legendX + 256, legendY);
    ctx.stroke();
    ctx.fillStyle = rollingAvgColor;
    ctx.fillText("60-Frame Avg", legendX + 264, legendY + 6);

    ctx.strokeStyle = targetColor;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    ctx.beginPath();
    ctx.moveTo(legendX + 460, legendY);
    ctx.lineTo(legendX + 496, legendY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = targetColor;
    ctx.fillText("60 FPS Target", legendX + 504, legendY + 6);

    ctx.fillStyle = text;
    ctx.font = "600 26px Arial";
    ctx.fillText("Frame Time (ms)", x + 14, y + 24);
    ctx.fillText("X-Axis: Frame Count / Time", plotX + plotW - 340, y + h - 54);

    ctx.save();
    ctx.translate(x + 32, y + h / 2 + 120);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Frame Time (ms)", 0, 0);
    ctx.restore();
  }

  private drawYAxisGuideBox(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    text: string,
    subText: string,
    border: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = text;
    ctx.font = "700 28px Arial";
    ctx.fillText("Y-Axis Guide", x + 22, y + 42);

    const rows = [
      "< 16.67ms: Meets 60 FPS target",
      "16.67-25ms: Minor drops",
      "> 25ms: Noticeable hitching",
      "Spikes: Transient frame stalls",
      "Flat line: Stable pacing"
    ];

    ctx.fillStyle = subText;
    ctx.font = "500 21px Arial";
    let yy = y + 88;
    for (const row of rows) {
      const usedLines = this.drawWrappedText(ctx, row, x + 22, yy, w - 40, 30);
      yy += usedLines * 30 + 14;
    }
  }

  private drawNarrativeBox(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    text: string,
    subText: string,
    border: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = text;
    ctx.font = "700 28px Arial";
    ctx.fillText("Narrative", x + 22, y + 42);

    const narrative = [
      "This chart visualizes per-frame render cadence across a fixed benchmark window.",
      "Short vertical excursions indicate transient frame-time spikes; repeated peaks usually align with compute pressure or memory sync.",
      "A compressed line near or under 16.67ms indicates stable 60 FPS behavior.",
      "Wider variance indicates unstable pacing and inconsistent simulation throughput."
    ];

    ctx.fillStyle = subText;
    ctx.font = "500 21px Arial";
    let yy = y + 85;
    for (const line of narrative) {
      const usedLines = this.drawWrappedText(ctx, line, x + 22, yy, w - 44, 31);
      yy += usedLines * 31 + 22;
    }
  }

  private drawStatsSidebar(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    stats: SummaryStats,
    text: string,
    subText: string,
    border: string,
    good: string,
    bad: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = text;
    ctx.font = "700 28px Arial";
    ctx.fillText("Statistics", x + 20, y + 42);

    const avgColor = stats.avgFrameTime <= 16.67 ? good : bad;
    const lowColor = stats.onePercentLowFrameTime <= 16.67 ? good : bad;

    const cardTop = y + 70;
    const cardGap = 22;
    const cardH = 170;

    this.drawMetricCard(ctx, x + 18, cardTop, w - 36, cardH, "Avg FPS", stats.avgFps.toFixed(1), avgColor, subText, border);
    this.drawMetricCard(
      ctx,
      x + 18,
      cardTop + cardH + cardGap,
      w - 36,
      cardH,
      "Avg Frame Time",
      `${stats.avgFrameTime.toFixed(2)} ms`,
      avgColor,
      subText,
      border
    );
    this.drawMetricCard(
      ctx,
      x + 18,
      cardTop + (cardH + cardGap) * 2,
      w - 36,
      cardH,
      "1% Lows",
      `${stats.onePercentLowFps.toFixed(1)} FPS`,
      lowColor,
      subText,
      border
    );

    ctx.fillStyle = subText;
    ctx.font = "500 18px Arial";
    ctx.fillText(`1% low frame-time: ${stats.onePercentLowFrameTime.toFixed(2)} ms`, x + 24, y + h - 28);
  }

  private drawMetricCard(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
    value: string,
    valueColor: string,
    subText: string,
    border: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = subText;
    ctx.font = "600 20px Arial";
    ctx.fillText(label, x + 16, y + 34);

    ctx.fillStyle = valueColor;
    ctx.font = "700 44px Arial";
    ctx.fillText(value, x + 16, y + 108);
  }

  private drawFooter(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    input: ExportInput,
    text: string,
    subText: string,
    border: string
  ): void {
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const colW = w / 3;
    ctx.beginPath();
    ctx.moveTo(x + colW, y);
    ctx.lineTo(x + colW, y + h);
    ctx.moveTo(x + colW * 2, y);
    ctx.lineTo(x + colW * 2, y + h);
    ctx.stroke();

    this.drawFooterColumn(
      ctx,
      x + 20,
      y + 36,
      colW - 40,
      "Project Details",
      [
        `Project Name: ${input.settings.projectName ?? "Boid Boys"}`,
        `Group Name: ${input.settings.groupName ?? "Boid Boys"}`,
        `Boid Count: ${input.settings.boidCount}`,
      ],
      text,
      subText
    );

    this.drawFooterColumn(
      ctx,
      x + colW + 20,
      y + 36,
      colW - 40,
      "Parameter Settings",
      [
        `Separation Weight: ${input.settings.separationWeight.toFixed(2)}`,
        `Alignment Weight: ${input.settings.alignmentWeight.toFixed(2)}`,
        `Cohesion Weight: ${input.settings.cohesionWeight.toFixed(2)}`,
        `Max Speed: ${input.settings.maxSpeed.toFixed(2)}`,
        `Update Freq: ${input.settings.updateFrequency ?? "N/A"}`,
        `Avg Sim: ${input.metrics.avgSimTime.toFixed(2)} ms`,
        `Avg Render: ${input.metrics.avgRenderTime.toFixed(2)} ms`,
      ],
      text,
      subText
    );

    this.drawFooterColumn(
      ctx,
      x + colW * 2 + 20,
      y + 36,
      colW - 40,
      "Hardware & Software",
      [
        `CPU: ${input.hardware.cpu}`,
        `GPU: ${input.hardware.gpu}`,
        `OS: ${input.hardware.os}`,
        "Platform: Chrome / WebGPU",
      ],
      text,
      subText
    );
  }

  private drawFooterColumn(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    maxWidth: number,
    title: string,
    lines: string[],
    text: string,
    subText: string
  ): void {
    ctx.fillStyle = text;
    ctx.font = "700 28px Arial";
    ctx.fillText(title, x, y);

    ctx.fillStyle = subText;
    ctx.font = "500 22px Arial";
    let yy = y + 42;
    for (const line of lines) {
      const usedLines = this.drawWrappedText(ctx, line, x, yy, maxWidth, 30);
      yy += usedLines * 30 + 8;
    }
  }

  private drawWrappedText(
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    maxWidth: number,
    lineHeight: number
  ): number {
    const words = text.split(" ");
    let line = "";
    let linesDrawn = 0;

    for (let i = 0; i < words.length; i++) {
      const candidate = line ? `${line} ${words[i]}` : words[i];
      const width = ctx.measureText(candidate).width;
      if (width > maxWidth && line) {
        ctx.fillText(line, x, y);
        linesDrawn += 1;
        line = words[i];
        y += lineHeight;
      } else {
        line = candidate;
      }
    }

    if (line) {
      ctx.fillText(line, x, y);
      linesDrawn += 1;
    }

    return linesDrawn;
  }

  private formatDateStamp(date: Date): string {
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, "0");
    const dd = String(date.getDate()).padStart(2, "0");
    return `${yyyy}${mm}${dd}`;
  }

  private formatBoidLabel(count: number): string {
    if (count >= 1000) {
      return `${Math.round(count / 1000)}K`;
    }
    return `${count}`;
  }

  private sanitizeFilePart(input: string): string {
    return input.replace(/[/\\?%*:|"<>]/g, "").trim().replace(/\s+/g, "_");
  }

  private promptBenchmarkNameWithPreview(dateStamp: string, boidLabel: string): Promise<string | null> {
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.style.position = "fixed";
      overlay.style.inset = "0";
      overlay.style.background = "rgba(0,0,0,0.35)";
      overlay.style.display = "flex";
      overlay.style.alignItems = "center";
      overlay.style.justifyContent = "center";
      overlay.style.zIndex = "10000";

      const modal = document.createElement("div");
      modal.style.width = "520px";
      modal.style.background = "#FFFFFF";
      modal.style.border = "1px solid #DADADA";
      modal.style.borderRadius = "12px";
      modal.style.boxShadow = "0 20px 60px rgba(0,0,0,0.2)";
      modal.style.padding = "18px";
      modal.style.fontFamily = "Arial, sans-serif";

      const title = document.createElement("h3");
      title.textContent = "Benchmark Name";
      title.style.margin = "0 0 12px 0";
      title.style.fontSize = "20px";

      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Enter label (e.g. RTX4070_TestA)";
      input.style.width = "100%";
      input.style.padding = "10px";
      input.style.border = "1px solid #CCCCCC";
      input.style.borderRadius = "8px";
      input.style.fontSize = "16px";

      const help = document.createElement("div");
      help.textContent = "Illegal filename characters are removed automatically.";
      help.style.marginTop = "8px";
      help.style.color = "#666";
      help.style.fontSize = "13px";

      const preview = document.createElement("div");
      preview.style.marginTop = "12px";
      preview.style.padding = "10px";
      preview.style.borderRadius = "8px";
      preview.style.background = "#F5F7FA";
      preview.style.fontSize = "14px";
      preview.style.color = "#222";

      const actions = document.createElement("div");
      actions.style.display = "flex";
      actions.style.justifyContent = "flex-end";
      actions.style.gap = "10px";
      actions.style.marginTop = "14px";

      const cancel = document.createElement("button");
      cancel.textContent = "Cancel";
      cancel.style.padding = "8px 12px";
      cancel.style.border = "1px solid #CCC";
      cancel.style.background = "#FFF";
      cancel.style.borderRadius = "8px";
      cancel.style.cursor = "pointer";

      const save = document.createElement("button");
      save.textContent = "Export PNG";
      save.style.padding = "8px 12px";
      save.style.border = "none";
      save.style.background = "#2F80ED";
      save.style.color = "#FFF";
      save.style.borderRadius = "8px";
      save.style.cursor = "pointer";

      const updatePreview = () => {
        const safe = this.sanitizeFilePart(input.value || "Untitled");
        const name = safe || "Untitled";
        preview.textContent = `Filename: ${dateStamp}_${boidLabel}_Boids_${name}.png`;
      };

      const finish = (value: string | null) => {
        window.removeEventListener("keydown", onEsc);
        overlay.remove();
        resolve(value);
      };

      const onEsc = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          finish(null);
        }
      };

      input.addEventListener("input", updatePreview);
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          const safe = this.sanitizeFilePart(input.value || "Untitled") || "Untitled";
          finish(safe);
        }
      });

      cancel.addEventListener("click", () => finish(null));
      save.addEventListener("click", () => {
        const safe = this.sanitizeFilePart(input.value || "Untitled") || "Untitled";
        finish(safe);
      });

      window.addEventListener("keydown", onEsc);

      updatePreview();
      modal.appendChild(title);
      modal.appendChild(input);
      modal.appendChild(help);
      modal.appendChild(preview);
      actions.appendChild(cancel);
      actions.appendChild(save);
      modal.appendChild(actions);
      overlay.appendChild(modal);
      document.body.appendChild(overlay);
      input.focus();
    });
  }

  private canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Failed to create PNG blob."));
          return;
        }
        resolve(blob);
      }, "image/png");
    });
  }

  private downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  private downloadJson(payload: BenchmarkExportPayload, filename: string): void {
    const jsonText = JSON.stringify(payload, null, 2);
    const blob = new Blob([jsonText], { type: "application/json" });
    this.downloadBlob(blob, filename);
  }
}
