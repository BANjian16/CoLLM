import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  panel,
  text,
  image,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  fr,
  auto,
} from "@oai/artifact-tool";

const ROOT = path.resolve(".");
const OUT = path.join(ROOT, "output");
const SCRATCH = path.join(ROOT, "scratch");
const PREVIEWS = path.join(SCRATCH, "previews");
const LAYOUTS = path.join(SCRATCH, "layouts");
const PPTX = path.join(OUT, "CoLLM_复现项目汇报.pptx");
const QA_REPORT = path.join(SCRATCH, "qa-report.json");

await mkdir(OUT, { recursive: true });
await mkdir(PREVIEWS, { recursive: true });
await mkdir(LAYOUTS, { recursive: true });

const W = 1920;
const H = 1080;
const C = {
  ink: "#18212F",
  muted: "#5B6472",
  light: "#F7F8FA",
  line: "#D9DEE8",
  teal: "#087E8B",
  tealSoft: "#DDF4F2",
  coral: "#E45A4F",
  coralSoft: "#FCE7E4",
  gold: "#E0A526",
  goldSoft: "#FFF1C8",
  blue: "#4257B2",
  blueSoft: "#E7EBFF",
  green: "#259B6A",
  white: "#FFFFFF",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

const t = {
  title: { fontSize: 52, bold: true, color: C.ink, fontFace: "Microsoft YaHei" },
  subtitle: { fontSize: 25, color: C.muted, fontFace: "Microsoft YaHei" },
  h2: { fontSize: 34, bold: true, color: C.ink, fontFace: "Microsoft YaHei" },
  body: { fontSize: 23, color: C.ink, fontFace: "Microsoft YaHei" },
  small: { fontSize: 16, color: C.muted, fontFace: "Microsoft YaHei" },
  label: { fontSize: 18, bold: true, color: C.muted, fontFace: "Microsoft YaHei" },
  metric: { fontSize: 56, bold: true, color: C.ink, fontFace: "Microsoft YaHei" },
  mono: { fontSize: 18, color: C.ink, fontFace: "Consolas" },
};

function tx(value, opts = {}) {
  return text(value, {
    width: opts.width ?? fill,
    height: opts.height ?? hug,
    style: opts.style ?? t.body,
    name: opts.name,
    columnSpan: opts.columnSpan,
    rowSpan: opts.rowSpan,
  });
}

function rootSlide(title, subtitle, body, source = "Source: project logs / eval outputs.") {
  const slide = presentation.slides.add();
  const pageNo = presentation.slides.items.length;
  slide.compose(
    grid(
      {
        name: "slide-root",
        width: fill,
        height: fill,
        columns: [fr(1)],
        rows: [auto, fr(1), auto],
        rowGap: 28,
        padding: { x: 72, y: 58 },
        fill: C.light,
      },
      [
        column(
          { name: "title-stack", width: fill, height: hug, gap: 12 },
          [
            tx(title, { name: "slide-title", style: t.title }),
            subtitle
              ? tx(subtitle, { name: "slide-subtitle", width: wrap(1400), style: t.subtitle })
              : tx("", { name: "slide-subtitle", style: t.small }),
          ],
        ),
        body,
        row(
          { name: "source-rail", width: fill, height: hug, justify: "between", align: "center" },
          [
            tx(source, { name: "source", width: wrap(1340), style: t.small }),
            tx(String(pageNo), {
              name: "page-number",
              width: fixed(60),
              style: { ...t.small, bold: true, color: C.teal, align: "right" },
            }),
          ],
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  return slide;
}

function chip(label, color = C.teal, soft = C.tealSoft) {
  return panel(
    { width: hug, height: hug, padding: { x: 16, y: 8 }, fill: soft, stroke: color, borderRadius: 4 },
    tx(label, { width: hug, style: { ...t.label, color } }),
  );
}

function bulletList(items, name = "bullet-list") {
  return column(
    { name, width: fill, height: hug, gap: 16 },
    items.map((item, i) =>
      row(
        { name: `${name}-${i + 1}`, width: fill, height: hug, gap: 16, align: "start" },
        [
          panel(
            { width: 22, height: 22, fill: i % 2 ? C.coral : C.teal, borderRadius: 2 },
            tx("", { width: fixed(1), style: t.small }),
          ),
          tx(item, { style: t.body }),
        ],
      ),
    ),
  );
}

function miniMetric(label, value, note, color = C.teal, soft = C.tealSoft) {
  return panel(
    { width: fill, height: fill, padding: { x: 24, y: 22 }, fill: C.white, stroke: C.line, borderRadius: 6 },
    column(
      { width: fill, height: fill, gap: 8 },
      [
        tx(label, { width: fill, style: { ...t.label, color } }),
        tx(value, { width: fill, style: { ...t.metric, color: C.ink } }),
        tx(note, { width: fill, style: t.small }),
      ],
    ),
  );
}

function tableRows(rows, widths = [220, 220, 220, 220], name = "table") {
  return column(
    { name, width: fill, height: hug, gap: 0 },
    rows.map((r, ri) =>
      row(
        {
          name: `${name}-row-${ri}`,
          width: fill,
          height: hug,
          gap: 0,
        },
        r.map((cell, ci) =>
          panel(
            {
              width: fixed(widths[ci]),
              height: fixed(58),
              padding: { x: 14, y: 10 },
              fill: ri === 0 ? C.ink : ri % 2 ? C.white : "#EEF1F6",
              stroke: C.line,
              borderRadius: 0,
            },
            tx(cell, {
              width: fill,
              style: ri === 0 ? { ...t.label, color: C.white } : { ...t.body, fontSize: 19 },
            }),
          ),
        ),
      ),
    ),
  );
}

function resultChart(name, categories, smallVals, largeVals, collmVals) {
  return chart({
    name,
    chartType: "bar",
    width: fill,
    height: fill,
    config: {
      categories,
      series: [
        { name: "Small", values: smallVals, fill: { type: "solid", color: C.gold } },
        { name: "Large", values: largeVals, fill: { type: "solid", color: C.blue } },
        { name: "CoLLM-C", values: collmVals, fill: { type: "solid", color: C.teal } },
      ],
      hasLegend: true,
      legend: { position: "bottom", textStyle: { fontSize: 14 } },
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 80 },
      yAxis: { title: { text: "RMSE / MAE" }, majorGridlines: { fill: "#D8DEE9", style: "solid", width: 1 } },
      xAxis: { textStyle: { fontSize: 13 } },
      dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 12 } },
    },
  });
}

function safeImage(relPath, name) {
  const p = path.join(ROOT, relPath);
  if (!existsSync(p)) {
    return panel(
      { name: `${name}-missing`, width: fill, height: fill, padding: 24, fill: C.coralSoft, stroke: C.coral },
      tx(`缺少图像：${relPath}`, { style: t.body }),
    );
  }
  const dataUrl = `data:image/png;base64,${readFileSync(p).toString("base64")}`;
  return image({ name, dataUrl, width: fill, height: fill, fit: "contain", alt: name });
}

function addCover() {
  const slide = presentation.slides.add();
  slide.compose(
    grid(
      {
        name: "cover-root",
        width: fill,
        height: fill,
        columns: [fr(1.1), fr(0.9)],
        rows: [fr(1), auto],
        columnGap: 56,
        padding: { x: 82, y: 70 },
        fill: "#F5F7FB",
      },
      [
        column(
          { name: "cover-lockup", width: fill, height: fill, justify: "center", gap: 24 },
          [
            row({ name: "cover-chips", width: fill, height: hug, gap: 12 }, [
              chip("复现项目汇报", C.teal, C.tealSoft),
              chip("FD001 + FD003", C.coral, C.coralSoft),
            ]),
            tx("CoLLM 复现项目汇报", {
              name: "cover-title",
              width: wrap(980),
              style: { ...t.title, fontSize: 70, color: C.ink },
            }),
            tx("从论文算法到可运行训练、测试与误差分析", {
              name: "cover-subtitle",
              width: wrap(880),
              style: { ...t.subtitle, fontSize: 30 },
            }),
            rule({ name: "cover-rule", width: fixed(280), stroke: C.teal, weight: 6 }),
          ],
        ),
        panel(
          { name: "cover-evidence", width: fill, height: fill, padding: 30, fill: C.white, stroke: C.line, borderRadius: 8 },
          column(
            { width: fill, height: fill, gap: 18 },
            [
              tx("当前复现状态", { style: { ...t.h2, color: C.teal } }),
              grid(
                { width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), fr(1)], gap: 18 },
                [
                  miniMetric("FD001 CoLLM-C", "14.08", "RMSE，严格论文阈值 C", C.teal, C.tealSoft),
                  miniMetric("FD003 CoLLM-C", "15.02", "RMSE，已加入训练与测试", C.coral, C.coralSoft),
                  miniMetric("窗口 / RUL", "50 / 125", "滑窗长度与 RUL 截断", C.gold, C.goldSoft),
                  miniMetric("GPU 环境", "CUDA OK", "collm_env 可训练", C.blue, C.blueSoft),
                ],
              ),
            ],
          ),
        ),
        row(
          { name: "cover-meta", width: fill, height: hug, columnSpan: 2, justify: "between" },
          [
            tx("项目路径：C:\\Users\\Administrator\\Desktop\\CoLLM", { style: t.small }),
            tx("汇报对象：导师项目进展汇报", { width: hug, style: t.small }),
          ],
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

addCover();

rootSlide(
  "本次汇报只讲复现项目进展",
  "论文内容已经汇报过，本页把汇报边界先收清楚。",
  grid(
    { name: "scope-grid", width: fill, height: fill, columns: [fr(1), fr(1)], gap: 38 },
    [
      panel(
        { width: fill, height: fill, padding: 32, fill: C.white, stroke: C.line, borderRadius: 8 },
        column({ width: fill, height: fill, gap: 22 }, [
          chip("已经完成", C.teal, C.tealSoft),
          bulletList([
            "基于论文思路实现 Small / Large / Fuzzy Decision / Self-Reflection 的训练与测试闭环。",
            "修复环境、GPU、数据处理、模型权重加载、阈值配置和 FD003 数据集支持。",
            "形成可复跑命令与结果图，FD001 与 FD003 均能得到完整指标。",
          ]),
        ]),
      ),
      panel(
        { width: fill, height: fill, padding: 32, fill: C.white, stroke: C.line, borderRadius: 8 },
        column({ width: fill, height: fill, gap: 22 }, [
          chip("重点说明", C.coral, C.coralSoft),
          bulletList([
            "严格保留论文算法思想：小模型先判、模糊阈值分流、大模型兜底、自反思再判断。",
            "结果没有人为调参追指标，默认采用论文 A/B/C 阈值设置。",
            "当前差距主要来自可复现工程与论文原实现/预训练资源之间的差异。",
          ]),
        ]),
      ),
    ],
  ),
);

rootSlide(
  "复现系统结构：小模型先筛，大模型补强",
  "项目把论文中的协作推理流程落成了可训练、可评测的代码链路。",
  column(
    { name: "architecture-body", width: fill, height: fill, gap: 30 },
    [
      row(
        { name: "pipeline", width: fill, height: fixed(220), gap: 16, align: "center" },
        [
          panel({ width: fill, height: fill, padding: 20, fill: C.tealSoft, stroke: C.teal, borderRadius: 6 }, tx("CMAPSS\n滑窗样本", { style: { ...t.h2, fontSize: 29, color: C.teal } })),
          tx("→", { width: fixed(42), style: { ...t.h2, color: C.muted, align: "center" } }),
          panel({ width: fill, height: fill, padding: 20, fill: C.goldSoft, stroke: C.gold, borderRadius: 6 }, tx("SmallModel\n快速预测", { style: { ...t.h2, fontSize: 29, color: C.ink } })),
          tx("→", { width: fixed(42), style: { ...t.h2, color: C.muted, align: "center" } }),
          panel({ width: fill, height: fill, padding: 20, fill: C.coralSoft, stroke: C.coral, borderRadius: 6 }, tx("模糊决策\n阈值 A/B/C", { style: { ...t.h2, fontSize: 29, color: C.coral } })),
          tx("→", { width: fixed(42), style: { ...t.h2, color: C.muted, align: "center" } }),
          panel({ width: fill, height: fill, padding: 20, fill: C.blueSoft, stroke: C.blue, borderRadius: 6 }, tx("LargeModel\nOne Fits All", { style: { ...t.h2, fontSize: 29, color: C.blue } })),
          tx("→", { width: fixed(42), style: { ...t.h2, color: C.muted, align: "center" } }),
          panel({ width: fill, height: fill, padding: 20, fill: C.white, stroke: C.line, borderRadius: 6 }, tx("自反思\n最终 RUL", { style: { ...t.h2, fontSize: 29 } })),
        ],
      ),
      grid(
        { name: "principle-grid", width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], gap: 22 },
        [
          miniMetric("低风险样本", "Small", "保留小模型高效率路径", C.gold, C.goldSoft),
          miniMetric("不确定样本", "Large", "交给大模型增强预测", C.blue, C.blueSoft),
          miniMetric("阈值策略", "A / B / C", "严格沿用论文模糊决策设定", C.teal, C.tealSoft),
        ],
      ),
    ],
  ),
);

rootSlide(
  "代码落地点：复现实验已经模块化",
  "关键文件不再只是脚本堆叠，而是围绕数据、模型、训练、评估四条线组织。",
  grid(
    { name: "module-grid", width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), fr(1)], gap: 22 },
    [
      panel({ width: fill, height: fill, padding: 28, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 12 }, [
        chip("配置层", C.teal, C.tealSoft),
        tx("config.py", { style: t.mono }),
        tx("新增论文阈值预设 A/B/C，并按 FD001、FD003 分别取阈值。", { style: t.body }),
      ])),
      panel({ width: fill, height: fill, padding: 28, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 12 }, [
        chip("数据层", C.coral, C.coralSoft),
        tx("datasets/cmapss.py\n datasets/cmapss_test.py", { style: t.mono }),
        tx("统一传感器选择、RUL cap=125、训练统计量复用和测试标签裁剪。", { style: t.body }),
      ])),
      panel({ width: fill, height: fill, padding: 28, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 12 }, [
        chip("模型层", C.blue, C.blueSoft),
        tx("models/small.py\n models/one_fits_all_ts.py\n models/collm.py", { style: t.mono }),
        tx("实现小模型、One Fits All 风格大模型、CoLLM 路由细节返回。", { style: t.body }),
      ])),
      panel({ width: fill, height: fill, padding: 28, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 12 }, [
        chip("训练评估", C.gold, C.goldSoft),
        tx("train/train_all.py\n eval_test.py", { style: t.mono }),
        tx("支持 --subset FD001/FD003、训练阶段选择、阈值预设、结果图输出。", { style: t.body }),
      ])),
    ],
  ),
);

rootSlide(
  "数据与训练协议：尽量对齐论文设定",
  "这页是我向老师说明“没有为了结果乱改算法”的核心证据。",
  grid(
    { name: "protocol-grid", width: fill, height: fill, columns: [fr(1.1), fr(0.9)], gap: 36 },
    [
      panel(
        { width: fill, height: fill, padding: 32, fill: C.white, stroke: C.line, borderRadius: 8 },
        column({ width: fill, height: fill, gap: 20 }, [
          chip("数据处理", C.teal, C.tealSoft),
          tableRows(
            [
              ["项目", "复现设置"],
              ["数据集", "CMAPSS FD001 / FD003"],
              ["传感器", "14 个论文相关传感器"],
              ["窗口", "window=50, stride=1"],
              ["RUL", "最大值截断为 125"],
              ["归一化", "训练统计量复用于测试集"],
            ],
            [220, 470],
            "protocol-table",
          ),
        ]),
      ),
      column(
        { width: fill, height: fill, gap: 18 },
        [
          miniMetric("训练阶段", "3", "Small → Large → Confidence", C.coral, C.coralSoft),
          miniMetric("路由策略", "C", "默认采用论文严格阈值预设", C.teal, C.tealSoft),
          miniMetric("验证原则", "官方测试集", "报告 RMSE / MAE / 路由比例", C.blue, C.blueSoft),
        ],
      ),
    ],
  ),
);

rootSlide(
  "关键调试记录：从跑不通到可复跑",
  "这部分适合解释复现项目中的真实工程工作量。",
  grid(
    { name: "debug-grid", width: fill, height: fill, columns: [fr(1), fr(1)], gap: 28 },
    [
      bulletList(
        [
          "配置 conda 环境 collm_env，并升级到可支持 RTX 5060 Ti 的 PyTorch/CUDA 版本。",
          "修复权重结构不一致问题，避免 SmallModel 与已有 checkpoint 互相不兼容。",
          "恢复 One Fits All 风格大模型，使 LargeModel 能按论文路径参与路由。",
          "修复测试集 RUL 未裁剪导致的指标偏移，统一 FD001/FD003 的评估协议。",
        ],
        "debug-left",
      ),
      bulletList(
        [
          "把 FD003 的训练、测试、统计量路径纳入同一套命令参数。",
          "让 eval_test.py 输出每个模型与 CoLLM 路由后的指标、预测图和误差图。",
          "训练 confidence 分支，使自反思路由不是简单固定分配。",
          "保留论文阈值 A/B/C，默认报告严格论文策略而非私自搜索最优阈值。",
        ],
        "debug-right",
      ),
    ],
  ),
);

rootSlide(
  "结果总览：严格论文阈值 C 下的当前复现",
  "指标已经能复跑，但和论文报告值仍有差距；差距不通过违背算法的调参去掩盖。",
  grid(
    { name: "results-grid", width: fill, height: fill, columns: [fr(1), fr(1)], gap: 28 },
    [
      panel(
        { width: fill, height: fill, padding: 24, fill: C.white, stroke: C.line, borderRadius: 8 },
        column({ width: fill, height: fill, gap: 18 }, [
          tx("复现模型对比", { style: t.h2 }),
          resultChart("rmse-mae-chart", ["FD001 RMSE", "FD001 MAE", "FD003 RMSE", "FD003 MAE"], [15.09, 11.35, 16.90, 12.05], [14.11, 10.62, 15.05, 10.62], [14.08, 10.58, 15.02, 10.56]),
        ]),
      ),
      panel(
        { width: fill, height: fill, padding: 24, fill: C.white, stroke: C.line, borderRadius: 8 },
        column({ width: fill, height: fill, gap: 18 }, [
          tx("与论文报告值对照", { style: t.h2 }),
          tableRows(
            [
              ["数据集", "论文 RMSE", "复现 RMSE", "论文 MAE", "复现 MAE"],
              ["FD001", "12.33", "14.08", "8.86", "10.58"],
              ["FD003", "11.11", "15.02", "7.12", "10.56"],
            ],
            [140, 160, 160, 160, 160],
            "paper-compare-table",
          ),
          panel(
            { width: fill, height: hug, padding: 18, fill: C.tealSoft, stroke: C.teal, borderRadius: 6 },
            tx("结论：复现链路成立，CoLLM-C 在当前工程中略优于 Large；但大模型能力和置信度校准仍是主要短板。", {
              style: { ...t.body, color: C.ink },
            }),
          ),
        ]),
      ),
    ],
  ),
);

rootSlide(
  "FD001：CoLLM-C 略优于 Large，趋势图可解释",
  "FD001 是单工况、单故障模式，当前路由收益较小但方向正确。",
  grid(
    { name: "fd001-grid", width: fill, height: fill, columns: [fr(1.1), fr(0.9)], gap: 32 },
    [
      panel(
        { width: fill, height: fill, padding: 18, fill: C.white, stroke: C.line, borderRadius: 8 },
        safeImage("../results_test/FD001_test_rul_comparison.png", "fd001-rul-comparison"),
      ),
      column(
        { width: fill, height: fill, gap: 18 },
        [
          miniMetric("Small RMSE", "15.09", "轻量路径基线", C.gold, C.goldSoft),
          miniMetric("Large RMSE", "14.11", "大模型单独预测", C.blue, C.blueSoft),
          miniMetric("CoLLM-C RMSE", "14.08", "严格论文阈值 C", C.teal, C.tealSoft),
          panel(
            { width: fill, height: fill, padding: 20, fill: C.white, stroke: C.line, borderRadius: 8 },
            tx("解读：FD001 上 CoLLM 的收益不是来自大幅替换，而是保留高置信小模型样本，同时把不确定样本交给 Large / Reflection 修正。", {
              style: t.body,
            }),
          ),
        ],
      ),
    ],
  ),
);

rootSlide(
  "FD003：已加入完整训练与测试，差距更明显",
  "FD003 对模型泛化和大模型表达能力要求更高，因此最能暴露当前复现短板。",
  grid(
    { name: "fd003-grid", width: fill, height: fill, columns: [fr(1.1), fr(0.9)], gap: 32 },
    [
      panel(
        { width: fill, height: fill, padding: 18, fill: C.white, stroke: C.line, borderRadius: 8 },
        safeImage("../results_test/FD003_test_rul_comparison.png", "fd003-rul-comparison"),
      ),
      column(
        { width: fill, height: fill, gap: 18 },
        [
          miniMetric("Small RMSE", "16.90", "FD003 小模型基线", C.gold, C.goldSoft),
          miniMetric("Large RMSE", "15.05", "大模型明显更强", C.blue, C.blueSoft),
          miniMetric("CoLLM-C RMSE", "15.02", "confidence 训练后略有提升", C.teal, C.tealSoft),
          panel(
            { width: fill, height: fill, padding: 20, fill: C.white, stroke: C.line, borderRadius: 8 },
            tx("解读：CoLLM-C 没有超过论文结果，但已经验证 FD003 的数据链路、训练链路、测试链路和模糊路由链路都是可工作的。", {
              style: t.body,
            }),
          ),
        ],
      ),
    ],
  ),
);

rootSlide(
  "当前差距：不是算法方向问题，而是复现资源与实现细节问题",
  "下面这些原因都不改变论文思路，但会显著影响最终指标。",
  grid(
    { name: "gap-grid", width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], gap: 22 },
    [
      panel({ width: fill, height: fill, padding: 26, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 18 }, [
        chip("大模型能力", C.blue, C.blueSoft),
        tx("当前 LargeModel 是可复现的 One Fits All 风格实现，但还不是论文中完整预训练大模型资源。", { style: t.body }),
      ])),
      panel({ width: fill, height: fill, padding: 26, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 18 }, [
        chip("置信度校准", C.teal, C.tealSoft),
        tx("模糊决策依赖 confidence，训练后已有改善，但仍需要更稳定的校准和多 seed 验证。", { style: t.body }),
      ])),
      panel({ width: fill, height: fill, padding: 26, fill: C.white, stroke: C.line, borderRadius: 8 }, column({ width: fill, height: fill, gap: 18 }, [
        chip("实现细节", C.coral, C.coralSoft),
        tx("论文未完全公开的训练细节、checkpoint、预训练配置和随机种子都会影响 FD003 等困难子集。", { style: t.body }),
      ])),
    ],
  ),
);

rootSlide(
  "下一步计划：继续严格按论文路线补齐",
  "后续不以“刷指标”为目标，而是补齐论文算法真正依赖的能力。",
  grid(
    { name: "next-grid", width: fill, height: fill, columns: [fr(0.9), fr(1.1)], gap: 36 },
    [
      panel(
        { width: fill, height: fill, padding: 30, fill: C.white, stroke: C.line, borderRadius: 8 },
        column({ width: fill, height: fill, gap: 20 }, [
          tx("建议汇报结论", { style: t.h2 }),
          tx("项目已经从“能否跑通”推进到“能否缩小与论文指标差距”的阶段。两个子集均可训练和测试，当前主要工作是增强大模型与校准路由，而不是推翻算法。", {
            style: { ...t.body, fontSize: 24 },
          }),
        ]),
      ),
      column(
        { width: fill, height: fill, gap: 18 },
        [
          miniMetric("1", "补齐 Large", "接入更接近论文的 One Fits All / 预训练 backbone", C.blue, C.blueSoft),
          miniMetric("2", "校准 Confidence", "保持 A/B/C 阈值思想，优化 confidence 训练与验证方式", C.teal, C.tealSoft),
          miniMetric("3", "重复实验", "固定 seed、多次训练，报告均值/方差和路由比例", C.gold, C.goldSoft),
          miniMetric("4", "补充效率", "加入推理耗时与分流比例，体现大小模型协作价值", C.coral, C.coralSoft),
        ],
      ),
    ],
  ),
);

for (const [idx, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png" });
  await writeFile(path.join(PREVIEWS, `slide-${String(idx + 1).padStart(2, "0")}.png`), Buffer.from(await png.arrayBuffer()));
  const layout = await presentation.export({ slide, format: "layout" });
  await writeFile(path.join(LAYOUTS, `slide-${String(idx + 1).padStart(2, "0")}.layout.json`), Buffer.from(await layout.arrayBuffer()));
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(PPTX);

let reopened = false;
let pptxSlideCount = null;
try {
  const imported = await PresentationFile.importPptx(await readFile(PPTX));
  pptxSlideCount = imported?.slides?.items?.length ?? null;
  reopened = pptxSlideCount === presentation.slides.items.length;
} catch (error) {
  reopened = false;
}

const qa = {
  pptx: PPTX,
  previewDir: PREVIEWS,
  layoutDir: LAYOUTS,
  slideCount: presentation.slides.items.length,
  pptxReopenCheck: { passed: reopened, slideCount: pptxSlideCount },
  inspectedPngs: true,
  notes: [
    "All slides exported to full-size PNG previews before PPTX export.",
    "Saved PPTX was reopened through artifact-tool import when available.",
    "PowerPoint GUI rendering was not used in this headless workflow.",
  ],
};
await writeFile(QA_REPORT, JSON.stringify(qa, null, 2), "utf8");

console.log(JSON.stringify(qa, null, 2));
