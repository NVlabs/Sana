import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { t } from "./studio.js";

function timingWidget(node) {
  if (!node.solTiming) {
    const element = document.createElement("pre");
    element.style.cssText = "white-space:pre-wrap;font:13px/18px monospace;margin:0;padding:8px;box-sizing:border-box;overflow:auto;color:#ddd";
    node.solTiming = node.addDOMWidget("timing", "pre", element, {
      serialize: false, getMinHeight: () => 64, getMaxHeight: () => 64,
    });
    node.setSize([node.size[0], Math.max(node.size[1], 420)]);
  }
  return node.solTiming.element;
}

app.registerExtension({
  name: "SolEngine.Timing",
  setup() {
    // Live elapsed time while a Sol-H3 node runs; the final value comes from the engine report.
    let timer = null;
    api.addEventListener("executing", ({ detail }) => {
      clearInterval(timer);
      const id = detail?.node ?? detail;
      const node = id != null ? app.graph.getNodeById(Number(id)) : null;
      if (node?.comfyClass !== "SolH3Generate") return;
      const element = timingWidget(node), started = performance.now();
      timer = setInterval(() => {
        element.textContent = `${t("Generating…", "生成中…")} ${((performance.now() - started) / 1000).toFixed(1)} s`;
        node.setDirtyCanvas(true, false);
      }, 100);
    });
    api.addEventListener("execution_error", () => clearInterval(timer));
    api.addEventListener("execution_interrupted", () => clearInterval(timer));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SolH3Generate") return;
    const previous = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      previous?.apply(this, arguments);
      if (!message?.text?.[0]) return;
      const data = JSON.parse(message.text[0]);
      timingWidget(this).textContent =
        `${t("Request → MP4", "请求 → MP4")}: ${data.e2e_s.toFixed(2)} s\n` +
        (data.cold_start ? `${t("Startup + warmup", "启动 + 预热")}: ${data.startup_and_warmup_s.toFixed(2)} s` : t("Model already loaded", "模型已加载"));
      this.setDirtyCanvas(true, true);
    };
  },
});
