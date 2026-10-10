# CodeGameEngine — TORCS 几何版本

Three.js 赛车与 14 类语义流导出。此版本使用 TORCS 的 E-Track 3 赛道、car1-stock1 车体及原始纹理；赛道起伏、道路边界和车体轮廓来自原始几何。车辆由本项目的街机动力学和自动驾驶控制，未播放或复制训练集的语义帧。

```sh
npm install
npm run dev
```

打开 http://127.0.0.1:5191 。W/A/S/D 或方向键驾驶，空格手刹，R 重置；页面可切换自动驾驶、录制和实时语义流。录制输出位于 `exports/`，默认 719 帧、24 fps、1344×768，约 30 秒。

- `public/palette.json`：与 TORCS 数据一致的 14 类 RGB 色表。
- `src/native-assets.js`：原始资产加载与语义材质。透明纹理仅作二值裁切；语义输出不混色、不抗锯齿。
- `src/native-world.js`：原始路面高度采样及闭合道路中心线。
- `src/main.js`：驾驶、相机、对手车和逐帧导出。

原始资产来自 https://github.com/jzbontar/torcs 的 `data/tracks/road/e-track-3` 和 `data/cars/models/car1-stock1`。模型转为 JSON；SGI RGB 纹理转为 PNG。赛道作者 Eric Espie、Bernhard Wymann，车辆作者 Bernhard Wymann。艺术资产及其格式转换结果遵循 [Free Art License](https://artlibre.org/licence/lal/en/)；原始声明保存在 `licenses/`。源代码与艺术资产的许可应分别保留。

在线 H3 render 的配置、权重清单和启动方式见本模块顶层 README.md。点击语义流按钮后，前端通过 /runtime 代理连接常驻 GPU worker；离线录制导出入口仍保留。
