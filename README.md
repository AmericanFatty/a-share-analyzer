# A股买入持有收益分析工具

一个基于 Streamlit 的可视化工具，用于分析：

- 指定股票在某个买入日后持有 `5/20/120` 个交易日的收益率
- 截至今天（最新交易日）的收益率
- 与上证指数同区间对比（大盘收益、超额收益、是否跑赢）
- Excel 批量导入后，对全部关注股票进行表现统计和可视化

## 功能概览

- 单只股票分析（核心仅两个输入）：
  - `股票名称/股票代码`
  - `买入日期`
- Excel 批量导入：
  - 自动识别 `买入日期` + `股票名称` 或 `股票代码`
  - 透传 `板块` 与 `备注` 到最终导出文档（板块首列、备注末列）
- 可视化看板：
  - 周期收益卡片
  - 绝对收益与超额收益分布、Top/Bottom 排名
  - 总体胜率、平均收益、明细表下载

## 本地运行

1. 创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. 启动应用：

```bash
.venv/bin/streamlit run app.py
```

## iPhone 使用（外出可访问）

### 1) 页面适配

- 应用内侧边栏提供 `移动端模式（iPhone推荐）` 开关
- 打开后会使用更紧凑的布局，适合手机屏幕查看

### 2) 外网访问方式

仅在本机运行时，默认只能你本机访问。若要在外面用 iPhone 打开，需要把应用发布到公网。

推荐做法（长期稳定）：部署到云平台（如 Streamlit Community Cloud / Render / Railway），得到一个 HTTPS 链接后，iPhone 直接访问即可。

临时做法（短期）：在家里电脑运行并使用内网穿透工具生成临时公网链接（例如 cloudflared / ngrok）。

## Streamlit Cloud 部署步骤

1. 在 GitHub 新建仓库（例如 `a-share-analyzer`）。
2. 本地推送代码到 GitHub：

```bash
cd /Users/lixiaoxuan/Desktop/back
git init
git add .
git commit -m "init: a-share analyzer"
git branch -M main
git remote add origin https://github.com/<你的用户名>/a-share-analyzer.git
git push -u origin main
```

3. 打开 `https://share.streamlit.io/`，登录后点击 `Create app`。
4. 选择你的仓库、分支 `main`、入口文件 `app.py`。
5. 在 `Advanced settings` 里把 Python 版本选成 `3.11`（推荐）。
6. 点击 `Deploy`，等待构建完成后会得到 `https://xxx.streamlit.app` 链接。

之后你在 iPhone 上直接打开这个链接即可使用。

## Excel 列要求

文件中需要包含：

- `买入日期`
- `股票名称` 或 `股票代码`

可选列：

- `板块`
- `备注`

示例：

| 板块 | 股票名称 | 买入日期 | 备注 |
| --- | --- | --- | --- |
| 白酒 | 贵州茅台 | 2024-01-15 | 核心仓位 |
| 新能源 | 宁德时代 | 2024-06-03 | 成长跟踪 |

## 计算口径说明

- 买入价：买入日（若休市则顺延到下一个交易日）的前复权收盘价
- `5/20/120` 日收益：买入日后第 `5/20/120` 个交易日收盘价相对买入价的涨跌幅
- 截至今天收益：最新交易日收盘价相对买入价的涨跌幅
- 大盘比较：以上周期同步计算上证指数收益，并给出超额收益（股票-上证）与跑赢判断
- 导出结果：百分比字段会带 `%` 符号
