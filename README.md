# jioasebanyan

一个基于多 Agent 的小说模拟系统：
- `master agent` 负责世界初始化、POV 观察生成与回合裁定
- `sub agents` 扮演角色进行思考与行动
- `recorder/reviewer` 在模拟后进入成稿与审核循环
- 内置 Chroma 主观记忆库、JSONL 原文档案、Typer CLI

## 环境要求

- Python 3.11+

## 安装

```bash
python -m pip install -e ".[dev]"
```

## 配置

1. 复制环境变量模板并填写：

```bash
cp .env.example .env
```

2. 在 `.env` 中设置你的 `ANTHROPIC_API_KEY`（不要提交真实 key）。

3. 复制并编辑配置文件：

```bash
cp config.example.yaml config.yaml
```

默认 `provider.type` 为 `anthropic`，可按需改为 OpenAI-compatible。

## 运行模拟

```bash
novel-sim run --scenario scenarios/inn_rainy_night.yaml --reset
```

## 常用命令

查看当前会话状态：

```bash
novel-sim status
```

查看 trace（例如主 Agent 调用）：

```bash
novel-sim trace master
```

## 测试

```bash
pytest
```
