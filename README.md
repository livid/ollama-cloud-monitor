# Ollama Cloud Performance Monitor

A Flask dashboard and Click CLI that benchmarks Ollama Cloud models every 5 minutes and stores server-reported output-token throughput in an RRDtool database.

## CLI

Run commands with the Python 3.14 virtual environment:

```bash
cd /home/dev/ollama-cloud-monitor
.venv/bin/python monitor.py --help
.venv/bin/python monitor.py probe
.venv/bin/python monitor.py graph --period 24h --output /tmp/graph.png
.venv/bin/python monitor.py status
```

The API key is loaded from `/etc/ollama-cloud-monitor.env`; it is not stored in the source tree or crontab. The web process runs under systemd as `ollama-cloud-monitor.service`. A user crontab executes `bin/scheduled-probe` every 5 minutes.

Throughput uses Ollama's `eval_count` divided by `eval_duration` when available. Ollama Cloud currently supplies `total_duration` instead, so the monitor uses server-side end-to-end duration (network time is excluded).

## Per-model charts

Each dashboard card links to `/model/<model-id>`, which shows 24-hour, 7-day, and 30-day RRDtool graphs containing only that model.

## Model information

Model detail pages display architecture, parameter count, quantization, context window, and capabilities from Ollama's authenticated `/api/show` endpoint. Only these basic non-sensitive fields are cached in `data/model_info.json`; scheduled probes refresh metadata when the cache is more than 24 hours old. Run `.venv/bin/python monitor.py refresh-info` for an immediate refresh.
