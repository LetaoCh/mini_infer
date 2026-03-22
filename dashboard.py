STATS_DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini Infer Stats</title>
  <style>
    :root {
      --bg: #f5efe3;
      --panel: #fffaf0;
      --ink: #1d2b2a;
      --muted: #6d7b78;
      --accent: #b85c38;
      --accent-soft: #e8c9b8;
      --line: #d9c8b8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff7ea 0, transparent 28%),
        linear-gradient(135deg, #f6efe4, #ece0cf);
      color: var(--ink);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      margin-bottom: 18px;
      padding: 20px 22px;
      border: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.86);
      backdrop-filter: blur(10px);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 32px;
    }
    .sub {
      color: var(--muted);
      font-size: 15px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .card {
      padding: 14px;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .label {
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .value {
      margin-top: 8px;
      font-size: 30px;
      font-weight: bold;
    }
    .small {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
    .panel {
      padding: 16px;
      border: 1px solid var(--line);
      background: var(--panel);
      margin-bottom: 18px;
    }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 20px;
    }
    .bars {
      display: grid;
      gap: 12px;
    }
    .bar-row {
      display: grid;
      gap: 6px;
    }
    .bar-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }
    .track {
      overflow: hidden;
      height: 12px;
      background: #eadbcf;
      border-radius: 999px;
    }
    .fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #d98e5f);
      border-radius: 999px;
      transition: width 0.2s ease;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 8px;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--ink);
      font-size: 12px;
    }
    .mono {
      font-family: "SFMono-Regular", ui-monospace, Menlo, monospace;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Mini Infer Dashboard</h1>
      <div class="sub" id="headline">Loading...</div>
    </div>

    <div class="grid" id="cards"></div>

    <div class="panel">
      <h2>Queue View</h2>
      <div class="bars" id="bars"></div>
    </div>

    <div class="panel">
      <h2>Active Requests</h2>
      <table>
        <thead>
          <tr>
            <th>Request</th>
            <th>State</th>
            <th>Progress</th>
            <th>Prompt</th>
          </tr>
        </thead>
        <tbody id="active-body">
          <tr><td colspan="4" class="small">No active requests.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    function pct(numerator, denominator) {
      if (!denominator) return 0;
      return Math.max(0, Math.min(100, (numerator / denominator) * 100));
    }

    function card(label, value, small) {
      return `
        <div class="card">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
          <div class="small">${small || ""}</div>
        </div>
      `;
    }

    function barRow(label, value, maxValue) {
      const percent = pct(value, maxValue);
      return `
        <div class="bar-row">
          <div class="bar-head">
            <span>${label}</span>
            <span class="mono">${value} / ${maxValue}</span>
          </div>
          <div class="track"><div class="fill" style="width:${percent}%"></div></div>
        </div>
      `;
    }

    function render(data) {
      document.getElementById("headline").textContent =
        `${data.model_name} on ${data.device} | profile=${data.server_profile} | prefill=${data.prefill_mode} decode=${data.decode_mode} | uptime=${data.uptime_sec}s`;

      document.getElementById("cards").innerHTML = [
        card("Submitted", data.submitted_total, "total requests seen"),
        card("Completed", data.completed_total, "finished requests"),
        card("Rejected", data.rejected_total, "queue full"),
        card("Profile", data.server_profile, `tick log every ${data.tick_log_every}`),
        card("Active", data.active_requests, `batch size ${data.batch_size}`),
        card("Pending", data.pending_queue, `capacity ${data.queue_capacity}`),
        card("Ticks", data.decode_ticks_total, "scheduler steps"),
      ].join("");

      document.getElementById("bars").innerHTML = [
        barRow("Queue Occupancy", data.pending_queue, data.queue_capacity),
        barRow("Active Slots", data.active_requests, data.batch_size),
      ].join("");

      const rows = data.active_details.map((item) => `
        <tr>
          <td class="mono">${item.request_id}</td>
          <td><span class="pill">${item.state}</span></td>
          <td class="mono">${item.generated_tokens} / ${item.max_new_tokens}</td>
          <td>${item.prompt_preview}</td>
        </tr>
      `);

      document.getElementById("active-body").innerHTML =
        rows.length > 0
          ? rows.join("")
          : '<tr><td colspan="4" class="small">No active requests.</td></tr>';
    }

    async function refresh() {
      const response = await fetch("/stats.json");
      const data = await response.json();
      render(data);
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
