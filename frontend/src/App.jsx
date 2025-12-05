import React, { useEffect, useState, useMemo } from "react";
import "./index.css";

const API_BASE = "http://localhost:5000";

const App = () => {
  const [activeView, setActiveView] = useState("dashboard");
  const [alertFilter, setAlertFilter] = useState("all");
  const [timeFilter, setTimeFilter] = useState("today");

  // ====== DATA STATE ======
  const [stats, setStats] = useState({
    criticalAlerts: 0,
    warningAlerts: 0,
    resolvedToday: 0,
    detectionRate: "0%",
    activeCameras: 0,
    alertBadge: 0,
  });

  const [cameras, setCameras] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [threats, setThreats] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: "grid" },
    { id: "cameras", label: "Live Cameras", icon: "camera" },
    { id: "alerts", label: "Alerts", icon: "alert" },
    { id: "analytics", label: "Analytics", icon: "analytics" },
    { id: "map", label: "Map View", icon: "map" },
  ];

  // ====== FETCH DATA FROM BACKEND ======
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError("");

        const [statsRes, camsRes, alertsRes, threatsRes] = await Promise.all([
          fetch(`${API_BASE}/api/stats`),
          fetch(`${API_BASE}/api/cameras`),
          fetch(`${API_BASE}/api/alerts`),
          fetch(`${API_BASE}/api/threats?range=${timeFilter}`),
        ]);

        if (!statsRes.ok || !camsRes.ok || !alertsRes.ok || !threatsRes.ok) {
          throw new Error("Failed to fetch one or more resources");
        }

        const statsJson = await statsRes.json();
        const camsJson = await camsRes.json();
        const alertsJson = await alertsRes.json();
        const threatsJson = await threatsRes.json();

        setStats((prev) => ({
          ...prev,
          ...statsJson,
          detectionRate:
            typeof statsJson.detectionRate === "number"
              ? `${statsJson.detectionRate.toFixed(1)}%`
              : statsJson.detectionRate ?? prev.detectionRate,
        }));

        setCameras(camsJson || []);
        setAlerts(alertsJson || []);
        setThreats(threatsJson || []);
      } catch (err) {
        console.error(err);
        setError(err.message || "Error loading data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [timeFilter]); // re-fetch threats when timeFilter changes

  // Filter alerts on client
  const filteredAlerts = useMemo(() => {
    if (alertFilter === "all") return alerts;
    return alerts.filter((a) => a.type === alertFilter);
  }, [alerts, alertFilter]);

  return (
    <div className="app-root">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M12 2L2 7L12 12L22 7L12 2Z"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M2 17L12 22L22 17"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M2 12L12 17L22 12"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <div className="logo-text">
              <h1>SafetyVision AI</h1>
              <p className="subtitle">Urban Monitoring System</p>
            </div>
          </div>

          <div className="header-status">
            <div className="status-indicator">
              <span className="status-dot active"></span>
              <span>
                {loading
                  ? "Syncing with backend..."
                  : "All Systems Operational"}
              </span>
            </div>
            <div className="active-cameras">
              <span className="camera-count" id="activeCameras">
                {stats.activeCameras}
              </span>
              <span>Active Cameras</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="main-container">
        {/* Sidebar */}
        <aside className="sidebar">
          <nav className="nav-menu">
            {navItems.map((item) => (
              <button
                key={item.id}
                className={`nav-item ${activeView === item.id ? "active" : ""}`}
                data-view={item.id}
                onClick={() => setActiveView(item.id)}
              >
                {/* icons (same as before) */}
                {item.icon === "grid" && (
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <rect
                      x="3"
                      y="3"
                      width="7"
                      height="7"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="14"
                      y="3"
                      width="7"
                      height="7"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="3"
                      y="14"
                      width="7"
                      height="7"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="14"
                      y="14"
                      width="7"
                      height="7"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                )}
                {item.icon === "camera" && (
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M23 7L16 12L23 17V7Z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <rect
                      x="1"
                      y="5"
                      width="15"
                      height="14"
                      rx="2"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                )}
                {item.icon === "alert" && (
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <line
                      x1="12"
                      y1="9"
                      x2="12"
                      y2="13"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <line
                      x1="12"
                      y1="17"
                      x2="12.01"
                      y2="17"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                )}
                {item.icon === "analytics" && (
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <line
                      x1="12"
                      y1="20"
                      x2="12"
                      y2="10"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <line
                      x1="18"
                      y1="20"
                      x2="18"
                      y2="4"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <line
                      x1="6"
                      y1="20"
                      x2="6"
                      y2="16"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                )}
                {item.icon === "map" && (
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <circle
                      cx="12"
                      cy="10"
                      r="3"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                )}
                <span>{item.label}</span>
                {item.id === "alerts" && (
                  <span className="badge" id="alertBadge">
                    {stats.alertBadge}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {error && (
            <div
              style={{
                marginBottom: 10,
                padding: "8px 10px",
                borderRadius: 10,
                background: "rgba(248,113,113,0.12)",
                border: "1px solid rgba(248,113,113,0.4)",
                fontSize: "0.8rem",
              }}
            >
              Backend error: {error}
            </div>
          )}

          {/* Stats Grid */}
          <section className="stats-grid">
            <div className="stat-card critical">
              <div className="stat-icon">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                  <line
                    x1="12"
                    y1="8"
                    x2="12"
                    y2="12"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                  <line
                    x1="12"
                    y1="16"
                    x2="12.01"
                    y2="16"
                    strokeWidth="2"
                    stroke="currentColor"
                  />
                </svg>
              </div>
              <div className="stat-content">
                <h3 className="stat-value" id="criticalAlerts">
                  {stats.criticalAlerts}
                </h3>
                <p className="stat-label">Critical Alerts</p>
                <span className="stat-change negative">+2 from last hour</span>
              </div>
            </div>

            <div className="stat-card warning">
              <div className="stat-icon">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                </svg>
              </div>
              <div className="stat-content">
                <h3 className="stat-value" id="warningAlerts">
                  {stats.warningAlerts}
                </h3>
                <p className="stat-label">Warnings</p>
                <span className="stat-change">Active monitoring</span>
              </div>
            </div>

            <div className="stat-card success">
              <div className="stat-icon">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M22 11.08V12a10 10 0 11-5.93-9.14"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                  <polyline
                    points="22 4 12 14.01 9 11.01"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                </svg>
              </div>
              <div className="stat-content">
                <h3 className="stat-value" id="resolvedToday">
                  {stats.resolvedToday}
                </h3>
                <p className="stat-label">Resolved Today</p>
                <span className="stat-change positive">+12% efficiency</span>
              </div>
            </div>

            <div className="stat-card info">
              <div className="stat-icon">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                </svg>
              </div>
              <div className="stat-content">
                <h3 className="stat-value" id="detectionRate">
                  {stats.detectionRate}
                </h3>
                <p className="stat-label">Detection Accuracy</p>
                <span className="stat-change positive">AI Performance</span>
              </div>
            </div>
          </section>

          {/* Main Grid */}
          <div className="content-grid">
            {/* Live Camera Feed */}
            <section className="camera-grid-section">
              <div className="section-header">
                <h2>Live Camera Feeds</h2>
                <div className="section-actions">
                  <button className="btn-secondary" id="gridViewBtn">
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <rect
                        x="3"
                        y="3"
                        width="7"
                        height="7"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <rect
                        x="14"
                        y="3"
                        width="7"
                        height="7"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <rect
                        x="3"
                        y="14"
                        width="7"
                        height="7"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <rect
                        x="14"
                        y="14"
                        width="7"
                        height="7"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="camera-grid" id="cameraGrid">
                {cameras.length === 0 && !loading && (
                  <p style={{ fontSize: "0.8rem", color: "#6b7280" }}>
                    No cameras loaded.
                  </p>
                )}
                {cameras.map((cam) => (
                  <div key={cam.id} className="camera-tile">
                    <div className="camera-tile-header">
                      {cam.name || "Camera"} · {cam.location || "Unknown"}
                    </div>
                    {cam.thumbnail ? (
                      <img
                        src={cam.thumbnail}
                        alt={cam.name}
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                        }}
                      />
                    ) : (
                      <div
                        style={{
                          height: 140,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.8rem",
                          color: "#6b7280",
                        }}
                      >
                        Stream preview
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* Alert Feed */}
            <section className="alert-feed-section">
              <div className="section-header">
                <h2>Recent Alerts</h2>
                <div className="filter-buttons">
                  <button
                    className={`filter-btn ${
                      alertFilter === "all" ? "active" : ""
                    }`}
                    data-filter="all"
                    onClick={() => setAlertFilter("all")}
                  >
                    All
                  </button>
                  <button
                    className={`filter-btn ${
                      alertFilter === "critical" ? "active" : ""
                    }`}
                    data-filter="critical"
                    onClick={() => setAlertFilter("critical")}
                  >
                    Critical
                  </button>
                  <button
                    className={`filter-btn ${
                      alertFilter === "warning" ? "active" : ""
                    }`}
                    data-filter="warning"
                    onClick={() => setAlertFilter("warning")}
                  >
                    Warning
                  </button>
                </div>
              </div>
              <div className="alert-feed" id="alertFeed">
                {filteredAlerts.length === 0 && !loading && (
                  <p style={{ fontSize: "0.8rem", color: "#6b7280" }}>
                    No alerts for this filter.
                  </p>
                )}
                {filteredAlerts.map((alert) => (
                  <div key={alert.id} className="alert-item">
                    <div className="alert-item-header">
                      <span className="alert-title">{alert.title}</span>
                      <span
                        className={
                          "alert-badge " +
                          (alert.type === "warning"
                            ? "warning"
                            : alert.type === "info"
                            ? "info"
                            : "")
                        }
                      >
                        {alert.type?.toUpperCase()}
                      </span>
                    </div>
                    <div className="alert-meta">
                      <span>{alert.camera || "Unknown camera"}</span>
                      <span>
                        {alert.time} · {alert.zone || "Unknown zone"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Threat Detection Stats */}
          <section className="threat-stats-section">
            <div className="section-header">
              <h2>Threat Detection Overview</h2>
              <select
                className="time-filter"
                id="timeFilter"
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
              >
                <option value="today">Today</option>
                <option value="week">This Week</option>
                <option value="month">This Month</option>
              </select>
            </div>
            <div className="threat-grid" id="threatGrid">
              {threats.length === 0 && !loading && (
                <p style={{ fontSize: "0.8rem", color: "#6b7280" }}>
                  No threat stats available.
                </p>
              )}
              {threats.map((th) => (
                <div key={th.id} className="threat-card">
                  <span className="threat-label">{th.label}</span>
                  <span className="threat-value">
                    {typeof th.value === "number" ? th.value : "--"}
                  </span>
                  <span className="threat-trend">{th.trend}</span>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};

export default App;
