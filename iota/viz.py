"""
IOTA Visualization Module
========================
Visualization utilities for semantic maps and Soulprints.
"""

import json
import html as html_escape
from pathlib import Path
from typing import Optional

import numpy as np


class AtlasVisualizer:
    """
    Visualize IOTA semantic maps and Soulprints.
    
    Supports:
    - HTML interactive visualizations
    - Static plots (requires matplotlib)
    - JSON exports for custom tooling
    """
    
    def __init__(self, map=None, identity=None):
        self.map = map
        self.identity = identity
    
    def to_html(self, output_path: Optional[str] = None) -> str:
        """
        Generate an interactive HTML visualization.
        
        This creates a standalone HTML file that shows:
        - Concept network (what concepts are connected)
        - Invariant markers (which are stable)
        - Comparison view (if comparing models)
        """
        html = self._generate_html_template()
        # Render the graph in the document so local previews need no CDN or JS.
        pairs = list(self.map.concept_directions.values()) if self.map else []
        height = max(400, len(pairs) * 38 + 50)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {height}" width="100%" role="img" aria-label="Concept pairs and cross-context consistency">']
        for i, pair in enumerate(pairs):
            y = 40 + i * 38
            score = pair.metrics.get("loo_cosine_mean", 0)
            a = html_escape.escape(pair.concept_a)
            b = html_escape.escape(pair.concept_b)
            svg.append(f'<g><title>{a} / {b}: LOO cosine {score:.3f}</title><line x1="200" y1="{y}" x2="440" y2="{y}" stroke="#667eea" stroke-width="{2 + max(0, score) * 5}"/><circle cx="200" cy="{y}" r="6" fill="#8c9eff"/><circle cx="440" cy="{y}" r="6" fill="#bb9aff"/><text x="185" y="{y+5}" text-anchor="end" fill="#e0e0e0" font-size="14">{a}</text><text x="455" y="{y+5}" fill="#e0e0e0" font-size="14">{b}</text><text x="320" y="{y-8}" text-anchor="middle" fill="#a7b5ff" font-size="11">{score:.3f}</text></g>')
        svg.append('</svg>')
        html = html.replace("{{NETWORK_SVG}}", ''.join(svg) if pairs else 'No concept pairs available')
        
        # Inject data
        if self.map:
            map_json = json.dumps(self.map.to_dict())
            html = html.replace("{{MAP_DATA}}", map_json)
        else:
            html = html.replace("{{MAP_DATA}}", "{}")
        
        if self.identity:
            identity_json = json.dumps(self.identity.to_dict())
            html = html.replace("{{IDENTITY_DATA}}", identity_json)
        else:
            html = html.replace("{{IDENTITY_DATA}}", "{}")
        
        if output_path:
            Path(output_path).write_text(html)
            print(f"[AtlasVisualizer] Saved HTML to {output_path}")
        
        return html
    
    def _generate_html_template(self) -> str:
        """Generate the HTML template for visualization."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IOTA - Semantic Atlas</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem;
            border-bottom: 1px solid #2a2a4a;
        }
        .header h1 { 
            font-size: 2rem; 
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .header p { color: #888; font-size: 0.9rem; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        .card {
            background: #12121a;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 1.5rem;
        }
        .card h2 { 
            font-size: 1.1rem; 
            color: #667eea;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .metric { 
            display: flex; 
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid #2a2a4a;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #888; }
        .metric-value { font-weight: 600; color: #667eea; }
        .marker {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 3px solid #667eea;
        }
        .marker-title { font-weight: 600; margin-bottom: 0.25rem; }
        .marker-desc { font-size: 0.85rem; color: #888; }
        .marker-strength { 
            font-size: 0.8rem; 
            color: #667eea;
            margin-top: 0.5rem;
        }
        .network-container {
            max-height: 820px;
            overflow: auto;
            min-height: 400px;
            background: #0a0a0f;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .empty-state { color: #666; font-style: italic; }
        .model-badge {
            display: inline-block;
            background: #2a2a4a;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>IOTA Atlas</h1>
        <p>Mapping the Semantic Identity of AI Models</p>
    </div>
    <div class="container">
        <div id="model-info" class="card" style="margin-bottom: 2rem;">
            <h2>Model Information</h2>
            <div class="metric">
                <span class="metric-label">Model</span>
                <span class="metric-value" id="model-name">Loading...</span>
            </div>
            <div class="metric">
                <span class="metric-label">Timestamp</span>
                <span class="metric-value" id="model-timestamp">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Concepts Mapped</span>
                <span class="metric-value" id="n-concepts">-</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Concept Network</h2><p style="color:#888;font-size:0.85rem;margin-bottom:1rem">Lines connect the tested concept pairs; labels show LOO cosine. Layout does not represent semantic distance.</p>
                <div id="network" class="network-container">
                    {{NETWORK_SVG}}
                </div>
            </div>
            
            <div class="card">
                <h2>Unconfirmed Probe Candidates</h2>
                <div id="markers">
                    <span class="empty-state">No invariant markers yet</span>
                </div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 2rem;">
            <h2>Identity Summary</h2>
            <p id="identity-summary" style="color: #888; line-height: 1.6;">
                Loading...
            </p>
        </div>
    </div>
    
    <script>
        const mapData = {{MAP_DATA}};
        const identityData = {{IDENTITY_DATA}};
        
        // Populate model info
        if (mapData.model_name) {
            document.getElementById('model-name').textContent = mapData.model_name;
        }
        if (mapData.timestamp) {
            document.getElementById('model-timestamp').textContent = mapData.timestamp;
        }
        if (mapData.concept_directions) {
            document.getElementById('n-concepts').textContent = Object.keys(mapData.concept_directions).length;
        }
        
        // Populate markers
        if (identityData.candidate_markers && identityData.candidate_markers.length > 0) {
            const markersDiv = document.getElementById('markers');
            markersDiv.innerHTML = '';
            identityData.candidate_markers.slice(0, 5).forEach(marker => {
                const div = document.createElement('div');
                div.className = 'marker';
                div.innerHTML = `
                    <div class="marker-title">${marker.concepts_involved.join(' vs ')}</div>
                    <div class="marker-desc">${marker.description}</div>
                    <div class="marker-strength">LOO cosine: ${marker.strength.toFixed(3)} | Confidence: unmeasured</div>
                `;
                markersDiv.appendChild(div);
            });
        }
        
        // Identity summary
        if (identityData.identity_summary) {
            document.getElementById('identity-summary').textContent = identityData.identity_summary;
        }
        
    </script>
</body>
</html>"""
    
    def to_json_report(self, output_path: Optional[str] = None) -> dict:
        """Export a comprehensive JSON report."""
        report = {
            "report_type": "IOTA Semantic Atlas Report",
            "generated_at": self.map.timestamp if self.map else None,
        }
        
        if self.map:
            report["map"] = self.map.to_dict()
        
        if self.identity:
            report["identity"] = self.identity.to_dict()
        
        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2))
            print(f"[AtlasVisualizer] Saved JSON report to {output_path}")
        
        return report
    
    def summary_table(self) -> str:
        """Generate a summary table for console output."""
        if not self.map or not self.identity:
            return "No data to display"
        
        lines = [
            "=" * 60,
            "IOTA SEMANTIC ATLAS SUMMARY",
            "=" * 60,
            f"Model: {self.map.model_name}",
            f"Timestamp: {self.map.timestamp}",
            f"Concepts Mapped: {len(self.map.concept_directions)}",
            "-" * 60,
            "UNCONFIRMED PROBE CANDIDATES:",
            "-" * 60,
        ]
        
        for i, marker in enumerate(self.identity.candidate_markers[:5], 1):
            lines.append(
                f"{i}. {marker.concepts_involved[0]} vs {marker.concepts_involved[1]} "
                f"(LOO cosine: {marker.strength:.3f}, confidence: unmeasured)"
            )
        
        lines.extend([
            "-" * 60,
            "IDENTITY SUMMARY:",
            "-" * 60,
            self.identity.identity_summary or "No summary available",
            "=" * 60,
        ])
        
        return "\n".join(lines)
