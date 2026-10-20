/* Multiples are measured against each active direction's own 1x vector. */
(() => {
  function combine(directions, coefficients, scale, mode = 'norm') {
    const native = Array(directions[0].native_direction.length).fill(0), projected = [0, 0, 0];
    const active = directions.filter(c => (coefficients[c.id] || 0) !== 0);
    for (const c of active) {
      const value = coefficients[c.id];
      c.native_direction.forEach((v, i) => native[i] += value * v);
      c.direction.forEach((v, i) => projected[i] += value * v);
    }
    const limit = Math.max(0, ...active.map(c => Math.abs(coefficients[c.id])));
    const cap = scale * Math.max(1, limit), raw = Math.hypot(...native), projections = {};
    for (const c of active) {
      let dot = 0, squared = 0;
      c.native_direction.forEach((v, i) => { dot += native[i] * v; squared += v * v; });
      if (!(squared > 0) || !Number.isFinite(squared)) {
        if (mode === 'direction_projection') throw new Error('已启用方向的 1× 向量必须有限且非零。');
        continue;
      }
      projections[c.id] = dot / squared;
    }
    const peak = Math.max(0, ...Object.values(projections).map(Math.abs));
    let factor = 1;
    if (mode === 'norm') factor = Math.min(1, cap / Math.max(raw, 1e-12));
    else if (mode === 'direction_projection' && peak > limit) factor = limit / peak;
    return {
      native_direction: native.map(v => v * factor), projected_direction: projected.map(v => v * factor),
      raw_norm: raw, norm: raw * factor, cap, norm_cap: mode === 'norm' ? cap : null,
      standard_cap: scale, regularization: mode, total_strength_cap_enabled: mode !== 'none',
      cap_factor: factor, projection_limit: limit, raw_projections: projections,
      projections: Object.fromEntries(Object.entries(projections).map(([id, v]) => [id, v * factor])),
      max_raw_projection: peak, max_projection: peak * factor,
      limiting_axis: peak ? Object.keys(projections).reduce((a, b) => Math.abs(projections[a]) >= Math.abs(projections[b]) ? a : b) : null,
      effective_coefficients: Object.fromEntries(Object.entries(coefficients).map(([id, v]) => [id, v * factor])),
      weights_updated: false
    };
  }
  window.IOTARegularization = {combine};
})();
