/**
 * Parameter controls for boid simulation
 * Provides UI sliders and getters/setters for simulation parameters
 */

export class SimulationControls {
  constructor(params) {
    this.params = params;
    this.controlPanel = null;
    this.sliders = {};
    this.initUI();
  }

  initUI() {
    // Create control panel
    this.controlPanel = document.createElement('div');
    this.controlPanel.id = 'control-panel';
    this.controlPanel.style.cssText = `
      position: fixed;
      bottom: 10px;
      left: 10px;
      z-index: 10;
      color: #0f0;
      background: rgba(0, 0, 0, 0.8);
      padding: 12px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 12px;
      max-width: 280px;
      max-height: 90vh;
      overflow-y: auto;
      border: 1px solid #0f0;
    `;

    // Add title
    const title = document.createElement('div');
    title.textContent = '⚙️ Parameters (Press P to toggle)';
    title.style.cssText = 'font-weight: bold; margin-bottom: 8px; text-align: center;';
    this.controlPanel.appendChild(title);

    // Create controls for each parameter
    for (const [key, config] of Object.entries(this.params)) {
      this.createControl(key, config);
    }

    document.body.appendChild(this.controlPanel);

    // Toggle panel with P key
    window.addEventListener('keydown', (event) => {
      if (event.key.toLowerCase() === 'p') {
        this.controlPanel.style.display = 
          this.controlPanel.style.display === 'none' ? 'block' : 'none';
      }
    });
  }

  createControl(key, config) {
    const container = document.createElement('div');
    container.style.cssText = 'margin: 6px 0; padding: 6px 0; border-bottom: 1px solid #044;';

    // Label
    const label = document.createElement('div');
    label.textContent = `${key}:`;
    label.style.cssText = 'font-weight: bold; margin-bottom: 3px;';
    container.appendChild(label);

    // Value display
    const valueDisplay = document.createElement('div');
    valueDisplay.style.cssText = 'color: #0f0; margin-bottom: 3px; font-size: 11px;';
    valueDisplay.textContent = `${config.value.toFixed(config.decimals || 1)}`;
    container.appendChild(valueDisplay);

    // Slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = config.min;
    slider.max = config.max;
    slider.step = config.step || (config.max - config.min) / 100;
    slider.value = config.value;
    slider.style.cssText = `
      width: 100%;
      cursor: pointer;
      height: 5px;
      accent-color: #0f0;
    `;

    slider.addEventListener('input', (e) => {
      const newValue = parseFloat(e.target.value);
      config.value = newValue;
      valueDisplay.textContent = `${newValue.toFixed(config.decimals || 1)}`;
      if (config.onChange) {
        config.onChange(newValue);
      }
    });

    container.appendChild(slider);
    this.sliders[key] = { slider, config };
    this.controlPanel.appendChild(container);
  }

  getValue(key) {
    return this.params[key].value;
  }

  setValue(key, value) {
    if (this.params[key]) {
      this.params[key].value = value;
      if (this.sliders[key]) {
        this.sliders[key].slider.value = value;
      }
    }
  }

  getAll() {
    const result = {};
    for (const [key, config] of Object.entries(this.params)) {
      result[key] = config.value;
    }
    return result;
  }
}
