/**
 * Unified UI system for boid simulations
 * Combines info display and parameter controls in a single integrated panel
 */

export class SimulationUI {
  constructor(params = {}, buttons = {}) {
    this.params = params;
    this.buttons = buttons;
    this.mainPanel = null;
    this.infoPanels = {};
    this.sliders = {};
    this.buttonElements = {};
    this.initUI();
  }

  initUI() {
    // Create main panel container
    this.mainPanel = document.createElement('div');
    this.mainPanel.id = 'simulation-ui';
    this.mainPanel.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      z-index: 10;
      color: #0f0;
      background: rgba(0, 0, 0, 0.8);
      padding: 12px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 12px;
      max-width: 300px;
      max-height: 90vh;
      overflow-y: auto;
      border: 1px solid #0f0;
    `;

    // Info section
    const infoSection = document.createElement('div');
    infoSection.id = 'info-section';
    infoSection.style.cssText = 'margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #0f0;';
    this.mainPanel.appendChild(infoSection);

    // Create info panels
    const infoPanelIds = ['info-app', 'info-fps', 'info-boids', 'info-step', 'info-gpu'];
    for (const id of infoPanelIds) {
      const panel = document.createElement('div');
      panel.id = id;
      panel.style.cssText = 'line-height: 1.5;';
      infoSection.appendChild(panel);
      this.infoPanels[id] = panel;
    }

    // Buttons section (if any buttons provided)
    if (Object.keys(this.buttons).length > 0) {
      const buttonsSection = document.createElement('div');
      buttonsSection.id = 'buttons-section';
      buttonsSection.style.cssText = 'margin: 12px 0; padding: 12px 0; border-bottom: 1px solid #0f0; display: flex; flex-direction: row; justify-content: center; gap: 4px;';
      this.mainPanel.appendChild(buttonsSection);

      for (const [key, config] of Object.entries(this.buttons)) {
        this.createButton(key, config, buttonsSection);
      }
    }

    // Controls section title
    if (Object.keys(this.params).length > 0) {
      const controlsTitle = document.createElement('div');
      controlsTitle.textContent = '⚙️ Parameters (Press P to toggle)';
      controlsTitle.style.cssText = 'font-weight: bold; margin-bottom: 8px; margin-top: 8px; text-align: center;';
      this.mainPanel.appendChild(controlsTitle);

      // Create controls for each parameter
      for (const [key, config] of Object.entries(this.params)) {
        this.createControl(key, config);
      }
    }

    document.body.appendChild(this.mainPanel);

    // Toggle panel with P key
    window.addEventListener('keydown', (event) => {
      if (event.key.toLowerCase() === 'p') {
        const controlsSection = this.mainPanel.querySelector('[id="controls-section"]');
        if (controlsSection) {
          controlsSection.style.display = 
            controlsSection.style.display === 'none' ? 'block' : 'none';
        }
      }
    });
  }

  createButton(key, config, parent) {
    const button = document.createElement('button');
    button.id = `btn-${key}`;
    // square icon-only button by default; fall back to label if no icon provided
    button.style.cssText = `
      margin: 4px 4px 4px 0;
      width: 30px;
      height: 30px;
      background: #0f0;
      color: #000;
      border: none;
      border-radius: 4px;
      font-family: monospace;
      font-weight: bold;
      cursor: pointer;
      transition: background 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    `;

    // icon support: static or state-based
    // set title for hover accessibility
    button.title = config.label || key;

    if (config.iconStates && config.iconStates.default) {
      button.innerHTML = config.iconStates.default;
    } else if (config.icon) {
      button.innerHTML = config.icon;
    } else {
      button.textContent = config.label || key;
    }

    button.addEventListener('mouseover', () => {
      button.style.background = '#0f0cc';
    });
    button.addEventListener('mouseout', () => {
      button.style.background = '#0f0';
    });

    button.addEventListener('click', () => {
      if (config.onClick) {
        config.onClick();
      }
    });

    parent.appendChild(button);
    this.buttonElements[key] = button;
  }

  createControl(key, config) {
    let controlsSection = this.mainPanel.querySelector('[id="controls-section"]');
    if (!controlsSection) {
      controlsSection = document.createElement('div');
      controlsSection.id = 'controls-section';
      this.mainPanel.appendChild(controlsSection);
    }

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
    controlsSection.appendChild(container);
  }

  setInfo(panelId, text) {
    if (this.infoPanels[panelId]) {
      this.infoPanels[panelId].textContent = text;
    }
  }

  getValue(key) {
    return this.params[key]?.value;
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

  setButtonState(key, state) {
    if (!this.buttonElements[key]) return;
    const btn = this.buttonElements[key];
    const cfg = this.buttons[key] || {};
    // change icon if iconStates provided
    if (cfg.iconStates && cfg.iconStates[state]) {
      btn.innerHTML = cfg.iconStates[state];
    }
    // simple visual feedback
    if (state === 'active' || state === 'playing') {
      btn.style.background = '#00ff00';
    } else if (state === 'inactive' || state === 'paused') {
      btn.style.background = '#ffff00';
    }
  }
}
